# 精读 Megatron 源码（7）：数据并行完全精读——DDP 缓冲、Bucket 与 finalize_model_grads

> 源文件：`megatron/core/distributed/distributed_data_parallel.py`、`param_and_grad_buffer.py`、`finalize_model_grads.py`

---

## 1. 为什么不用 torch.nn.parallel.DistributedDataParallel

PyTorch 自带的 `DDP` 对于标准模型已经足够，但 Megatron 的训练场景有几个特殊需求，导致必须自己实现 DDP：

| 原因 | 说明 |
|------|------|
| **混合精度 main_grad** | Megatron 使用 BF16 参数但 FP32 主梯度（`main_grad`），torch DDP 不支持这种异构梯度缓冲区 |
| **TP+PP+DP 嵌套** | torch DDP 对 AllReduce 的触发时机不适合流水线并行（PP 内的 microbatch 循环需要精确控制何时同步） |
| **Distributed Optimizer** | torch DDP 默认 AllReduce，无法切换为 ReduceScatter+AllGather（Distributed Optimizer 所需） |
| **Bucket 粒度控制** | torch DDP 的 bucket 是自动划分的，无法与 PP rank 0 only 的规则联动（见第 3 节） |
| **Gradient accumulation fusion** | Megatron 需要把梯度计算与累加融合为单个 CUDA kernel |
| **CP 组搭载 DP** | CP 组的梯度需要与 DP 梯度一起同步（dp-cp 联合 AllReduce），torch DDP 的钩子机制难以扩展 |

---

## 2. ParamAndGradBuffer 的设计

```
文件: megatron/core/distributed/param_and_grad_buffer.py
```

所有参数的梯度被连续分配在一块大内存缓冲区（`_ParamAndGradBuffer`）里，而不是每个参数单独一块。这有几个好处：

1. **减少小 tensor 通信开销**：把数千个参数的梯度合并成一个或几个大 tensor 再 AllReduce，通信效率大幅提升。
2. **减少内存碎片**：连续内存有利于 CUDA 内存分配器和 NCCL 的 DMA 传输。
3. **支持异步通信**：可以在一个 bucket 就绪时立即发起通信，与后续 bucket 的梯度计算重叠（`overlap_grad_reduce`）。

### 2.1 Bucket 划分规则

```python
# megatron/core/distributed/param_and_grad_buffer.py
# bucket_size 默认由 ddp_config.bucket_size 控制（通常 40MB 或 125MB）
```

参数按照**注册顺序的逆序**（即反向传播的顺序）填充 bucket：当一个 bucket 的所有参数梯度都就绪时，触发该 bucket 的 AllReduce/ReduceScatter。

**PP rank 0 only 规则**：当 `overlap_grad_reduce=True` 时，只有 PP rank 0（流水线的第一个 stage）的 bucket 才启用异步重叠。其他 PP stage 等待所有 microbatch 完成后再同步梯度。原因是非 rank 0 的 stage 在流水线 warmup/cooldown 期间梯度尚未完整累积，提前触发会导致梯度错误。

```python
# megatron/core/distributed/distributed_data_parallel.py
# 源码中的相关判断
if self.ddp_config.overlap_grad_reduce:
    # Only overlap on pp rank 0 buckets
    if parallel_state.get_pipeline_model_parallel_rank() == 0:
        bucket_group.start_grad_sync()
```

### 2.2 main_grad 与 FP32 梯度

```python
# _ParamAndGradBuffer 中，当 use_distributed_optimizer=False 时：
# grad_data 是 BF16 的梯度缓冲区
# 但参数的 .main_grad 属性指向一个 FP32 缓冲区

# 在 distributed_data_parallel.py 中：
if self.ddp_config.use_distributed_optimizer:
    # DistOpt 情况：使用 ReduceScatter，梯度在 FP32 中累积
    ...
else:
    # 标准 DDP：main_grad 是 FP32，AllReduce 后拷贝给参数的 .grad
    param.main_grad = grad_buffer_fp32_view
```

---

## 3. overlap_grad_reduce 工作原理

```python
# distributed_data_parallel.py
class DistributedDataParallel:
    def __init__(self, ...):
        self.ddp_config = ddp_config
        # 为每个参数注册反向传播钩子
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_param_hook(param, ...)
                )
```

每次反向传播中，当某个参数的梯度计算完成时，钩子函数被调用：

```python
def param_hook(param):
    # 通知 bucket 该参数的梯度已就绪
    bucket_group = self.param_to_bucket_group[param]
    bucket_group.register_grad_ready(param)

    # 当 bucket 内所有参数都就绪时，触发异步通信
    if bucket_group.all_params_ready():
        bucket_group.start_grad_sync()  # 异步 AllReduce 或 ReduceScatter
```

`finish_grad_sync` 在 `finalize_model_grads` 中调用，等待所有异步通信完成：

```python
def finish_grad_sync(self, force_all_reduce=False):
    """Wait for all bucket's allreduce/reducescatter to finish."""
    for bucket_group in self.bucket_groups:
        bucket_group.finish_grad_sync()
```

---

## 4. Distributed Optimizer 的内存节省

标准 Adam 优化器每个参数需要：
- 1 份 FP32 主参数（master param）
- 1 份 FP32 梯度
- 2 份 FP32 优化器状态（m, v）

共 4 × sizeof(FP32) × N_params = 16 bytes/参数。

Distributed Optimizer 把这 4 份都按 DP 维度分片，每个 rank 只负责 N_params/DP 个参数的优化：

```
内存占用 = 16 × N_params / DP  bytes（优化器状态 + master param + grad）
         + 2 × N_params            bytes（BF16 模型参数，每个 rank 完整复制）
```

对于 DP=8 的情况，优化器内存从 16N 降到 2N + 16N/8 = 4N，减少约 4 倍。

### Distributed Optimizer 的 ReduceScatter + AllGather 流程

```
反向传播结束时：
  ReduceScatter → 每个 rank 得到 1/DP 的梯度总和

优化器更新（每个 rank 更新自己负责的 1/DP 参数）

正向传播开始前：
  AllGather → 每个 rank 获得完整的更新后参数
```

这比 AllReduce 的通信量多了 AllGather 这一步，但允许把梯度和参数的通信与计算重叠。

---

## 5. dp-cp 组的含义

在启用 Context Parallel 时，CP 组内的 rank 持有相同的权重，因此需要跨 CP rank 同步梯度。Megatron 把 CP 维度"搭"在 DP 上：

```python
# parallel_state.py
_DATA_PARALLEL_GROUP_WITH_CP = None  # dp-cp 联合组

# finalize_model_grads.py
dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
```

`with_context_parallel=True` 返回的是 `dp-cp` 联合组，它的大小是 `DP × CP`。在 `finalize_model_grads` 中，梯度同步使用这个联合组，确保跨 CP rank 的梯度也被正确平均。

**直觉**：CP 把一个序列切成 CP 份，每份由不同的 GPU 处理，但权重是相同的。反向传播产生的梯度来自不同的序列片段，需要全部加起来才是完整的梯度。CP 组就相当于额外的 DP 维度，所以直接用 `dp-cp` 联合做 AllReduce 最自然。

---

## 6. finalize_model_grads 八步流程

```python
# megatron/core/distributed/finalize_model_grads.py  第 494-614 行
def finalize_model_grads(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
```

| 步骤 | 函数/操作 | 说明 |
|------|---------|------|
| 1 | `model_chunk.finish_grad_sync()` | 等待所有异步 AllReduce/ReduceScatter 完成 |
| 2 | `_allreduce_conditional_embedding_grads()` | 条件 embedding（DiT 等多模态模型的时间步嵌入）跨 PP 同步 |
| 3 | `_allreduce_non_tensor_model_parallel_grads()` | SP LayerNorm 梯度跨 TP 做 AllReduce（sum），以及 `average_gradients_across_tp_domain=True` 的参数做 AVG |
| 4 | `_allreduce_word_embedding_grads()` | Word embedding 梯度在 first+last PP stage 之间同步（通过 embedding group） |
| 5 | `_allreduce_position_embedding_grads()` | 位置编码梯度在 PP first stage 内同步 |
| 6 | `_update_router_expert_bias()` | MoE router 的 expert bias 更新（`moe_router_enable_expert_bias=True` 时） |
| 7 | `reset_model_temporary_tensors()` | 清理临时张量（例如 FlexTron router 梯度） |
| 8 | 梯度归一化（`num_tokens`） | 若 `num_tokens` 不为 None，把所有梯度除以总 token 数（per-token loss 归一化） |

### 步骤 3 的详细逻辑：SP LayerNorm 梯度

在 SP 模式下，LayerNorm 的参数（`weight`、`bias`）不是 TP 切分的，但梯度是从序列分片的激活反向传播而来，所以每个 TP rank 只有梯度的一部分：

```
LayerNorm weight grad（每个 TP rank）: sum(dL/dγ for tokens in my_sequence_slice)

需要 AllReduce → sum all TP ranks → 完整的 dL/dγ
```

```python
# finalize_model_grads.py  第 452-458 行
elif (config.sequence_parallel and getattr(param, "sequence_parallel", False)) or (
    config.qk_layernorm and ("q_layernorm" in name or "k_layernorm" in name)
):
    grad_attr = _get_main_grad_attr(param)
    grad = getattr(param, grad_attr)
    params_sum.append(param)
    grads_sum.append(grad.data)
```

被标记为 `sequence_parallel=True` 的参数（通常是 LayerNorm 和 QK LayerNorm 的参数）进入 `params_sum` 列表，最后做 SUM AllReduce。

### 步骤 4 的详细逻辑：Word Embedding 梯度同步

```python
# finalize_model_grads.py  第 164-199 行
def _allreduce_word_embedding_grads(model, config, embd_group, pp_group):
    """All-reduce word-embedding gradients across the first and last PP stages."""
    if embd_group is None:
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
```

Word embedding 在 PP 的 first stage 和 last stage 共享权重（`share_embeddings_and_output_weights=True`），但两个 stage 上的梯度是独立计算的。`_allreduce_word_embedding_grads` 把这两个 stage 的梯度相加，确保权重更新一致。

如果 `share_embeddings_and_output_weights=False`（词表和 lm_head 用独立权重），则跳过这一步。

---

## 7. 梯度归一化（per-token）

```python
# finalize_model_grads.py  第 596-614 行
if num_tokens is not None:
    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor.
    # this number will be the total number of non-padded tokens in the global batch.
    scaling_factor = 1.0 / num_tokens
    for model_chunk in model:
        for param in get_attr_wrapped_model(model_chunk, 'parameters')():
            if param.requires_grad:
                grad_attr = _get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                if grad is not None:
                    grad.mul_(scaling_factor)
```

这一步把每个梯度都乘以 `1/num_tokens`，实现 per-token 的 loss 归一化。`num_tokens` 是全局 batch 中所有非 padding token 的数量，需要在 loss 函数中统计并通过 `forward_backward_func` 的返回值传出。

---

## 8. 静默错误梯度排查清单

数据并行最难调试的是**静默梯度错误**：训练可以运行，但模型收敛曲线异常（比 baseline 慢、或震荡、或 loss 不降）。以下是系统排查流程：

### 8.1 验证进程组成员一致

```python
# 在每个 rank 上打印 DP 组成员
dp_group = parallel_state.get_data_parallel_group()
dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
print(f"[rank {torch.distributed.get_rank()}] DP group: {dp_ranks}")
```

如果不同 rank 的 DP 组成员不一致，说明进程组构造有 bug。

### 8.2 比较 DP=1 与 DP=N 的梯度

```python
# 步骤 1：用 DP=1 运行一步，记录每个参数的梯度
grads_dp1 = {name: param.grad.clone() for name, param in model.named_parameters()}

# 步骤 2：用 DP=N 运行一步（相同数据），记录 rank 0 的梯度
grads_dpN = {name: param.grad.clone() for name, param in model.named_parameters()}

# 步骤 3：比较（应该相等，误差在 FP16/BF16 精度范围内）
for name in grads_dp1:
    diff = (grads_dp1[name] - grads_dpN[name]).abs().max()
    if diff > 1e-3:
        print(f"MISMATCH: {name}, max_diff={diff}")
```

### 8.3 检查 SP LayerNorm 梯度是否被 AllReduce

```python
# 验证 LayerNorm weight 的 sequence_parallel 属性
for name, param in model.named_parameters():
    if 'layer_norm' in name or 'layernorm' in name:
        print(f"{name}: sequence_parallel={getattr(param, 'sequence_parallel', False)}")
```

如果 `sequence_parallel=False` 但 `config.sequence_parallel=True`，说明 LayerNorm 的属性没有正确设置，梯度不会被 AllReduce，导致 DP replica 之间的 LayerNorm 参数逐渐发散。

### 8.4 检查 Embedding 权重同步

```python
# 在 finalize_model_grads 之后，验证 first stage 和 last stage 的 embedding weight 梯度相同
if parallel_state.is_pipeline_first_stage():
    emb_grad = model.language_model.embedding.word_embeddings.weight.main_grad
    print(f"[first stage] emb grad norm: {emb_grad.norm()}")

if parallel_state.is_pipeline_last_stage():
    output_grad = model.language_model.output_layer.weight.main_grad
    print(f"[last stage] output grad norm: {output_grad.norm()}")
```

两者应该相等（如果 `share_embeddings_and_output_weights=True`）。

### 8.5 验证梯度 AllReduce 已完成

```python
# 在优化器 step 之前，强制同步所有通信
torch.cuda.synchronize()
for model_chunk in model:
    model_chunk.finish_grad_sync()
# 然后再做梯度裁剪和优化器步
```

---

## 9. Distributed Optimizer 内存 O(params/DP) 的精确计算

设模型参数量为 `P`，数据类型混合精度（BF16 参数 + FP32 optimizer state）：

```
标准 Adam（不使用 DistOpt）：
  BF16 参数：      2P  bytes
  FP32 主参数：    4P  bytes
  FP32 梯度：      4P  bytes
  Adam m (FP32)：  4P  bytes
  Adam v (FP32)：  4P  bytes
  总计：           18P bytes（约 18 bytes/参数）

Distributed Optimizer（DP=8）：
  BF16 参数（每 rank 完整复制）：  2P  bytes
  FP32 主参数（每 rank 1/8）：     4P/8 = P/2  bytes
  FP32 梯度（每 rank 1/8）：       P/2  bytes
  Adam m（每 rank 1/8）：          P/2  bytes
  Adam v（每 rank 1/8）：          P/2  bytes
  总计：                           2P + 2P = 4P  bytes（节省 ~4.5x）
```

这就是在大规模训练中 Distributed Optimizer 几乎必不可少的原因。

---

## 10. 练习题

**题目 1**：为什么 `finalize_model_grads` 中 SP LayerNorm 梯度做 SUM AllReduce 而不是 AVG AllReduce？提示：考虑 SP 下每个 rank 计算梯度的方式。

**题目 2**：在 TP=4, DP=2, CP=2, PP=1 的配置下：
1. `dp-cp` 组的大小是多少？
2. `dp` 组的大小是多少？
3. `finalize_model_grads` 中 SP LayerNorm 梯度使用哪个组做 AllReduce？
4. Word embedding 梯度不需要 AllReduce，为什么？（提示：PP=1）

**题目 3**：阅读 `param_and_grad_buffer.py` 中 `_ParamAndGradBucket` 类，解释 `gradient_scaling_factor` 字段的作用，以及它在 MoE 和非 MoE 模型中的值分别是多少。

**题目 4**：设 global_batch_size=512，micro_batch_size=4，PP=4，DP=2，那么 `num_microbatches=512/(4×2)/4=32`。在 `finalize_model_grads` 最后一步梯度归一化中，`num_tokens` 应该是多少（假设序列长度 S=2048，无 padding）？这个值是在哪里计算并传入的？

---

## 12. DistributedDataParallel 的初始化流程

```python
# megatron/core/distributed/distributed_data_parallel.py
class DistributedDataParallel(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        ...
    ):
```

初始化的主要步骤：

**步骤 1：参数分类**

把模型参数分为两类：
- `params_with_grad`：需要梯度的参数（通常是所有可训练参数）
- 专家参数（`is_expert=True`）与非专家参数分别处理，因为它们使用不同的通信组

**步骤 2：构造 `_ParamAndGradBuffer`**

```python
self.buffers = {}
for dtype, params in params_by_dtype.items():
    self.buffers[dtype] = _ParamAndGradBuffer(
        ddp_config=ddp_config,
        param_dtype=dtype,
        params=params,
        data_parallel_group=data_parallel_group,
        bucket_size=ddp_config.bucket_size,
        ...
    )
```

每种数据类型（BF16、FP32）各有一个 buffer。Buffer 内部按参数大小和 bucket_size 划分 `_ParamAndGradBucket`。

**步骤 3：注册反向传播钩子**

```python
for param in self.module.parameters():
    if param.requires_grad:
        param.register_post_accumulate_grad_hook(
            self._make_param_hook(param, self.param_to_buffer)
        )
```

`register_post_accumulate_grad_hook` 是 PyTorch 2.1+ 的新 API，在每个参数的梯度累积完成后触发（包括多个 microbatch 的 `loss.backward()` 后）。

---

## 13. no_sync 上下文管理器的作用

在流水线并行中，有 `num_microbatches` 个 microbatch 需要依次执行。在最后一个 microbatch 之前，梯度不应该立即 AllReduce（因为还有梯度没算完）。Megatron 用 `no_sync` 上下文来实现这一点：

```python
# schedules.py  第 2248-2262 行
def disable_grad_sync():
    """Disable asynchronous grad reductions"""
    nonlocal no_sync_context
    if no_sync_context is None:
        no_sync_context = no_sync_func()
        no_sync_context.__enter__()

def enable_grad_sync():
    """Enable asynchronous grad reductions"""
    nonlocal no_sync_context
    if no_sync_context is not None:
        no_sync_context.__exit__(None, None, None)
        no_sync_context = None

disable_grad_sync()  # 开始时禁用
```

在 1F1B 稳态最后一个 microbatch 的 backward 之前启用：

```python
# megatron/core/pipeline_parallel/schedules.py  第 2420-2422 行
if num_warmup_microbatches == 0 and last_iteration:
    if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
        enable_grad_sync()
```

`no_sync_func` 通常是 `DDP.no_sync()`，它暂时禁用 bucket 触发的异步 AllReduce，让多个 microbatch 的梯度在本地累积后再统一同步。

---

## 14. 梯度裁剪（grad clip）与 DistOpt 的交互

```python
# megatron/training/training.py（训练循环中）
optimizer.step()
# 其中 optimizer.step() 内部会调用 finalize_model_grads 和梯度裁剪
```

在 Distributed Optimizer 中，梯度裁剪（`clip_grad_norm`）需要特别处理：

1. ReduceScatter 后每个 rank 只有部分梯度
2. 计算全局梯度 L2 范数需要先做 AllReduce（sum of squares）
3. 然后每个 rank 根据全局范数裁剪自己负责的梯度分片

```python
# megatron/core/optimizer/optimizer.py
def clip_grad_norm(parameters, max_norm, norm_type=2.0, ...):
    # 计算局部 norm^2
    total_norm_sq = sum(p.grad.norm()**2 for p in local_params)
    # AllReduce 得到全局 norm^2
    torch.distributed.all_reduce(total_norm_sq, group=dp_group)
    total_norm = total_norm_sq.sqrt()
    # 裁剪
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in local_params:
            p.grad.mul_(clip_coef)
```

---

## 15. 小结

Megatron 的数据并行实现不是简单地套用 torch DDP，而是针对多维并行的特殊需求做了深度定制：

1. **`ParamAndGradBuffer`**：连续内存布局 + bucket 机制，实现梯度的异步 AllReduce/ReduceScatter，只有 PP rank 0 的 bucket 才启用 overlap。
2. **`main_grad` FP32**：参数是 BF16，梯度累积在 FP32 缓冲区，避免梯度下溢。
3. **Distributed Optimizer**：把 AllReduce 换为 ReduceScatter + AllGather，每个 rank 只持有 1/DP 的优化器状态，内存从 O(18P) 降到 O(2P + 2P/DP)。
4. **dp-cp 联合组**：CP 的梯度归约与 DP 合并，通过 `get_data_parallel_group(with_context_parallel=True)` 访问。
5. **`finalize_model_grads` 八步**：完成 DDP 之外的所有梯度同步工作——SP LayerNorm、embedding 对齐、位置编码、MoE expert bias、梯度归一化。

理解了这五个机制，就能系统地理解 Megatron 为什么比 torch DDP 更适合超大模型的训练，也能快速定位梯度相关的 bug。
