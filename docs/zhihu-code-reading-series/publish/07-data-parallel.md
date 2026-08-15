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
# megatron/core/distributed/distributed_data_parallel.py  第 68-72 行
if ddp_config.bucket_size is None:
    ddp_config.bucket_size = max(40000000, 1000000 * dp_group.size())
# Set bucket_size to infinity if overlap_grad_reduce is False.
if not ddp_config.overlap_grad_reduce:
    ddp_config.bucket_size = None
```

默认 bucket 大小 = `max(40MB, 1MB × DP_size)`。DP=64 时 bucket_size=64 MB，这是因为大规模集群下 NCCL ring-reduce 的分块需要足够大才能保持带宽效率。

参数按照**注册顺序的逆序**（即反向传播的顺序）填充 bucket：当一个 bucket 的所有参数梯度都就绪时，触发该 bucket 的 AllReduce/ReduceScatter。

**PP rank 0 only 规则**：

```python
# megatron/core/distributed/distributed_data_parallel.py  第 101-106 行
if disable_bucketing or pp_rank > 0:
    self.bucket_size = None  # 禁用 bucket，不做异步重叠
```

只有 PP rank 0（流水线的第一个 stage）的 bucket 才启用异步重叠。其他 PP stage 等待所有 microbatch 完成后再同步梯度。原因：非 rank 0 的 stage 在流水线 warmup/cooldown 期间梯度尚未完整累积，提前触发会导致梯度错误。

### 2.2 main_grad 与 FP32 梯度

```python
# _ParamAndGradBuffer 中：
# BF16 参数存储在 param.data
# FP32 梯度存储在 param.main_grad（指向 grad_buffer 的 FP32 视图）

# 反向传播钩子（_make_backward_post_hook）中：
if param.grad is not None and not param.grad_added_to_main_grad:
    param.main_grad.add_(param.grad.data)  # 把 BF16 梯度累加到 FP32 main_grad
param.grad = None  # 清空 param.grad，节省内存
```

---

## 3. finish_grad_sync / register hooks 的工作机制

### 3.1 梯度钩子注册

DDP 在初始化时为每个参数注册反向传播后钩子（`_make_backward_post_hook`）：

```python
# megatron/core/distributed/distributed_data_parallel.py  第 349-373 行
for param in self.module.parameters():
    if param.requires_grad:
        # expand 获取 AccumulateGrad 节点
        param_tmp = param.expand_as(param)
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        grad_acc.register_hook(self._make_backward_post_hook(param))
        self.grad_accs.append(grad_acc)
```

这里用的是 `AccumulateGrad.register_hook`（比 `param.register_hook` 更早触发，在 autograd 图节点级别），确保每次 `.backward()` 完成后立即触发钩子。

### 3.2 backward_post_hook 的执行流程

```python
# megatron/core/distributed/distributed_data_parallel.py  第 449-478 行
def hook(*unused):
    if param in self.param_to_bucket_group:
        # 1. 把 param.grad 累加到 main_grad（FP32）
        if param.grad is not None and not param.grad_added_to_main_grad:
            param.main_grad.add_(param.grad.data)
        param.grad = None   # 释放 BF16 梯度缓冲区

        # 2. 通知 bucket_group 该参数的梯度已就绪
        if self.ddp_config.overlap_grad_reduce:
            self.param_to_bucket_group[param].register_grad_ready(
                param, self.force_all_reduce
            )
            # 当 bucket 内所有参数都就绪时，
            # register_grad_ready 内部触发 start_grad_sync()（异步通信）
```

### 3.3 finish_grad_sync 的调用路径

```python
# megatron/core/distributed/distributed_data_parallel.py  第 545-555 行
def finish_grad_sync(self, force_all_reduce=False):
    """等待所有异步 AllReduce/ReduceScatter 完成"""
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
```

调用时序：

```
backward() → 各参数的 backward_post_hook →
  → bucket 就绪 → start_grad_sync（异步 AllReduce/RS） →
  → finalize_model_grads() →
      model_chunk.finish_grad_sync() →  ← 在此等待异步通信完成
      _allreduce_non_tensor_model_parallel_grads()
      _allreduce_word_embedding_grads()
      ...
```

---

## 4. overlap_param_gather

当 `ddp_config.overlap_param_gather=True` 时，Distributed Optimizer 的参数 AllGather（从分片参数恢复完整参数）可以与 forward 计算重叠：

```python
# megatron/core/distributed/distributed_data_parallel.py  第 295-306 行
if self.ddp_config.overlap_param_gather:
    for bucket_groups in [self.bucket_groups, self.expert_parallel_bucket_groups]:
        num_bucket_groups = len(bucket_groups)
        for i in range(1, num_bucket_groups):
            # 设置 next_param_gather_bucket_group 链表
            # AllGather 在 forward pre-hook 中按 bucket 顺序触发
            bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                bucket_groups[num_bucket_groups - i - 1]
            )
```

`enable_forward_pre_hook()` 会在每个 sub-module 的 forward 开始前触发，确保该 module 需要的参数 AllGather 已完成：

```python
# megatron/core/distributed/distributed_data_parallel.py  第 384-394 行
def enable_forward_pre_hook(self):
    for module in self.module.modules():
        self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
            self._make_forward_pre_hook()
        )
```

这使得 AllGather 与上一层的 forward 计算重叠，理论上可以完全隐藏 AllGather 延迟（在深度足够的网络中）。

---

## 5. Distributed Optimizer 的内存节省

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

## 6. 内存计算 Worked Example：7B 参数 BF16 + Adam

模型：LLaMA-7B（P=7B 参数），数据精度：BF16，优化器：Adam，DP=8，不使用 DistOpt

### 6.1 不使用 Distributed Optimizer（DP=8）

```
每个 rank 的内存（字节）：
  BF16 参数：   7B × 2 = 14 GB
  FP32 主参数： 7B × 4 = 28 GB
  FP32 梯度：   7B × 4 = 28 GB  （main_grad 缓冲区）
  Adam m：      7B × 4 = 28 GB
  Adam v：      7B × 4 = 28 GB
  ─────────────────────────────
  总计（非 DistOpt）： 126 GB    ← 需要多卡 HBM
```

### 6.2 使用 Distributed Optimizer（DP=8）

```
每个 rank 的内存（字节）：
  BF16 参数（完整副本）：  7B × 2  = 14 GB
  FP32 主参数（1/8）：     7B/8 × 4 = 3.5 GB
  FP32 梯度（1/8）：       7B/8 × 4 = 3.5 GB
  Adam m（1/8）：           7B/8 × 4 = 3.5 GB
  Adam v（1/8）：           7B/8 × 4 = 3.5 GB
  ─────────────────────────────
  总计（DistOpt, DP=8）：  28 GB    ← 节省约 4.5×
```

关键洞察：即使 DistOpt 把优化器状态分片，**BF16 参数本身还是每个 rank 完整保存**（因为 forward 需要完整参数）。只有 FP32 的优化器状态和 FP32 主参数是分片的。

### 6.3 BF16 训练中 FP32 主参数的作用

BF16（2 bytes/param）的精度约 3.3 位有效小数位，而 Adam 的参数更新步幅（learning rate × momentum / variance）通常远小于 BF16 的精度阈值。如果直接在 BF16 上更新，微小的梯度步长会被舍入为零，导致参数停止更新（"梯度消失"的精度版本）。FP32 主参数（4 bytes/param）有 7 位有效小数位，足以保留微小更新。

---

## 7. dp-cp 组的含义

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

## 8. finalize_model_grads 完整流程详解

```python
# megatron/core/distributed/finalize_model_grads.py  第 494-614 行
def finalize_model_grads(model, num_tokens=None, pg_collection=None, force_all_reduce=False):
```

### 步骤 1：finish_grad_sync——等待异步通信完成

```python
for model_chunk in model:
    model_chunk.finish_grad_sync(force_all_reduce=force_all_reduce)
```

- 所使用的进程组：`intra_dp_cp_group`（非专家参数），`intra_expt_dp_group`（专家参数）
- 操作：AllReduce 或 ReduceScatter（取决于 `use_distributed_optimizer`）
- 时序依赖：此步骤等待所有 backward_post_hook 触发的异步通信全部完成

### 步骤 2：_allreduce_conditional_embedding_grads

```python
_allreduce_conditional_embedding_grads(model, config, pp_group)
```

- 所使用的进程组：`pp_group`（PP 组）
- 用途：DiT（Diffusion Transformer）等模型中，时间步嵌入（timestep embedding）的梯度需要在所有 PP stage 间同步

### 步骤 3：_allreduce_non_tensor_model_parallel_grads（SP LayerNorm 梯度）

```python
_allreduce_non_tensor_model_parallel_grads(model, config, tp_group)
```

- 所使用的进程组：`tp_group`（TP 组）
- 操作：SUM AllReduce（不是 AVG）
- 哪些参数受影响：`param.sequence_parallel=True` 的参数（通常是 LayerNorm 的 weight/bias）

在 SP 模式下，LayerNorm 的参数（`weight`、`bias`）不是 TP 切分的，但梯度是从序列分片的激活反向传播而来，所以每个 TP rank 只有梯度的一部分：

```
LayerNorm weight grad（每个 TP rank）: sum(dL/dγ for tokens in my_sequence_slice)

需要 AllReduce → sum all TP ranks → 完整的 dL/dγ
```

**为什么是 SUM 而不是 AVG？** 因为每个 TP rank 的梯度是不同 token 的贡献之和，不是同一梯度的复制。汇总的正确语义是"加起来"（SUM），而不是"平均"（AVG）。与之对比，DP 的梯度 AllReduce 是 AVG（每个 DP replica 处理的是不同数据，梯度是独立估计，需要平均）。

### 步骤 4：_allreduce_word_embedding_grads（Word Embedding 梯度同步）

```python
_allreduce_word_embedding_grads(model, config, embd_group, pp_group)
```

- 所使用的进程组：`embd_group`（embedding 组，只包含 PP first 和 last stage 的 rank）
- 触发条件：PP > 1 且 `share_embeddings_and_output_weights=True`

```python
# finalize_model_grads.py  第 164-201 行
def _allreduce_word_embedding_grads(model, config, embd_group, pp_group):
    """All-reduce word-embedding gradients across the first and last PP stages."""
    if embd_group is None:
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
```

**为什么 first + last stage 需要特殊同步？**
- first stage 的 input embedding（`word_embeddings.weight`）在 forward 时把 token IDs 映射为 embedding 向量，并通过 PP chain 传播到 last stage。
- last stage 的 lm_head（`output_layer.weight`）与 input embedding 共享权重（tied weights）。
- 两处的梯度来自不同计算路径（forward embedding 的梯度和 lm_head 的梯度），必须相加才是完整的 dL/d(embedding_weight)。
- `_allreduce_word_embedding_grads` 通过 embedding_group（只含 first+last stage）的 AllReduce 完成这个相加。

如果 `share_embeddings_and_output_weights=False`，则跳过此步骤。

### 步骤 5：_allreduce_position_embedding_grads

```python
_allreduce_position_embedding_grads(model, config, pos_emb_group, pp_group)
```

- 所使用的进程组：`pos_emb_group`（位置编码组，只含 PP first stage）
- 用途：当位置编码 weight 跨 PP 共享时（非典型场景），同步其梯度

### 步骤 6-7：MoE Expert Bias 更新 / FlexTron router 梯度

```python
if config.moe_router_enable_expert_bias:
    _update_router_expert_bias(model, config, tp_dp_cp_group=tp_dp_cp_group)
reset_model_temporary_tensors(config, model)
```

- 所使用的进程组：`tp_dp_cp_group`（TP+DP+CP 联合组）
- 用途：MoE load balancing 的 expert bias 更新（类似 DeepSeek MoE 的 dynamic expert routing 修正）

### 步骤 8：梯度归一化（per-token loss）

当 loss 按 token 平均（或按非 padding token 计数）时，反向得到的梯度还要统一除以全局 `num_tokens`，否则不同 DP/PP 布局下有效学习率会漂。

```python
# megatron/core/distributed/finalize_model_grads.py  第 595-614 行
if num_tokens is not None:
    # 1. num_tokens 只在 PP last stage 有值，先 broadcast 到全 PP 组
    last_rank = get_pp_last_rank(pp_group)
    torch.distributed.broadcast(num_tokens, src=last_rank, group=pp_group)

    # 2. 跨 DP 组 AllReduce（sum 出全局 token 数）
    torch.distributed.all_reduce(num_tokens, group=dp_cp_group)

    # 3. 梯度 ÷ num_tokens（等价于每个参数 grad *= 1/num_tokens）
    safe_num_tokens = torch.clamp(num_tokens, min=1)
    scaling = 1.0 / safe_num_tokens
    for model_chunk in model:
        model_chunk.scale_gradients(scaling)
```

`num_tokens` 是全局 batch 中所有非 padding token 的数量，需在 loss 函数中统计，经 `forward_backward_func` 返回值传入 `finalize_model_grads`。

---

## 9. 各步骤使用的进程组速查

| 步骤 | 函数 | 进程组 | 大小 |
|------|------|------|------|
| 1 | `finish_grad_sync` | `intra_dp_cp_group` | DP×CP / num_instances |
| 2 | conditional embedding | `pp_group` | PP |
| 3 | SP LayerNorm | `tp_group` | TP |
| 4 | word embedding | `embd_group` | 2（first+last stage per TP shard） |
| 5 | position embedding | `pos_emb_group` | 1（first stage only） |
| 6 | expert bias | `tp_dp_cp_group` | TP×DP×CP |
| 8 | num_tokens broadcast | `pp_group` | PP |
| 8 | num_tokens AllReduce | `dp_cp_group` | DP×CP |

---

## 10. Expert DP vs Dense DP 对比表

| 属性 | Dense DP（非MoE参数） | Expert DP（MoE专家参数） |
|------|------|------|
| 进程组 | `dp_cp_group`（大小 DP×CP） | `expt_dp_group`（大小 expert_DP） |
| 通信操作 | AllReduce / ReduceScatter | AllReduce / ReduceScatter（在 expert_DP 维度） |
| 梯度缩放因子 | 1/（DP×CP） | edp_size/dp_size × 1/edp_size = 1/dp_size |
| Buffer | `self.buffers` | `self.expert_parallel_buffers` |
| 触发条件 | `param.allreduce=True` | `param.allreduce=False`（is_expert） |

Expert 参数（`is_expert=True`）使用单独的 `expert_parallel_buffers` 和 `expert_parallel_bucket_groups`，在 `finish_grad_sync` 时两者都会被处理：

```python
# distributed_data_parallel.py 第 554-555 行
for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
    bucket_group.finish_grad_sync(force_all_reduce=force_all_reduce)
```

关键区别：expert 参数在不同 EP rank 上的值是不同的（每个 EP rank 持有不同 expert 的权重），而 dense 参数在所有 DP replica 上完全相同。因此 expert DP 的 AllReduce 语义与 dense DP 相同（平均各 replica 的梯度），但进程组只包含持有相同 expert 权重的 rank（即 expert DP 组）。

---

## 11. 梯度归一化（per-token）

> **本节与上文重复，正文已合并进 §8「步骤 8」**（含 `num_tokens` 的 PP broadcast、DP×CP AllReduce、`clamp` 与 `scale_gradients`）。  
> 若只关心 per-token 归一化，直接回看该步骤即可，此处不再重贴代码。

---

## 12. no_sync 上下文管理器的作用

在流水线并行中，有 `num_microbatches` 个 microbatch 需要依次执行。在最后一个 microbatch 之前，梯度不应该立即 AllReduce（因为还有梯度没算完）。Megatron 用 `no_sync` 上下文来实现这一点：

```python
# distributed_data_parallel.py  第 480-491 行
@contextmanager
def no_sync(self):
    """Context manager that turns off gradient synchronization."""
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.is_last_microbatch = False  # 禁用同步触发
    try:
        yield
    finally:
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = True  # 恢复同步触发
```

`is_last_microbatch=False` 时，即使 bucket 内所有参数梯度就绪，`register_grad_ready` 也不会触发 `start_grad_sync()`。这确保了多个 microbatch 的梯度在本地累积后再统一同步。

---

## 13. 梯度裁剪（grad clip）与 DistOpt 的交互

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

## 14. DistributedDataParallel 的初始化流程

```python
# megatron/core/distributed/distributed_data_parallel.py
class DistributedDataParallel(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
        full_param_layout: Optional[FullParamLayout] = None,
    ):
```

初始化的主要步骤：

**步骤 1：参数分类**

把模型参数分为两类：
- `params_with_grad`：需要梯度的参数（通常是所有可训练参数）
- 专家参数（`is_expert=True`）与非专家参数分别处理，因为它们使用不同的通信组

**步骤 2：构造 `_ParamAndGradBuffer`**

```python
self.buffers = []
self.expert_parallel_buffers = []
for buffer_key, (params, param_indices) in buffer_groups.items():
    if buffer_key.is_expert_parallel:
        data_parallel_group = self.intra_expt_dp_group
        scaling_factor = expert_gradient_scaling_factor
    else:
        data_parallel_group = self.intra_dp_cp_group
        scaling_factor = gradient_scaling_factor

    buffer = _ParamAndGradBuffer(
        self.ddp_config, param_dtype, grad_dtype, params_with_names,
        data_parallel_group, self.bucket_size, param_to_name, scaling_factor, ...
    )
```

每种数据类型（BF16、FP32）各有一个 buffer。Buffer 内部按参数大小和 bucket_size 划分 `_ParamAndGradBucket`。

**步骤 3：注册反向传播钩子**

使用 `AccumulateGrad.register_hook`（底层 autograd 节点钩子），确保每次 `.backward()` 结束时触发梯度累加和（可能的）异步通信。

**步骤 4：`overlap_param_gather` 设置**

若开启 `overlap_param_gather`，设置 `next_param_gather_bucket_group` 链表，并注册 forward pre-hook，使 AllGather 与 forward 计算重叠。

---

## 15. 静默错误梯度排查清单

数据并行最难调试的是**静默梯度错误**：训练可以运行，但模型收敛曲线异常（比 baseline 慢、或震荡、或 loss 不降）。以下是系统排查流程：

### 15.1 验证进程组成员一致

```python
# 在每个 rank 上打印 DP 组成员
dp_group = parallel_state.get_data_parallel_group()
dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
print(f"[rank {torch.distributed.get_rank()}] DP group: {dp_ranks}")
```

### 15.2 比较 DP=1 与 DP=N 的梯度

```python
# 步骤 1：用 DP=1 运行一步，记录每个参数的梯度
grads_dp1 = {name: param.main_grad.clone() for name, param in model.named_parameters()
             if hasattr(param, 'main_grad') and param.main_grad is not None}

# 步骤 2：用 DP=N 运行一步（相同数据），记录 rank 0 的梯度
grads_dpN = {name: param.main_grad.clone() for name, param in model.named_parameters()
             if hasattr(param, 'main_grad') and param.main_grad is not None}

# 步骤 3：比较（应该相等，误差在 BF16 精度范围内）
for name in grads_dp1:
    if name in grads_dpN:
        diff = (grads_dp1[name] - grads_dpN[name]).abs().max()
        if diff > 1e-3:
            print(f"MISMATCH: {name}, max_diff={diff:.4e}")
```

### 15.3 检查 SP LayerNorm 梯度是否被 AllReduce

```python
for name, param in model.named_parameters():
    if 'layer_norm' in name or 'layernorm' in name:
        sp = getattr(param, 'sequence_parallel', False)
        print(f"{name}: sequence_parallel={sp}")
```

### 15.4 检查 Embedding 权重同步

```python
if parallel_state.is_pipeline_first_stage():
    emb_grad = model.language_model.embedding.word_embeddings.weight.main_grad
    print(f"[first stage] emb grad norm: {emb_grad.norm():.4f}")

if parallel_state.is_pipeline_last_stage():
    output_grad = model.language_model.output_layer.weight.main_grad
    print(f"[last stage] output grad norm: {output_grad.norm():.4f}")
```

两者应该相等（如果 `share_embeddings_and_output_weights=True`）。

---

## 16. 静默错误决策树

以下决策树帮助系统定位"loss 异常"的根本原因：

```
loss 异常（比预期高/发散/不下降）
│
├─ 检查 1：DP=1 时 loss 是否正常？
│   └─ 否 → 问题出在模型本身或数据，与并行无关
│   └─ 是 → 继续
│
├─ 检查 2：比较 DP=1 和 DP=N 在相同数据上的梯度（见 15.2 节）
│   └─ 梯度不一致 →
│       ├─ 检查 SP LayerNorm 属性（见 15.3 节）
│       ├─ 检查 embedding 梯度同步（见 15.4 节）
│       └─ 检查 Gloo group barrier 是否卡住（进程组构造问题）
│   └─ 梯度一致 → 继续
│
├─ 检查 3：per-token 归一化是否正确？
│   └─ 打印 num_tokens，确认 PP AllReduce 后值是否等于 global_batch_tokens
│   └─ 若 num_tokens 在某些 rank 上为 0 → PP broadcast 有问题
│
├─ 检查 4：是否有 NaN/Inf 梯度？
│   └─ for p in model.parameters(): print(p.main_grad.isnan().any())
│   └─ 有 NaN → 检查 loss scale（FP16/BF16 数值溢出）
│
└─ 检查 5：Distributed Optimizer 分片是否正确？
    └─ 打印 rank 0 的 optimizer state，与 DP=1 的 state 比较
    └─ ReduceScatter 后的梯度是否等于 AllReduce 后梯度的 1/DP 子集
```

---

## 17. FAQ（数据并行常见问题 12 条）

**Q1：为什么 DDP 的 bucket 大小默认随 DP 大小增加？**

DP=64 时 NCCL ring-reduce 的每个节点分到的数据块 = bucket_size / 64。如果 bucket_size 过小（如 40 MB），每节点只有 0.6 MB，通信效率低（latency-bound 而非 bandwidth-bound）。随 DP 增大 bucket_size，保证每节点分配足够大的块，维持带宽效率。

**Q2：`overlap_grad_reduce=True` 只对 PP rank 0 有效的原因？**

PP rank > 0 的 stage 在流水线 warmup/1F1B/cooldown 期间，梯度是分批计算的（每个 microbatch 贡献一部分梯度）。提前触发 AllReduce（在所有 microbatch 完成之前）会同步不完整的梯度，导致参数更新错误。PP rank 0 是 first stage，其 backward 序列恰好与"梯度累积完毕"时机对齐，可以安全地异步 overlap。

**Q3：SP LayerNorm 梯度为什么要做 SUM 而不是 AVG？**

参见第 8 节步骤 3 的解释。简言之：SP 把序列切分，每个 TP rank 的 LayerNorm 梯度是不同 token 的贡献之和，SUM 才是正确的聚合。

**Q4：`gradient_scaling_factor` 在何时等于 `1.0`？**

当 `ddp_config.average_in_collective=True` 时，scaling_factor=1.0（由 NCCL 的 AVG collective 内部做平均）。当 `average_in_collective=False`（默认）时，scaling_factor=1/dp_size，在通信前预乘。

**Q5：如果关闭 `use_distributed_optimizer`，梯度是 AllReduce 还是 ReduceScatter？**

关闭时用 AllReduce（每个 rank 获得完整梯度）。开启时用 ReduceScatter（每个 rank 只获得 1/DP 的梯度分片），随后各 rank 独立更新自己负责的 1/DP 参数，再通过 AllGather 恢复完整参数。

**Q6：`main_grad` 和 `param.grad` 的区别？**

`param.grad` 是 PyTorch autograd 系统分配的 BF16 梯度缓冲区（与参数数据类型相同）。`param.main_grad` 是 Megatron 的 FP32 主梯度缓冲区（在 `_ParamAndGradBuffer` 中分配）。反向传播后，`backward_post_hook` 把 `param.grad` 累加到 `param.main_grad`，然后清空 `param.grad`。

**Q7：`zero_grad_buffer()` 在哪里调用？**

在每个训练 step 开始时调用，清零 `main_grad` 缓冲区。位置在 `megatron/training/training.py` 的 `train_step` 函数里，在 `forward_backward_func` 调用之前。

**Q8：`broadcast_params()` 有什么用？**

训练开始时（step 0 之前），确保所有 DP replica 的初始参数相同。通常在 checkpoint 加载后调用，防止不同 rank 的初始参数因随机性差异而不一致。

**Q9：`gradient_accumulation_fusion` 影响 DDP 的 bucket 吗？**

直接影响。开启 `gradient_accumulation_fusion` 后，`param.grad` 不再创建，梯度直接累加到 `param.main_grad`（FP32 buffer 中）。DDP 的 bucket 检测"梯度就绪"的方式也需要改变（通过 `grad_added_to_main_grad` 标志而不是检查 `param.grad is not None`）。

**Q10：Expert DP 梯度 AllReduce 与 dense DP 梯度 AllReduce 同时进行时，会有竞争吗？**

不会，因为两者使用不同的 CUDA stream 和不同的 NCCL communicator。Expert buffer 和 dense buffer 的异步通信独立调度，不存在资源竞争。

**Q11：如何验证 Distributed Optimizer 的梯度分片是否正确？**

```python
# 在 ReduceScatter 后，rank r 应该持有
# 全局梯度的 [r*shard_size:(r+1)*shard_size] 部分
# （按参数注册顺序的 flat view）
shard_size = total_grad_size // dp_size
expected_grad = all_grads[rank * shard_size: (rank + 1) * shard_size]
actual_grad = optimizer.shard_grad  # 从 DistOpt 暴露的接口
assert torch.allclose(expected_grad, actual_grad, atol=1e-5)
```

**Q12：`num_distributed_optimizer_instances > 1` 是什么场景？**

用于超大规模集群（DP > 512）时，把 DP 组进一步分成 N 个子组，每个子组内做独立的 ReduceScatter，最后通过 inter_partial_dp_group 做一次 AllReduce 合并。这减少了单次通信的 rank 数量，降低 NCCL AlltoAll 的延迟，适合 DP 极大（如 DP=1024）的场景。

---

## 18. Distributed Optimizer 内存 O(params/DP) 的精确计算

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

## 19. 练习题

**题目 1**：为什么 `finalize_model_grads` 中 SP LayerNorm 梯度做 SUM AllReduce 而不是 AVG AllReduce？提示：考虑 SP 下每个 rank 计算梯度的方式。

**题目 2**：在 TP=4, DP=2, CP=2, PP=1 的配置下：
1. `dp-cp` 组的大小是多少？
2. `dp` 组的大小是多少？
3. `finalize_model_grads` 中 SP LayerNorm 梯度使用哪个组做 AllReduce？
4. Word embedding 梯度不需要 AllReduce，为什么？（提示：PP=1）

**题目 3**：阅读 `param_and_grad_buffer.py` 中 `_ParamAndGradBucket` 类，解释 `gradient_scaling_factor` 字段的作用，以及它在 MoE 和非 MoE 模型中的值分别是多少（当 `average_in_collective=False` 时）。

**题目 4**：设 global_batch_size=512，micro_batch_size=4，PP=4，DP=2，那么 `num_microbatches=512/(4×2)/4=16`。在 `finalize_model_grads` 最后一步梯度归一化中，`num_tokens` 应该是多少（假设序列长度 S=2048，无 padding）？这个值是在哪里计算并传入的？

**题目 5**：在 7B 参数 BF16 模型 + DistOpt + DP=16 的配置下，每个 rank 的优化器相关内存（FP32 主参数 + FP32 梯度 + Adam m + Adam v）是多少 GB？与第 6 节 DP=8 的结果相比，内存节省了多少？

---

## 20. 小结

Megatron 的数据并行实现不是简单地套用 torch DDP，而是针对多维并行的特殊需求做了深度定制：

1. **`ParamAndGradBuffer`**：连续内存布局 + bucket 机制，实现梯度的异步 AllReduce/ReduceScatter，只有 PP rank 0 的 bucket 才启用 overlap。FP32 `main_grad` 通过 `backward_post_hook` 从 BF16 `param.grad` 累加而来，`no_sync` 上下文在多 microbatch 间抑制提前同步。
2. **`main_grad` FP32**：参数是 BF16，梯度累积在 FP32 缓冲区，避免梯度下溢。`overlap_param_gather` 进一步把 DistOpt 的 AllGather 与 forward 计算重叠。
3. **Distributed Optimizer**：把 AllReduce 换为 ReduceScatter + AllGather，每个 rank 只持有 1/DP 的优化器状态，内存从 O(18P) 降到 O(2P + 2P/DP)。7B BF16 模型在 DP=8 时内存从 126 GB 降到 28 GB。
4. **dp-cp 联合组**：CP 的梯度归约与 DP 合并，通过 `get_data_parallel_group(with_context_parallel=True)` 访问。Expert DP 使用独立的 `expt_dp_group` 和 `expert_parallel_buffers`，两者在 `finish_grad_sync` 时独立处理。
5. **`finalize_model_grads` 八步**：完成 DDP 之外的所有梯度同步工作——SP LayerNorm（SUM AllReduce，TP组），embedding 对齐（first+last stage），位置编码，MoE expert bias，梯度 per-token 归一化（PP broadcast + DP AllReduce）。每步使用的进程组各不相同，静默错误往往来自其中某步用错了进程组。

理解了这五个机制，就能系统地理解 Megatron 为什么比 torch DDP 更适合超大模型的训练，也能快速定位梯度相关的 bug。

---

## 21. 附录：DDP 关键配置项速查表

```python
@dataclass
class DistributedDataParallelConfig:
    # 梯度同步相关
    overlap_grad_reduce: bool = False        # 异步重叠梯度归约与反向计算
    bucket_size: Optional[int] = None        # None=由大小自动推断；overlap_grad_reduce=False时无效
    average_in_collective: bool = False      # True: NCCL AVG collective（GPU内平均）；False: 手动预乘 1/dp

    # DistOpt 相关
    use_distributed_optimizer: bool = False  # ReduceScatter+AllGather 替代 AllReduce
    num_distributed_optimizer_instances: int = 1  # >1: 进一步切分 DistOpt 分片

    # AllGather 重叠相关
    overlap_param_gather: bool = False       # 把 AllGather 与 forward 计算重叠

    # 梯度精度
    grad_reduce_in_fp32: bool = True         # FP32 梯度缓冲区（主梯度）

    # 高级选项
    delay_wgrad_compute: bool = False        # 延迟 wgrad 计算（用于特定融合优化）
    reduce_scatter_with_fp32_accumulation: bool = False  # RS 时在 FP32 中累积

    # NCCL 用户缓冲区（UB）
    nccl_ub: bool = False                    # 使用 NCCL User Buffer（NVLink 上进一步降低延迟）
```

---

## 22. 附录：梯度同步的通信量估算

以 LLaMA-7B（P=7B 参数，BF16 梯度）为例：

| 场景 | 操作 | 通信量（每 rank）|
|------|------|------|
| 标准 AllReduce（DP=8） | AllReduce | P × 2 bytes × 2(send+recv) / DP = 7B×2×2/8 = 3.5 GB |
| DistOpt ReduceScatter（DP=8） | ReduceScatter | P × 2 bytes × (DP-1)/DP = 7B×2×7/8 = 12.25 GB |
| DistOpt AllGather（DP=8） | AllGather | P × 2 bytes × (DP-1)/DP = 12.25 GB |
| DistOpt 合计 | RS + AG | ~24.5 GB |

看起来 DistOpt 通信量更大（AllReduce 3.5 GB vs DistOpt 24.5 GB），但这是因为标准计算方式不同：

- AllReduce 实际 = 2 × P × 2 / DP（ring 通信，每个 rank 发送 + 接收 P/DP 大小的块，共 2(DP-1)/DP 轮）
- DistOpt RS = AllReduce 的一半（只做 reduce 不 gather），AG = AllReduce 的另一半

两者等效，DistOpt 没有额外通信量，只是把通信分成了两个阶段，便于在两者之间插入优化器更新步骤。

---

## 23. 附录：PP rank > 0 时梯度同步的时机

非 PP rank 0 的 stage 不启用 `overlap_grad_reduce`，其梯度同步在 `finalize_model_grads` 的 `finish_grad_sync` 中以同步方式完成。这不会影响性能，因为非 rank 0 的 stage 的梯度同步发生在 PP 调度的"气泡"期间（cooldown 阶段），此时 GPU 无论如何都处于空闲状态：

```
PP=4 时 stage 0（first）的时间线：
  warmup(3步F) → 1F1B稳态 → cooldown(3步B) → finish_grad_sync → finalize
                                              ↑ 此时通信可与计算重叠（rank 0）

PP=4 时 stage 3（last）的时间线：
  4步F → 4步B → finish_grad_sync → finalize
               ↑ 此时 cooldown = 0，需要同步等待通信完成
```

stage 3 的梯度同步是阻塞的（等待 ReduceScatter 完成），但由于整个 step 的关键路径是 stage 0（warmup 时间最长），stage 3 的阻塞同步不会成为瓶颈。

---

## 24. 附录：`gradient_accumulation_fusion` 与 DDP 的协同

当同时启用 `gradient_accumulation_fusion`（wgrad 直接累加到 `main_grad`）和 `overlap_grad_reduce`（异步通信）时，需要额外注意：

1. `gradient_accumulation_fusion` 直接把梯度写入 `main_grad`（不经过 `param.grad`）
2. DDP 的 `backward_post_hook` 原本检查 `param.grad is not None`，但现在 `param.grad=None`
3. 解决方案：参数设置了 `param.grad_added_to_main_grad=True` 标志，钩子用这个标志判断梯度是否已就绪

```python
# backward_post_hook 的逻辑
if param.grad is not None and not param.grad_added_to_main_grad:
    param.main_grad.add_(param.grad.data)   # 手动融合（非GA-fusion路径）
param.grad = None

# GA-fusion 路径：param.grad_added_to_main_grad=True 由 CUDA kernel 设置
# 此时 param.grad 从未设置，main_grad 在 GEMM kernel 内直接累加
```

两种路径的最终效果相同：`main_grad` 包含 FP32 梯度，`param.grad=None`，DDP 可以安全地触发通信。

---

## 25. 附录：DistOpt 与 Layer-wise Optimizer 的关系

Megatron 还支持 Layer-wise Distributed Optimizer（`LayerWiseDistributedOptimizer`），把每层的参数更新与该层的 backward 重叠：

```
Layer N backward → layer N 的 ReduceScatter → layer N 参数更新 → layer N AllGather
Layer N-1 backward → ... （与上面并行进行）
```

这进一步减少了优化器步骤的总延迟，但实现更复杂（需要 per-layer 的通信控制）。Layer-wise 优化器由 `overlap_param_gather=True` + 额外的 layer-by-layer 调度实现，源码见 `megatron/core/optimizer/distrib_optimizer.py`。与本文讨论的标准 DistOpt 相比，Layer-wise 模式在通信延迟上有额外收益，但增加了调度复杂度和内存开销（每层需要独立的 AllGather buffer）。

---

## 26. 附录：zero_grad_buffer 的调用时机与重要性

```python
# distributed_data_parallel.py  第 567-581 行
def zero_grad_buffer(self):
    """Zeros out all grad buffers. Needs to be called at the beginning of each training iteration."""
    for param in self.params_with_grad:
        param.grad_added_to_main_grad = False
    for buffer in self.buffers + self.expert_parallel_buffers:
        buffer.reset()  # 将 main_grad 清零
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.reset()  # 重置 bucket 状态（is_last_microbatch 等）
```

**忘记调用 `zero_grad_buffer()` 的后果**：
- `main_grad` 不清零 → 本 step 的梯度叠加了上一 step 的梯度 → 参数更新量翻倍（或更大）→ loss 发散
- 这是一种典型的"第一步正常，第二步开始发散"的 bug

调用位置（`megatron/training/training.py`，训练循环伪代码）：

```python
for batch in dataloader:
    optimizer.zero_grad()           # 调用 DDP.zero_grad_buffer()
    loss = forward_backward(batch)
    optimizer.step()                # 包含 finalize_model_grads + clip_grad + adam_step
```

---

## 27. 附录：broadcast_params 的使用场景

`broadcast_params()` 把所有参数从 DP rank 0 广播到其他所有 rank，确保参数一致性：

```python
# distributed_data_parallel.py  第 583-598 行
def broadcast_params(self):
    for param in self.module.parameters():
        is_expert = not getattr(param, 'allreduce', True)
        dp_group = self.expt_dp_group if is_expert else self.dp_cp_group
        torch.distributed.broadcast(
            param.data,
            src=torch.distributed.get_global_rank(dp_group, 0),
            group=dp_group,
        )
```

使用场景：
1. **训练开始时**（step 0 前）：若不同 rank 用不同随机种子初始化参数，需要 broadcast 统一
2. **加载 checkpoint 后**：若 checkpoint 只在 rank 0 上加载，需要 broadcast 同步到其他 rank
3. **测试代码中**：确保测试环境参数一致性

注意：Megatron 的标准训练流程通过 `initialize_params` 或 checkpoint 加载时统一处理，通常不需要手动调用 `broadcast_params`。

---

## 28. 小结（数据并行相关 API 速查）

| API | 位置 | 调用时机 |
|------|------|------|
| `zero_grad_buffer()` | DDP | 每 step 开始前 |
| `no_sync()` | DDP | 多 microbatch 期间（PP/梯度累积） |
| `finish_grad_sync()` | DDP | `finalize_model_grads` 步骤 1 |
| `scale_gradients()` | DDP | `finalize_model_grads` 步骤 8 |
| `broadcast_params()` | DDP | 训练开始/checkpoint 加载后 |
| `start_param_sync()` | DDP | DistOpt AllGather（下一 step 开始前） |
| `finalize_model_grads()` | finalize_model_grads.py | 每 step 反向传播结束后 |
