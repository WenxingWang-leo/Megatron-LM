# 精读 Megatron 源码（6）：流水线并行完全精读——1F1B、气泡、P2P 与 VPP

> 源文件：`megatron/core/pipeline_parallel/schedules.py`、`p2p_communication.py`

---

## 1. 为什么需要流水线并行

单台 GPU 装不下一个数百亿参数的模型，即使 TP 能切分单层，层数太多时激活内存依然爆炸。流水线并行（Pipeline Parallelism，PP）把 Transformer 的层按 stage 分配给不同的 GPU：stage 0 做第 1-N/PP 层，stage 1 做第 N/PP+1-2N/PP 层，以此类推。数据沿 stage 方向流动，后一个 stage 的前向激活由前一个 stage 的 GPU 通过 **P2P send/recv** 发送过来。

这把模型内存从 O(N_params) 降到 O(N_params/PP)，代价是出现"流水线气泡"（bubble）——在调度的暖机和冷却阶段部分 GPU 处于空闲状态。

---

## 2. get_forward_backward_func 的调度选择逻辑

```python
# megatron/core/pipeline_parallel/schedules.py  第 48-163 行
def get_forward_backward_func(
    pp_size=None, vp_size=None, schedule_pg_collection=None
):
    if isinstance(schedule_pg_collection, MultiModuleProcessGroupCollection):
        return forward_backward_pipelining_without_interleaving  # 多模态桥接

    if pp_size is None and vp_size is None:
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving  # VPP
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving  # 标准 1F1B
    else:
        forward_backward_func = forward_backward_no_pipelining  # 单机，无 PP
    return forward_backward_func
```

三条路径：

| 条件 | 函数 | 特点 |
|------|------|------|
| PP=1 | `forward_backward_no_pipelining` | 无流水线，最简单 |
| PP>1, VPP=None | `forward_backward_pipelining_without_interleaving` | 标准 1F1B |
| PP>1, VPP>1 | `forward_backward_pipelining_with_interleaving` | VPP 交错，气泡更少 |

---

## 3. 标准 1F1B 调度

### 3.1 warmup 公式推导

```python
# megatron/core/pipeline_parallel/schedules.py  第 2264-2267 行
num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
num_microbatches_remaining = num_microbatches - num_warmup_microbatches
```

其中 `total_stages = PP`，`current_stage = pp_rank`（0-indexed）。

设 PP=4，m=num_microbatches，stage r 的 warmup 轮数为：

```
warmup(r) = min(m, PP - r - 1)
          = min(m, 4 - r - 1)

stage 0: warmup = min(m, 3)
stage 1: warmup = min(m, 2)
stage 2: warmup = min(m, 1)
stage 3: warmup = min(m, 0) = 0
```

**直觉**：最后一个 stage（stage PP-1）无需等待任何数据就可以立即开始正向传播，所以 warmup=0。越靠前的 stage，需要先把越多的 microbatch 推送进流水线才能开始做 1F1B 循环。

### 3.2 完整时间线：PP=4，m=8

下表用 F 表示正向（Forward），B 表示反向（Backward），空格表示气泡（idle）：

```
时钟步：  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
stage 0: F0   F1   F2  F3B0  F4B1  F5B2  F6B3  F7B4  B5   B6   B7
stage 1:      F0   F1   F2   F3B0  F4B1  F5B2  F6B3  F7B4 B5   B6   B7
stage 2:           F0   F1   F2    F3B0  F4B1  F5B2  F6B3 F7B4 B5   B6   B7
stage 3:                F0   F1    F2    F3    F4    B4   B5B6  B7B8 B6   B7
```

更精确的时间线（每行是一个 stage，列是时间步，F/B 后的数字是 microbatch 编号）：

```
         t=1  t=2  t=3  t=4  t=5  t=6  t=7  t=8  t=9  t=10 t=11 t=12 t=13 t=14
stage 0:  F0   F1   F2  [1F1B: F3B0  F4B1  F5B2  F6B3  F7B4]  B5   B6   B7
stage 1:  __   F0   F1   F2  [F3B0  F4B1  F5B2  F6B3  F7B4]   B5   B6   B7
stage 2:  __   __   F0   F1   F2  [F3B0  F4B1  F5B2  F6B3  F7B4] B5  B6   B7
stage 3:  __   __   __   F0   F1   F2   F3   F4   B4   B5   B6  B7   B8*  __
```

（* = stage 3 最后做 B8，实际 m=8 时共 8 个 microbatch）

**气泡（bubble）**：stage 0 在步骤 1-3（warmup 期间）没有 backward；stage 3 在开始 backward 之前有 3 步等待 forward 填满流水线。气泡比例约为：

```
bubble_fraction ≈ (PP - 1) / (m + PP - 1)
```

当 `m >> PP` 时气泡趋近于 0，这就是为什么增大 `num_microbatches` 能提升 PP 效率。

### 3.3 为什么 stage PP-1 warmup=0

stage PP-1 处于流水线末端，第一个 microbatch 的 forward 一到达就立刻可以计算 loss 并做 backward，不需要等待其他 stage"先走"，所以 warmup 步数为 0。反过来，stage 0 是源头，它需要先把 PP-1 个 microbatch 的 forward 推入流水线，确保所有下游 stage 都有活可做，才开始自己的 1F1B 节奏。

---

## 4. P2PCommunicator API

P2P 通信由 `P2PCommunicator` 类封装，位于 `megatron/core/pipeline_parallel/p2p_communication.py`：

```python
class P2PCommunicator:
    def recv_forward(self, tensor_shapes, is_first_stage) -> List[Tensor]:
        """从上一个 stage 接收正向激活。第一个 stage 返回 None。"""

    def send_forward(self, output_tensor, is_last_stage):
        """把正向激活发送给下一个 stage。最后一个 stage 丢弃（loss 不需要发送）。"""

    def recv_backward(self, tensor_shapes, is_last_stage) -> List[Tensor]:
        """从下一个 stage 接收反向梯度。最后一个 stage 返回 None。"""

    def send_backward(self, input_tensor_grad, is_first_stage):
        """把输入梯度发送给上一个 stage。第一个 stage 丢弃。"""

    def send_forward_recv_backward(self, output_tensor, tensor_shapes, is_last_stage):
        """合并操作：同时发送正向激活 + 接收反向梯度。
        这是 1F1B 稳态的关键优化：把两次通信合并为一次双向传输。"""

    def send_backward_recv_forward(self, input_tensor_grad, tensor_shapes, is_first_stage):
        """合并操作：同时发送反向梯度 + 接收下一个正向激活。"""
```

在 1F1B 稳态循环中：

```python
# megatron/core/pipeline_parallel/schedules.py  第 2404-2406 行
output_tensor_grad = p2p_communicator.send_forward_recv_backward(
    output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
)
```

`send_forward_recv_backward` 把发送正向激活和接收反向梯度合并为一次 `isend + irecv` 调用，减少通信延迟（尤其是当 PP 通信在 InfiniBand 上时，合并操作可以减少小消息的开销）。

---

## 5. loss_func 只在最后一个 stage 调用

```python
# megatron/core/pipeline_parallel/schedules.py  第 2335-2350 行（forward_step 内部）
output_tensor, num_tokens = forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    ...
    is_last_stage=p2p_communicator.is_pp_last_stage,
)
```

在 `forward_step` 内部：

```python
def forward_step(forward_step_func, data_iterator, model, ...):
    output_tensor, loss_func = forward_step_func(data_iterator, model)

    if is_last_stage:
        # loss_func 返回 (loss_tensor, loss_dict) 或 (loss_tensor, loss_dict, num_tokens)
        output_tensor, loss_reduced = loss_func(output_tensor)
```

`loss_func` 返回 2 元组或 3 元组：

```python
def loss_func(loss_mask, output_tensor):
    loss = compute_loss(output_tensor, loss_mask)
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}  # 2-tuple

# 或者 3-tuple（包含 token 计数，用于 per-token loss 归一化）
def loss_func(loss_mask, output_tensor):
    loss, num_tokens = compute_loss_with_tokens(output_tensor, loss_mask)
    return loss, {'lm loss': loss}, num_tokens  # 3-tuple
```

非最后 stage 的 `forward_step` 直接返回 `(output_tensor, 0)`（num_tokens=0），`output_tensor` 是激活张量，随后通过 P2P 发送给下一个 stage。

---

## 6. Virtual Pipeline Parallelism（VPP）

### 6.1 基本概念

VPP 把每个物理 GPU 分配为 V 个 virtual stage，减少气泡的同时增加 P2P 通信次数。

以 **16 层，PP=4，V=2** 为例：

```
标准 PP（V=1，每个 GPU 4 层）：
  GPU 0: layers  1- 4
  GPU 1: layers  5- 8
  GPU 2: layers  9-12
  GPU 3: layers 13-16

VPP（V=2，每个 GPU 2+2=4 层，但分为 2 个 virtual stage）：
  GPU 0: layers  1- 2  +  layers  9-10
  GPU 1: layers  3- 4  +  layers 11-12
  GPU 2: layers  5- 6  +  layers 13-14
  GPU 3: layers  7- 8  +  layers 15-16
```

交错式调度下，一个 microbatch 的数据路径变为：

```
GPU0(v0) → GPU1(v0) → GPU2(v0) → GPU3(v0)
→ GPU0(v1) → GPU1(v1) → GPU2(v1) → GPU3(v1)
```

### 6.2 VPP 下模型表示为 list

```python
# 标准 PP：model 是单个 nn.Module
model: torch.nn.Module

# VPP：model 是长度为 V 的列表
model: List[torch.nn.Module]   # model[v] = virtual stage v 的层
```

在 `forward_backward_pipelining_with_interleaving` 中：

```python
for k in range(num_model_chunks):
    model_chunk = model[k]
    ...
```

### 6.3 VPP 的气泡公式

```
bubble_fraction(VPP) ≈ (PP - 1) / (V × m + PP - 1)
                     ≈ 1/V × bubble_fraction(standard PP)
```

VPP=2 时气泡减半，但每个 microbatch 的 P2P 通信次数翻倍（标准 PP 每 microbatch 做 2×(PP-1) 次 P2P，VPP 做 2×V×(PP-1) 次）。因此 VPP 对 P2P 带宽要求更高。

### 6.4 VPP 约束

```python
# VPP 要求 PP > 1（源码第 740-744 行）
if virtual_pipeline_model_parallel_size is not None:
    if not pipeline_model_parallel_size > 1:
        raise RuntimeError(...)

# 层数必须被 PP×V 整除（在 GPT model 配置中检查）
assert num_layers % (pipeline_model_parallel_size * vpp_size) == 0
```

---

## 7. deallocate_output_tensor

```python
# megatron/core/pipeline_parallel/schedules.py  第 166-196 行
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's .data field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage.'''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    ...
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

正向激活被发送给下一个 stage 之后，它仍然需要保留在内存里（用于反向传播计算梯度）。但实际数据已经不需要了——反向时需要的是 `.grad_fn` 引用，不是 `.data`。`deallocate_output_tensor` 把 `.data` 替换为一个标量张量（只占 4 字节），释放激活的主体内存，保留计算图节点。

```python
# megatron/core/pipeline_parallel/schedules.py  第 2356 行
deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
```

这个优化可以显著减少流水线 warmup 阶段的内存峰值（warmup 期间有 `num_warmup_microbatches` 个正向激活同时存活）。

---

## 8. 性能调优：num_microbatches 与 PP 的关系

关键经验规则：

**规则 1**：气泡比例 ≈ `(PP-1) / (m + PP-1)`，所以 `m` 至少要是 `PP` 的几倍才能把气泡压到 5% 以下。例如 PP=8 时，`m ≥ 152` 才能保证气泡 < 5%。

**规则 2**：每个 microbatch 的数据量（`micro_batch_size × seq_length`）决定了 P2P 通信的数据量。PP 通信延迟与激活 tensor 大小成正比：

```
activation_size = micro_batch_size × seq_length × hidden_size × bytes_per_element
```

对于 bf16，hidden=4096，seq=2048，mbs=1：

```
activation_size ≈ 1 × 2048 × 4096 × 2 = 16 MB
```

16 MB 在 InfiniBand 上需要 ~80μs，远低于一个 forward pass 的时间（通常 >1ms），所以 PP 通信开销通常可以接受。

**规则 3**：VPP 减少气泡的代价是增加 P2P 通信次数。当 TP 通信已经占用大量带宽时，VPP 的额外 P2P 通信可能成为瓶颈。通常建议先尝试增大 `m`（global batch size / micro batch size），而不是直接上 VPP。

---

## 9. 完整示例：PP=2，m=4 的时间线

```
阶段设置：
  PP = 2, m = 4（4 个 microbatch）
  stage 0 warmup = min(4, 2-0-1) = 1
  stage 1 warmup = min(4, 2-1-1) = 0

时间步：  t1   t2   t3   t4   t5   t6
stage 0:  F0  [F1B0  F2B1  F3B2]  B3
stage 1:  F0   F1   F2   F3   B3   B2   B1   B0

更准确的格式（考虑通信延迟）：

stage 0: F0 →send→ F1 ←recv B0→ F2 ←recv B1→ F3 ←recv B2→ B3
stage 1:      ←recv F0→ ←recv F1→ F2 ←recv F3→  B3→send→  B2→send→  B1→send→  B0→send
```

气泡仅出现在 stage 0 的开头（1 步 warmup）和末尾（1 步 cooldown），共 2 步，总时间 10 步，气泡率 = 2/10 = 20%。当 m=8 时，气泡率降到 2/14 ≈ 14%。

---

## 10. combined_1f1b 与 hybrid_cp_schedule

Megatron 还提供了两个高级调度变体：

### 10.1 combined_1f1b（`combined_1f1b.py`）

把流水线内的正向和反向合并成一个 kernel，减少 CPU-GPU 同步开销。当 PP=1 时使用 `combined_1f1b_schedule_for_no_pipelining`，当 PP>1 时使用 `combined_1f1b_schedule_for_interleaved_pipelining`。

### 10.2 hybrid_cp_schedule（`hybrid_cp_schedule.py`）

当同时启用 CP（Context Parallel）和 PP 时，CP 的 AllGather/ReduceScatter（用于 KV 交换）和 PP 的 P2P 通信可能竞争带宽。`hybrid_context_parallel_forward_backward` 把 CP 的 KV 通信与 PP 的 P2P 通信交替安排，减少带宽冲突。

---

## 11. 调试流水线并行的常见问题

### 问题 1：`assert len(model) == 1` 报错

```
AssertionError: non-interleaved pipeline-parallel schedule does not support model chunking
```

**原因**：使用了 `forward_backward_pipelining_without_interleaving` 但传入了 `model = [chunk0, chunk1]`（VPP 格式）。确保 VPP 场景下调用 `forward_backward_pipelining_with_interleaving`。

### 问题 2：NCCL 通信 hang（所有 rank 卡在 P2P）

**诊断**：

```python
# 在 p2p_communication.py 中添加日志
print(f"[rank {rank}] about to recv from rank {src_rank}")
torch.distributed.recv(tensor, src=src_rank, group=pp_group)
print(f"[rank {rank}] recv done")
```

**常见原因**：stage 数量与 PP 不匹配，或者某个 stage 异常退出导致其他 rank 永久等待。

### 问题 3：loss 只在 stage PP-1 上，如何在 stage 0 上记录？

```python
# 在 forward_data_store 中收集 loss
# schedules.py 会把 loss 写入 forward_data_store
# 在训练循环中：
losses = [x['lm loss'] for x in forward_data_store if 'lm loss' in x]
# 注意：非最后 stage 的 forward_data_store 是空的
if is_pp_last_stage():
    log_loss(losses)
```

### 问题 4：激活检查点（activation checkpointing）与 PP 的交互

```python
# schedules.py 第 2277-2279 行
max_outstanding_backprops = None
if config.num_microbatches_with_partial_activation_checkpoints is not None:
    max_outstanding_backprops = num_warmup_microbatches + 1
```

PP warmup 阶段有 `num_warmup_microbatches` 个正向激活同时保存在内存里。激活检查点通过 `checkpoint_activations_microbatch` 参数控制哪些 microbatch 需要重新计算激活，从而减少 warmup 期间的内存峰值。

---

## 12. 练习题

**题目 1**：请画出 PP=2，m=4 的完整时间线（格式参考第 9 节）。标出所有 F、B 操作和 P2P 通信方向，计算气泡率。

**题目 2**：PP=4，m=4 时，各 stage 的 warmup 轮数分别是多少？总气泡率是多少？

**题目 3**：VPP=2，PP=4，m=8 时，气泡率比标准 PP（VPP=1）减少了多少？额外的 P2P 通信次数是多少（与标准 PP 相比）？

**题目 4**：阅读 `forward_backward_pipelining_with_interleaving` 的源码，找到 VPP 下 `model_chunk_id` 的计算逻辑，解释它如何决定哪个 `model[v]` 处理当前的 microbatch。

---

## 14. UCC 后端与零 SM 通信

Megatron 支持为 PP 通信使用 UCC（Unified Collective Communication）后端，而不是默认的 NCCL：

```python
# megatron/core/parallel_state.py  第 1027-1046 行
if pipeline_model_parallel_comm_backend == "ucc":
    # The UCC backend provides two key benefits:
    # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
    # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
    #    with overlapping compute kernels.
    if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
        assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] != "1", \
            "UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1"
```

UCC 的两个优势：

1. **更高带宽利用率**：UCC 在 InfiniBand 上比 NCCL 有更好的带宽利用率，尤其在消息较大时。
2. **零 SM 占用**：UCC 通过 CPU 发起通信，不占用 GPU SM 资源，避免 P2P 通信与 TP GEMM 争抢 SM。

使用方法：

```python
initialize_model_parallel(
    ...,
    pipeline_model_parallel_comm_backend="ucc",
)
# 同时需要设置 CUDA_DEVICE_MAX_CONNECTIONS > 1（例如 8）
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
```

UCC 模式下还需要设置 UCX 相关环境变量，Megatron 会自动处理这些设置（第 1061-1077 行）。

---

## 15. 小结

流水线并行的核心是用 1F1B 调度减少空闲时间：

1. **warmup 公式**：`warmup(r) = min(m, PP - r - 1)`，最后一个 stage warmup=0，最前一个 stage warmup=PP-1（当 m≥PP-1 时）。
2. **P2PCommunicator**：把正向和反向的 P2P 通信封装为 `send_forward`、`recv_backward`、`send_forward_recv_backward` 等 API，稳态下通过合并发送+接收操作减少延迟。
3. **loss_func 只在最后 stage 调用**：返回 2 元组 `(loss, loss_dict)` 或 3 元组 `(loss, loss_dict, num_tokens)`；其他 stage 只传递激活 tensor。
4. **VPP** 把气泡减少约 1/V，代价是 P2P 通信量增加 V 倍；模型表示为 `List[nn.Module]`。
5. **deallocate_output_tensor** 在激活发送后立即释放数据内存，只保留计算图节点，减少 warmup 期内存峰值。

下一篇（07）将深入数据并行：DDP 缓冲、bucket 设计、`finalize_model_grads` 的八步流程，以及 Distributed Optimizer 的内存节省原理。
