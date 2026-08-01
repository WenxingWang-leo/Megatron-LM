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

## 3. 标准 1F1B 调度伪代码详注

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

### 3.2 标准 1F1B 三阶段伪代码

以下伪代码对应 `forward_backward_pipelining_without_interleaving` 的三个主体循环：

```python
# ======== 阶段一：Warmup（只做 Forward）========
for i in range(num_warmup_microbatches):
    # 1. 从上一 stage 接收激活（first stage 返回 None）
    input_tensor = p2p_communicator.recv_forward(recv_shapes, is_first_stage)

    # 2. 执行本 stage 的 forward
    output_tensor, num_tokens = forward_step(...)

    # 3. 把激活发给下一 stage（last stage 不发送）
    p2p_communicator.send_forward(output_tensor, is_last_stage)

    # 4. 保存 (input_tensor, output_tensor) 留给 backward 用
    input_tensors.append(input_tensor)
    output_tensors.append(output_tensor)
    deallocate_output_tensor(output_tensor)  # 释放激活 data，保留计算图

# ======== 阶段二：1F1B 稳态（每步 1 Forward + 1 Backward）========
for i in range(num_microbatches_remaining):
    last_iteration = (i == num_microbatches_remaining - 1)

    # 接收下一个 forward 的激活
    input_tensor = p2p_communicator.recv_forward(recv_shapes, is_first_stage)

    # 执行 forward
    output_tensor, num_tokens = forward_step(...)

    # 同时发送 forward 激活 + 接收 backward 梯度（合并通信）
    output_tensor_grad = p2p_communicator.send_forward_recv_backward(
        output_tensor, send_shapes, is_last_stage
    )

    # 从队列取出历史 (input, output) 做 backward
    input_tensor  = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    # 最后一个 1F1B 步骤前，开启异步梯度同步
    if last_iteration:
        enable_grad_sync()

    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

    if last_iteration:
        p2p_communicator.send_backward(input_tensor_grad, is_first_stage)
    else:
        # 同时发送 backward 梯度 + 接收下一 forward 激活（合并通信）
        input_tensor = p2p_communicator.send_backward_recv_forward(
            input_tensor_grad, recv_shapes, is_first_stage
        )

# ======== 阶段三：Cooldown（只做 Backward）========
for i in range(num_warmup_microbatches):
    if i == num_warmup_microbatches - 1:
        enable_grad_sync()  # 最后一个 backward 前开启同步

    input_tensor  = input_tensors.pop(0)
    output_tensor = output_tensors.pop(0)

    # 从下一 stage 接收 backward 梯度
    output_tensor_grad = p2p_communicator.recv_backward(send_shapes, is_last_stage)

    input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad)

    # 发送 backward 梯度给上一 stage
    p2p_communicator.send_backward(input_tensor_grad, is_first_stage)
```

关键设计点：
- `send_forward_recv_backward`（第二阶段）和 `send_backward_recv_forward`（第二阶段尾部）把两次通信合并为一次 isend+irecv，降低通信延迟。
- `enable_grad_sync` 在最后一个 backward 前调用，确保梯度已全部累积再做 AllReduce。

---

## 4. 气泡率公式与数值表

### 4.1 气泡率公式

```
bubble_fraction ≈ (PP - 1) / (m + PP - 1)
```

其中 m = num_microbatches（每个 PP stage 处理的 microbatch 数量，= global_batch_size / micro_batch_size / DP / PP）。

当 `m >> PP` 时气泡趋近于 0，这就是为什么增大 `num_microbatches` 能提升 PP 效率。

### 4.2 气泡率数值表（PP=4）

| m（microbatch 数） | bubble_fraction | 有效计算比例 |
|---:|---:|---:|
| PP=4（m=4） | 3/7 ≈ 43% | 57% |
| 2×PP（m=8） | 3/11 ≈ 27% | 73% |
| 4×PP（m=16） | 3/19 ≈ 16% | 84% |
| 8×PP（m=32） | 3/35 ≈ 9% | 91% |
| 无穷大 | 0% | 100% |

**工程经验**：为把气泡率压到 5% 以下，m 至少需要 `(PP-1)/0.05 - (PP-1) = (PP-1)×19 ≈ 57` 个 microbatch（PP=4 时）。对于 PP=8，需要 m ≥ 133。

### 4.3 完整时间线：PP=4，m=8

下表用 F/B 加 microbatch 编号表示操作，`_` 表示气泡：

```
时间步：  0    1    2    3    4    5    6    7    8    9   10   11   12   13
stage 0: F0   F1   F2  [F3B0 F4B1 F5B2 F6B3 F7B4] B5   B6   B7   _    _
stage 1: _    F0   F1   F2  [F3B0 F4B1 F5B2 F6B3 F7B4] B5   B6   B7   _
stage 2: _    _    F0   F1   F2  [F3B0 F4B1 F5B2 F6B3 F7B4] B5   B6   B7
stage 3: _    _    _    F0   F1   F2   F3   F4   B4   B5   B6   B7   _    _
         ↑____________________↑                             ↑_________↑
              warmup (气泡)                                  cooldown (气泡)
```

气泡总步数 = warmup + cooldown 中每个 stage 的空闲步数。对 stage 0：3 步 warmup + 0 步 cooldown（warmup 恰好等于 m 的余量，不超出）。

---

## 5. P2PCommunicator API

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

## 6. P2P 通信的 tensor_shape：与 SP 的交互

`get_tensor_shapes` 函数计算 P2P 传输的 tensor 形状：

```python
# megatron/core/pipeline_parallel/schedules.py  第 2095-2124 行
def get_tensor_shapes(*, seq_length, micro_batch_size, ..., tp_group, cp_group):
    if config.variable_seq_lengths:
        # packed seq / variable-length 模式：形状动态交换，返回 [()]
        tensor_shapes.append(())
        return tensor_shapes

    effective_seq_length = decoder_seq_length or seq_length
    effective_seq_length = effective_seq_length // cp_group.size()  # CP 切分

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()  # SP 切分

    tensor_shapes.append((effective_seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes
```

以 SP=True，TP=4，CP=2，S=4096，mbs=1，H=4096 为例：

```
effective_seq = 4096 // 2 = 2048   （CP 切分后）
effective_seq = 2048 // 4 = 512    （SP 切分后）

P2P tensor shape = (512, 1, 4096)
P2P 数据量 = 512 × 1 × 4096 × 2 bytes = 4 MB
```

与非 SP（shape=(2048, 1, 4096)，16 MB）相比，SP 模式下 P2P 通信量减少 4×（TP 倍）。这是 SP 的另一个优势：不仅节省激活内存，还减少了 PP 通信量。

### 6.1 variable_seq_lengths（packed seq）支持

当 `config.variable_seq_lengths=True` 时，`get_tensor_shapes` 返回空元组 `[()]`，P2P 通信会在运行时动态协商 tensor 形状（通过先发送 shape metadata，再发送 data）。这对于 packed sequence（多个不同长度的序列 pack 进一个 microbatch）是必须的，因为每个 microbatch 的实际 token 数不同，无法静态推断形状。

---

## 7. loss_func 只在最后一个 stage 调用

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

`num_tokens` 会在 `finalize_model_grads` 中用于 per-token loss 归一化（第 7 篇会详细讲解）。由于 num_tokens 只在 last stage 有值，`finalize_model_grads` 会先通过 PP 组 broadcast 把它广播到所有 stage，再通过 DP 组 AllReduce 汇总全局 token 数。

---

## 8. Virtual Pipeline Parallelism（VPP）

### 8.1 基本概念

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

### 8.2 VPP 下模型表示为 list

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

### 8.3 get_schedule_table 直觉：V=2，PP=2，m=4 的例子

`get_schedule_table` 生成 `(microbatch_id, model_chunk_id)` 序列：

```python
# megatron/core/pipeline_parallel/schedules.py  第 954-981 行
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    schedule_table = []
    for min_mb_id in range(0, num_microbatches, microbatch_group_size_per_vp_stage):
        schedule_table.extend(
            [(mb_id, chunk_id)
             for chunk_id in range(num_model_chunks)
             for mb_id in range(min_mb_id, min_mb_id + microbatch_group_size_per_vp_stage)]
        )
    return schedule_table
```

以 V=2（num_model_chunks=2），PP=2，m=4，microbatch_group_size=1 为例：

```
min_mb_id=0:  [(0,0), (0,1)]  → mb0 先做 chunk0，再做 chunk1
min_mb_id=1:  [(1,0), (1,1)]  → mb1 先做 chunk0，再做 chunk1
min_mb_id=2:  [(2,0), (2,1)]
min_mb_id=3:  [(3,0), (3,1)]

完整 schedule_table（按时间顺序）:
  microbatch_id   | 0 0 1 1 2 2 3 3
  model_chunk_id  | 0 1 0 1 0 1 0 1
```

`get_model_chunk_id(virtual_mb_id, forward=True)` 从该表取第 `virtual_mb_id` 项的 `model_chunk_id`；backward 时则取镜像（`V - chunk_id - 1`），确保反向传播顺序正确。

### 8.4 VPP 的气泡公式

```
bubble_fraction(VPP) ≈ (PP - 1) / (V × m + PP - 1)
                     ≈ 1/V × bubble_fraction(standard PP)
```

VPP=2 时气泡减半，但每个 microbatch 的 P2P 通信次数翻倍（标准 PP 每 microbatch 做 2×(PP-1) 次 P2P，VPP 做 2×V×(PP-1) 次）。因此 VPP 对 P2P 带宽要求更高。

### 8.5 VPP 约束

```python
# VPP 要求 PP > 1（源码第 740-744 行）
if virtual_pipeline_model_parallel_size is not None:
    if not pipeline_model_parallel_size > 1:
        raise RuntimeError(...)

# 层数必须被 PP×V 整除（在 GPT model 配置中检查）
assert num_layers % (pipeline_model_parallel_size * vpp_size) == 0
```

---

## 9. deallocate_output_tensor

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

## 10. 性能调优：num_microbatches 与 PP 的关系

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

## 11. 完整示例：PP=2，m=4 的时间线（1F1B）

```
阶段设置：
  PP = 2, m = 4（4 个 microbatch）
  stage 0 warmup = min(4, 2-0-1) = 1
  stage 1 warmup = min(4, 2-1-1) = 0

# 按时间列对齐（Fi/Bi = microbatch i 的前向/反向；编号从 1 起）
时间 →     1    2    3    4    5    6    7    8
stage 0   F1   F2   B1   F3   B2   F4   B3   B4
stage 1        F1   B1   F2   B2   F3   B3   F4   B4
```

关键依赖：stage1 上某个 `Fk` 做完后，同一拍或紧接着做 `Bk`，再把梯度发回 stage0；  
**stage0 的 `Bk` 不能与 stage1 的 `Fk` 画在同一列**（否则违反激活/梯度依赖）。

带通信注解的同一调度：

```
stage 0: F1 →send→ F2 ←recv B1→ F3 ←recv B2→ F4 ←recv B3→ B4
stage 1:      ←recv F1→ B1→send→ F2 ... F3 ... F4 → B4→send→
```

气泡主要出现在 stage0 开头（warmup 空等 stage1）与末尾 cooldown；  
近似气泡率 `(PP-1)/m = 1/4 = 25%`。m 越大，该比例越低。

---

## 12. hang 调试：P2P 死锁定位

### 12.1 P2P 死锁的典型原因

1. **send/recv 顺序不对称**：stage i 先调用 `send`，stage i+1 也先调用 `send`，而不是 `recv`。两个 stage 互相等待对方先接收，形成死锁。

   Megatron 通过严格的 send-recv 配对顺序（源码中通过 `p2p_communicator` 统一管理）避免这一问题，但自定义 schedule 需要小心。

2. **PP 数量与实际进程数不匹配**：例如 PP=4 但只启动了 3 个进程，NCCL P2P 会永久等待第 4 个 rank。

3. **某个 rank 在流水线执行期间 OOM 崩溃**：OOM 通常以 Python 异常退出，不发送通知给其他 rank，导致其他 rank hang 在 recv。

### 12.2 调试步骤

```python
# 方法 1：在每次 P2P 操作前后添加打印
# 找到哪一步 hang 住了

import megatron.core.pipeline_parallel.p2p_communication as p2p

# 包装 recv_forward
original_recv_forward = p2p.recv_forward
def debug_recv_forward(*args, **kwargs):
    rank = torch.distributed.get_rank()
    print(f"[rank {rank}] about to recv_forward")
    result = original_recv_forward(*args, **kwargs)
    print(f"[rank {rank}] recv_forward done")
    return result
p2p.recv_forward = debug_recv_forward
```

```bash
# 方法 2：设置 NCCL 超时（防止无限 hang）
export NCCL_TIMEOUT=120  # 120 秒超时
# 超时后 NCCL 会抛出异常并打印哪个操作超时
```

```python
# 方法 3：检查 p2p 通信的 send/recv 配对
# 在 stage r，检查：
#   - recv_forward（从 stage r-1 接收）是否与 stage r-1 的 send_forward 配对
#   - send_forward（到 stage r+1）是否与 stage r+1 的 recv_forward 配对
print(f"[rank {rank}] pp_rank={pp_rank}, prev_rank={prev_rank}, next_rank={next_rank}")
```

### 12.3 常见死锁场景

**场景**：自定义 schedule 中，两个相邻 stage 都先 send 再 recv：

```python
# stage 0 做：
send_forward(...)   # 等待 stage 1 的 recv
recv_backward(...)  # 等待 stage 1 的 send

# stage 1 做：
send_backward(...)  # 等待 stage 0 的 recv ← 死锁！
recv_forward(...)   # 等待 stage 0 的 send
```

修复：确保相邻 stage 的 send/recv 严格配对，即 stage i 的 `send_forward` 对应 stage i+1 的 `recv_forward`，两者必须几乎同时（在同一时钟步内）调用。

---

## 13. combined_1f1b 与 hybrid_cp_schedule

Megatron 还提供了两个高级调度变体：

### 13.1 combined_1f1b（`combined_1f1b.py`）

把流水线内的正向和反向合并成一个 kernel，减少 CPU-GPU 同步开销。当 PP=1 时使用 `combined_1f1b_schedule_for_no_pipelining`，当 PP>1 时使用 `combined_1f1b_schedule_for_interleaved_pipelining`。

### 13.2 hybrid_cp_schedule（`hybrid_cp_schedule.py`）

当同时启用 CP（Context Parallel）和 PP 时，CP 的 AllGather/ReduceScatter（用于 KV 交换）和 PP 的 P2P 通信可能竞争带宽。`hybrid_context_parallel_forward_backward` 把 CP 的 KV 通信与 PP 的 P2P 通信交替安排，减少带宽冲突。

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

---

## 15. 调试流水线并行的常见问题

### 问题 1：`assert len(model) == 1` 报错

```
AssertionError: non-interleaved pipeline-parallel schedule does not support model chunking
```

**原因**：使用了 `forward_backward_pipelining_without_interleaving` 但传入了 `model = [chunk0, chunk1]`（VPP 格式）。确保 VPP 场景下调用 `forward_backward_pipelining_with_interleaving`。

### 问题 2：NCCL 通信 hang（所有 rank 卡在 P2P）

参考第 12 节的调试步骤。

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

## 16. FAQ（流水线并行常见问题 10 条）

**Q1：为什么最后一个 PP stage 的 warmup=0？**

最后一个 stage 处于流水线末端，第一个 microbatch 的激活到达时就可以立即计算 loss 并开始 backward，不需要等待下游 stage（因为它就是下游的终点）。

**Q2：PP 气泡率为 20% 时，MFU（模型 FLOP 利用率）大约下降多少？**

MFU 下降约 20%（气泡期间 GPU 完全空闲）。实际上还需考虑 P2P 通信开销，总 MFU 损失通常在 15-25% 之间（取决于 m/PP 比值）。

**Q3：VPP=2 时 P2P 通信次数翻倍，带宽够用吗？**

对于单机 NVLink（900 GB/s），通常够用。对于多机 InfiniBand（200 Gbps ≈ 25 GB/s），16 MB 激活 × 2V × 2方向 = 64 MB per microbatch，在 25 GB/s 下需要 2.6 ms，可能超过 stage 计算时间。建议在 IB 环境下谨慎使用 VPP。

**Q4：SP 模式下 PP P2P 传输的 tensor 更小（S/TP），通信效率更高吗？**

是的，SP + PP 组合下 P2P 的 tensor shape 是 `(S/TP/CP, mbs, H)`，大小为标准 PP 的 `1/(TP×CP)` 倍。这是在高 TP/CP 场景下 PP 通信开销实际很小的原因。

**Q5：`deallocate_pipeline_outputs` 默认是开启的吗？什么时候要关闭？**

默认 False（不释放）。开启后（设为 True）可以节省 warmup 期的内存。唯一需要关闭的场景是调试时需要检查中间激活值。

**Q6：1F1B 和 GPipe 的主要区别是什么？**

GPipe 的 warmup 阶段做完所有 microbatch 的 forward，再做所有 backward（F1F2...FmB1B2...Bm），气泡率约 `(PP-1)/m`，且同时存活 m 个 microbatch 的激活，内存消耗很大。1F1B 在稳态时交替 forward+backward，最多只有 PP-1 个 microbatch 的激活同时存活，内存友好。

**Q7：如何确认我的 PP schedule 没有数值错误？**

用 PP=1 和 PP=2 分别训练相同步数（相同数据），比较 loss 曲线。如果两者完全一致（BF16 精度范围内），则 PP schedule 正确。如果 loss 从第 1 步就不同，说明进程组或数据分配有问题。

**Q8：`forward_only=True` 时 P2P 通信有何不同？**

`forward_only=True`（用于推理或评估）时，只做 forward 的 send/recv，不做 backward 的 send/recv，通信量减半。同时不保存 `input_tensors`/`output_tensors`，不存激活，内存减少约 PP 倍。

**Q9：`no_sync_func` 在 PP 中起什么作用？**

在流水线的多个 microbatch 处理期间，梯度不应该立即 AllReduce（因为还有梯度没算完）。`no_sync_func`（通常是 `DDP.no_sync()`）在 warmup 和 1F1B 期间抑制梯度同步，直到最后一个 microbatch 的 backward 完成才开启同步。

**Q10：PP 和 activation recomputation 同时使用时，每个 stage 的内存峰值是多少？**

使用 full activation checkpointing 时，每个 stage 在 warmup 阶段最多保存 `num_warmup_microbatches + 1` 个 microbatch 的激活（recomputation 只保留 checkpoint，不保留完整激活）。实际内存 ≈ `(PP_rank_offset + 1) × per_microbatch_activation × compression_ratio`（checkpointing 通常压缩 8-20×）。

---

## 17. 性能调优清单

运行 PP 训练前，按以下顺序检查：

1. **m/PP 比值**：确保 `num_microbatches >= PP`，最好 `>= 4×PP`，使气泡率 < 20%。
2. **P2P 后端**：单机内（NVLink 环境）用 NCCL；多机间（IB 环境）考虑 UCC（需评估）。
3. **`deallocate_pipeline_outputs`**：开启以节省内存（生产训练推荐开启）。
4. **VPP 权衡**：若 m < 4×PP，考虑 VPP=2 来减少气泡；若 IB 带宽有限，不要用 VPP。
5. **activation checkpointing**：PP warmup 阶段激活内存峰值 = `num_warmup_microbatches × per_mb_activation`，若峰值过高请开启 checkpointing。
6. **SP + PP 组合**：SP 减少 P2P 数据量（1/TP 倍），同时节省激活内存，两者协同效果好。
7. **variable_seq_lengths**：若使用 packed seq，确保开启此选项；动态 shape 会增加少量 CPU 开销。

---

## 20. 附录 A：PP 通信量精确计算

以下给出 PP 通信量的精确公式，帮助估算 IB 带宽需求：

```
Per-microbatch P2P data (bytes):
  = seq_per_rank × micro_batch_size × hidden_size × bytes_per_element
  = (S / CP / TP_if_SP) × mbs × H × dtype_bytes

Total P2P per step (bytes, standard PP, no VPP):
  = num_microbatches × (PP - 1) × 2 × per_mb_data × 2  # ×2 for forward+backward
  # Note: (PP-1) P2P hops, 2 for send+recv per hop, 2 for forward+backward

Total P2P per step (bytes, VPP):
  = num_microbatches × V × (PP - 1) × 2 × per_mb_data × 2  # V倍 P2P 次数
```

**数值示例**（PP=4，m=16，V=2，SP=True，TP=4，S=4096，mbs=1，H=4096，BF16）：

```
seq_per_rank = 4096 / 4 = 1024
per_mb_data  = 1024 × 1 × 4096 × 2 = 8 MB
total_P2P_VPP = 16 × 2 × 3 × 2 × 8 × 2 = 3072 MB ≈ 3 GB per step

若 step 时间 ≈ 300 ms（约 10B 参数模型），IB 带宽需求 = 3 GB / 0.3 s = 10 GB/s
200 Gbps IB = 25 GB/s，有 2.5× 余量，足够
```

---

## 21. 附录 B：`no_sync_context` 生命周期图

```
训练 step 开始
  │
  ├─ disable_grad_sync()        ← 禁用所有 bucket 的异步通信
  │   no_sync_context.__enter__()
  │
  ├─ Warmup 阶段（只有 forward）
  │   [多个 microbatch 的梯度在本地累积，不触发 AllReduce]
  │
  ├─ 1F1B 稳态
  │   [每步 1 forward + 1 backward，梯度继续本地累积]
  │
  ├─ 最后一个 backward 前：enable_grad_sync()
  │   no_sync_context.__exit__()
  │   ← 开启 bucket 级别的异步 AllReduce
  │
  ├─ Cooldown 阶段最后一个 backward 前：enable_grad_sync()
  │   （非 first stage 可能在此处触发）
  │
  └─ finalize_model_grads()     ← 等待所有异步通信完成
```

这个设计保证了：梯度只在所有 microbatch 都处理完后才同步，避免了提前同步导致的梯度不完整问题。

---

## 22. 附录 C：激活检查点与 PP 内存峰值表

以 PP=4，H=4096，mbs=1，S=2048，BF16 为例（每层激活约 16 MB，full recompute 压缩 16×）：

| stage_rank | warmup 数 | peak 激活（无 checkpointing） | peak 激活（full recompute） |
|:-----------:|:---------:|:---:|:---:|
| 0 | 3 | 3 × 4层 × 16MB = 192 MB | 3 × 4层 × 1MB = 12 MB |
| 1 | 2 | 2 × 4层 × 16MB = 128 MB | 2 × 4层 × 1MB = 8 MB |
| 2 | 1 | 1 × 4层 × 16MB = 64 MB | 1 × 4层 × 1MB = 4 MB |
| 3 | 0 | 0（last stage 不保存 warmup） | 0 |

激活检查点在 PP 下的收益最大：stage 0（warmup=PP-1 最多）的激活从 192 MB 降到 12 MB（~16× 压缩）。这解释了为什么在大 PP 配置下激活检查点几乎是必需的。

---

## 23. 附录 D：PP 与 Gradient Accumulation 的关系

PP 中的 `num_microbatches` 等价于 gradient accumulation steps：

```
global_batch_size = micro_batch_size × DP × num_microbatches × PP
```

等效地：

```
num_microbatches = global_batch_size / (micro_batch_size × DP × PP)
```

这意味着增大 `global_batch_size` 或减小 `micro_batch_size`/`DP` 都会增大 `num_microbatches`，从而降低 PP 气泡率。在设计训练配置时，优先通过增大 `global_batch_size` 来同时获得：
1. 更小的气泡率（更高的 PP 效率）
2. 更稳定的梯度（更大的 batch 在统计上更稳定）
3. 更好的通信计算重叠（更多的 microbatch 并发）

权衡点：`global_batch_size` 过大会增加单步训练延迟（更多 microbatch 意味着更长的 step），同时可能影响收敛速度（learning rate 调整困难）。工业界通常把 `num_microbatches` 控制在 PP-16×PP 之间作为平衡点。

---

## 24. 深入：interleaved schedule 的 warmup 计算

对于 VPP 场景，warmup microbatch 数的计算比标准 1F1B 复杂（以下为逻辑近似，源码第 896-930 行）：

```python
# warmup = (PP_size - PP_rank - 1) * group_size  （标准 PP 的贡献）
#        + (num_model_chunks - 1) * group_size    （额外 virtual chunk 的贡献）
num_warmup = (PP_size - PP_rank - 1) * group_size
           + (V - 1) * group_size
```

以 V=2，PP=4，m=4，group_size=1 为例：

```
stage 0: warmup = (4-0-1)*1 + (2-1)*1 = 3 + 1 = 4
stage 1: warmup = (4-1-1)*1 + (2-1)*1 = 2 + 1 = 3
stage 2: warmup = (4-2-1)*1 + (2-1)*1 = 1 + 1 = 2
stage 3: warmup = (4-3-1)*1 + (2-1)*1 = 0 + 1 = 1
```

VPP 的 stage 3 warmup=1（而标准 PP 时 stage 3 warmup=0），这是因为 V=2 时即使最后一个物理 stage，也需要等 virtual chunk 0 先走一遍再处理 virtual chunk 1。

---

## 25. 完整 PP 配置速查表

| 参数 | 范围 | 影响 | 建议 |
|------|------|------|------|
| `pipeline_model_parallel_size` (PP) | 2-64 | 模型层切分、气泡率 | 越大气泡越多，需增大 m 补偿 |
| `virtual_pipeline_model_parallel_size` (V) | None, 2-16 | 气泡率/P2P通信量 | V=2 通常已够；IB 带宽有限时慎用 |
| `num_microbatches` (m) | 1-∞ | 气泡率、内存 | 目标: m >= 4×PP（气泡<20%） |
| `pipeline_model_parallel_comm_backend` | nccl, ucc | P2P 带宽/SM占用 | NVLink 用 nccl；IB 可试 ucc |
| `deallocate_pipeline_outputs` | False/True | warmup 内存 | 生产环境建议 True |
| `num_microbatches_with_partial_activation_checkpoints` | None/int | warmup 内存 | 内存紧张时设非 None |
| `variable_seq_lengths` | False/True | packed seq 支持 | 使用 packed seq 时必须 True |

---

## 27. 调试实战：最小复现脚本

以下脚本在 CPU（gloo 后端）上模拟 PP=2 的 1F1B 通信模式，用于在无 GPU 的环境下验证逻辑：

```python
import os, torch, torch.distributed as dist
from functools import partial

def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    pp_rank = rank  # 每个进程对应一个 PP stage
    is_first = (pp_rank == 0)
    is_last  = (pp_rank == world_size - 1)
    next_rank = rank + 1 if not is_last else None
    prev_rank = rank - 1 if not is_first else None

    # 模拟 warmup=1（PP=2，stage 0 warmup=1，stage 1 warmup=0）
    tensor = torch.tensor([float(rank)], dtype=torch.float32)

    if is_first:
        # Stage 0 warmup: 发送 forward 激活
        dist.send(tensor, dst=next_rank)
        print(f"[stage {rank}] warmup: sent forward")
    else:
        # Stage 1: 接收 forward 激活
        recv_buf = torch.zeros(1)
        dist.recv(recv_buf, src=prev_rank)
        print(f"[stage {rank}] warmup: recv forward = {recv_buf.item()}")

    dist.barrier()
    print(f"[stage {rank}] P2P test passed")
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
```

运行此脚本确认 P2P send/recv 顺序正确，是调试更复杂 schedule 的第一步。

---

## 26. 练习题

**题目 2**：PP=4，m=4 时，各 stage 的 warmup 轮数分别是多少？总气泡率是多少？

**题目 3**：VPP=2，PP=4，m=8 时，气泡率比标准 PP（VPP=1）减少了多少？额外的 P2P 通信次数是多少（与标准 PP 相比）？

**题目 4**：阅读 `get_schedule_table` 的源码（第 954-981 行），对 V=2，PP=2，m=4，microbatch_group_size=1 手工生成完整的 `schedule_table`（参考本文第 8.3 节），验证结果与本文一致。

**题目 5**：在 PP=4，m=8，SP=True，TP=4，CP=1，S=4096，H=4096 的配置下，计算每个 microbatch 的 P2P 通信量（字节数），以及整个 step 的总 P2P 通信量。

---

## 19. 小结

流水线并行的核心是用 1F1B 调度减少空闲时间：

1. **warmup 公式**：`warmup(r) = min(m, PP - r - 1)`，最后一个 stage warmup=0，最前一个 stage warmup=PP-1（当 m≥PP-1 时）。三阶段（warmup/1F1B稳态/cooldown）各有清晰职责：warmup 填充流水线，稳态用 `send_forward_recv_backward` 合并双向通信，cooldown 清空剩余 backward。
2. **P2PCommunicator**：把正向和反向的 P2P 通信封装为 `send_forward`、`recv_backward`、`send_forward_recv_backward` 等 API，稳态下通过合并发送+接收操作减少延迟。P2P tensor shape 受 SP 和 CP 影响（`S/TP/CP`）。
3. **loss_func 只在最后 stage 调用**：返回 2 元组 `(loss, loss_dict)` 或 3 元组 `(loss, loss_dict, num_tokens)`；num_tokens 通过 PP broadcast + DP AllReduce 传播到所有 rank。
4. **VPP** 把气泡减少约 1/V，代价是 P2P 通信量增加 V 倍；模型表示为 `List[nn.Module]`；`get_schedule_table` 生成 `(microbatch_id, model_chunk_id)` 的执行序列。
5. **deallocate_output_tensor** 在激活发送后立即释放数据内存，只保留计算图节点，减少 warmup 期内存峰值。
6. **死锁防护**：严格的 send/recv 配对顺序、统一的 `P2PCommunicator` 接口、以及 NCCL 超时设置是避免 PP hang 的三道防线。

下一篇（07）将深入数据并行：DDP 缓冲、bucket 设计、`finalize_model_grads` 的八步流程，以及 Distributed Optimizer 的内存节省原理。
