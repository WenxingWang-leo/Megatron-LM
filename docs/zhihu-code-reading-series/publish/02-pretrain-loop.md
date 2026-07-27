# 精读 Megatron 源码（2）：完整拆解一次训练——从 `__main__` 到 `optimizer.step`

> **前置阅读**：本文假设你已读过第 1 篇（地图篇），熟悉 TP/PP/DP/microbatch/global batch 等基本概念。本篇目标：用"探针法"从程序入口一路追踪，直到参数被更新，在脑海中建立完整的训练循环印记。

---

## 0. 阅读策略：探针法

"探针法"的思路很简单：在关键函数入口加一行 `print`，把程序跑起来，观察输出。即使没有 GPU，我们也可以在脑子里"跑"这个过程——带着"此刻是哪张卡在执行，数据形状是什么"的问题，逐行阅读。

我们的线索是这条调用链：

```
__main__
  └─ parse_and_validate_args()
  └─ gpt_config_from_args()
  └─ pretrain()
       └─ initialize_megatron()
       └─ setup_model_and_optimizer()
       └─ train()
            └─ train_step()
                 └─ zero_grad_buffer()
                 └─ optimizer.zero_grad()
                 └─ forward_backward_func()
                      └─ forward_step() [pretrain_gpt.py]
                           └─ get_batch()
                           └─ model(tokens, ...)
                 └─ optimizer.step()
                 └─ opt_param_scheduler.step()
```

---

## 1. `__main__` 块：程序入口的每一行

文件：`/workspace/pretrain_gpt.py`，第 495-530 行。

```python
if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()                          # (1)

    # 打印版本信息
    print_rank_0(f'> PyTorch version ... {get_torch_version()}')
    print_rank_0(f'> Megatron-Core version ... {mcore_version}')

    set_startup_timestamps(...)                              # (2) 记录启动时间戳

    setattr(train_valid_test_datasets_provider,              # (3) 标记分布式数据集
            "is_distributed", True)

    # 可选：inprocess restart 包装
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)  # (4)

    args = parse_and_validate_args(                          # (5) 解析并验证所有参数
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )

    model_cfg = gpt_config_from_args(args)                   # (6) 构建 TransformerConfig
    full_config = pretrain_cfg_container_from_args(args, model_cfg)  # (7)

    pretrain(                                                 # (8) 启动训练
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
```

**逐行解析**：

- **(1) 时间戳**：`_PROGRAM_START_TIME`（文件最顶部）和 `_MAIN_ENTRY_TIME` 之差 = Python 导入所有库的耗时。在大集群上，这可以超过 60 秒。
- **(5) `parse_and_validate_args`**：这是**最重要的第一步**。它不仅解析命令行参数，还在 `validate_args` 中做大量合法性检查，并**计算派生参数**（如 `data_parallel_size`）。下一节详解。
- **(6) `gpt_config_from_args`**：将 `args` 转换为 `TransformerConfig` dataclass。这是 Megatron Core 层的入口——从这里开始，代码不再直接读 `args`，而是读 `config`。
- **(8) `pretrain`**：一切的起点，见第 3 节。

---

## 2. `validate_args`：参数验证的"守门人"

文件：`/workspace/megatron/training/arguments.py`，第 388 行起。

这个函数做了超过 400 行的检查，但最关键的是以下几条：

### 2.1 计算 DP size

```python
# megatron/training/arguments.py, 第 411-425 行
total_model_size = (
    args.tensor_model_parallel_size *     # TP
    args.pipeline_model_parallel_size *   # PP
    args.context_parallel_size            # CP
)
assert args.world_size % total_model_size == 0
args.data_parallel_size = args.world_size // total_model_size
```

**公式**：`DP = world_size / (TP × PP × CP)`

这里 CP（Context Parallelism）也参与了"模型并行"的总大小计算，因为 CP 也是将一个序列的 tokens 分配到多张卡上，每张卡只看一段，所以和 TP/PP 一样，同一 CP rank 的多张卡持有的是"同一份模型"的一部分。

### 2.2 验证 batch size

```python
# 伪代码（实际分散在多处）
num_microbatches = global_batch_size / (micro_batch_size * data_parallel_size)
assert num_microbatches == int(num_microbatches)  # 必须是整数
```

如果 `global_batch_size = 128`，`micro_batch_size = 2`，`DP = 8`，则 `num_microbatches = 128 / (2 × 8) = 8`。这 8 个 microbatch 会在流水线中依次流动。

### 2.3 验证注意力头整除性

```python
# megatron/training/arguments.py, 第 1249-1251 行
if args.kv_channels is None:
    assert args.hidden_size % args.num_attention_heads == 0
    args.kv_channels = args.hidden_size // args.num_attention_heads
```

还有隐含的检查：`num_attention_heads % TP == 0`（在 ColumnParallelLinear 的 QKV 投影时强制）。这是一个不在 validate_args 里但会在模型构建时触发的检查。

---

## 3. `pretrain()`：训练总指挥

文件：`/workspace/megatron/training/training.py`，第 1013 行。

```python
def pretrain(cfg_container, train_valid_test_dataset_provider, model_type,
             forward_step_func, ...):
    # 步骤 1: 初始化 Megatron（分布式 + 随机种子）
    initialize_megatron(...)

    args = get_args()

    # 步骤 2: 构建模型、优化器、LR scheduler
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider_func, model_type, ...
    )

    # 步骤 3: 构建数据集和 dataloader
    # (train/valid/test datasets, build samplers)

    # 步骤 4: 可选地从 checkpoint 恢复
    iteration = 0
    if args.load:
        iteration, num_floating_point_operations_so_far = load_checkpoint(...)

    # 步骤 5: 训练主循环
    iteration, num_floating_point_operations_so_far = train(
        forward_step_func, model, optimizer, opt_param_scheduler, ...
    )
```

整个过程是线性的：**初始化 → 建模 → 准备数据 → 恢复断点 → 训练**。

---

## 4. `initialize_megatron()`：分布式初始化的全过程

文件：`/workspace/megatron/training/initialize.py`，第 42 行。

调用链：

```
initialize_megatron()
  └─ _initialize_distributed()
       └─ torch.distributed.init_process_group()      # (A) 初始化 torch.distributed
       └─ mpu.initialize_model_parallel(              # (B) 创建所有并行进程组
               tensor_model_parallel_size=TP,
               pipeline_model_parallel_size=PP,
               ...)
            └─ megatron/core/parallel_state.py
  └─ _set_random_seed()                               # (C) 设置随机种子
```

### 4.1 进程组初始化（关键！）

`mpu.initialize_model_parallel` 会创建以下进程组（Process Group）：

| 进程组类型 | 含义 | 举例（TP=2, PP=2, DP=4, world=16） |
|-----------|------|-----------------------------------|
| TP group | 协作完成单层矩阵乘法 | `[0,1], [2,3], [4,5], ...` |
| PP group | 流水线各 stage | `[0,4], [1,5], [2,6], ...` |
| DP group | 梯度同步 | `[0,2,8,10], [1,3,9,11], ...` |
| CP group | 序列切分（可选） | 取决于 CP size |
| EP group | Expert 分组（可选） | 取决于 num_experts |

每个 rank 只属于**每种类型的一个**进程组。`mpu.get_tensor_model_parallel_group()` 返回**当前 rank 所在的 TP group**。

### 4.2 随机种子：不同 PP stage 使用不同种子

文件：`/workspace/megatron/training/initialize.py`，第 402-423 行。

```python
def _set_random_seed(seed_, ...):
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    seed = seed_ + (100 * pp_rank)   # ← 关键！不同 PP stage 种子不同
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensor_parallel.model_parallel_cuda_manual_seed(seed, ...)
```

**为什么不同 PP stage 需要不同种子？**

Dropout 是有随机性的。如果所有 PP stage 用同一个种子，那么第 1 层和第 9 层的 Dropout 模式会完全相同，破坏了层间的独立性，影响模型质量。

**记忆公式**：`seed_for_pp_rank_r = base_seed + 100 × r`

---

## 5. `setup_model_and_optimizer()`：模型构建与 PP pre/post_process

文件：`/workspace/megatron/training/training.py`，第 1999 行 → `get_model`，第 1691 行。

### 5.1 pre_process / post_process 表（PP=4 时）

| PP Stage | PP Rank | `pre_process` | `post_process` | 持有的组件 |
|----------|---------|---------------|----------------|-----------|
| 0 | 0 | **True** | False | `embedding` + layers 0~7 |
| 1 | 1 | False | False | layers 8~15 |
| 2 | 2 | False | False | layers 16~23 |
| 3 | 3 | False | **True** | layers 24~31 + `output_layer` + loss |

**代码逻辑（非 VPP 情形）**：

```python
# megatron/training/training.py, 第 1741-1748 行
pre_process = is_pp_first_stage(pg_collection.pp)   # PP rank == 0?
post_process = is_pp_last_stage(pg_collection.pp)   # PP rank == last?
model = model_provider_func(
    pre_process=pre_process,
    post_process=post_process,
    ...
)
```

### 5.2 VPP 时的 pre_process / post_process（PP=4, VPP=2）

VPP 时，每个 PP rank 持有 2 个 chunk（虚拟阶段），每个 chunk 是一个独立的 GPTModel 实例：

```python
# megatron/training/training.py, 第 1718-1739 行
for i in range(vp_size):  # i = 0, 1 (VPP=2)
    pre_process = is_pp_first_stage() and is_vp_first_stage(vp_stage=i, vp_size=vp_size)
    post_process = is_pp_last_stage() and is_vp_last_stage(vp_stage=i, vp_size=vp_size)
    model.append(model_provider_func(pre_process=pre_process,
                                     post_process=post_process,
                                     vp_stage=i))
```

| PP rank | VPP chunk (vp_stage) | `pre_process` | `post_process` |
|---------|---------------------|---------------|----------------|
| 0 | 0 | **True** | False |
| 0 | 1 | False | False |
| 1 | 0 | False | False |
| 1 | 1 | False | False |
| 2 | 0 | False | False |
| 2 | 1 | False | False |
| 3 | 0 | False | False |
| 3 | 1 | False | **True** |

---

## 6. `train_step()`：单步训练的全部操作

文件：`/workspace/megatron/training/training.py`，第 2290 行。

```python
def train_step(forward_step_func, data_iterator, model, optimizer,
               opt_param_scheduler, config, forward_backward_func, ...):

    # ── 步骤 1：清零梯度缓冲区 ──
    for model_chunk in model:
        model_chunk.zero_grad_buffer()    # 清零分布式优化器的 grad buffer

    # ── 步骤 2：清零优化器梯度 ──
    optimizer.zero_grad()

    # ── 步骤 3：前向 + 反向传播（所有 microbatch）──
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
    )

    # ── 步骤 4：参数更新 ──
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    # ── 步骤 5：LR scheduler 步进 ──
    if update_successful:
        opt_param_scheduler.step(increment=get_num_microbatches())

    return losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
```

### 6.1 为什么有两个 "zero grad"？

`zero_grad_buffer()` 清零的是 **DDP/DistributedOptimizer 的梯度缓冲区**（`param.main_grad`），这个缓冲区在内存中是连续的大块 buffer，专为高效通信设计。

`optimizer.zero_grad()` 清零的是 **PyTorch 标准的 `param.grad`**。

在 Megatron 的 DistributedOptimizer 模式下，`param.main_grad` 才是真正存梯度的地方，`param.grad` 可能指向同一块内存或为 None。**必须两个都清零**，否则梯度会在多步之间累积。

---

## 7. `get_batch()`：数据是怎么到达每张卡的

文件：`/workspace/pretrain_gpt.py`，第 97 行。

```python
def get_batch(data_iterator, vp_stage=None):
    args = get_args()
    cp_size = args.context_parallel_size
    tp_rank = mpu.get_tensor_model_parallel_rank()

    # ── 关键判断：中间 PP stage 直接返回 None ──
    if not is_first_or_last_pipeline_stage(vp_stage):
        return [None for _ in BATCH_KEYS]

    # ── 只有 TP rank 0 从 data_iterator 取数据 ──
    batch = {}
    if tp_rank == 0:
        batch = next(data_iterator)
        for key in BATCH_KEYS:
            batch[key] = batch[key].cuda(non_blocking=True)

    # ── 广播给同 TP group 的其他 rank ──
    batch = get_batch_on_this_tp_rank(
        batch,
        broadcast_src_rank=mpu.get_tensor_model_parallel_src_rank(),
        broadcast_group=mpu.get_tensor_model_parallel_group(),
        ...
    )

    # ── CP split：如果 CP > 1，按序列切分 ──
    batch = get_batch_on_this_cp_rank(batch, ...)

    return [batch[key] for key in BATCH_KEYS]
```

### 7.1 `BATCH_KEYS` 字段解释

```python
BATCH_KEYS = [
    "attention_mask",      # [1, 1, seq_len, seq_len]，因果掩码
    "cu_seqlens",          # packed sequence 的累积序列长度（SFT 模式）
    "cu_seqlens_padded",   # 对齐后的 cu_seqlens
    "hybrid_cp_group",     # Hybrid CP 的进程组
    "labels",              # [batch, seq_len]，next-token labels
    "local_cp_size",       # 本地 CP 大小
    "loss_mask",           # [batch, seq_len]，哪些位置计入 loss
    "max_seqlen",          # packed sequence 的最大长度
    "position_ids",        # [batch, seq_len]，位置 ID
    "tokens",              # [batch, seq_len]，输入 token IDs
]
```

**注意字母序**：BATCH_KEYS 按字母序排列，这样 `sorted(batch.keys())` 的结果和 `BATCH_KEYS` 一致，保证多处解包时的顺序一致性。

### 7.2 中间 PP stage 为什么返回 None？

流水线并行的中间 stage（PP rank 1, 2, ...最后-1）不需要数据，它们的输入来自**上一个 stage 通过 P2P 通信传来的激活值**。所以：

- PP stage 0（`pre_process=True`）：需要 `tokens`, `position_ids`, `attention_mask`
- PP stage 中间：只需要 `None`（激活从 P2P 得到）
- PP stage 最后（`post_process=True`）：需要 `labels`, `loss_mask`

但如果使用了 `cu_seqlens`（packed sequence），中间 stage 也需要 `cu_seqlens` 用于计算注意力，所以会有特殊处理。

---

## 8. `forward_step()`：前向步的返回值契约

文件：`/workspace/pretrain_gpt.py`，第 282 行。

```python
def forward_step(data_iterator, model: GPTModel, return_schedule_plan=False):
    # ... 获取 batch ...
    
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )
    
    # !! 关键：返回 (output_tensor, loss_func_partial)
    return output_tensor, partial(loss_func, loss_mask, model=model)
```

**返回值的契约**：

1. `output_tensor`：
   - 若 `labels` 不为 None（最后 PP stage）：形状为 `[batch, seq_len]`，每个位置的交叉熵损失
   - 若 `labels` 为 None（中间 PP stage）：形状为 `[seq_len, batch, hidden_size]`，隐藏状态
   - 注意：在 Megatron 中，Transformer 的张量形状习惯是 `[S, B, H]`（Sequence-first），而不是 PyTorch 常见的 `[B, S, H]`

2. `partial(loss_func, loss_mask, model=model)`：
   - 是一个**偏函数**，等待 `output_tensor` 来计算最终标量 loss
   - **只有最后 PP stage 才真正执行这个函数**，其他 stage 的这个偏函数会被 schedule 忽略

这个设计允许 `forward_backward_func`（流水线调度）将 forward 和 loss 的计算解耦，便于在多个 microbatch 之间做流水线调度。

---

## 9. `loss_func()`：loss 计算详解

文件：`/workspace/pretrain_gpt.py`，第 209 行。

```python
def loss_func(loss_mask, output_tensor, model=None):
    # output_tensor: [batch * seq_len]，每个 token 的交叉熵
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    
    # 只对 loss_mask==1 的位置计算 loss（padding 和 EOS 不计）
    loss = torch.sum(losses * loss_mask)
    
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), 
                                     num_tokens.view(1)])}
    
    return loss, num_tokens, report
```

**重要细节**：

- `output_tensor` 是**每个 token 的 loss**（交叉熵），不是标量
- `loss_mask` 用来屏蔽填充 token 和不需要预测的位置
- 返回值中的 `report` 是用于日志记录的字典，包含 loss 总和和有效 token 数
- `loss / num_tokens` 才是真正的平均 loss（normalized loss）

---

## 10. 数值例子：全程追踪一个 batch

### 配置
- `world_size = 256`，`TP = 8`，`PP = 4`，`CP = 1`
- `DP = 256 / (8 × 4 × 1) = 8`
- `micro_batch_size = 2`，`global_batch_size = 128`
- `num_microbatches = 128 / (2 × 8) = 8`
- `seq_length = 4096`，`hidden_size = 8192`，`num_layers = 32`

### 每张卡持有多少层？

```
总层数 = 32
PP 切分：每个 PP stage 持有 32 / 4 = 8 层
```

| PP rank | 层 | 累计参数（粗估，仅 Transformer 层） |
|---------|----|------------------------------------|
| 0 | 0-7 | ~16B × (8/32) = ~4B |
| 1 | 8-15 | ~4B |
| 2 | 16-23 | ~4B |
| 3 | 24-31 | ~4B |

注：这里是 32 层 / PP=4 = 8 层/stage，实际参数大小取决于 hidden_size 等。

### 数据在 get_batch 中的变化

```
原始数据（在 TP rank 0 + PP first/last stage 上）:
  tokens: [2, 4096]  (micro_batch_size=2, seq_len=4096)

CP 切分（CP=1，不切）:
  tokens: [2, 4096]  (不变)

TP 广播后（所有 TP rank 都有相同 tokens）:
  tokens: [2, 4096]  (相同副本，每个 TP rank 都有)

中间 PP stage:
  return [None, None, None, None, None, None, None, None, None, None]
  (10 个 BATCH_KEYS 对应的 None)
```

### 一步 train_step 的时间线

```
时刻 T=0: 所有 rank 进入 train_step
T=1: zero_grad_buffer(), optimizer.zero_grad()
T=2: forward_backward_func 开始（num_microbatches=8）
  T=2.1: microbatch #1 在 PP stage 0 前向
  T=2.2: microbatch #1 激活 P2P 发送到 PP stage 1
  T=2.3: microbatch #2 在 PP stage 0 前向（流水线填充）
  ...（1F1B schedule，8 个 microbatch 依次流水）...
  T=2.n: 反向传播逆向流回
T=3: DP group 内梯度 AllReduce（或 ReduceScatter）
T=4: optimizer.step()（在 DP shard 内更新参数）
T=5: opt_param_scheduler.step()（更新 LR）
T=6: 返回 losses_reduced
```

---

## 11. 调用图（ASCII 版）

```
pretrain_gpt.py __main__
│
├─ parse_and_validate_args()         [arguments.py]
│   └─ validate_args()
│       ├─ DP = world/(TP*PP*CP)
│       ├─ num_microbatches 整除性检查
│       └─ batch size 合理性检查
│
├─ gpt_config_from_args()            [argument_utils.py]
│   └─ 返回 TransformerConfig(hidden_size, num_layers, ...)
│
└─ pretrain()                        [training.py]
    │
    ├─ initialize_megatron()         [initialize.py]
    │   ├─ _initialize_distributed()
    │   │   ├─ torch.distributed.init_process_group()
    │   │   └─ mpu.initialize_model_parallel(TP, PP, CP, ...)
    │   └─ _set_random_seed(base_seed + 100*pp_rank)
    │
    ├─ setup_model_and_optimizer()   [training.py]
    │   ├─ get_model()
    │   │   ├─ [VPP] for i in range(vp_size):
    │   │   │     pre = is_pp_first & is_vp_first(i)
    │   │   │     post = is_pp_last & is_vp_last(i)
    │   │   │     model_chunk = model_provider(pre, post, vp_stage=i)
    │   │   └─ [no VPP] model_provider(pre, post)
    │   ├─ wrap_with_ddp(model)
    │   └─ get_megatron_optimizer(model)
    │
    └─ train()                       [training.py]
        └─ for iteration in range(train_iters):
               └─ train_step()
                   ├─ model.zero_grad_buffer()
                   ├─ optimizer.zero_grad()
                   ├─ forward_backward_func(         [schedules.py]
                   │     num_microbatches=N,
                   │     forward_step_func=forward_step,
                   │     ...)
                   │   └─ for each microbatch:
                   │       ├─ forward_step(data_iter, model)  [pretrain_gpt.py]
                   │       │   ├─ get_batch()
                   │       │   │   ├─ [TP rank 0] next(data_iter)
                   │       │   │   ├─ broadcast to TP group
                   │       │   │   └─ [CP>1] split along seq dim
                   │       │   └─ model(tokens, pos_ids, mask, labels)
                   │       │         └─ GPTModel.forward()   [gpt_model.py]
                   │       └─ loss_func(loss_mask, output)   [pretrain_gpt.py]
                   │
                   ├─ optimizer.step()
                   └─ opt_param_scheduler.step()
```

---

## 12. Debug 断点表：在哪里加断点最有收益

| 断点位置 | 文件 | 行号（参考） | 能观察到什么 |
|---------|------|------------|------------|
| `validate_args` 末尾 | `arguments.py` | ~800 | `args.data_parallel_size`, `args.num_microbatches` |
| `_initialize_distributed` 末尾 | `initialize.py` | ~350 | 进程组是否正确创建，打印 rank 映射 |
| `_set_random_seed` | `initialize.py` | ~423 | 每个 rank 的实际 seed 值 |
| `get_model` 的 `pre_process/post_process` | `training.py` | ~1741 | 哪个 rank 持有 embedding/output_layer |
| `get_batch` 返回前 | `pretrain_gpt.py` | ~181 | batch shape，None 验证 |
| `forward_step` 返回前 | `pretrain_gpt.py` | ~358 | output_tensor shape |
| `train_step` 的 `optimizer.step()` | `training.py` | ~2428 | update_successful, grad_norm |

**推荐的最小调试命令**：

```python
# 在 train_step 开头加：
import torch.distributed as dist
print(f"[rank {dist.get_rank()}] "
      f"pp_rank={mpu.get_pipeline_model_parallel_rank()} "
      f"tp_rank={mpu.get_tensor_model_parallel_rank()} "
      f"dp_rank={mpu.get_data_parallel_rank()} "
      f"iteration={iteration}")
```

这一行输出就能验证进程组配置是否正确。

---

## 13. 常见问题 FAQ

**Q：为什么 `forward_step` 的 `output_tensor` 在中间 PP stage 是 hidden state，而不是 loss？**

A：只有最后 PP stage 才有 `labels`，才能计算 loss。中间 stage 的 `model.forward(labels=None)` 返回 hidden state `[S, B, H]`，这个 hidden state 会通过 P2P 通信传给下一个 stage。

**Q：DistributedOptimizer 的 `zero_grad_buffer` 和普通的 `optimizer.zero_grad()` 有什么区别？**

A：Megatron 的 DistributedOptimizer 使用连续内存缓冲区（contiguous buffer）存储梯度，以便高效地进行集合通信（ReduceScatter）。`zero_grad_buffer()` 清零这个大 buffer，而 `optimizer.zero_grad()` 清零 `param.grad`。两者都需要清零，因为它们可能是不同的内存区域。

**Q：流水线 bubble 是什么，怎么产生的？**

A：在 1F1B（one-forward-one-backward）调度中，流水线填满需要 PP-1 个 microbatch，排空也需要 PP-1 个。这个"填满和排空"的过程中，某些 stage 是空闲的，这段空闲时间就是 bubble。bubble 占比 ≈ `(PP-1) / num_microbatches`。

---

## 14. 课后练习

**练习 1**：计算以下配置的 `num_microbatches`：
- `world_size=32`, `TP=4`, `PP=2`, `CP=1`
- `micro_batch_size=4`, `global_batch_size=256`

答案：DP = 32/(4×2×1) = 4，num_microbatches = 256/(4×4) = 16

**练习 2**：在 `pretrain_gpt.py` 的 `get_batch` 中，如果 `cp_size > 1`，序列会怎么被切分？提示：查看 `get_batch_on_this_cp_rank` 的实现（`megatron/core/utils.py`）。

**练习 3**：阅读 `megatron/core/num_microbatches_calculator.py` 的 `get_num_microbatches()` 函数。Megatron 支持动态调整 global batch size（rampup），在 rampup 阶段，`num_microbatches` 是如何变化的？

**练习 4（实验题）**：启动一个 `TP=2, PP=1` 的 2 卡训练，在 `get_batch` 中打印 `tp_rank` 和 `batch` 是否为 None。验证你对"TP rank 0 取数据，广播给 TP rank 1"这个过程的理解。

**练习 5（思考题）**：如果 `forward_step` 的返回值不是 `(output_tensor, loss_func)` 的元组，而是直接返回 `loss`（标量），流水线调度会遇到什么问题？

---

## 15. 本文小结

通过本文，你应该能够：

1. **背诵** `__main__` 块的 8 步：时间戳 → 版本 → inprocess → parse_args → config → pretrain
2. **解释** `validate_args` 的三大核心检查：DP 计算、batch 整除性、heads 整除性
3. **画出** `pretrain()` 的 5 个阶段：初始化 → 建模 → 数据 → 恢复 → 训练
4. **理解** `train_step` 的 5 步：zero_grad_buffer → zero_grad → forward_backward → optimizer.step → scheduler.step
5. **解释** 为什么 `get_batch` 中间 PP stage 返回 None，以及 TP 广播的必要性
6. **追踪** 一个具体数值例子（world=256 TP=8 PP=4 DP=8）的全流程

下一篇，我们将进入 `GPTModel.forward()` 的内部，拆解 Config/Spec 体系和每一层的计算。

---

*上一篇：[精读 Megatron 源码（1）：从零建立心智模型](./01-megatron-map.md)*
*下一篇：[精读 Megatron 源码（3）：GPTModel 全拆解](./03-gptmodel-spec.md)*
