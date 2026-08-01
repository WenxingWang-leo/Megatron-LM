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
- **(4) `inprocess_restart`**：这是 NVIDIA 容错训练扩展（`nvidia_resiliency_ext`）的 hook。当某张 GPU 出现 CUDA OOM 或其他错误时，可以在不重启整个作业的情况下重新初始化这个 rank。普通训练中可以忽略这一步。
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

## 3. `pretrain()`：训练总指挥——六个阶段带时间戳

文件：`/workspace/megatron/training/training.py`，第 1013 行。

`pretrain()` 函数超过 500 行，但结构清晰地分为六个阶段。以下是带时间戳记录的注解版：

```python
def pretrain(cfg_container, train_valid_test_dataset_provider, model_type,
             forward_step_func, ...):

    # ── 阶段 0：进入时间戳 ──
    _STARTUP_TIMESTAMPS['pretrain_entry'] = time.time()

    # ── 阶段 1：容错与 JIT 设置 ──
    ft_integration.setup()                # 容错（FaultTolerance）初始化
    set_jit_fusion_options(...)           # 设置 JIT 融合（减少 CUDA kernel 启动开销）

    # ── 阶段 2：初始化 Megatron ──
    initialize_megatron(...)              # 分布式 + 随机种子
    # 打印初始化耗时（全局同步后的最小启动时间）
    # "time to initialize megatron (seconds): X.XXX"

    # ── 阶段 3：构建模型与优化器 ──
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_type, model_provider_func=model_provider, ...
    )
    timers('model-and-optimizer-setup').stop()
    # 打印: "after model, optimizer, and learning rate scheduler are built"

    # ── 阶段 4：构建数据集和 dataloader ──
    timers('train/valid/test-data-iterators-setup', log_level=0).start(barrier=True)
    train_data_iterator, valid_data_iterator, test_data_iterator = (
        build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
    )
    timers('train/valid/test-data-iterators-setup').stop()
    # 打印: "after dataloaders are built"

    # ── 阶段 5：可选地从 checkpoint 恢复 ──
    iteration = 0
    if args.load:
        iteration, num_floating_point_operations_so_far = load_checkpoint(
            model, optimizer, opt_param_scheduler
        )
    # 打印: "after loading checkpoint"（如果有 checkpoint）

    # ── 阶段 6：训练主循环 ──
    if not cfg_container.validation.skip_train and args.do_train:
        iteration, num_floating_point_operations_so_far = train(
            forward_step_func, model, optimizer, opt_param_scheduler, ...
        )
    # 训练结束后进行最终验证和测试
```

**从日志识别阶段**：运行 Megatron 时，可以通过以下 rank 0 输出识别当前所处阶段：

```
time to initialize megatron (seconds): X.XXX    → 阶段 2 完成
after model, optimizer, ...                      → 阶段 3 完成
after dataloaders are built                      → 阶段 4 完成
successfully loaded checkpoint from ...          → 阶段 5 完成
[2024-01-01 00:00:01.000000] iteration      1/...→ 阶段 6 开始
```

### 3.1 checkpoint 恢复时 args 的交互

从 checkpoint 恢复时有一个关键行为：`load_checkpoint` 会将 checkpoint 中保存的 `consumed_train_samples`、`consumed_valid_samples`、`iteration` 等状态恢复到 `args` 中。

```python
# megatron/training/checkpointing.py（精简）
def load_checkpoint(model, optimizer, opt_param_scheduler, ...):
    state_dict = load_state_dict_from_disk(args.load)
    
    # 恢复 iteration 计数
    iteration = state_dict.get('iteration', 0)
    
    # 恢复消耗样本数（用于数据集采样的确定性恢复）
    if 'consumed_train_samples' in state_dict:
        args.consumed_train_samples = state_dict['consumed_train_samples']
    
    # 恢复优化器状态（包括 Adam 的 m/v 矩阵和 LR scheduler 状态）
    optimizer.load_state_dict(state_dict['optimizer'])
    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
    
    return iteration, state_dict.get('num_floating_point_operations_so_far', 0)
```

**为什么需要恢复 `consumed_train_samples`？**

Megatron 的数据集使用确定性采样——给定相同的 `consumed_train_samples` 和随机种子，总能生成相同的数据序列。这保证了即使重启训练，数据流也能从中断处精确续接，不会重复或遗漏样本。

**恢复后 `num_microbatches` 的一致性检查**：

```python
# megatron/core/num_microbatches_calculator.py（StepBatchsizeNumMicroBatchesCalculator）
def update(self, consumed_samples, consistency_check=True, ...):
    self.current_global_batch_size = self._get_batch_size_for_samples(consumed_samples)
    if consistency_check:
        assert (
            self.current_global_batch_size % self.micro_batch_times_data_parallel_size == 0
        ), ...
```

从 checkpoint 恢复时，`update_num_microbatches(consumed_samples, consistency_check=True)` 会验证当前 batch size 能被 `mbs × DP` 整除。如果你在恢复后改变了 `DP`（例如增加机器），这个检查会失败，提示配置不一致。

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
        model_chunk.force_all_reduce = save_wgrads_in_this_iteration

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
    timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers('optimizer').stop()

    # ── 步骤 5：LR scheduler 步进 ──
    if update_successful:
        opt_param_scheduler.step(increment=get_num_microbatches())

    return losses_reduced, skipped_iter, grad_norm, num_zeros_in_grad
```

### 6.1 为什么有两个 "zero grad"？

`zero_grad_buffer()` 清零的是 **DDP/DistributedOptimizer 的梯度缓冲区**（`param.main_grad`），这个缓冲区在内存中是连续的大块 buffer，专为高效通信设计。

`optimizer.zero_grad()` 清零的是 **PyTorch 标准的 `param.grad`**。

在 Megatron 的 DistributedOptimizer 模式下，`param.main_grad` 才是真正存梯度的地方，`param.grad` 可能指向同一块内存或为 None。**必须两个都清零**，否则梯度会在多步之间累积。

### 6.2 `update_successful` 的含义

`optimizer.step()` 返回三个值：
- `update_successful`：本步是否真的更新了参数（False 表示梯度溢出，跳过本步）
- `grad_norm`：梯度的全局范数（用于监控训练稳定性）
- `num_zeros_in_grad`：梯度中零元素的数量（调试用）

仅当 `update_successful=True` 时，LR scheduler 才步进。这确保了当训练过程中出现 NaN/Inf 梯度（在 fp16 混合精度训练时常见）时，学习率调度不会错误地前进。

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

## 10. `get_num_microbatches` 与梯度累积的精确关系

文件：`/workspace/megatron/core/num_microbatches_calculator.py`

### 10.1 公式推导

```
num_microbatches = global_batch_size / (micro_batch_size × data_parallel_size)
```

这个公式直接来自 `ConstantNumMicroBatchesCalculator.__init__`：

```python
micro_batch_times_data_parallel_size = micro_batch_size * data_parallel_size
num_micro_batches = global_batch_size // micro_batch_times_data_parallel_size
```

**物理意义**：
- `global_batch_size`：每次参数更新消耗的总样本数
- `micro_batch_size × data_parallel_size`：一个 microbatch 轮次中，所有 DP replica 共同处理的样本数
- 因此，`num_microbatches` 就是"需要多少轮 microbatch 才能凑齐一个 global batch"

### 10.2 梯度累积的实际发生位置

很多人误以为 Megatron 有"显式的梯度累积步骤"，实际上并没有——梯度累积是**隐式发生**在以下两个地方：

1. **流水线中的多次反向传播**：`forward_backward_func` 内部对每个 microbatch 调用一次反向传播，每次反向都将梯度**累加**到 `param.main_grad`（而不是覆盖）。这就是梯度累积。

2. **DP AllReduce/ReduceScatter**：在所有 `num_microbatches` 个微批次都跑完之后，才触发 DP 组内的梯度同步（AllReduce 或 ReduceScatter）。

```python
# 伪代码展示梯度累积位置
for microbatch_idx in range(num_microbatches):
    output = model.forward(microbatch)
    loss = loss_func(output)
    loss.backward()                    # 梯度累加到 param.main_grad

# num_microbatches 个微批次全部完成后
optimizer.step()                       # 参数更新（内部触发 DP 梯度同步）
```

### 10.3 `num_microbatches` 的动态变化（阶梯调度）

使用 `--step-batch-size-schedule` 时，`num_microbatches` 会在训练过程中变化。调用时序是：

```
train_step 开始
  ↓
get_num_microbatches()   ← 读取当前值
  ↓
forward_backward_func(num_microbatches=...)
  ↓
optimizer.step()
  ↓
update_num_microbatches(consumed_samples)  ← 更新计算器（可能切换到新的 gbs）
```

**数值示例**：假设 `--step-batch-size-schedule "0:768 250B:1536"`, `mbs=1`, `DP=8`, `seq=2048`:
- 消耗样本 < 250B/2048 = 122M 时：gbs=768, num_microbatches = 768/(1×8) = 96
- 消耗样本 ≥ 122M 时：gbs=1536, num_microbatches = 1536/(1×8) = 192

---

## 11. timers / 日志 / 吞吐量指标解读

### 11.1 训练日志的典型格式

Megatron 在每 `--log-interval` 步打印一次日志（默认 100 步），格式如下：

```
[2024-01-01 00:05:23.456789]
iteration     100/  1000 |
consumed samples:        800 |
elapsed time per iteration (ms): 1234.5 |
throughput per GPU (TFLOP/s/GPU): 312.4 |
learning rate: 1.000000E-04 |
global batch size:   128 |
lm loss: 3.456789E+00 |
loss scale: 65536.0 |
grad norm: 1.234 |
num zeros: 0 |
```

**逐字段解释**：

| 字段 | 含义 | 正常范围 |
|------|------|----------|
| `elapsed time per iteration (ms)` | 单步平均耗时 | 取决于硬件，几百到几千 ms |
| `throughput per GPU (TFLOP/s/GPU)` | 每卡每秒 TFLOP，衡量计算效率（MFU 的分子） | A100 理论峰值 312 TFLOP/s；通常能达到 35-55% |
| `lm loss` | 语言模型交叉熵损失 | 初始约 log(vocab_size)，e.g. log(32000)≈10；训练中逐步下降 |
| `loss scale` | fp16 动态损失缩放系数 | 65536 是正常值；持续降低说明梯度溢出频繁 |
| `grad norm` | 参数梯度的全局 L2 范数 | 通常 0.5-5.0；持续 > 10 或 NaN 需警惕 |
| `num zeros` | 梯度中零元素数量 | 小数值正常；极高可能说明训练崩溃 |

### 11.2 `throughput` 的计算方式

吞吐量（TFLOP/s/GPU）由 `num_floating_point_operations` 计算（`megatron/training/training.py`，第 2764 行）：

```python
throughput = num_floating_point_operations(args, batch_size, ...) \
             / (elapsed_time_per_iteration * 10**12 * args.world_size)
```

`num_floating_point_operations` 使用一个估算公式，大致是：
```
FLOPs ≈ 6 × num_parameters × num_tokens_per_step
       (前向 2× + 反向 4×，对于 Transformer 的矩阵乘法)
```

**如何提高 MFU**：
- 增大 `micro_batch_size` 或 `num_microbatches`（更大的 GEMM 批次）
- 减少 PP bubble（增大 `num_microbatches`，或使用 VPP）
- 启用 Flash Attention（减少注意力的内存带宽瓶颈）
- 启用 SP（Sequence Parallel，减少 LayerNorm 的计算开销）
- 启用 FP8（在 H100 上理论 FLOPs 翻倍）

### 11.3 `StragglerDetector`：落后者检测

`megatron/core/utils.py` 中的 `StragglerDetector` 是一个单例类，用于检测集群中的"落后者"（某张卡比其他卡慢的情况）：

```python
# megatron/training/training.py
stimer = StragglerDetector()

# 在每个 train_step 中：
with stimer:
    losses_reduced = forward_backward_func(...)
```

通过 CUDA events 精确计时每张卡的前向/反向时间，然后收集所有 rank 的数据，报告最慢的若干 rank 及其延迟。这对于排查集群故障（某张卡性能下降、网络抖动等）非常有用。启用方式：`--straggler-ctrlr-port <port>` 或通过 `StragglerDetector.configure()` 方法。

---

## 12. 数值例子 A：world=256 TP=8 PP=4 DP=8 配置全程追踪

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

## 13. 数值例子 B：world=8 TP=2 PP=2 DP=2 小规模追踪（每 rank 视角）

这个小例子让你能在脑子里完整模拟所有 8 个 rank 的行为。

**配置**：
- `world_size=8`, `TP=2`, `PP=2`, `CP=1`
- `DP = 8 / (2×2×1) = 2`
- `micro_batch_size=1`, `global_batch_size=8`
- `num_microbatches = 8 / (1×2) = 4`
- `seq_length=128`, `num_layers=4`（每 PP stage 2 层）

**Rank 分配**（Megatron 默认的进程组布局）：

```
TP group 0: ranks [0, 1]
TP group 1: ranks [2, 3]
TP group 2: ranks [4, 5]
TP group 3: ranks [6, 7]

PP group 0: ranks [0, 1, 4, 5]  (TP rank 0 + TP rank 2)
PP group 1: ranks [2, 3, 6, 7]  (TP rank 1 + TP rank 3)

在 PP group 0 中:
  PP stage 0: ranks [0, 1]   → pre_process=True,  post_process=False
  PP stage 1: ranks [4, 5]   → pre_process=False, post_process=True

DP group:
  DP pair 1: PP group 0 的 rank 0,1 与 PP group 1 的 rank 2,3 → DP group [0,2] 和 [1,3]
  DP pair 2: PP group 0 的 rank 4,5 与 PP group 1 的 rank 6,7 → DP group [4,6] 和 [5,7]
```

**4 个 microbatch 的流水线时序（PP=2，真正的 1F1B）**：

> 注意：下面这张才是 Megatron `without_interleaving` 的 1F1B。  
> **错误画法**（曾误写成「全前向再全反向」，且 stage0 在 stage1 还在做 `F4` 时就开始 `B4`）是不合法的：  
> stage1 必须先完成某个 microbatch 的 Forward，再做该 microbatch 的 Backward，并把梯度 P2P 回 stage0，stage0 才能开始对应的 Backward。

```
warmup(r) = min(m, PP - r - 1)
  → stage0 (ranks 0,1): warmup=1
  → stage1 (ranks 4,5): warmup=0

时间 →     1    2    3    4    5    6    7    8
rank 0,1  F1   F2   B1   F3   B2   F4   B3   B4
rank 4,5       F1   B1   F2   B2   F3   B3   F4   B4
```

依赖读法（以 microbatch 4 为例）：

1. t=6：stage0 做完 `F4`，把激活 P2P 发给 stage1  
2. t=7：stage1 做 `F4`，立刻做 `B4`，把梯度 P2P 回 stage0  
3. t=8：stage0 收到梯度后做 `B4`

因此 **stage1 上「`F4` 之后紧接着 `B4`」**；**stage0 的 `B4` 必须更晚一拍**，绝不能和 stage1 的 `F4` 画在同一列。

若画成 GPipe（先灌满全部 Forward，再统一 Backward），正确依赖应是：

```
时间 →     1    2    3    4    5    6    7    8    9
rank 0,1  F1   F2   F3   F4   ·    B4   B3   B2   B1
rank 4,5       F1   F2   F3   F4   B4   B3   B2   B1
```

这里 stage1 在 t=5 做 `F4`、t=6 做 `B4`；stage0 在 t=5 空等（`·`），t=7 才做 `B4`。  
Megatron 默认训练路径用的是上面的 **1F1B**，不是 GPipe。

阶段划分（1F1B）：

- `F1` = microbatch 1 的前向；`B1` = microbatch 1 的反向  
- **warm-up**：stage0 先推 `F1`（t=1）；stage1 warmup=0  
- **稳态 1F1B**：交替 Forward/Backward（stage0：`F2 B1 F3 B2 F4 B3`；stage1：`F1 B1 ... F3 B3`）  
- **cooldown**：收尾 Backward（stage0 的 `B4`；stage1 的 `F4 B4` 落在末尾两拍）

流水线 bubble 比例（近似）≈ `(PP-1) / num_microbatches = (2-1)/4 = 25%`

**每个 rank 在 `get_batch` 中的行为**：

| rank | 角色 | `tokens` 值 |
|------|------|-------------|
| 0, 1 | PP stage 0, TP rank 0/1 | rank 0 从 data_iterator 取数据，广播给 rank 1 |
| 4, 5 | PP stage 1, TP rank 0/1 | 返回 None（等 P2P 激活） |
| 2, 3 | PP stage 0 的 DP replica（另一条流水线） | rank 2 取数据，广播给 rank 3 |
| 6, 7 | PP stage 1 的 DP replica | 返回 None |

---

## 14. `train()` 主循环的内部结构

`train()` 函数（`training.py` 第 2087 行起）是 `pretrain()` 调用 `train_step` 的入口，它还管理验证、checkpoint 保存和日志输出。

```python
def train(forward_step_func, model, optimizer, opt_param_scheduler, ...):
    args = get_args()
    timers = get_timers()

    # 每 --eval-interval 步做一次验证
    # 每 --save-interval 步保存一次 checkpoint
    # 每 --log-interval 步打印一次日志

    done_with_persistent_ckpt = False
    iteration = args.iteration   # 从 checkpoint 恢复的起始 iteration

    timers('interval-time', log_level=0).start(barrier=True)

    for iteration in range(args.iteration, args.train_iters):
        # ── 前置：更新 num_microbatches（阶梯 batch size 调度）──
        update_num_microbatches(consumed_samples=args.consumed_train_samples,
                                consistency_check=True)

        # ── 核心：执行一步训练 ──
        losses_reduced, skipped_iter, should_checkpoint, should_exit, ..., \
            grad_norm, num_zeros_in_grad, num_fp_ops = train_step(
            forward_step_func, train_data_iterator, model, optimizer,
            opt_param_scheduler, config, forward_backward_func, iteration,
        )

        # ── 后置：更新计数器 ──
        iteration += 1
        args.consumed_train_samples += get_current_global_batch_size()
        num_floating_point_operations_so_far += num_fp_ops

        # ── 验证 ──
        if args.eval_interval and iteration % args.eval_interval == 0:
            evaluate_and_print_results(prefix, forward_step_func, ...)

        # ── 保存 checkpoint ──
        if args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer, ...)

        # ── 日志输出 ──
        if iteration % args.log_interval == 0:
            training_log(loss_dict, total_loss_dict, learning_rate, ...)

    return iteration, num_floating_point_operations_so_far
```

### 关键计数器说明

| 计数器 | 每步增量 | 保存在 | 用途 |
|--------|---------|--------|------|
| `iteration` | +1 | `args.iteration` | 控制训练循环终止、checkpoint 保存间隔 |
| `consumed_train_samples` | +`current_gbs` | `args.consumed_train_samples` | 数据集的确定性采样起点；batch size 调度 |
| `num_floating_point_operations_so_far` | +`num_fp_ops` | 不保存在 args，传给 ckpt | 计算吞吐量（TFLOP/s）；记录总计算量 |

`consumed_train_samples` 是**恢复训练最关键的状态之一**：它确保 checkpoint 重载后，数据集从正确的位置续接，不会重复或遗漏任何样本。

### `training_log` 打印的内容

```python
# megatron/training/training.py，第 2781-2835 行（精简版）
log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}]"
log_string += ' iteration {:8d}/{:8d} |'.format(iteration, args.train_iters)
log_string += ' consumed samples: {:12d} |'.format(args.consumed_train_samples)
log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(...)
if args.log_throughput:
    log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
log_string += f' learning rate: {learning_rate:.6E} |'
log_string += f' global batch size: {batch_size:5d} |'
# ... 各损失项
log_string += f' loss scale: {loss_scale:.1f} |'
if grad_norm is not None:
    log_string += f' grad norm: {grad_norm:.3f} |'
```

通过这个输出，可以监控：
- `elapsed time per iteration`：单步耗时（越稳定越好）
- `throughput per GPU`：计算效率（越高越好）
- `lm loss`：语言模型损失（应稳定下降）
- `loss scale`：fp16 动态缩放（若持续降低，说明梯度爆炸）
- `grad norm`：梯度范数（异常高 → 不稳定，NaN → 崩溃）

---

## 15. 调用图（ASCII 版）

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
    ├─ [可选] load_checkpoint()      [checkpointing.py]
    │   ├─ 恢复 iteration 计数
    │   ├─ 恢复 consumed_train_samples
    │   └─ 恢复 optimizer + scheduler 状态
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

## 16. "step 0 loss 是 NaN" 诊断树

遇到训练一开始就 NaN，按以下顺序排查：

```
loss 在 step 0 是 NaN？
    │
    ├─ 检查 grad_norm 是否也是 NaN ──→ 是
    │   │
    │   ├─ 是 fp16/bf16 训练吗？
    │   │   ├─ 是 → 检查 loss scale：若很小（< 1），说明溢出频繁
    │   │   │         解决：降低 lr，检查 weight init，或切换到 bf16
    │   │   └─ 否 → 检查输入数据是否有 NaN（tokens 全零等异常情况）
    │   │
    │   └─ 检查 TP/PP > 1 时的数值：
    │       └─ 是否 only 某些 rank 的 grad_norm 是 NaN？
    │           ├─ 是 → P2P 通信数据损坏，检查网络/NCCL 版本
    │           └─ 否 → 所有 rank 均 NaN，继续往下排查
    │
    ├─ loss 是 NaN 但 grad_norm 正常 ──→ 
    │   └─ 检查 loss_func：loss_mask 是否全零（没有有效 token）
    │       └─ 可能原因：数据格式错误，labels 全为 padding id
    │
    └─ 怀疑是模型初始化问题
        ├─ 检查 weight init：scale 是否合理（LLaMA 用 std=0.02）
        ├─ 检查是否有 logit 爆炸：在 _postprocess 中打印 logits 的 max/min
        └─ 尝试禁用 recompute（--no-recompute-granularity）看是否影响
```

**最常见的 NaN 原因及解决方案**：

1. **学习率过大**：将 `--lr` 降低 10 倍
2. **fp16 溢出**：改用 `--bf16`（bf16 不需要 loss scaling，不会溢出）
3. **位置编码 overflow**：RoPE 的 `rotary_base` 过小，在超长序列上可能溢出，尝试增大 `--rotary-base`
4. **embedding 初始化**：vocab embedding 的 `std` 参数与 LLM 规模不匹配，检查 `config.init_method`
5. **数据问题**：token ids 超出 vocab_size，或 loss_mask 全为 0

---

## 17. Debug 断点表：在哪里加断点最有收益

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

## 18. 常见问题 FAQ（12 题）

**Q1：为什么 `forward_step` 的 `output_tensor` 在中间 PP stage 是 hidden state，而不是 loss？**

A：只有最后 PP stage 才有 `labels`，才能计算 loss。中间 stage 的 `model.forward(labels=None)` 返回 hidden state `[S, B, H]`，这个 hidden state 会通过 P2P 通信传给下一个 stage。

**Q2：DistributedOptimizer 的 `zero_grad_buffer` 和普通的 `optimizer.zero_grad()` 有什么区别？**

A：Megatron 的 DistributedOptimizer 使用连续内存缓冲区（contiguous buffer）存储梯度，以便高效地进行集合通信（ReduceScatter）。`zero_grad_buffer()` 清零这个大 buffer，而 `optimizer.zero_grad()` 清零 `param.grad`。两者都需要清零，因为它们可能是不同的内存区域。

**Q3：流水线 bubble 是什么，怎么产生的？**

A：在 1F1B（one-forward-one-backward）调度中，流水线填满需要 PP-1 个 microbatch，排空也需要 PP-1 个。这个"填满和排空"的过程中，某些 stage 是空闲的，这段空闲时间就是 bubble。bubble 占比 ≈ `(PP-1) / num_microbatches`。

**Q4：`update_successful=False` 时，参数没有更新，LR 也没有步进。那这一步的 microbatch 数据是否被"浪费"了？**

A：是的，那些数据被"跳过"了（consumed_samples 不会增加）。这在 fp16 训练中偶有发生，通常不影响最终效果，因为 `loss_scale` 会在溢出后自动调整，使后续步骤恢复正常。如果 `skipped_iters` 占比超过 5%，就需要排查稳定性问题。

**Q5：`opt_param_scheduler.step(increment=get_num_microbatches())` 为什么传的是 microbatch 数而不是 1？**

A：LR scheduler 基于"已消耗步数"来调整 LR。在 rampup 阶段（动态 gbs），`num_microbatches` 可能随着 gbs 增大而增大，scheduler 需要感知到这个变化，确保学习率的衰减/warmup 按照实际"等效步数"进行。

**Q6：如果不使用 `DistributedOptimizer`，梯度同步在哪里发生？**

A：如果使用标准的 DDP（`DistributedDataParallel`），梯度同步发生在 `backward()` 调用中，通过钩子（`grad_fn.register_hook`）自动触发 AllReduce。Megatron 的 DDP 实现延迟了这个时机，在 `finalize_model_grads()` 中统一触发，以支持梯度桶的重叠通信。

**Q7：什么是 `consumed_train_samples`，它如何影响数据采样？**

A：`consumed_train_samples` 记录到目前为止总共处理了多少个训练样本。Megatron 的数据集使用这个值作为 sampler 的起点，确保 checkpoint 恢复后数据流精确续接。公式：`consumed_train_samples += global_batch_size`（每步加一个 gbs）。

**Q8：如果我修改了 PP 大小重新加载 checkpoint 会怎样？**

A：通常会失败。checkpoint 中保存的参数按 PP 切分存储（每个 PP rank 只保存自己的 layers），如果改变 PP 大小，你需要使用 `tools/checkpoint/` 下的转换脚本合并/重新切分 checkpoint。直接加载会因 state_dict 的 key 不匹配而报错。

**Q9：`global_batch_size` 会影响模型的收敛质量吗？**

A：会。Megatron 的 gbs 直接决定每步的有效样本数，相当于"批大小"，影响梯度的噪声水平。通常 gbs 增大后需要相应调高学习率（如线性缩放规则：`lr ∝ sqrt(gbs)` 或 `lr ∝ gbs`），否则训练可能变慢。

**Q10：`timers` 是什么，日志里 `interval-time` 代表什么？**

A：`timers` 是 `megatron/training/global_vars.py` 中的 `Timers` 对象，通过 `get_timers()` 访问。`interval-time` timer 在每个 `--log-interval` 步的 log 时重置，测量从上一次 log 到现在的总时间，除以步数得到每步平均耗时。

**Q11：`pretrain()` 结束后会打印什么，怎么知道训练是否正常完成？**

A：正常完成会依次打印：
1. 最后一次 log（iteration = train_iters）
2. `after training is done`
3. `done with training ...`（如果有验证，还会打印 valid/test 结果）

如果训练提前终止（如 timeout、OOM），则不会打印这些行，可以通过日志文件的最后几行判断。

**Q12：`inprocess_restart` 是什么情况下会生效？**

A：当某个 rank 遇到 CUDA 错误或其他异常时，`inprocess_restart` 机制（来自 `nvidia_resiliency_ext`）可以让所有 rank 重置到上一个 checkpoint 并重新开始训练，而无需终止整个 SLURM/K8s 作业。`maybe_wrap_for_inprocess_restart(pretrain)` 在 `pretrain` 函数外包装了一个重试循环。如果没有安装 `nvidia_resiliency_ext`，这个调用是 no-op，返回原始的 `pretrain` 函数。

---

## 19. 课后练习

**练习 1**：计算以下配置的 `num_microbatches`：
- `world_size=32`, `TP=4`, `PP=2`, `CP=1`
- `micro_batch_size=4`, `global_batch_size=256`

答案：DP = 32/(4×2×1) = 4，num_microbatches = 256/(4×4) = 16

**练习 2**：在 `pretrain_gpt.py` 的 `get_batch` 中，如果 `cp_size > 1`，序列会怎么被切分？提示：查看 `get_batch_on_this_cp_rank` 的实现（`megatron/core/utils.py`）。

**练习 3**：阅读 `megatron/core/num_microbatches_calculator.py` 的 `get_num_microbatches()` 函数。Megatron 支持动态调整 global batch size（`--step-batch-size-schedule`），在切换阶段，`num_microbatches` 是如何变化的？

**练习 4（实验题）**：启动一个 `TP=2, PP=1` 的 2 卡训练，在 `get_batch` 中打印 `tp_rank` 和 `batch` 是否为 None。验证你对"TP rank 0 取数据，广播给 TP rank 1"这个过程的理解。

**练习 5（思考题）**：如果 `forward_step` 的返回值不是 `(output_tensor, loss_func)` 的元组，而是直接返回 `loss`（标量），流水线调度会遇到什么问题？

**练习 6（诊断题）**：训练 10 步后 loss 稳定在 10.8 不下降（约等于 log(50000)），而你的词表大小是 50000。这意味着什么，可能的原因有哪些？

---

## 20. 本文小结

通过本文，你应该能够：

1. **背诵** `__main__` 块的 8 步：时间戳 → 版本 → inprocess → parse_args → config → pretrain
2. **解释** `validate_args` 的三大核心检查：DP 计算、batch 整除性、heads 整除性
3. **画出** `pretrain()` 的 6 个阶段：容错初始化 → 建模 → 数据 → checkpoint 恢复 → 训练 → 评估
4. **理解** `train_step` 的 5 步：zero_grad_buffer → zero_grad → forward_backward → optimizer.step → scheduler.step
5. **解释** 为什么 `get_batch` 中间 PP stage 返回 None，以及 TP 广播的必要性
6. **追踪** 两个具体数值例子（world=256 和 world=8）的全流程
7. **使用诊断树**排查 step 0 NaN 的常见原因
8. **解读训练日志**中的 throughput、grad_norm、loss_scale 字段

下一篇，我们将进入 `GPTModel.forward()` 的内部，拆解 Config/Spec 体系和每一层的计算。

---

*上一篇：[精读 Megatron 源码（1）：从零建立心智模型](./01-megatron-map.md)*
*下一篇：[精读 Megatron 源码（3）：GPTModel 全拆解](./03-gptmodel-spec.md)*
