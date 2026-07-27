# 精读 Megatron 源码（8）：数据管线、DistributedOptimizer 与 Dist Checkpoint 完全精读

> **专栏**：Megatron 源码精读 · 第 8 篇  
> **核心文件**：  
> - `megatron/core/datasets/indexed_dataset.py`  
> - `megatron/core/datasets/gpt_dataset.py`  
> - `megatron/core/datasets/blended_megatron_dataset_builder.py`  
> - `megatron/core/optimizer/__init__.py`  
> - `megatron/core/optimizer/optimizer.py`  
> - `megatron/core/optimizer/distrib_optimizer.py`  
> - `megatron/core/dist_checkpointing/mapping.py`  
> - `megatron/core/dist_checkpointing/serialization.py`  
> - `megatron/training/checkpointing.py`  
>
> **上一篇**：[精读 Megatron 源码（7）：数据并行完全精读](./07-data-parallel.md)  
> **下一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)

---

前面七篇把"怎么算、怎么切、怎么同步梯度"全部覆盖。训练闭环还差三块：

1. **样本怎样变成 batch**（文件格式 → DataLoader → get_batch）
2. **优化器状态怎样分片更新**（DistributedOptimizer / ZeRO-1）
3. **checkpoint 怎样按并行布局保存与恢复**（ShardedTensor 体系）

本篇按数据流从输入到存储的顺序精读，每节都给出「可运行推演」帮助深刻理解。

---

## 一、数据管线：从 `.bin/.idx` 到 `get_batch`

### 1.1 数据流全景

```
磁盘 .bin + .idx 文件
  ↓  IndexedDataset（mmap 只读）
  ↓  GPTDataset（shuffle 索引 + __getitem__）
  ↓  BlendedMegatronDatasetBuilder（多源按权重混合）
  ↓  DataLoader → iterator
  ↓  get_batch（TP broadcast、CP 切分、PP None）
  ↓  forward_step → 模型
```

涉及的关键文件如下表：

| 文件 | 职责 |
|------|------|
| `indexed_dataset.py` | 低层 `.bin/.idx` 随机访问 |
| `gpt_dataset.py` | 构建 tokens/labels/loss_mask/position_ids |
| `blended_megatron_dataset_builder.py` | 多源数据集按权重采样混合 |
| `pretrain_gpt.py` | `datasets_provider`、`get_batch` |

---

### 1.2 IndexedDataset：O(1) 随机访问的秘密

```
文件: megatron/core/datasets/indexed_dataset.py
```

原始语料经过预处理工具（`tools/preprocess_data.py`）生成两个文件：

| 文件 | 内容 |
|------|------|
| `prefix.bin` | 所有 token id 的二进制流，dtype 通常 `uint16`/`int32` |
| `prefix.idx` | 元数据：每条序列的 **长度**（`sequence_lengths`）和 **字节偏移**（`pointers`） |

`IndexedDataset.__init__` 读取 `.idx`：

```python
# indexed_dataset.py 核心逻辑（简化）
self._index = _IndexReader(idx_path)      # 解析 header + lengths + pointers
self._bin_buffer = numpy.memmap(          # mmap，不占 RAM
    bin_path, dtype=self._index.dtype, mode='r'
)
```

**为什么是 O(1) 访问？**  
`.idx` 文件存储了每条序列相对于 `.bin` 文件头的字节偏移量 `pointers[i]`。取第 `i` 条序列时：

```python
ptr   = self._index.pointers[i]           # 字节偏移，O(1) 数组查找
size  = self._index.sequence_lengths[i]   # 序列长度，O(1)
numpy_slice = self._bin_buffer[ptr : ptr + size]
```

这是经典的"索引文件 + 数据文件"设计，常见于 LMDB / LevelDB 等系统。

**mmap 的作用**：操作系统的页缓存负责管理哪些页真正在内存中，训练程序以"访问普通 numpy 数组"的方式使用，不需要手动 `seek`/`read`。数据集远大于 GPU 节点内存时，mmap 是最优选择。

**对象存储支持**：`object_storage_utils.py` 实现了 S3 / MSC 路径下先把 `.idx` 缓存到本地再 mmap 的逻辑，用 `is_object_storage_path(dataset_path)` 分支控制。

---

### 1.3 GPTDataset.__getitem__：tokens/labels 是怎么拼出来的

```python
# megatron/core/datasets/gpt_dataset.py  第 230-350 行（精简）
def __getitem__(self, idx):
    text = self._get_text(idx)   # shape: [seq_len + add_extra_token]

    if self.config.add_extra_token_to_sequence:   # 默认 True
        tokens = text[:-1].contiguous()           # [0 .. seq_len-1]
        labels = text[1:].contiguous()            # [1 .. seq_len]
    else:
        tokens = text
        labels = torch.roll(text, shifts=-1, dims=0)
        labels[-1] = self._pad_token_id

    attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        tokens,
        self.config.eod_token_id,
        self.config.reset_position_ids,
        self.config.reset_attention_mask,
        self.config.eod_mask_loss,
    )
    ...
    return {"tokens": tokens, "labels": labels,
            "loss_mask": loss_mask, "position_ids": position_ids, ...}
```

**核心设计决策 `add_extra_token_to_sequence=True`**：

预处理时，每条 sample 从数据集中取 `seq_len + 1` 个 token。这样 `tokens[i]` 与 `labels[i]` 分别是序列的第 `i` 和第 `i+1` 个 token，实现自回归语言模型的标准 next-token 预测，且 tokens 和 labels **均为完整的 seq_len 长度**，不浪费任何 token。

**`loss_mask` 是怎么来的**：

```python
# _get_ltor_masks_and_position_ids
loss_mask = torch.ones(tokens.shape, dtype=torch.float)
if eod_mask_loss:
    loss_mask[tokens == eod_token_id] = 0.0   # EOD token 不计入损失
```

之后 `__getitem__` 还会把 PAD token 对应位置的 `loss_mask` 清零，确保填充位置不影响梯度。

**position_ids**：默认是 `[0, 1, 2, ..., seq_len-1]`；若设置了 `reset_position_ids=True`，遇到 EOD token 会重置为 0，适合把多篇文档拼接训练时保持每篇独立的位置信息。

---

### 1.4 worked example：手工推演 text=[10,20,30,40,50]

假设 `seq_len=4`，`add_extra_token=True`，`eod_token_id=0`，无 PAD：

```
text           = [10, 20, 30, 40, 50]   # 5 个 token（seq_len+1）
tokens         = [10, 20, 30, 40]       # text[:-1]
labels         = [20, 30, 40, 50]       # text[1:]
loss_mask      = [1., 1., 1., 1.]       # 无 EOD/PAD → 全 1
position_ids   = [0, 1, 2, 3]
```

模型用 `tokens` 做 forward，用 `labels` 计算交叉熵，`loss_mask` 对每个位置的 loss 加权：

```
cross_entropy(logit[i], labels[i]) * loss_mask[i]
```

---

### 1.5 shuffle 索引与可缓存性

`GPTDataset.__init__` 调用 `_build_document_sample_shuffle_indices`，生成三个数组：

| 数组 | 含义 |
|------|------|
| `document_index` | 每个 epoch 文档的访问顺序（随机打乱） |
| `sample_index` | 每个 sample 对应的文档范围（支持跨文档拼接） |
| `shuffle_index` | sample 级别的访问顺序 |

这三个数组**会被缓存到磁盘**（`.npy` 文件），缓存 key 包含所有影响采样的参数。这意味着：

1. **第一次运行**：从头构建，速度慢（数十亿 token 数据集可能需要数分钟）。
2. **之后再运行**：直接加载缓存，启动极快。
3. **可复现性**：只要参数相同，同一 seed 产生完全相同的 shuffle，方便复现实验。

---

### 1.6 BlendedMegatronDatasetBuilder：多源混合采样

```
文件: megatron/core/datasets/blended_megatron_dataset_builder.py
```

CLI 支持：

```bash
--data-path 0.6 /path/to/pile 0.3 /path/to/c4 0.1 /path/to/books
```

`BlendedMegatronDatasetBuilder.build` 的逻辑：

1. 对每个数据源独立构建 `GPTDataset`（或 `MockGPTDataset`）。
2. 按权重比例 `[0.6, 0.3, 0.1]` 预先计算每个数据集应贡献多少 sample。
3. 构建 `BlendedDataset`，内部维护一个映射 `global_idx → (dataset_id, local_idx)`，确保按权重混合。

**mock 模式**：当没有真实数据时，可以使用 `MockGPTDataset`：

```python
# blended_megatron_dataset_builder.py
if config.mock:
    return MockGPTDataset(config)
```

`MockGPTDataset.__getitem__` 返回随机 token id，适合调试模型代码（不依赖数据预处理），也是新手入手 Megatron 的最简路径。

---

### 1.7 get_batch：让数据"懂并行"

```python
# pretrain_gpt.py  BATCH_KEYS 定义（第 83-95 行附近）
BATCH_KEYS = [
    "attention_mask",
    "cu_seqlens",
    "labels",
    "local_cp_size",
    "loss_mask",
    "position_ids",
    "tokens",
    ...
]
```

`get_batch` 的完整逻辑分三步：

**步骤 1：只有 TP rank 0 和 PP 首尾 stage 从 DataLoader 读数据**

```python
# pretrain_gpt.py get_batch
if mpu.get_tensor_model_parallel_rank() != 0:
    # TP 的非 rank 0 收广播即可
    ...
if not mpu.is_pipeline_first_stage() and not mpu.is_pipeline_last_stage():
    return [None for _ in BATCH_KEYS]   # 中间 PP stage 不需要真实数据
```

对于流水线并行中间 stage，返回全 None，因为它们的输入来自上一 stage 的激活，而非原始输入。**这是 PP 中最容易忽略的细节**：如果你断点调试，会发现中间 stage 的 `tokens` 是 None，并不是 bug。

**步骤 2：TP rank 0 广播给同 TP group 其他 rank**

```python
batch = get_batch_on_this_tp_rank(
    data_iterator,
    is_pipeline_last_stage=mpu.is_pipeline_last_stage(),
)
```

`get_batch_on_this_tp_rank` 内部用 `torch.distributed.broadcast` 把 TP rank 0 读到的 batch 广播给 TP group 里其他 rank，保证所有 TP rank 处理完全相同的序列。

**步骤 3：CP 切分序列**

```python
batch = get_batch_on_this_cp_rank(
    batch,
    is_hybrid_cp=...,
    cp_group=get_context_parallel_group(),
)
```

Context Parallel（CP=N）把序列维度切 N 份，每个 CP rank 只处理自己那段。切分策略是"zigzag"交错，原因在第 10 篇详述。切完后每个 rank 拿到的 `tokens` 形状从 `[B, seq_len]` 变为 `[B, seq_len/CP]`。

---

## 二、优化器：DistributedOptimizer / ZeRO-1

### 2.1 优化器层次结构

```
get_megatron_optimizer
  └─ _get_param_groups（分 dense / expert 两类）
  └─ _get_megatron_optimizer_based_on_param_groups
       ├─ MixedPrecisionOptimizer（BF16 模型参数 / FP32 主参数）
       │    └─ DistributedOptimizer（可选，ZeRO-1 分片）
       └─ ChainedOptimizer（多个优化器串联，针对 dense+expert 各一个）
```

**为什么分 dense 和 expert 两个优化器？**

在 MoE 模型中，expert 参数只存在于少数 GPU（EP 分组）上，它们的梯度**不需要 AllReduce**（只需在 EP 组内 ReduceScatter）。因此：

```python
# optimizer/__init__.py  第 349 行附近
is_expert_parallel = not getattr(param, 'allreduce', True)
```

非 expert 参数的 `allreduce=True`（默认），expert 参数的 `allreduce=False`，由此分成两个 `param_groups`，喂给两个独立的 `_get_megatron_optimizer_based_on_param_groups`，最终用 `ChainedOptimizer` 串联：

```python
return ChainedOptimizer(results)
```

---

### 2.2 MixedPrecisionOptimizer：BF16 训练的"双份参数"

```
文件: megatron/core/optimizer/optimizer.py
```

BF16 训练的核心矛盾：BF16 有足够的动态范围做 forward/backward，但精度只有约 3 位十进制，**累积优化器更新时会丢失细节**。

解决方案：

| 类型 | 设备 | 用途 |
|------|------|------|
| 模型参数（BF16） | GPU | forward / backward 计算 |
| master 参数（FP32） | GPU | 优化器状态存储 + 参数更新 |

`MixedPrecisionOptimizer` 管理这两份参数的同步：

```python
# optimizer.py  step 流程（简化）
def step(self):
    # 1. 把 BF16 梯度 unscale 并（可选）clip
    self._unscale_main_grads_and_check_for_nan()
    self.clip_grad_norm(...)
    # 2. 在 FP32 master 参数上做优化器更新（Adam / SGD）
    self._fp32_optimizer.step()
    # 3. 把更新后的 FP32 master 参数复制回 BF16 模型参数
    self._copy_main_params_to_model_params()
```

`_unscale_main_grads_and_check_for_nan`：把 BF16 梯度（`param.main_grad`）除以 loss scale，同时检测 Inf/NaN，若出现则跳过本次 step（loss scale 自动缩减）。

---

### 2.3 DistributedOptimizer：ZeRO-1 的 Megatron 实现

```
文件: megatron/core/optimizer/distrib_optimizer.py
```

**ZeRO-1 的核心思路**：把优化器状态（Adam 的 m、v 以及 master 参数）在 DP 组的所有 rank 间均匀切分，每个 rank 只负责更新自己持有的那一片参数，更新完后用 AllGather 把更新后的参数广播给所有 rank。

Megatron 的实现：

**阶段 1：Reduce-Scatter 梯度**

```
[rank0 grad_buf] = [g0, g1, g2, g3, ..., g15]   # DP=4, buffer 16 elems
                    ↓ ReduceScatter（sum + scatter）
rank0 拿到 sum(g0..g3) [0..3]
rank1 拿到 sum(g4..g7) [4..7]
rank2 拿到 sum(g8..g11) [8..11]
rank3 拿到 sum(g12..g15) [12..15]
```

**阶段 2：每个 rank 更新自己的那片参数**

```python
# distrib_optimizer.py  step 核心（简化）
for model_group, main_group in zip(self.model_float16_groups, self.main_param_groups):
    for model_param, main_param in zip(model_group, main_group):
        # 每个 rank 只更新自己 shard 对应的 main_param
        ...
self._fp32_optimizer.step()
```

**阶段 3：AllGather 参数**

```python
# 更新完 main_param 后，把各 rank 的 shard 合并回完整参数
self._all_gather_model_params()
```

**内存收益**：设 DP=N，模型参数量 P（FP32 master），优化器状态量约 2P（Adam 的 m+v）。ZeRO-1 下每 rank 只需存 3P/N（master params + m + v），而非 3P。

**`Range` 类**：`distrib_optimizer.py` 开头定义的 `Range(start, end)` 就是用来描述每个 DP rank 拥有 grad buffer 的哪个片段：

```python
class Range:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start
```

`_build_model_gbuf_param_range_map` 把整个 grad buffer 按 DP rank 数均匀切割，建立 `param → Range` 的映射，这样 ReduceScatter 后每个 rank 直接操作自己的 `Range`。

---

### 2.4 worked example：DP=4，buffer 16 个元素

```
整个 grad buffer:  [g0 g1 g2 g3 | g4 g5 g6 g7 | g8 g9 g10 g11 | g12 g13 g14 g15]
DP 切分（每 rank 4 个）：
  rank0 owns: [0..3]   → 持有 master_param[0..3]，更新后 AllGather
  rank1 owns: [4..7]   → 持有 master_param[4..7]，...
  rank2 owns: [8..11]
  rank3 owns: [12..15]

内存对比（16 元素 FP32 = 64 bytes）：
  standard opt:   3 * 64 bytes = 192 bytes / rank（每个 rank 存全量 m, v, fp32）
  DistOpt ZeRO-1: 3 * 16 bytes = 48 bytes  / rank（每 rank 只存 1/4）
```

更多细节参见官方文档：`docs/user-guide/features/dist_optimizer.md`。

---

### 2.5 ChainedOptimizer：多个优化器的透明串联

```python
# optimizer/optimizer.py  ChainedOptimizer
class ChainedOptimizer:
    def step(self):
        for opt in self.chained_optimizers:
            opt.step()
    def zero_grad(self, ...):
        for opt in self.chained_optimizers:
            opt.zero_grad(...)
```

对外接口与单个优化器完全一致，内部串联调用。dense 参数优化器 + expert 参数优化器 → 一个 `ChainedOptimizer` 传给训练主循环。

---

## 三、Distributed Checkpoint：并行感知的存档

### 3.1 为什么需要专门的分布式 checkpoint？

普通 `torch.save(model.state_dict())` 存储的是单进程视角的完整参数，对于 Megatron 这种多维并行来说有几个问题：

1. **TP/PP 切片**：每个 rank 只有参数的一个切片，无法简单 save 全量。
2. **换并行度**：从 TP=2 训练的 checkpoint 换到 TP=4 继续训练，需要重新切分。
3. **内存压力**：如果让 rank 0 聚合全量参数再保存，大模型会 OOM。

Megatron 的解决方案是 **`megatron/core/dist_checkpointing/`** 提供的 `ShardedTensor` 体系：每个 rank 描述"我持有的这块 tensor 在全局中的位置"，然后分别写盘，读取时按需重新拼接。

---

### 3.2 ShardedTensor 六大字段

```python
# dist_checkpointing/mapping.py
@dataclass
class ShardedTensor:
    key: str                         # 全局唯一 key，如 "decoder.layers.0.self_attention.query.weight"
    data: Optional[torch.Tensor]     # 本 rank 持有的局部 tensor
    dtype: torch.dtype               # tensor 数据类型
    local_shape: Tuple[int, ...]     # 本 rank tensor 的 shape
    global_shape: Tuple[int, ...]    # 全局（跨所有 rank）的完整 shape
    global_offset: Tuple[int, ...]   # 本 rank tensor 在全局 tensor 中的偏移（元素数）
    axis_fragmentations: Optional[Tuple[int, ...]]  # 每个轴被切分成多少份
    replica_id: ReplicaId = 0        # 副本编号（DP 副本时有用）
```

**直觉理解**：想象一个 `[4096, 4096]` 的 weight matrix 被 TP=4 横切成 4 份，每份 `[1024, 4096]`：

| rank | local_shape | global_offset | axis_fragmentations |
|------|-------------|---------------|---------------------|
| 0 | (1024, 4096) | (0, 0) | (4, 1) |
| 1 | (1024, 4096) | (1024, 0) | (4, 1) |
| 2 | (1024, 4096) | (2048, 0) | (4, 1) |
| 3 | (1024, 4096) | (3072, 0) | (4, 1) |

`global_slice()` 方法直接返回这块 tensor 对应全局 tensor 的 `slice(1024, 2048)` 等。

---

### 3.3 ShardedTensor.from_rank_offsets：最常用的构造方式

```python
# 典型用法（来自模型的 sharded_state_dict 方法）
ShardedTensor.from_rank_offsets(
    key="model.decoder.layers.0.mlp.fc1.weight",
    data=self.weight,                    # 本 rank 持有的切片
    (0, tp_rank, tp_size),               # 第 0 轴：tp_rank / tp_size
    replica_id=dp_rank,                  # DP 副本用 replica_id 区分
)
```

`from_rank_offsets` 根据 `(axis, rank, size)` 三元组自动推算 `global_offset` 和 `global_shape`，无需手动计算。

---

### 3.4 保存流程：generate_state_dict → save

```python
# training/checkpointing.py  save_checkpoint（简化）
def save_checkpoint(iteration, model, optimizer, ...):
    state_dict = generate_state_dict(
        args, model, optimizer, opt_param_scheduler, ...
    )
    # state_dict 里的 tensor 已经被替换成 ShardedTensor
    dist_checkpointing.save(
        sharded_state_dict=state_dict,
        checkpoint_dir=checkpoint_path,
        sharded_strategy=...
    )
```

`generate_state_dict` 调用 `model.sharded_state_dict()`，模型里每层的 `sharded_state_dict` 方法把自己的参数包装成 `ShardedTensor`，最终聚合成一个嵌套字典。

`dist_checkpointing.save`（`serialization.py`）按策略把 dict 写到磁盘：

- `torch` 策略：`torch.save` 每个 rank 独立写一个文件。
- `zarr` / `fully_parallel` 策略：并行写，速度更快，支持大规模。

---

### 3.5 加载流程：需要先有"空壳" state_dict

**加载的关键约束**：`dist_checkpointing.load` 需要传入一个已有正确 `ShardedTensor` 结构的 state_dict（但 `data=None`），才能知道"当前训练配置期望什么形状"，并把磁盘上存的 shard 映射到正确位置：

```python
# training/checkpointing.py  load_checkpoint（简化）
# 先用当前并行度构建空壳
empty_state_dict = model.sharded_state_dict(sharded_offsets=[])
# 再让框架做 resharding
loaded = dist_checkpointing.load(
    sharded_state_dict=empty_state_dict,
    checkpoint_dir=load_path,
)
model.load_state_dict(loaded)
```

这就是"**save 时记录了每块 tensor 在全局中的位置，load 时按新的并行度重新拼接**"的机制，即 resharding。

---

### 3.6 worked example：TP=2 保存，TP=4 加载的 resharding 思路

```
保存时（TP=2）：
  rank0 写: key="layers.0.attn.q_proj.weight", local_shape=(2048,4096), global_offset=(0,0), global_shape=(4096,4096)
  rank1 写: key="layers.0.attn.q_proj.weight", local_shape=(2048,4096), global_offset=(2048,0), global_shape=(4096,4096)

加载时（TP=4）：
  当前 rank 期望: local_shape=(1024,4096), global_offset=(rank*1024, 0)
  框架读取磁盘文件中 key="layers.0.attn.q_proj.weight"，找到覆盖 global_offset 范围的 shard，
  切割出本 rank 需要的那 [1024,4096] 片段，加载进来。
```

整个 resharding 由 `dist_checkpointing/strategies/fully_parallel.py` 中的坐标映射逻辑完成，用户无需手写任何 split/cat。

---

### 3.7 常见 resume 陷阱

| 陷阱 | 现象 | 排查 |
|------|------|------|
| 忘记恢复 RNG 状态 | 不同 resume 点的 dropout 行为不同，导致损失曲线跳变 | `args.rng_state` 是否在 `state_dict` 中 |
| 未恢复优化器状态 | 从低 loss 突然跳高，后缓慢恢复 | `load_optimizer_state=True` 是否生效 |
| args 版本不匹配 | 并行度、序列长度等与 checkpoint 不符导致形状错误 | 加载时检查 `args` 中关键字段与 checkpoint 一致 |
| DP rank 副本计数错误 | `replica_id` 不为 0 的副本被重复写入 | `ShardedTensor(replica_id=dp_rank)` 检查 |
| `allow_shape_mismatch=False` | TP 度改变时全局 shape 不一致报错 | 分析哪些参数需要设 `allow_shape_mismatch=True` |

---

## 四、端到端时间线：第一次迭代 vs. 稳态

```
启动阶段（仅一次）：
  ① 解析 args → 初始化 parallel_state（进程组）
  ② 构建 GPTDataset（或读缓存 shuffle 索引）
  ③ 搭建模型、初始化 DistributedOptimizer
  ④ 若有 checkpoint → load（包含 resharding）

第一次迭代：
  ① get_batch → TP broadcast → CP split
  ② forward（PP schedule：warmup microbatches → 1F1B → cooldown）
  ③ backward（bucket 梯度累积 → ReduceScatter → AllReduce）
  ④ finalize_model_grads（PP 流水线残余梯度）
  ⑤ optimizer.step（unscale → clip → ZeRO-1 ReduceScatter → update → AllGather）
  ⑥ （可选）save_checkpoint

稳态迭代（与第一次的差别）：
  - shuffle 索引已缓存，DataLoader 更快
  - CUDA kernels 已 JIT 编译完成
  - NCCL 通信路由已建立
  - 每次 step 均匀稳定，主要开销：compute + communication overlap
```

---

## 五、Debug 检查清单

遇到数据/优化器/checkpoint 问题时，按以下顺序排查：

```
数据相关：
  □ IndexedDataset header magic bytes 是否正确（MMIDIDX\x00\x00）？
  □ shuffle 索引缓存文件是否与当前 seed/seq_len 匹配？
  □ get_batch 返回的 tokens shape 是否 [B, seq_len/CP]（CP 已切分）？
  □ 中间 PP stage 的 tokens 是否 None（正常行为）？

优化器相关：
  □ loss scale 是否持续下降（检测到大量 NaN 梯度）？
  □ DistributedOptimizer 的 ReduceScatter 是否在正确的 DP group 上执行？
  □ expert 参数的 allreduce 属性是否为 False？

Checkpoint 相关：
  □ ShardedTensor key 是否与 checkpoint 文件中的 key 匹配？
  □ global_shape 是否与期望的完整参数尺寸一致？
  □ 加载后是否调用了 optimizer.reload_model_params()？
```

---

## 六、五道练习题

1. **数据**：修改 `GPTDataset.__getitem__`，让所有 EOD token 后的第一个 token 的 `loss_mask` 也置为 0（模拟不预测文档开头）。需要修改哪个函数？有什么副作用？

2. **数据**：在 `MockGPTDataset` 中模拟一个"始终返回相同序列"的模式，用于测试 loss 是否单调下降（过拟合单条样本）。需要改几行代码？

3. **优化器**：假设 DP=8，grad buffer 有 800 个元素。手写每个 rank 的 `Range(start, end)`，以及 ReduceScatter 后每个 rank 拿到多少个梯度。

4. **优化器**：`MixedPrecisionOptimizer._copy_main_params_to_model_params` 做的是 FP32 → BF16 的精度截断。解释为什么这个截断不会导致"参数震荡"（提示：Adam 的更新量通常很小）。

5. **Checkpoint**：如果你要把 TP=4、PP=2 训练的 checkpoint 加载到 TP=2、PP=4，需要同时处理 TP 和 PP 两个维度的 resharding。`ShardedTensor` 的哪些字段需要变化？`axis_fragmentations` 应该怎么设？

---

## 七、本篇小结

| 模块 | 关键设计 | 记忆钩子 |
|------|----------|---------|
| IndexedDataset | `.idx` 存偏移，mmap `.bin`，O(1) 访问 | "索引 + mmap = 无限大数据集" |
| GPTDataset | `add_extra_token` 做 next-token 位移 | "多取一个 token，shift 一位" |
| get_batch | TP rank 0 加载 → broadcast → CP 切分 | "TP 广播，CP 切割，PP 填 None" |
| MixedPrecisionOptimizer | BF16 模型 / FP32 master，step 后复制回去 | "双份参数，FP32 负责更新" |
| DistributedOptimizer | ReduceScatter 梯度 → 分片更新 → AllGather 参数 | "ZeRO-1：scatter 梯度，gather 参数" |
| ShardedTensor | 6 个字段描述局部 tensor 在全局的位置 | "key + offset + shape = 全局坐标" |
| dist_checkpointing | save 各自写，load 按新并行度 reshard | "存时记坐标，读时重拼接" |

---

> **上一篇**：[精读 Megatron 源码（7）：数据并行完全精读](./07-data-parallel.md)  
> **下一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)
