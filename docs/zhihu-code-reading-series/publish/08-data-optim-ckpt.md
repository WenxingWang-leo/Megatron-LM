# 精读 Megatron 源码（8）：数据管线、DistributedOptimizer 与 Dist Checkpoint 完全精读

> **专栏**：Megatron 源码精读 · 第 8 篇  
> **核心文件**：  
> - `megatron/core/datasets/indexed_dataset.py`  
> - `megatron/core/datasets/gpt_dataset.py`  
> - `megatron/core/datasets/blended_megatron_dataset_builder.py`  
> - `megatron/training/datasets/sft_dataset.py`  
> - `megatron/training/datasets/fim_dataset.py`  
> - `megatron/core/optimizer/__init__.py`  
> - `megatron/core/optimizer/optimizer.py`  
> - `megatron/core/optimizer/distrib_optimizer.py`  
> - `megatron/core/dist_checkpointing/mapping.py`  
> - `megatron/core/dist_checkpointing/serialization.py`  
> - `megatron/core/dist_checkpointing/strategies/filesystem_async.py`  
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
  ↓  GPTDataset / SFTDataset / GPTFIMDataset（shuffle 索引 + __getitem__）
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
| `sft_dataset.py` | 对话指令微调数据集，携带 `cu_seqlens` |
| `fim_dataset.py` | Fill-In-the-Middle 数据增强 |
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
| `prefix.idx` | 元数据：每条序列的 **长度**、**字节偏移** 以及文档边界 |

#### 1.2.1 `.idx` 文件精确布局

`_IndexWriter.__enter__` 写入 header，`_IndexWriter.write` 写入主体。下表给出每个字段的 **精确字节位置和 dtype**：

| 偏移 | 大小 | dtype | 含义 |
|------|------|-------|------|
| 0 | 9 bytes | `bytes` | Magic header `MMIDIDX\x00\x00`（常量，用于校验文件格式） |
| 9 | 8 bytes | `<Q`（uint64 LE） | 版本号，固定为 `1` |
| 17 | 1 byte | `<B`（uint8） | dtype 代码（1=uint8, 4=int32, 8=uint16，见 `DType` 枚举） |
| 18 | 8 bytes | `<Q` | 序列总数 `sequence_count` |
| 26 | 8 bytes | `<Q` | 文档总数 `document_count` |
| 34 | `4 * sequence_count` | `int32[]` | 每条序列的 token 长度 `sequence_lengths` |
| 34 + 4·N | `8 * sequence_count` | `int64[]` | 每条序列在 `.bin` 中的字节偏移 `sequence_pointers` |
| 34 + 12·N | `8 * document_count` | `int64[]` | 文档边界：每个文档最后一个序列的序号 `document_indices` |
| 34 + 12·N + 8·D | `1 * sequence_count`（可选） | `int8[]` | 序列模式（多模态时使用） |

`_IndexReader.__init__` 在构造时按照上表逐字段解析，并把 `sequence_lengths` 和 `sequence_pointers` **mmap 到 numpy 数组**，使访问单条序列成为一次 O(1) 数组下标操作。

#### 1.2.2 为什么是 O(1) 访问

```python
# indexed_dataset.py _IndexReader.__getitem__（精简）
ptr   = self.sequence_pointers[i]           # 字节偏移，O(1)
size  = self.sequence_lengths[i]            # 序列长度，O(1)
numpy_slice = self._bin_buffer[ptr : ptr + size]
```

这是经典的"索引文件 + 数据文件"设计，LMDB / LevelDB 等系统广泛使用。

**mmap 的作用**：操作系统的页缓存负责管理哪些页真正在内存中，训练程序以"访问普通 numpy 数组"的方式使用，不需要手动 `seek`/`read`。数据集远大于 GPU 节点内存时，mmap 是最优选择。

**对象存储支持**：`object_storage_utils.py` 实现了 S3 / MSC 路径下先把 `.idx` 缓存到本地再 mmap 的逻辑，用 `is_object_storage_path(dataset_path)` 分支控制。

---

### 1.3 GPTDataset.__getitem__：tokens/labels 是怎么拼出来的

```python
# megatron/core/datasets/gpt_dataset.py（精简）
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

这三个数组**会被缓存到磁盘**（`.npy` 文件）。缓存文件名格式包含影响采样的所有参数，例如：

```
{prefix}_doc_{N_epoch}ep_{num_docs}docs_{seq_len}sl_{seed}s.npy
{prefix}_sample_{N_epoch}ep_{num_docs}docs_{seq_len}sl_{seed}s.npy
{prefix}_shuffle_{N_epoch}ep_{num_docs}docs_{seq_len}sl_{seed}s.npy
```

这意味着：
1. **第一次运行**：从头构建，速度慢（数十亿 token 数据集可能需要数分钟）。
2. **之后再运行**：直接加载缓存，启动极快。
3. **可复现性**：只要参数相同，同一 seed 产生完全相同的 shuffle，方便复现实验。

**cache 文件什么时候失效**：修改了 `seed`、`num_samples`、`sequence_length`、`num_epochs`、文档数任一参数，hash 不同，旧缓存不会被删除，会生成新文件。务必定期清理陈旧缓存，避免磁盘占满。

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
2. **归一化权重**：用户给的可以不归一，框架内部 `normalize_weights` 确保 `sum(weights) == 1.0`。
3. 按权重比例 `[0.6, 0.3, 0.1]` 预先计算每个数据集应贡献多少 sample（总 sample 数乘以权重取整）。
4. 构建 `BlendedDataset`，内部维护一个映射 `global_idx → (dataset_id, local_idx)`，确保按权重混合。

**权重如何变成采样次数**：

```python
# blended_megatron_dataset_builder.py（精简）
normalized_weights = normalize_weights(weights)
dataset_sample_counts = [
    int(num_samples * w) for w in normalized_weights
]
# 修正舍入误差：剩余 sample 全部分给最大权重数据集
remainder = num_samples - sum(dataset_sample_counts)
dataset_sample_counts[largest_weight_idx] += remainder
```

**BlendedDataset 内部 index 映射**：构建时生成一个长度为 `num_samples` 的 `dataset_index` 数组，每个位置存 `(dataset_id, local_idx)`。`BlendedDataset.__getitem__(i)` 直接按此数组查找，无需运行时计算权重。

**mock 模式**：当没有真实数据时，可以使用 `MockGPTDataset`：

```python
if config.mock:
    return MockGPTDataset(config)
```

`MockGPTDataset.__getitem__` 返回随机 token id，适合调试模型代码（不依赖数据预处理），也是新手入手 Megatron 的最简路径。

---

### 1.7 SFTDataset vs GPTDataset：有监督微调的差异

```
文件: megatron/training/datasets/sft_dataset.py
```

`SFTDataset` 是 `MegatronDataset` 的直接子类（非 `IndexedDataset` 驱动），其底层数据源是 `SFTLowLevelDataset`（基于 Hugging Face `datasets` 库加载 jsonl）。

**核心差异对比**：

| 特性 | GPTDataset | SFTDataset |
|------|-----------|-----------|
| 底层存储 | `.bin/.idx` (IndexedDataset) | `.jsonl`（Hugging Face datasets） |
| 输入格式 | token id 序列 | `{"messages": [...]}` 对话列表 |
| 标签策略 | 全部 token 都参与 loss | 只有 assistant 回复位置参与 loss（`IGNORE_INDEX=-100`） |
| `cu_seqlens` | 无（默认） | 有（标记每条对话边界，packed 序列） |
| 跨文档拼接 | 支持（sample_index 多文档 span） | 支持（`split_conversations` 迭代填充） |
| position_ids | 可跨文档累积或 reset | 每条对话从 0 开始，多对话 pack 时各自递增 |

**SFTDataset.__getitem__ 核心流程**：

```python
# sft_dataset.py（精简）
def __getitem__(self, idx):
    merged_conversations = self.dataset[int(self.indices[idx])]
    split_conversations = self._split_conversations(merged_conversations)

    pack_tokens, pack_targets, pack_positions = [], [], []
    cu_seqlens = [0]

    for conversation in split_conversations:
        tokens, targets = tokenizer.tokenize_conversation(conversation, return_target=True)
        pack_tokens.extend(tokens.tolist())
        pack_targets.extend(targets.tolist())
        pack_positions.extend(range(len(tokens)))
        cu_seqlens.append(len(pack_tokens))   # 每条对话的结束位置

        if len(pack_tokens) >= pack_length + 1:
            break  # 截断

    # 对齐：tokens=input[:-1], labels=targets[1:]
    input_ids    = torch.tensor(pack_tokens[:-1], dtype=torch.int64)
    labels       = torch.tensor(pack_targets[1:], dtype=torch.int64)
    loss_mask    = torch.ones(pack_length, dtype=torch.float32)
    loss_mask[labels == pad]          = 0.0   # 填充不计损失
    loss_mask[labels == IGNORE_INDEX] = 0.0   # prompt 不计损失
    ...
    return {'tokens': input_ids, 'labels': labels, 'cu_seqlens': padded_cu_seqlens, ...}
```

`cu_seqlens` 是 packed sequence（THD 格式）的边界数组，`[0, L1, L1+L2, ...]`。FlashAttention 用它区分不同对话，避免跨对话的注意力计算。

---

### 1.8 GPTFIMDataset：Fill-In-the-Middle 数据增强

```
文件: megatron/training/datasets/fim_dataset.py
```

`GPTFIMDataset` 继承 `GPTDataset`，覆写 `_query_document_sample_shuffle_indices`，在读取原始 token 序列后以概率 `fim_rate` 对每段（遇到 EOD 则按文档分段）执行 FIM 变换。

**PSM vs SPM 两种格式**：

| 格式 | 排列 | 说明 |
|------|------|------|
| PSM（Prefix-Suffix-Middle） | `<PRE> prefix <SUF> suffix <MID> middle` | 较常用 |
| SPM（Suffix-Prefix-Middle） | `<SUF> suffix <PRE> prefix <MID> middle` | `fim_spm_rate` 控制比例 |

FIM 不改变序列**长度**（不足补 pad，过长截断），因此与 `GPTDataset` 的 `add_extra_token` / `shuffle_index` 逻辑完全兼容。在 `pretrain_gpt.py` 中通过 `--fim-rate 0.5` 参数启用：

```bash
--fim-rate 0.5 --fim-spm-rate 0.2 \
--fim-extra-tokens '{"prefix":"<PRE>","middle":"<MID>","suffix":"<SUF>","pad":"<PAD>","eod":"<EOD>"}'
```

---

### 1.9 get_batch：让数据"懂并行"

```python
# pretrain_gpt.py  BATCH_KEYS（第 83-95 行附近）
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
if mpu.get_tensor_model_parallel_rank() != 0:
    ...  # TP 的非 rank 0 收广播即可
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

Context Parallel（CP=N）把序列维度切 N 份，每个 CP rank 只处理自己那段。切完后每个 rank 拿到的 `tokens` 形状从 `[B, seq_len]` 变为 `[B, seq_len/CP]`。

---

### 1.10 cu_seqlens 在 forward_step 中的完整处理路径

当数据集返回 `cu_seqlens`（SFTDataset / 开启 packed sequence 的配置）时，`forward_step` 在调用模型前需要把它包装成 `PackedSeqParams`：

```python
# pretrain_gpt.py forward_step（第 310-334 行，精简）
packed_seq_params = None
if cu_seqlens is not None:
    cu_seqlens = cu_seqlens.squeeze(0)   # (1, N) → (N,)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",                # token-head-dim 格式，FlashAttention 要求
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=int(max_seqlen.item()),
        max_seqlen_kv=int(max_seqlen.item()),
        local_cp_size=int(local_cp_size.item()) if local_cp_size is not None else None,
        cp_group=hybrid_cp_group,
        tokens_per_sample=args.seq_length,
    )
```

`PackedSeqParams` 是传递给 `TransformerEngine` 中 FlashAttention kernel 的参数对象。`cu_seqlens_q` 数组（`[0, L1, L1+L2, ...]`）让 attention kernel 知道"第几个 token 属于第几条序列"，从而**避免跨序列的注意力计算**（跨序列的 attention 不符合 SFT 语义）。

**THD 格式说明**：
- 普通（无 packing）：`tokens shape = [B, S, H]`（batch × seq × hidden）
- Packed（packing）：`tokens shape = [1, T, H]`（1 × total_tokens × hidden），T = 所有序列总长度之和，`cu_seqlens` 标记边界

**调试技巧**：若发现 SFT 微调的 loss 异常高，检查 `cu_seqlens` 是否正确。错误的 `cu_seqlens` 会导致跨序列注意力污染，表现为早期 loss 偏低（"虚假地看到了别人的 context"），之后推理时却表现很差。

---

### 1.11 数据管线失败模式速查表

| 现象 | 最可能原因 | 排查命令/方法 |
|------|------------|--------------|
| `IndexError: index out of bounds` | shuffle cache 文件与新数据集参数不一致 | 删除 `*.npy` cache 文件重新生成 |
| 启动极慢（构建 shuffle 索引卡住） | 第一次运行大数据集，正在生成 npy | 查看 rank 0 日志，搜索 "Building shuffle index" |
| 所有 rank loss 完全一致 | TP broadcast 意外发到错误的 group | 检查 `mpu.get_tensor_model_parallel_group()` 与 DataLoader 的搭配 |
| 中间 PP stage 报 `None` 异常 | 使用了不支持 None batch 的自定义模块 | 在 PP 中间 stage 所有操作必须支持 `None` 输入 |
| SFT loss 不降 | `IGNORE_INDEX=-100` 没正确设置 | 检查 tokenizer 的 `tokenize_conversation` 是否在 prompt 位置设了 -100 |
| FIM 数据 loss 异常高 | `fim_extra_tokens` 不在词表中 | 打印 `prefix_tok_id` 等，确认 tokenizer 正确映射 |
| Packed seq attention 计算错误 | `cu_seqlens` padding 没剥离 | `cu_seqlens.squeeze(0)` 后检查末尾是否是真实边界而非填充值 |
| 跨 epoch shuffle 结果不同 | `seed` 参数不同或 cache 未命中 | 检查 cache 文件名是否包含当前 seed |

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
# optimizer/__init__.py
is_expert_parallel = not getattr(param, 'allreduce', True)
```

非 expert 参数的 `allreduce=True`（默认），expert 参数的 `allreduce=False`，由此分成两个 `param_groups`，喂给两个独立的 `_get_megatron_optimizer_based_on_param_groups`，最终用 `ChainedOptimizer` 串联。

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

**ZeRO-1 的核心思路**：把优化器状态（Adam 的 m、v 以及 master 参数）在 DP 组的所有 rank 间均匀切分，每个 rank 只负责更新自己持有的那片参数，更新完后用 AllGather 把更新后的参数广播给所有 rank。

**阶段 1：Reduce-Scatter 梯度**

```
[rank0 grad_buf] = [g0, g1, g2, g3, ..., g15]   # DP=4, buffer 16 elems
                    ↓ ReduceScatter（sum + scatter）
rank0 拿到 sum(g0..g3)   [0..3]
rank1 拿到 sum(g4..g7)   [4..7]
rank2 拿到 sum(g8..g11)  [8..11]
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

---

### 2.4 参数到 shard 的映射直觉：Range 类

`distrib_optimizer.py` 开头定义的 `Range(start, end)` 描述每个 DP rank 拥有 grad buffer 的哪个片段：

```python
class Range:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start
```

`_build_model_gbuf_param_range_map` 把整个 grad buffer 按 DP rank 数均匀切割，建立 `param → Range` 的映射：

```
整个 grad buffer（flat view）:
  [param_A: 0..99] [param_B: 100..199] [param_C: 200..399] ...
                                              ↓ DP=4 切分
  rank0 Range: [0..99]     → 负责 param_A 全部 + param_B 前半
  rank1 Range: [100..199]  → 负责 param_B 后半
  rank2 Range: [200..299]  → 负责 param_C 前半
  rank3 Range: [300..399]  → 负责 param_C 后半
```

**注意**：单个参数可能被跨两个 rank 分割（如 param_A 被 rank0 和 rank1 共同持有 master copy）。`_build_model_gbuf_param_range_map` 记录了每个 rank 负责每个参数的哪个 sub-range，更新时精确操作对应 slice。

---

### 2.5 worked example：DP=4，buffer 16 个元素

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

---

### 2.6 保存 / 加载优化器状态的常见陷阱

DistributedOptimizer 的 checkpoint 包含 FP32 master 参数和 Adam (m, v)。以下是最常遇到的坑：

| 陷阱 | 现象 | 排查 |
|------|------|------|
| 未调用 `reload_model_params()` | 加载 checkpoint 后 BF16 模型参数未从 FP32 master 同步，loss 跳变 | 加载 optimizer state 后确认 `distrib_optimizer.reload_model_params()` 被调用 |
| DP 度改变 | Range 映射与旧 checkpoint 不一致，参数无法对齐 | DistributedOptimizer 与 ShardedTensor reshard 是两套逻辑，DP 改变时 opt state 需要单独处理（通常只能舍弃，从 warm start 恢复） |
| 加载 FP32 master 参数失败 | checkpoint 只保存了 BF16 模型参数而未保存 master params | 确认 `args.save_optimizer` / `state_dict['optimizer']` 不为空 |
| Loss scale 状态未恢复 | resume 后 loss scale 从 1.0 重新开始，前期 loss 有抖动 | `state_dict['loss_scaler']` 中包含 `scale_value` 和 `growth_tracker` |
| 分布式保存中途失败 | checkpoint 目录损坏（只写了部分 rank） | 使用双 buffer checkpoint（交替写 `iter_X` 和 `iter_Y`），加载时优先最新完整目录 |

---

### 2.7 ChainedOptimizer：多个优化器的透明串联

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
- `fully_parallel` 策略：`FullyParallelSaveStrategyWrapper` 用贪心算法在所有 rank 间分配保存任务，最大化 I/O 并行度，减少总写入时间。

---

### 3.5 异步 checkpoint（filesystem_async）

Megatron 提供 `FileSystemWriterAsync`（`strategies/filesystem_async.py`）实现**异步 checkpoint**：

```
同步 checkpoint：
  训练暂停 → save → 训练恢复     （有停顿）

异步 checkpoint：
  训练继续 → 后台线程执行 I/O → 不阻塞训练   （zero downtime）
```

**工作流程**：
1. 主进程调用 `write_data`，把需要写的数据序列化到内存缓冲区。
2. 调用 `get_save_function_and_args` 获取写入函数和参数。
3. 在**后台子进程**（`mp.spawn`）中调用 `writer_proxy_func`，用多线程并发写入文件。
4. 通过 `_results_queue` 把写入结果返回主进程。

**使用方式**（CLI 配置）：

```bash
--async-save           # 开启异步 checkpoint
--ckpt-format torch_dist   # 使用 torch.distributed.checkpoint 格式
```

**注意事项**：
- 异步保存期间，参数可能被下一个 step 修改；框架通过 `copy_on_write` 避免数据竞争。
- 保存进程崩溃不会影响主训练进程，但需要额外的健康检查逻辑。
- `HAVE_PSUTIL = True` 时可监控保存进程的内存使用。

---

### 3.6 加载流程：需要先有"空壳" state_dict

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

### 3.7 worked example：TP=2 保存，TP=4 加载的 resharding 思路

```
保存时（TP=2）：
  rank0 写: key="layers.0.attn.q_proj.weight"
            local_shape=(2048,4096), global_offset=(0,0), global_shape=(4096,4096)
  rank1 写: key="layers.0.attn.q_proj.weight"
            local_shape=(2048,4096), global_offset=(2048,0), global_shape=(4096,4096)

加载时（TP=4）：
  当前 rank0 期望: local_shape=(1024,4096), global_offset=(0,0)
  当前 rank1 期望: local_shape=(1024,4096), global_offset=(1024,0)
  ...

框架动作（fully_parallel.py exchange_utils）：
  读磁盘文件，发现 global_offset=(0,0), size=(2048,4096)
  → 切割出 rank0 需要的 [0:1024, :] 和 rank1 需要的 [1024:2048, :]
  → 通过 exchange_by_distribution 发给对应 rank
```

整个 resharding 由 `dist_checkpointing/strategies/fully_parallel.py` 中的坐标映射逻辑完成，用户无需手写任何 split/cat。

---

### 3.8 TP 度改变时，哪些配置必须匹配，哪些可以不同

这是工程中最常见的问题之一：

| 配置项 | 能否改变 | 说明 |
|--------|---------|------|
| `tensor_model_parallel_size` (TP) | ✅ 可改变 | dist_checkpointing reshard 处理 |
| `pipeline_model_parallel_size` (PP) | ✅ 可改变 | PP 切层，每层参数 key 不变 |
| `num_layers` | ❌ 不能改变 | key 包含层编号，层数不同会匹配失败 |
| `hidden_size` | ❌ 不能改变 | 参数形状与存储值不匹配 |
| `num_attention_heads` | ❌ 不能改变 | attention weight 形状与 TP 绑定 |
| `vocab_size` | ❌ 不能改变 | embedding weight 形状不匹配 |
| `num_moe_experts` | ❌ 不能改变 | expert weight key 绑定专家数 |
| `expert_parallel_size` (EP) | ✅ 可改变 | 类似 TP reshard，按 EP 重新分配专家 |
| `data_parallel_size` (DP) | ✅ 可改变 | replica_id 区分副本，DP 改变只影响副本筛选 |
| 优化器状态（Adam m/v） | ⚠️ 通常丢弃 | DP 改变时 DistOpt shard 不可恢复，需从 warm start |

**最安全的做法**：TP/PP 发生改变时，优先只用模型参数，丢弃优化器状态，从学习率 warm-start 继续训练（`--no-load-optim --no-load-rng`）。

---

### 3.9 常见 resume 陷阱

| 陷阱 | 现象 | 排查 |
|------|------|------|
| 忘记恢复 RNG 状态 | 不同 resume 点的 dropout 行为不同，导致损失曲线跳变 | `args.rng_state` 是否在 `state_dict` 中 |
| 未恢复优化器状态 | 从低 loss 突然跳高，后缓慢恢复 | `load_optimizer_state=True` 是否生效 |
| args 版本不匹配 | 并行度、序列长度等与 checkpoint 不符导致形状错误 | 加载时检查 `args` 中关键字段与 checkpoint 一致 |
| DP rank 副本计数错误 | `replica_id` 不为 0 的副本被重复写入 | `ShardedTensor(replica_id=dp_rank)` 检查 |
| `allow_shape_mismatch=False` | TP 度改变时全局 shape 不一致报错 | 分析哪些参数需要设 `allow_shape_mismatch=True` |
| 异步 checkpoint 未完成即退出 | 后台保存进程还在运行但主进程已退出 | 确认 `finalize_async_save()` 被调用 |

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
  ⑥ （可选）save_checkpoint（同步或异步）

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
  □ SFT 场景：cu_seqlens 末尾是否正确填充到 pack_length？
  □ FIM 场景：fim_rate + fim_spm_rate 之和是否 <= 1.0？

优化器相关：
  □ loss scale 是否持续下降（检测到大量 NaN 梯度）？
  □ DistributedOptimizer 的 ReduceScatter 是否在正确的 DP group 上执行？
  □ expert 参数的 allreduce 属性是否为 False？
  □ 加载 optimizer state 后是否调用了 reload_model_params()？

Checkpoint 相关：
  □ ShardedTensor key 是否与 checkpoint 文件中的 key 匹配？
  □ global_shape 是否与期望的完整参数尺寸一致？
  □ TP/PP 改变时，num_layers/hidden_size 是否保持不变？
  □ 异步 checkpoint 是否正常完成（检查后台进程状态）？
```

---

## 六、十二道 FAQ

**Q1：`.idx` 文件的 magic header 有什么用？**  
A：`_IndexReader.__init__` 在加载时校验前 9 字节是否等于 `MMIDIDX\x00\x00`，防止误加载非 IndexedDataset 格式的文件（例如 `.bin` 被当 `.idx` 用），提前报错而非产生难以调试的运行时错误。

**Q2：为什么 `sequence_pointers` 用 `int64` 而 `sequence_lengths` 用 `int32`？**  
A：`sequence_lengths` 是单条序列的 token 数，通常几百到几万，`int32` 足够。`sequence_pointers` 是累积字节偏移，对于 TB 级数据集可能超过 `2^32`，必须用 `int64`。

**Q3：BlendedDataset 的混合是训练中动态决定还是预先计算好的？**  
A：预先计算好。`BlendedDataset` 在构建时就生成了全部 `num_samples` 条记录的 `(dataset_id, local_idx)` 映射数组，`__getitem__` 时直接查表，无运行时随机性（可复现）。

**Q4：SFTDataset 中 `IGNORE_INDEX=-100` 和 GPTDataset 的 `loss_mask=0` 有什么区别？**  
A：语义相同，机制不同。GPTDataset 用 float `loss_mask * loss` 加权；SFTDataset 把 prompt 位置的 `labels` 设为 `-100`，PyTorch `CrossEntropyLoss(ignore_index=-100)` 自动跳过。两种方式等价，SFTDataset 的做法更契合 HuggingFace 生态。

**Q5：`cu_seqlens` 为什么要 pad 到 `pack_length + 1`？**  
A：DataLoader 的 `default_collate` 要求同一 batch 内的 tensor 形状一致。不同样本对话数量不同，`cu_seqlens` 长度不同，无法直接 stack。填充到 `pack_length + 1` 是最大可能长度，用 `pack_length` 值填充尾部（非真实边界），后续 `get_batch` 中会剥离多余的填充。

**Q6：DistributedOptimizer 中一个参数被两个 rank 分割时，各自如何更新并同步？**  
A：`_build_model_gbuf_param_range_map` 记录每个 rank 对参数的 sub-range。两个 rank 各持有 FP32 master 的不同片段，分别独立执行 Adam 更新（不通信）。AllGather 时自然拼合还原完整的 BF16 参数。

**Q7：换 TP 度时，如何知道哪些参数的 `key` 是稳定的？**  
A：`key` 是模型的 Python 属性路径，如 `"decoder.layers.0.mlp.fc1.weight"`，与 TP 度无关。只要层结构不变，`key` 就稳定。`global_shape` 和 `axis_fragmentations` 随 TP 度改变，但 resharding 逻辑就是靠 `key` + `global_offset` + `global_shape` 重新路由数据。

**Q8：`FullyParallelSaveStrategyWrapper` 如何决定哪个 rank 保存哪个 shard？**  
A：使用贪心算法（`determine_main_replica_uniform_distribution`）：统计每个 rank 需要保存的数据量，优先把任务分配给当前负载最轻的 rank，尽量均衡 I/O 压力。只有 `replica_id=0` 的 shard 真正写盘，其他副本跳过写入。

**Q9：异步 checkpoint 期间如果训练 crash，checkpoint 是否可用？**  
A：取决于保存是否完成。若 crash 发生在后台写入完成前，目录可能只有部分文件（checkpoint 不完整）。推荐使用双 buffer 策略：交替写 `iter_A` 和 `iter_B`，每次只在验证 `iter_A` 完整后才删除上一个 `iter_B`，始终保有一个完整的旧 checkpoint。

**Q10：`sequence_modes` 字段在 `.idx` 文件中是什么？**  
A：用于多模态数据集（`multimodal=True`）。每条序列有一个 `int8` 模式标记（如 0=文本，1=图像 token），模型或 DataLoader 可据此区分序列类型，做不同的预处理或 loss masking。

**Q11：GPTFIMDataset 的 FIM 变换会改变序列长度吗？**  
A：不会。`_fim_split_and_permute_sequence` 对序列重排后若比原序列短则补 pad，比原序列长则截断，确保输出长度 == 输入长度。这样 FIM 数据集的 `sequence_lengths` 与原始一致，`shuffle_index` 缓存逻辑完全复用。

**Q12：如何验证 TP=2→TP=4 resharding 是否正确（不训练，只检查加载）？**  
A：可用以下方式验证：
1. 用 `MockGPTDataset` 在 TP=2、2 层模型跑 100 步保存 checkpoint。
2. 换 TP=4 加载 checkpoint，打印各 rank 模型参数的 `sum()`。
3. TP=2 时 rank 0 权重 + rank 1 权重的 `sum()` 应等于 TP=4 时 rank 0..3 权重 `sum()` 之和（数值守恒）。
4. 用两个配置各跑 1 步（相同输入），loss 应完全一致。

---

## 七、八道练习题

1. **数据格式**：手写一个 20 条序列的 `.idx` 文件的 header（前 34 字节），序列长度均为 512，dtype 为 `uint16`（code=8）。计算第 15 条序列的 `sequence_pointer` 值。

2. **数据**：修改 `GPTDataset.__getitem__`，让所有 EOD token 后的第一个 token 的 `loss_mask` 也置为 0（模拟不预测文档开头）。需要修改哪个函数？有什么副作用？

3. **SFT 数据**：在 `SFTDataset.__getitem__` 中，如果一个 `pack_length=2048` 的窗口被截断，`cu_seqlens` 最后一个值会是多少？画出截断场景下 `pack_tokens` / `cu_seqlens` 的状态。

4. **BlendedDataset**：假设三个数据集权重为 `[3, 2, 1]`，总 sample 数为 `1000`，计算每个数据集分配到的 sample 数量（注意归一化和舍入误差修正）。

5. **优化器**：假设 DP=8，grad buffer 有 800 个元素。手写每个 rank 的 `Range(start, end)`，以及 ReduceScatter 后每个 rank 拿到多少个梯度。

6. **优化器**：`MixedPrecisionOptimizer._copy_main_params_to_model_params` 做的是 FP32 → BF16 的精度截断。解释为什么这个截断不会导致"参数震荡"（提示：Adam 的更新量通常很小）。

7. **Checkpoint**：用表格列出从 TP=2,PP=4 加载到 TP=4,PP=2 时，`ShardedTensor` 哪些字段会变化，哪些字段保持不变。

8. **异步 checkpoint**：在 `FileSystemWriterAsync` 的流程中，"主进程序列化数据到内存"和"后台进程写磁盘"之间，如果训练在写盘期间更新了参数，会发生什么？代码中用什么机制解决？

---

## 八、本篇小结

| 模块 | 关键设计 | 记忆钩子 |
|------|----------|---------|
| IndexedDataset | `.idx` 存偏移，mmap `.bin`，O(1) 访问 | "索引 + mmap = 无限大数据集" |
| `.idx` 布局 | header(9B) + version(8B) + dtype(1B) + seq_count + doc_count + lengths + pointers + doc_indices | "9+8+1+8+8+4N+8N+8D" |
| GPTDataset | `add_extra_token` 做 next-token 位移 | "多取一个 token，shift 一位" |
| SFTDataset | jsonl → 对话分段 → cu_seqlens + IGNORE_INDEX | "SFT = 对话边界 + prompt 不计 loss" |
| GPTFIMDataset | 以 fim_rate 概率变换序列为 PSM/SPM 格式 | "FIM = 重排三段，长度不变" |
| get_batch | TP rank 0 加载 → broadcast → CP 切分 | "TP 广播，CP 切割，PP 填 None" |
| cu_seqlens | packed seq 边界标记，forward_step 包装为 PackedSeqParams | "cu_seqlens = THD 格式的导航栏" |
| MixedPrecisionOptimizer | BF16 模型 / FP32 master，step 后复制回去 | "双份参数，FP32 负责更新" |
| DistributedOptimizer | ReduceScatter 梯度 → 分片更新 → AllGather 参数 | "ZeRO-1：scatter 梯度，gather 参数" |
| Range 类 | 描述每个 DP rank 负责 grad buffer 的哪段 | "Range = 每 rank 的优化器领地" |
| ShardedTensor | 6 个字段描述局部 tensor 在全局的位置 | "key + offset + shape = 全局坐标" |
| dist_checkpointing | save 各自写，load 按新并行度 reshard | "存时记坐标，读时重拼接" |
| 异步 checkpoint | 后台进程 I/O，不阻塞训练 | "训练继续，存档后台跑" |

---

> **上一篇**：[精读 Megatron 源码（7）：数据并行完全精读](./07-data-parallel.md)  
> **下一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)
