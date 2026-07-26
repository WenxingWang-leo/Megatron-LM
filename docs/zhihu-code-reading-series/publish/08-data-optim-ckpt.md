# 精读 Megatron 源码（8）：数据、DistOpt、Checkpoint——训练闭环的另外三块

> **专栏**：Megatron 源码精读 · 第 8 篇  
> **核心文件**：`datasets/gpt_dataset.py`、`optimizer/distrib_optimizer.py`、`dist_checkpointing/`

---

前面几篇把「怎么算、怎么切、怎么同步梯度」讲完了。还缺三块，训练才真正闭环：

1. **样本如何变成 batch**  
2. **优化器状态如何分片更新**  
3. **checkpoint 如何按并行布局保存/恢复**  

这篇按数据流顺序精读。

---

## 一、数据：从索引文件到 `get_batch`

### 关键路径

| 文件 | 角色 |
|------|------|
| `datasets/indexed_dataset.py` | `.bin` / `.idx` 随机访问 |
| `datasets/gpt_dataset.py` | `GPTDataset` 组 sample |
| `datasets/blended_megatron_dataset_builder.py` | 多源按权重混合 |
| `pretrain_gpt.py` | `datasets_provider`、`get_batch` |

### 数据流

```text
CLI blend / 路径
  → GPTDatasetConfig
  → BlendedMegatronDatasetBuilder.build
  → GPTDataset / MockGPTDataset / SFTDataset
  → DataLoader / iterator
  → get_batch（TP broadcast、CP 切分、PackedSeqParams）
  → forward_step → model
```

### 精读抓手

1. **`IndexedDataset`**：为什么能近似 O(1) 取文档；mmap 与对象存储路径差异。  
2. **`GPTDataset.__getitem__`**：如何拼出 tokens/labels/loss_mask/position_ids。  
3. **shuffle 索引可缓存**（`_build_document_sample_shuffle_indices`）——复现与启动耗时的关键点。  
4. **`get_batch`**：数据集输出如何变成「符合当前并行布局」的输入。  

官方补充：`docs/user-guide/data-loading.md`。

---

## 二、优化器：`DistributedOptimizer` 在分什么

### 关键路径

- `optimizer/__init__.py`：`get_megatron_optimizer`  
- `optimizer/optimizer.py`：`MegatronOptimizer`、`MixedPrecisionOptimizer`  
- `optimizer/distrib_optimizer.py`：`DistributedOptimizer`  

### 层次直觉

```text
get_megatron_optimizer
  → 基础算法（Adam / SGD / 新兴优化器…）
  → MixedPrecision（FP16/BF16 ↔ FP32 master）
  → 可选 DistributedOptimizer（状态按 DP 分片）
```

**DistributedOptimizer** 近似 ZeRO-1：梯度在 DP 维 reduce-scatter 后，每卡只更新自己分到的状态，再 all-gather 参数（时序以源码为准）。

它与第 7 篇的 grad buffer 布局强耦合——这也是 Megatron 坚持自定义 DDP 的原因之一。

文档：`docs/user-guide/features/dist_optimizer.md`。

### 精读抓手

- `MegatronOptimizer` 抽象：`zero_grad` / `step` / `clip` / `state_dict`  
- `MixedPrecisionOptimizer`：主权重与 unscale  
- `DistributedOptimizer`：分片映射与 param buffer 对应关系  
- MoE 时 expert 是否独立 param group  

---

## 三、Checkpoint：`ShardedTensor` 与并行无关存储

### 关键路径

- `dist_checkpointing/mapping.py`：`ShardedTensor`、`ShardedObject`  
- `dist_checkpointing/serialization.py`：`save` / `load`  
- `dist_checkpointing/strategies/`：后端  
- `training/checkpointing.py`：训练侧编排  

### 核心思想

本地 rank 不保存「一个完整巨大 state_dict」，而是保存：

```text
我持有的 shard
  + 全局形状 / offset / 并行元数据
```

加载时用**当前**并行布局去对齐**当时**保存的 shard，从而支持常见的「换 GPU 数 / 换 TP/PP 再启动」（能力边界以当前策略与版本为准）。

### 精读抓手

1. 模型侧 `sharded_state_dict()` 如何描述切分  
2. `save`：common vs sharded、async、metadata  
3. `load`：为何先要构建「空的 sharded_state_dict 当地图」  
4. resume 与 `parse_and_validate_args` 的交互  

---

## 三者如何嵌进 `pretrain`

```text
setup_model_and_optimizer
  → 模型已按 TP/PP 建好
  → get_megatron_optimizer
  →（可选）load checkpoint

train loop
  → iterators 提供 batch
  → train_step → … → optimizer.step
  → 定期 dist_checkpointing.save
```

笔记里建议把「第一次 iteration 之前」与「稳态 iteration」分成两段 timeline。

---

## 常见坑

1. shuffle 索引缓存与数据变更不一致 → 静默采到旧分布  
2. 只保存模型、忘记优化器/RNG → resume 后 loss 跳变  
3. 并行度变更后硬套非 sharded 旧 ckpt  
4. DistOpt 开启时仍按「每卡完整 Adam 状态」估显存  
5. SFT/FIM 与预训练 `get_batch` 字段不一致  

---

## 动手验证

1. 跟踪一个 mock sample：从 `__getitem__` 到 `get_batch` 返回的 key 列表。  
2. 一句话对比 DistOpt 与 FSDP 各自主要分片什么。  
3. 在 `serialization.py` 找到 `save`/`load` 签名，记下调用方必须先准备什么。  

下一篇进入当代训练热点路径：MoE——Router、Token Dispatcher 与专家并行。
