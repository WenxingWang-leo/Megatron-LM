# 知乎系列 08｜数据管线、DistributedOptimizer 与 Dist Checkpoint

> 目标：补齐训练闭环的三块「状态」：样本如何变成 batch、优化器状态如何分片、checkpoint 如何按并行布局保存与恢复。

---

## 一、数据：从索引文件到 `get_batch`

### 关键路径

| 文件 | 角色 |
|------|------|
| `megatron/core/datasets/indexed_dataset.py` | `.bin` / `.idx` 随机访问 |
| `megatron/core/datasets/gpt_dataset.py` | `GPTDataset` 组 sample |
| `megatron/core/datasets/blended_megatron_dataset_builder.py` | 多源按权重混合构建 |
| `megatron/core/datasets/blended_dataset.py` | 混合数据集 |
| `pretrain_gpt.py` | `train_valid_test_datasets_provider`、`get_batch` |
| `megatron/training/datasets/sft_dataset.py` 等 | SFT / FIM 变体 |

### 数据流

```text
CLI blend / 路径
  → GPTDatasetConfig（core_gpt_dataset_config_from_args）
  → BlendedMegatronDatasetBuilder.build
  → GPTDataset / MockGPTDataset / SFTDataset
  → DataLoader / iterator
  → get_batch（TP broadcast、CP 切分、PackedSeqParams）
  → forward_step → model
```

### 阅读抓手

1. **`IndexedDataset`**：为什么能 O(1) 取文档；mmap 与对象存储路径差异。  
2. **`GPTDataset.__getitem__`**：如何从 document 索引拼出定长（或打包）序列，生成 tokens/labels/loss_mask/position_ids。  
3. **`_build_document_sample_shuffle_indices`**：shuffle 索引可缓存——这是复现与启动耗时的关键点。  
4. **`get_batch`**：数据集输出如何变成「符合当前并行布局」的输入。

官方补充：`docs/user-guide/data-loading.md`、`data-preparation.md`。

---

## 二、优化器：`get_megatron_optimizer` 与 DistOpt

### 关键路径

- `megatron/core/optimizer/__init__.py`：`get_megatron_optimizer`  
- `megatron/core/optimizer/optimizer.py`：`MegatronOptimizer`、`MixedPrecisionOptimizer`、`ChainedOptimizer`  
- `megatron/core/optimizer/distrib_optimizer.py`：`DistributedOptimizer`  
- `megatron/core/optimizer/optimizer_config.py`  
- `megatron/core/optimizer_param_scheduler.py`（LR）

### 层次直觉

```text
get_megatron_optimizer
  → 按 param group 构建基础算法（Adam / SGD / 新兴优化器等）
  → 包上 MixedPrecision（FP16/BF16 ↔ FP32 master weights）
  → 可选 DistributedOptimizer（优化器状态按 DP 分片）
```

**DistributedOptimizer** 近似 ZeRO-1 思想：参数梯度在 DP 维 reduce-scatter 后，每卡只更新自己分到的状态分片，再 all-gather 参数（具体时序以源码为准）。  

它与第 07 篇的 grad buffer 布局强耦合——这也是 Megatron 坚持自定义 DDP 的原因之一。

文档：`docs/user-guide/features/dist_optimizer.md`。

### 阅读抓手

- `MegatronOptimizer` 抽象：`zero_grad` / `step` / `clip` / `state_dict`  
- `MixedPrecisionOptimizer`：主权重与梯度 unscale  
- `DistributedOptimizer`：分片映射、与 param buffer 的对应关系  
- 专家参数是否走单独 param group / 链式优化器（MoE 场景）

---

## 三、Checkpoint：`ShardedTensor` 与并行无关存储

### 关键路径

- `megatron/core/dist_checkpointing/mapping.py`：`ShardedTensor`、`ShardedObject`  
- `megatron/core/dist_checkpointing/serialization.py`：`save` / `load`  
- `megatron/core/dist_checkpointing/strategies/`：后端实现（如 torch_dist）  
- `megatron/training/checkpointing.py`：训练侧编排（何时存、存什么、怎么与 args 交互）

### 核心思想

本地 rank 并不保存「一个完整巨大 state_dict」，而是保存：

```text
我持有的 shard
  + 全局形状 / offset / 并行元数据
```

这样加载时可以用**当前**并行布局去对齐**当时**保存的 shard，实现常见的「换 GPU 数 / 换 TP/PP 再启动」（能力边界以当前策略与版本为准）。

### 阅读抓手

1. 模型侧 `sharded_state_dict()`（如 `GPTModel`）如何描述自己的切分  
2. `save`：common vs sharded、async 路径、metadata  
3. `load`：为何需要先构建「空的 sharded_state_dict 作为地图」  
4. 训练脚本中的 save interval、resume 与 `parse_and_validate_args` 的交互  

---

## 三者如何嵌进 `pretrain`

```text
setup_model_and_optimizer
  → 模型已按 TP/PP 建好
  → get_megatron_optimizer
  → （可选）load checkpoint → 恢复 model/optim/scheduler/rng

train loop
  → 数据 iterators 提供 batch
  → train_step → … → optimizer.step
  → 定期 dist_checkpointing.save
```

建议在笔记里把「第一次 iteration 之前」与「稳态 iteration」分成两段 timeline。

---

## 建议阅读顺序

```text
datasets_provider → BlendedMegatronDatasetBuilder → GPTDataset.__getitem__
  → get_batch
  → get_megatron_optimizer → DistributedOptimizer（先读类文档再读 step）
  → GPTModel.sharded_state_dict
  → dist_checkpointing.save/load
  → training/checkpointing.py 编排函数
```

---

## 常见坑

1. **shuffle 索引缓存与数据变更不一致** → 静默采到旧分布。  
2. **只保存模型、忘记优化器/RNG** → resume 后 loss 跳变。  
3. **并行度变更后硬套非 sharded 旧 ckpt** → 加载失败或错位。  
4. **DistOpt 开启时按「每卡完整 Adam 状态」估算显存** → 估算偏差。  
5. **SFT/FIM 数据集与预训练 `get_batch` 字段不一致** → 运行期 KeyError/shape 错。  

---

## 本周作业

1. 跟踪一个 mock sample：从 `GPTDataset.__getitem__` 到 `get_batch` 返回的 key 列表。  
2. 说明 DistOpt 与 FSDP 各自主要分片什么（一句话对比）。  
3. 在 `serialization.py` 找到 `save`/`load` 签名，记下「调用方必须先准备什么」。  

下一篇进入当代大模型训练的热点路径：MoE——Router、Token Dispatcher 与专家并行。
