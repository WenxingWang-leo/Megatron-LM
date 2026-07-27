# 知乎发布稿 · Megatron 源码精读（精通级）

本目录是 **面向零基础读者、目标写到「能独立读改 Megatron」** 的知乎专栏成稿。

相较上级提纲笔记（`../01`–`../10`），本目录文章：

- 单篇约 **450–700 行**，含概念从零、源码逐步拆解、数值例题、调试清单与练习题  
- 引用真实路径：`pretrain_gpt.py`、`megatron/core/...`  
- 可直接复制到知乎发布（建议先发 01–02 测反馈，再连载并行篇）

配套自学计划仍见：[`../00-reading-plan.md`](../00-reading-plan.md)

## 建议专栏名

**Megatron 源码精读**

## 十篇目录与体量

| 篇 | 文件 | 建议知乎标题 | 约行数 |
|----|------|----------------|--------|
| 01 | [01-megatron-map.md](./01-megatron-map.md) | 从零建立心智模型——地图、术语表与阅读路线 | ~500 |
| 02 | [02-pretrain-loop.md](./02-pretrain-loop.md) | 完整拆解一次训练——从 `__main__` 到 `optimizer.step` | ~640 |
| 03 | [03-gptmodel-spec.md](./03-gptmodel-spec.md) | GPTModel 全拆解——Config、Spec 到每一层计算 | ~700 |
| 04 | [04-parallel-state.md](./04-parallel-state.md) | parallel_state 完全指南——进程组如何把 GPU 编成网格 | ~510 |
| 05 | [05-tensor-parallel.md](./05-tensor-parallel.md) | 张量并行完全精读——mappings、Column/Row 与 Attention/MLP | ~480 |
| 06 | [06-pipeline-parallel.md](./06-pipeline-parallel.md) | 流水线并行完全精读——1F1B、气泡、P2P 与 VPP | ~460 |
| 07 | [07-data-parallel.md](./07-data-parallel.md) | 数据并行完全精读——DDP 缓冲、Bucket 与 finalize | ~480 |
| 08 | [08-data-optim-ckpt.md](./08-data-optim-ckpt.md) | 数据管线、DistributedOptimizer 与 Dist Checkpoint | ~630 |
| 09 | [09-moe.md](./09-moe.md) | MoE 完全精读——Router、Dispatcher、专家并行 | ~550 |
| 10 | [10-advanced.md](./10-advanced.md) | 从「读通主路径」到「能改框架」 | ~560 |

合计约 **5500+ 行** 精读正文。

## 建议阅读 / 发布节奏

1. **第 1 周**：01 地图 + 02 训练循环（建立全局调用链）  
2. **第 2 周**：03 模型积木（Config / Spec / Layer）  
3. **第 3–4 周**：04–07 五种并行（务必做文内数值题）  
4. **第 5 周**：08 数据与状态 + 09 MoE  
5. **第 6 周**：10 进阶与「第二轮深挖项目」  

每篇文末有练习题；做完再进下一篇，效果远好于只收藏。

## 在 Cursor Agent Window 打开

`Ctrl+P`（Mac：`Cmd+P`）搜索例如：

```text
publish/02-pretrain-loop
publish/05-tensor-parallel
```

## 发布到知乎小提示

1. 标题用上表「建议知乎标题」，可加前缀「精读 Megatron 源码（N）」  
2. 文首注明：基于 NVIDIA Megatron-LM / Megatron Core，以对应 commit/`main` 为准  
3. 代码块保持 Markdown；路径保留便于读者对照仓库  
4. 系列导航改成你专栏的真实链接  

## 诚实边界

「精通」仍需配合：**跑通示例、打断点、改配置做对照实验**。文档负责把主路径与关键细节讲透；动手验证负责把知识变成肌肉记忆。第 10 篇给出了可验收的第二轮深挖项目。
