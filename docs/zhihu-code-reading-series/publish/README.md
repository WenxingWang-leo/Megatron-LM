# 知乎发布稿 · Megatron 源码精读（精通级 · 扩写版）

本目录是 **面向零基础读者、目标写到「能独立读改 Megatron」** 的知乎专栏成稿。

当前体量约 **9600+ 行** 正文（十篇均在约 900–1100 行），在提纲导读之上补充了：

- 真实源码逐步拆解与关键摘录  
- 多组数值例题（进程组、shape、1F1B、MoE token 流等）  
- 对照表（HF vs Megatron、TE vs local、AllGather vs AllToAll…）  
- 排障决策树 / FAQ / 练习题  
- 第 10 篇含掌握度量表与自学日历  

配套计划：[`../00-reading-plan.md`](../00-reading-plan.md)

## 建议专栏名

**Megatron 源码精读**

## 十篇目录

| 篇 | 文件 | 建议知乎标题 | 约行数 |
|----|------|----------------|--------|
| 01 | [01-megatron-map.md](./01-megatron-map.md) | 从零建立心智模型——地图、术语表与阅读路线 | ~960 |
| 02 | [02-pretrain-loop.md](./02-pretrain-loop.md) | 完整拆解一次训练——从 `__main__` 到 `optimizer.step` | ~1090 |
| 03 | [03-gptmodel-spec.md](./03-gptmodel-spec.md) | GPTModel 全拆解——Config、Spec 到每一层计算 | ~1080 |
| 04 | [04-parallel-state.md](./04-parallel-state.md) | parallel_state 完全指南——进程组如何把 GPU 编成网格 | ~930 |
| 05 | [05-tensor-parallel.md](./05-tensor-parallel.md) | 张量并行完全精读——mappings、Column/Row 与 Attention/MLP | ~920 |
| 06 | [06-pipeline-parallel.md](./06-pipeline-parallel.md) | 流水线并行完全精读——1F1B、气泡、P2P 与 VPP | ~910 |
| 07 | [07-data-parallel.md](./07-data-parallel.md) | 数据并行完全精读——DDP 缓冲、Bucket 与 finalize | ~910 |
| 08 | [08-data-optim-ckpt.md](./08-data-optim-ckpt.md) | 数据管线、DistributedOptimizer 与 Dist Checkpoint | ~940 |
| 09 | [09-moe.md](./09-moe.md) | MoE 完全精读——Router、Dispatcher、专家并行 | ~910 |
| 10 | [10-advanced.md](./10-advanced.md) | 从「读通主路径」到「能改框架」 | ~910 |

## 建议阅读节奏

1. **第 1–2 周**：01–02（地图 + 完整训练链；做文内数值题）  
2. **第 3 周**：03（模型 / Spec / 形状追踪）  
3. **第 4–5 周**：04–07（并行四篇；务必手算进程组与 1F1B 时间线）  
4. **第 6 周**：08–09（数据状态 + MoE）  
5. **第 7–8 周**：10 + 第二轮深挖项目 / 跑通实验  

知乎发布可拆成「上/中/下」或按节连载；单篇较长时建议文首加目录。

## Agent Window 打开

`Ctrl+P` → 例如 `publish/02-pretrain-loop`

## 质量说明

扩写原则是 **可验证细节优先**：公式、进程组表、shape、调度时间线、源码分支均应对齐本仓库实现。若你发现与当前 `main` 漂移，以源码为准并在文首注明阅读的 commit。

「精通」仍需配合动手：跑 `examples/run_simple_mcore_train_loop.py`、打断点、改并行度做对照。文档把主路径讲透；实验把知识变成肌肉记忆。
