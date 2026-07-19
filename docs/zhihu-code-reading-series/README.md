# Megatron-LM 源码阅读计划与知乎系列文章

本目录包含两份材料：

1. **详细代码阅读计划**（`00-reading-plan.md`）  
2. **可直接发布到知乎的系列导读文章**（`01`–`10`）

内容基于本仓库真实源码结构（Megatron-LM / Megatron Core）整理，面向希望系统读通分布式训练主路径的工程师。

## 阅读顺序

| 序号 | 文件 | 说明 |
|------|------|------|
| 00 | [00-reading-plan.md](./00-reading-plan.md) | 分阶段阅读计划、必读清单、作业与完成标准 |
| 01 | [01-overview.md](./01-overview.md) | 开篇：代码地图与阅读方法论 |
| 02 | [02-startup-chain.md](./02-startup-chain.md) | 从 `pretrain_gpt.py` 走进训练主循环 |
| 03 | [03-gptmodel-transformer.md](./03-gptmodel-transformer.md) | `GPTModel` 与 Spec 驱动积木 |
| 04 | [04-parallel-state.md](./04-parallel-state.md) | 五种并行的进程组拓扑 |
| 05 | [05-tensor-parallel.md](./05-tensor-parallel.md) | 张量并行 Column/Row Linear |
| 06 | [06-pipeline-parallel.md](./06-pipeline-parallel.md) | 流水线并行 1F1B 调度 |
| 07 | [07-data-parallel.md](./07-data-parallel.md) | DDP、梯度桶与 finalize |
| 08 | [08-data-optim-ckpt.md](./08-data-optim-ckpt.md) | 数据、DistributedOptimizer、Dist Checkpoint |
| 09 | [09-moe.md](./09-moe.md) | MoE：Router / Dispatcher / EP |
| 10 | [10-advanced-roadmap.md](./10-advanced-roadmap.md) | CP、推理、RL 与继续深挖 |

## 发布到知乎的建议

- 专栏名建议：「Megatron 源码走读」或「Megatron-LM 代码阅读笔记」
- 每篇保留文首系列导航与文末作业
- 注明对应的仓库 commit / 版本，减少 API 漂移误导
- 公式与大段代码可酌情改为知乎支持的格式；文中路径保持与仓库一致便于读者跳转

## 与官方文档的关系

本系列是**源码导读**，不替代官方 User Guide。概念对齐请优先阅读 `docs/get-started/` 与 `docs/user-guide/`，再回到本系列对应篇目对照实现。
