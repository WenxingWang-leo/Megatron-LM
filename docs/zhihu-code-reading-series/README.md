# Megatron-LM 源码阅读计划与知乎系列

本目录材料目标：**让没读过 Megatron-LM 的人，能靠系统阅读 + 文内练习，把主路径读到可独立改框架的程度。**

## 优先读这里（精通级成稿）

→ **[`publish/`](./publish/)**（约 5500 行精读正文，十篇知乎成稿）

| 篇 | 内容 |
|----|------|
| 01 | 地图、完整术语表、阅读路线 |
| 02 | `__main__` → `train_step` → `optimizer.step` 全拆解 |
| 03 | GPTModel / Config / Spec / Layer 数值例题 |
| 04 | parallel_state 进程组手算 |
| 05 | TP mappings + Column/Row + Attention/MLP |
| 06 | PP 1F1B 时间线 + P2P + VPP |
| 07 | DDP buffer / Bucket / finalize 八步 |
| 08 | Dataset / DistOpt / Dist Checkpoint |
| 09 | MoE Router / Dispatcher / EP |
| 10 | CP/推理/RL/贡献与第二轮深挖项目 |

索引与发布说明：[`publish/README.md`](./publish/README.md)

## 辅助材料

| 文件 | 用途 |
|------|------|
| [`00-reading-plan.md`](./00-reading-plan.md) | 分阶段自学计划、必读清单、完成标准 |
| `01`–`10`（本目录根下） | 早期提纲笔记；**正式精读请用 `publish/`** |

## 打开方式（Agent Window）

`Ctrl+P` → 输入 `publish/01-megatron-map` 或任意篇文件名。

## 与官方文档

概念对齐优先看 `docs/get-started/`、`docs/user-guide/`；本系列负责**源码级拆解与例题**。
