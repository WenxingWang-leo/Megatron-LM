# Megatron-LM 源码阅读计划与知乎系列文章

本目录包含三类材料：

1. **详细代码阅读计划**：`00-reading-plan.md`  
2. **提纲式导读笔记**：`01`–`10`（结构清晰，适合对照源码自学）  
3. **知乎发布成稿**：[`publish/`](./publish/)（叙事更强，**优先复制这里的文章发知乎**）

内容基于本仓库真实源码结构（Megatron-LM / Megatron Core）。

## 想发知乎？从这里开始

→ **[`publish/README.md`](./publish/README.md)**

十篇成稿标题一览：

1. 别一上来啃 MoE，先把代码地图画清楚  
2. 从 `pretrain_gpt.py` 拆开一次训练步  
3. `GPTModel` 为什么能「换积木不换训练循环」  
4. `parallel_state`：五种并行到底切的是什么  
5. 张量并行：Column / Row 为什么必须配对  
6. 流水线并行：1F1B 是一张调度表  
7. 数据并行：为什么要自己写一套 DDP  
8. 数据、DistOpt、Checkpoint  
9. MoE：Router 与 Dispatcher  
10. 主路径读通之后下一站去哪  

## 提纲笔记索引（自学用）

| 序号 | 文件 | 说明 |
|------|------|------|
| 00 | [00-reading-plan.md](./00-reading-plan.md) | 分阶段阅读计划、必读清单、完成标准 |
| 01–10 | `01-*.md` … `10-*.md` | 提纲式导读（成稿见 `publish/`） |

## 与官方文档的关系

本系列是**源码精读**，不替代官方 User Guide。概念对齐请优先阅读 `docs/get-started/` 与 `docs/user-guide/`。
