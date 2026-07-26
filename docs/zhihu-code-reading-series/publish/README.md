# 知乎发布稿（Megatron 源码精读）

本目录是基于上级导读笔记（`../01`–`../10`、`../00-reading-plan.md`）**改写的知乎专栏成稿**，更偏叙事与精读节奏，可直接复制到知乎发布。

## 建议专栏名

**Megatron 源码精读**

## 发布顺序与标题

| 序号 | 文件 | 建议知乎标题 |
|------|------|----------------|
| 01 | [01-megatron-map.md](./01-megatron-map.md) | 精读 Megatron 源码（1）：别一上来啃 MoE，先把代码地图画清楚 |
| 02 | [02-pretrain-loop.md](./02-pretrain-loop.md) | 精读 Megatron 源码（2）：从 `pretrain_gpt.py` 拆开一次训练步 |
| 03 | [03-gptmodel-spec.md](./03-gptmodel-spec.md) | 精读 Megatron 源码（3）：`GPTModel` 为什么能「换积木不换训练循环」 |
| 04 | [04-parallel-state.md](./04-parallel-state.md) | 精读 Megatron 源码（4）：`parallel_state`——五种并行到底切的是什么 |
| 05 | [05-tensor-parallel.md](./05-tensor-parallel.md) | 精读 Megatron 源码（5）：张量并行——Column / Row 这一对为什么必须配对 |
| 06 | [06-pipeline-parallel.md](./06-pipeline-parallel.md) | 精读 Megatron 源码（6）：流水线并行——1F1B 不是玄学，是一张调度表 |
| 07 | [07-data-parallel.md](./07-data-parallel.md) | 精读 Megatron 源码（7）：数据并行——为什么 Megatron 要自己写一套 DDP |
| 08 | [08-data-optim-ckpt.md](./08-data-optim-ckpt.md) | 精读 Megatron 源码（8）：数据、DistOpt、Checkpoint——训练闭环的另外三块 |
| 09 | [09-moe.md](./09-moe.md) | 精读 Megatron 源码（9）：MoE——Router 决定去哪，Dispatcher 负责物流 |
| 10 | [10-advanced.md](./10-advanced.md) | 精读 Megatron 源码（10）：主路径读通之后，下一站去哪 |

## 发布到知乎时的小提示

1. **标题**直接用上表「建议知乎标题」。  
2. **文首**可加一句：基于 NVIDIA Megatron-LM / Megatron Core 源码精读，版本以仓库 `main` 为准。  
3. **代码块**：知乎支持 Markdown 代码块；文中的 `path:line` 引用可改成「文件路径 + 摘录」。  
4. **系列导航**：每篇已含「上一篇/下一篇」语境，发布时可改成你专栏的实际链接。  
5. **节奏**：建议每周 1–2 篇；先发 01–02 测反馈，再连载并行三部曲（04–07）。  

## 与上级文件的关系

| 上级文件 | 用途 |
|----------|------|
| `../00-reading-plan.md` | 系统学习计划（偏自学手册，不一定整篇发知乎） |
| `../01`–`../10` | 提纲式导读笔记 |
| `./01`–`./10`（本目录） | **知乎成稿，优先发这些** |
