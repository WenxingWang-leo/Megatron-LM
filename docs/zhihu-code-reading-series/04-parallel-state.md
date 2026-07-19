# 知乎系列 04｜`parallel_state`：五种并行的进程组拓扑

> 目标：看懂 `initialize_model_parallel` 如何把 `world_size` 切成 TP/PP/DP/CP/EP 进程组，并为后续 TP/PP/DP 文章打好坐标系。

---

## 为什么这篇必须先读

Megatron 里几乎所有「神奇行为」背后都是一句很土的话：

> 当前 rank 属于哪些 process group，就和谁通信。

`ColumnParallelLinear` 的 all-reduce、PP 的 P2P、DDP 的梯度桶、MoE 的 all-to-all——通信对象都来自 `parallel_state`（或更新式的 `ProcessGroupCollection`）。

文件：

- `megatron/core/parallel_state.py`  
- `megatron/core/process_groups_config.py`

---

## 五种并行各切什么

| 缩写 | 切分对象 | 直观理解 |
|------|----------|----------|
| **TP** Tensor Parallel | 单层内的权重/激活 | 一层拆到多卡一起算 |
| **PP** Pipeline Parallel | 模型深度（层） | 不同卡持有不同层 |
| **DP** Data Parallel | 数据样本 | 每卡一份（或分片）模型副本，刷不同 batch |
| **CP** Context Parallel | 序列长度 | 长序列切成 chunk |
| **EP** Expert Parallel | MoE 专家 | 不同卡持有不同 expert |

它们可以组合。核心约束写在初始化逻辑里：**world size 必须能被相关并行度乘积整除**。

对非专家路径，常见关系是：

```text
data_parallel_size = world_size / (TP * PP * CP)
```

（EP / expert-TP 有单独的 RankGenerator 与 expert DP，MoE 篇再展开。）

---

## 默认 order：`tp-cp-ep-dp-pp`

`initialize_model_parallel(..., order="tp-cp-ep-dp-pp")` 里的 `order` 决定 rank 如何被「正交分组」。

可以把它想成：在多维网格上给每个 GPU 编坐标，然后：

- 固定其他维、沿 TP 维走 → TP group  
- 沿 PP 维走 → PP group  
- ……

实现上的关键抽象是 `RankGenerator` 与 `generate_masked_orthogonal_rank_groups`：用 mask 选出某一并行维对应的 rank 列表，再 `create_group`（封装 `torch.distributed.new_group`）。

**读法建议**：

1. 先读 `RankGenerator.__init__` / `get_ranks`  
2. 再读 `initialize_model_parallel` 里创建 TP、PP、DP、CP、`dp-cp` 等 group 的循环  
3. 最后浏览 `get_tensor_model_parallel_group` 一类 getter  

---

## 手算一个小例子

假设：

```text
world_size = 16
TP=2, PP=2, CP=2, EP=1
→ DP = 16 / (2*2*2) = 2
```

对任意 rank，问三个问题：

1. 我的 TP 同伴是谁？（一起做 Column/Row 通信）  
2. 我的 PP 上一跳 / 下一跳是谁？（P2P 传激活）  
3. 我的 DP（常与 CP 组合为 `dp-cp`）同伴是谁？（梯度 all-reduce / reduce-scatter）  

如果你能对 rank=0 和 rank=7 各答一遍，这篇就算读懂了。

---

## `dp-cp` 为什么经常一起出现

Context Parallel 切的是序列，**权重通常仍在 CP 维上复制**。因此做数据并行梯度同步时，常常需要在「DP × CP」合成组上通信，避免漏同步。  

阅读 `initialize_model_parallel` 与 `finalize_model_grads` 时，看到 `dp-cp` 不要惊讶：它是「参数副本」视角下的数据并行组。

---

## `ProcessGroupCollection`：阅读与写作的新约定

文件：`process_groups_config.py`

新代码趋势是：把一组 `ProcessGroup` 收进 `ProcessGroupCollection`，从调用方注入到模块/调度器，而不是在深层模块里回调全局 `parallel_state`。

本仓库 `AGENTS.md` 也写了同样的指导原则：

- **允许**在 `parallel_state.py`、初始化/bootstrap、测试、迁移 fallback 等兼容点使用全局 getter  
- **避免**在 `megatron/core` 新生产逻辑里继续扩散 `parallel_state.get_*_group()`  

读老代码时你会两者并存；写新代码时请走注入式。

---

## Embedding ranks 与「边角」进程组

`initialize_model_parallel` 还支持 `get_embedding_ranks` / `get_position_embedding_ranks` 回调。  

GPT 预训练脚本里（`pretrain_gpt.py` 的 `get_embedding_ranks`）典型逻辑是：PP 首 stage（以及在共享 embedding 时的尾 stage，再加上 MTP ranks）需要参与 embedding 相关通信。  

这些「不是主路径五种并行、但很关键」的组，往往是 checkpoint / 共享权重 / MTP 出 bug 的温床。

---

## Virtual Pipeline（VPP）在拓扑中的位置

VPP 不额外创造一种「切分维度」，它改变的是：**同一个 PP rank 上持有多个 model chunk（交错层）**，从而改变 schedule 与气泡。  

初始化时若设置 `virtual_pipeline_model_parallel_size`，会要求 `PP > 1`，并影响后续 `get_forward_backward_func` 选择 interleaving 调度。  

拓扑上你仍先理解物理 PP group；VPP 放到第 06 篇与 schedule 一起看更合适。

---

## 阅读清单（本篇）

| 优先级 | 符号 | 看什么 |
|--------|------|--------|
| P0 | `initialize_model_parallel` | 组如何创建、断言条件 |
| P0 | `RankGenerator` | order 与 get_ranks |
| P0 | `ProcessGroupCollection` | 字段含义 |
| P1 | `get_*_group` / `get_*_world_size` / `get_*_rank` | 查询 API |
| P1 | expert 相关 `get_expert_*` | 为 MoE 篇铺垫 |
| P2 | SHARP / Gloo / hierarchical CP 参数 | 集群性能与特殊拓扑 |

---

## 常见坑

1. **并行度乘积 ≠ world size** → 初始化直接失败。  
2. **order 字符串漏掉 size>1 的维** → 分组错误或断言。  
3. **把 CP 当成「不影响通信组」** → 权重同步常看 `dp-cp`。  
4. **EP 与 CP 在同一 RankGenerator 上同时 >1** → 当前实现有限制（见源码注释/断言）。  
5. **只看 getter，不看创建过程** → 遇到自定义 order 时会懵。

---

## 本周作业

1. 写出 `TP=4, PP=2, CP=1, world=16` 时的 DP，并描述 rank 0 的 TP group 应有几人。  
2. 在 `initialize_model_parallel` 源码中找到创建 `tp` 与 `pp` group 的代码段，各贴（笔记里）五行关键逻辑。  
3. 打开 `ProcessGroupCollection`，列出与 TP/PP/DP 对应的字段名。  

下一篇把拓扑用起来：张量并行下的 `ColumnParallelLinear` / `RowParallelLinear`。
