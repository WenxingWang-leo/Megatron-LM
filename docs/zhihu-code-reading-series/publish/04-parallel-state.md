# 精读 Megatron 源码（4）：`parallel_state`——五种并行到底切的是什么

> **专栏**：Megatron 源码精读 · 第 4 篇  
> **核心文件**：`megatron/core/parallel_state.py`、`process_groups_config.py`

---

Megatron 里几乎所有「神奇行为」，背后都是一句很土的话：

> **当前 rank 属于哪些 process group，就和谁通信。**

`ColumnParallelLinear` 的 all-reduce、PP 的 P2P、DDP 的梯度桶、MoE 的 all-to-all——通信对象都来自 `parallel_state`（或更新式的 `ProcessGroupCollection`）。

这篇不讲数学证明，只讲：**进程组怎么建、默认 order 意味着什么、你该怎么手算一个小例子。**

---

## 五种并行各切什么

| 缩写 | 切分对象 | 一句话 |
|------|----------|--------|
| **TP** | 单层内权重/激活 | 一层拆到多卡一起算 |
| **PP** | 模型深度（层） | 不同卡持有不同层 |
| **DP** | 数据样本 | 副本（或分片）刷不同 batch |
| **CP** | 序列长度 | 长序列切成 chunk |
| **EP** | MoE 专家 | 不同卡持有不同 expert |

它们可以组合。初始化里有硬约束：**world size 必须能被相关并行度乘积整除**。

非专家路径常见关系：

```text
DP = world_size / (TP × PP × CP)
```

（EP / expert-TP 有单独的 RankGenerator，MoE 篇再展开。）

---

## 入口：`initialize_model_parallel`

签名很长，但核心参数你先认这些：

```547:561:megatron/core/parallel_state.py
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    # ...
    context_parallel_size: int = 1,
    # ...
    expert_model_parallel_size: int = 1,
    # ...
    order: str = "tp-cp-ep-dp-pp",
```

默认 `order="tp-cp-ep-dp-pp"` 决定 rank 如何被正交分组：在多维网格上给每个 GPU 编坐标，固定其他维、沿某一维走，就得到对应 process group。

实现上的关键抽象：

- `RankGenerator` / `get_ranks`  
- `generate_masked_orthogonal_rank_groups`  
- `create_group`（封装 `torch.distributed.new_group`）

**精读顺序建议**：`ProcessGroupCollection` 字段 → `RankGenerator` → `initialize_model_parallel` 建组循环 → getter API。

---

## 手算一个小例子（请真的算一遍）

```text
world_size = 16
TP=2, PP=2, CP=2, EP=1
→ DP = 16 / (2×2×2) = 2
```

对任意 rank，问三个问题：

1. 我的 **TP 同伴**是谁？（Column/Row 通信）  
2. 我的 **PP 上一跳 / 下一跳**是谁？（P2P 传激活）  
3. 我的 **DP（常为 dp-cp）同伴**是谁？（梯度同步）  

能对 rank=0 和 rank=7 各答一遍，这篇就算过关。

---

## 为什么经常看到 `dp-cp`

Context Parallel 切的是序列，**权重通常仍在 CP 维上复制**。因此做数据并行梯度同步时，常常要在「DP × CP」合成组上通信，避免漏同步。

阅读 `finalize_model_grads` 时看到 `dp-cp`，不要惊讶：那是「参数副本」视角下的数据并行组。

---

## `ProcessGroupCollection`：读与写的新约定

文件：`process_groups_config.py`

新趋势是：把一组 `ProcessGroup` 收进 `ProcessGroupCollection`，从调用方注入模块/调度器，而不是在深层回调全局 `parallel_state`。

本仓库贡献指南也写了：

- 兼容点（`parallel_state.py`、初始化、测试、迁移 fallback）可以用全局 getter  
- **`megatron/core` 新生产逻辑应避免继续扩散 `get_*_group()`**

读老代码时两者并存；写新代码请走注入式。

---

## 边角但不容忽视：embedding ranks

`initialize_model_parallel` 支持 `get_embedding_ranks` 回调。GPT 脚本里典型逻辑是：PP 首 stage（共享 embedding 时还有尾 stage，以及 MTP ranks）参与 embedding 相关通信。

这些组不是「主路径五种并行」，却是共享权重 / checkpoint / MTP 出 bug 的温床。

---

## Virtual PP 先放一放

VPP 不额外创造一种切分维，它改变的是：**同一 PP rank 持有多个 model chunk（层交错）**，从而改变 schedule 与气泡。拓扑上先理解物理 PP group；VPP 放到第 6 篇和 1F1B 一起看更合适。

---

## 常见坑

1. 并行度乘积 ≠ world size → 初始化直接失败  
2. order 漏掉 size>1 的维 → 分组错误  
3. 把 CP 当成「不影响通信组」→ 权重同步常看 `dp-cp`  
4. EP 与 CP 在同一 RankGenerator 上同时 >1 → 当前实现有限制  
5. 只看 getter、不看创建过程 → 自定义 order 时会懵  

---

## 动手验证

1. 写出 `TP=4, PP=2, CP=1, world=16` 时的 DP，并描述 rank 0 的 TP group 应有几人。  
2. 在 `initialize_model_parallel` 里找到创建 `tp` 与 `pp` group 的代码段。  
3. 打开 `ProcessGroupCollection`，列出与 TP/PP/DP 对应的字段名。  

下一篇把拓扑用起来：张量并行下的 `ColumnParallelLinear` / `RowParallelLinear`——Megatron 最经典、也最值得精读的一层。
