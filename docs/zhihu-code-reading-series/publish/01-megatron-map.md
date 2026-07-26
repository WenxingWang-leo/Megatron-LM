# 精读 Megatron 源码（1）：别一上来啃 MoE，先把代码地图画清楚

> **专栏**：Megatron 源码精读  
> **适合谁**：会写 PyTorch，用过 `DistributedDataParallel`，想搞懂大模型分布式训练框架到底怎么落地  
> **对应源码**：NVIDIA Megatron-LM / Megatron Core（以仓库 `main` 为准）

---

很多人打开 Megatron 仓库的第一反应是：文件也太多了吧。然后要么在 `transformer/moe` 里迷路，要么在两千行的 `schedules.py` 里劝退。

其实 Megatron 没那么玄。**它不是一个巨大的 `train.py`，而是两层结构**：

1. **Megatron Core**（`megatron/core/`）——可组合的训练积木：并行、Transformer 层、优化器、分布式 checkpoint。
2. **Megatron-LM**——站在 Core 上的参考训练栈：参数解析、数据、日志、ckpt 编排。你常看到的 `pretrain_gpt.py` 就在这一层。

把这两层分清，后面读代码会轻松一个数量级。

---

## 大模型框架要回答的三个问题

| 问题 | Megatron 落点 |
|------|----------------|
| 模型怎么表示？ | `GPTModel` + `TransformerLayer` + `ModuleSpec` |
| 算力怎么切开？ | TP / PP / DP / CP / EP + `parallel_state` |
| 一步训练怎么串起来？ | `pretrain` → `train_step` → `forward_backward_*` |

论文负责讲「为什么这样切」；源码负责讲「通信到底发给谁、shape 怎么变、bug 出在哪一行」。本专栏走后者。

---

## 十分钟建立代码地图

根目录先只记这些：

```text
pretrain_gpt.py          → 入口剧本（注入数据/前向/模型构建）
gpt_builders.py          → args 如何变成 GPTModel
megatron/training/       → 训练编排：initialize / pretrain / train_step
megatron/core/           → 核心库：并行、层、优化器、数据、ckpt
examples/                → 可跑示例
docs/                    → 官方文档（先对齐概念再用）
```

Core 里再记六个「街区」：

1. `parallel_state.py` —— 进程组从哪来  
2. `tensor_parallel/` —— 层内怎么切  
3. `pipeline_parallel/` —— 层间怎么排流水  
4. `distributed/` —— 梯度在数据并行维怎么同步  
5. `transformer/` + `models/` —— 算什么  
6. `optimizer/` + `dist_checkpointing/` —— 状态怎么更新和保存  

后面每一篇，基本就是把其中一个街区拆开。

---

## 精读方法论：五条就够

### 1. 先画调用链，再进函数体

不要打开 `schedules.py` 从头读到尾。先问：

- 谁调用了 `get_forward_backward_func`？
- 它根据什么条件返回不同 schedule？
- `train_step` 前后各发生了什么？

调用图画出来了，再钻进某一个 `forward_backward_pipelining_without_interleaving`。

### 2. 脑子里固定一组「小世界」配置

```text
world_size = 8
TP=2, PP=2, CP=1, EP=1
→ DP = 8 / (2 × 2 × 1) = 2
```

所有 shape、通信、microbatch，先在这组数字上推演，再放大到千卡。

### 3. 一次只拧一个旋钮

- 学 TP：关掉 PP / VPP / MoE  
- 学 PP：先非 interleaving，再 VPP  
- 学 MoE：先固定一种 dispatcher  

### 4. 用单测当说明书

`tests/unit_tests/` 按模块名搜索，往往比二手博客更接近当前代码行为。

### 5. 新代码别再到处 `get_*_group()`

仓库约定很清楚：在 `megatron/core` 里，新逻辑应优先接收 `ProcessGroupCollection`（或显式 `ProcessGroup`）并向下传。老代码里全局 getter 很多，阅读时认得出即可；写新代码请走注入式。

---

## 文档和源码怎么搭配

| 你想搞清… | 先看文档 | 再看代码 |
|-----------|----------|----------|
| 第一次跑通 | `docs/get-started/` | `examples/run_simple_mcore_train_loop.py` |
| 并行概念 | `docs/user-guide/parallelism-guide.md` | `parallel_state.py` |
| MoE | `docs/user-guide/features/moe.md` | `transformer/moe/` |

文档回答「推荐怎么用」；源码回答「边界条件卡在哪」。

---

## 本专栏目录

1. **开篇：代码地图**（本文）  
2. 从 `pretrain_gpt.py` 走进训练主循环  
3. `GPTModel`：Spec 驱动的 Transformer 积木  
4. `parallel_state`：五种并行的进程组  
5. 张量并行：Column / Row Parallel Linear  
6. 流水线并行：1F1B 与 microbatch  
7. 数据并行：DDP、梯度桶与 finalize  
8. 数据管线、DistributedOptimizer 与 Dist Checkpoint  
9. MoE：Router、Dispatcher 与专家并行  
10. 进阶：CP、推理、RL 与继续深挖  

配套阅读计划见同目录上级的 `00-reading-plan.md`。

---

## 写在最后

如果你只能做一件事：打开根目录 `README.md` 的 Project Structure，自己画一张只含十个目录的简图，再用三句话区分 Core 和 LM。

下一篇我们从 `pretrain_gpt.py` 的 `if __name__ == "__main__"` 出发，把「一次训练步」拆到源码级。

（欢迎收藏专栏，按篇跟读。有 GPU 的话，顺手跑一下 `examples/run_simple_mcore_train_loop.py`，后面讨论并行时会有感觉。）
