# 知乎系列 01｜开篇：Megatron 代码地图与阅读方法论

> 系列定位：跟着 NVIDIA Megatron-LM / Megatron Core **真实源码**走读，而不是只复述论文公式。  
> 适合读者：会写 PyTorch、了解 `DistributedDataParallel` 基本概念，想搞懂大模型分布式训练框架如何落地。

---

## 先给结论

Megatron 不是「一个巨大的 `train.py`」，而是两层：

1. **Megatron Core**（`megatron/core/`）：可组合的 GPU 训练积木——并行、Transformer 层、优化器、分布式 checkpoint、推理等。
2. **Megatron-LM**：站在 Core 之上的**参考训练栈**——参数解析、数据、日志、ckpt 编排、示例脚本。你日常看到的 `pretrain_gpt.py` 就属于这一层。

读源码时最容易犯的错，是把两层揉在一起，或者一上来钻进 MoE / FP8 / CUDA Graph。正确顺序是：**地图 → 启动链 → 模型积木 → 并行 → 数据与状态 → 专题**。

---

## 为什么值得精读 Megatron

大模型训练框架要同时回答三个问题：

| 问题 | Megatron 的落点 |
|------|-----------------|
| 模型怎么表示？ | `GPTModel` + `TransformerLayer` + `ModuleSpec` |
| 算力怎么切开？ | TP / PP / DP / CP / EP + `parallel_state` |
| 一步训练怎么串起来？ | `pretrain` → `train_step` → `forward_backward_*` |

它把「算法论文里的并行策略」落成了可以挂调试器的 Python/CUDA 代码。读通之后，你再看 DeepSpeed、FSDP、其他自研框架，会轻松很多——概念是共通的，只是 API 与工程取舍不同。

---

## 十分钟建立代码地图

根目录结构可以压缩成一张表：

```
pretrain_gpt.py          → 入口剧本
gpt_builders.py          → args 如何变成 GPTModel
megatron/training/       → 训练编排（initialize / pretrain / train_step）
megatron/core/           → 核心库（并行、层、优化器、数据、ckpt）
examples/                → 可跑示例（建议先跑 simple loop）
docs/                    → 官方文档（概念对齐用）
```

Core 内部再记六个「街区」即可：

1. `parallel_state.py`：进程组从哪里来  
2. `tensor_parallel/`：层内怎么切  
3. `pipeline_parallel/`：层间怎么排流水  
4. `distributed/`：梯度怎么在数据并行维同步  
5. `transformer/` + `models/`：算什么  
6. `optimizer/` + `dist_checkpointing/`：状态怎么存

本系列后续文章会按街区逐个拆开。

---

## 推荐阅读方法论（请直接照做）

### 1. 先调用链，后实现细节

不要打开 `schedules.py` 两千行从头读。先问：

- 谁调用了 `get_forward_backward_func`？
- 它根据什么条件返回不同的 schedule？
- `train_step` 里前后发生了什么？

把调用图画出来，再钻进某一个 `forward_backward_pipelining_without_interleaving`。

### 2. 固定一组「小世界」配置

建议脑内始终带着：

- `world_size = 8`
- `TP=2, PP=2, CP=1, EP=1` → 则 `DP = 8 / (2*2*1) = 2`

所有 shape、通信、microbatch 讨论都先在这组数字上推演，再放大。

### 3. 一次只拧一个旋钮

- 学 TP：关掉 PP/VPP/MoE  
- 学 PP：先非 interleaving，再 VPP  
- 学 MoE：先固定 `alltoall` 或 `allgather` 一种 dispatcher  

### 4. 用测试当说明书

`tests/unit_tests/` 里按模块名搜索，往往比博客更接近当前代码行为。

### 5. 警惕「全局 process group」依赖

仓库贡献指南明确：在 `megatron/core` 生产代码中，**新逻辑应优先接收 `ProcessGroupCollection`（或显式 `ProcessGroup`）并向下传递**，而不是继续到处调用 `parallel_state.get_*_group()`。阅读时你会大量看到历史写法；写新代码时请按新约定走。

---

## 和官方文档怎么配合

| 你想搞清… | 先读文档 | 再读代码 |
|-----------|----------|----------|
| 安装与第一次跑通 | `docs/get-started/` | `examples/run_simple_mcore_train_loop.py` |
| 并行策略概念 | `docs/user-guide/parallelism-guide.md` | `parallel_state.py` |
| MoE | `docs/user-guide/features/moe.md` | `transformer/moe/` |
| Context Parallel | `docs/user-guide/features/context_parallel.md` | Attention CP 路径 |

文档负责「为什么」与「推荐用法」；源码负责「到底怎么实现」与「边界条件在哪」。

---

## 本系列目录预告

1. **开篇：代码地图与方法论**（本文）  
2. 从 `pretrain_gpt.py` 走进训练主循环  
3. `GPTModel`：Spec 驱动的 Transformer 积木  
4. `parallel_state`：五种并行的进程组拓扑  
5. 张量并行：Column / Row Parallel Linear  
6. 流水线并行：1F1B 与微批次调度  
7. 数据并行：DDP、梯度桶与 finalize  
8. 数据管线、DistributedOptimizer 与 Dist Checkpoint  
9. MoE：Router、Dispatcher 与专家并行  
10. 进阶：CP、推理、RL 与继续深挖的路线  

同目录还有完整阅读计划：`00-reading-plan.md`。

---

## 本周作业

1. 打开根 `README.md` 的 Project Structure，自己画一版「只含 10 个目录」的简图。  
2. 读完 `megatron/core/README.md`，用三句话区分 Core 与 LM。  
3. （有 GPU）跑通 `examples/run_simple_mcore_train_loop.py`，记下启动命令与日志里出现的并行信息。  

下一篇我们会从 `pretrain_gpt.py` 的 `if __name__ == "__main__"` 开始，把训练主循环拆开。
