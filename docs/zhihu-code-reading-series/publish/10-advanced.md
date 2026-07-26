# 精读 Megatron 源码（10）：主路径读通之后，下一站去哪

> **专栏**：Megatron 源码精读 · 终篇  
> **定位**：不是再塞一篇同等厚度的专题，而是给你一张「第二轮深挖地图」

---

如果前面 1–9 篇对应的源码你都跟过，那你应当已经能：

1. 从 `pretrain_gpt.py` 指到 `train_step`  
2. 说明 Spec 如何换掉一层 Attention/MLP  
3. 手算一组 TP/PP/DP/CP 的组大小  
4. 指出 Column/Row、1F1B、DDP finalize、DistOpt、MoE dispatcher 的代码落点  

这已经超过「会跑脚本」，进入「能改框架」的层次。下面按优先级列出下一站。

---

## A. Context Parallel（CP）

**问题**：序列太长，激活与 attention 塞不进单卡。  
**手段**：沿序列维切分，并在 attention 时交换必要的 KV/上下文。

推荐路径：

1. `docs/user-guide/features/context_parallel.md`  
2. 回顾 `initialize_model_parallel` 中 CP 组与 `dp-cp`  
3. `SelfAttention` / core attention 中的 CP 分支；`get_batch_on_this_cp_rank`  

练习：固定 `TP=1, PP=1, CP=2`，只观察序列维如何切开、通信出现在前向哪一步。

---

## B. 推理栈

目录：`megatron/core/inference/`（另有顶层 `megatron/inference/`）

按 README 看：

- engines：生成循环如何驱动模型  
- contexts / KV：缓存抽象  
- server / controllers：对外服务  
- disaggregation / MoE inference：更前沿拆分  

目标若是训练框架研发，推理「知其入口」即可；目标若是在线生成性能，则应把 inference 提升到与 PP schedule 同级精读。

---

## C. RL / RLHF

- 代码：`megatron/rl/`、`examples/rl/`、`train_rl.py`  
- 文档：`docs/user-guide/features/megatron_rl.md`  

关注差异：rollout 交互、推理服务如何被训练进程调用、策略/参考模型的并行配置是否共享。建议在 dense SFT 路径非常熟之后再进。

---

## D. 精度与算子融合

| 主题 | 入口 |
|------|------|
| FP8 | `fp8_utils.py`、TE 路径 |
| FP4 | `fp4_utils.py` |
| Fusion | `megatron/core/fusions/` |
| CUDA Graph | `full_cuda_graph.py`、相关 docs |
| Recompute | `recompute.py`、TransformerConfig 字段 |

这类主题**正确性与性能强绑定硬件/TE 版本**。笔记务必记录 commit、CUDA、TE、PyTorch 版本。

---

## E. 模型多样性

| 方向 | 入口 |
|------|------|
| 多模态 | `examples/multimodal/`、`models/multimodal/` |
| Mamba / SSM | `pretrain_mamba.py`、`models/mamba/`、`ssm/` |
| Hybrid | `pretrain_hybrid.py`、`models/hybrid/` |
| 导出 | `core/export/`、`examples/export/` |

策略：复用你对 `GPTModel` + Spec 的理解，先看各 builder 的差异，而不是从零重读训练循环。

---

## F. 若你要给 Megatron 贡献代码

同时阅读：

- `docs/developer/contribute.md`  
- 根目录 `AGENTS.md`（含 `ProcessGroupCollection` 约定、draft PR、签名提交等）  

新代码优先：

```text
外部创建 ProcessGroupCollection → 传入模块 / schedule
```

而不是在深层继续 `parallel_state.get_*_group()`。

---

## 把系列变成长期习惯

1. **维护一张个人调用图**——每遇到新旗标只更新一个节点。  
2. **用单元测试当回归说明书**——改 TP 通信先找 `tests/unit_tests`。  
3. **笔记固定配置三元组**：`world/TP/PP/DP/CP/EP` + commit + 是否 MoE/VPP/DistOpt。  

### 推荐的「第二轮」精读（任选 2 个做深）

1. `forward_backward_pipelining_with_interleaving` 全文 + 纸面 schedule simulator  
2. `MoEAlltoAllTokenDispatcher` + 一次 NCCL 时间线 profiling  
3. `DistributedOptimizer.step` + grad buffer 字节级布局对照  
4. Dist checkpoint 换并行度加载实操  
5. Inference engine 的 KV cache 路径  

---

## 专栏回顾

| 篇 | 主题 | 核心落点 |
|----|------|----------|
| 1 | 地图与方法 | README / Core vs LM |
| 2 | 启动链 | `pretrain_gpt` → `train_step` |
| 3 | 模型积木 | `GPTModel` + Spec |
| 4 | 进程组 | `parallel_state` |
| 5 | TP | Column / Row Linear |
| 6 | PP | 1F1B schedules |
| 7 | DP | DDP + finalize |
| 8 | 数据/Optim/Ckpt | Dataset / DistOpt / ShardedTensor |
| 9 | MoE | Router / Dispatcher / EP |
| 10 | 进阶地图 | CP / 推理 / RL / 精度… |

---

## 结束语

Megatron 源码体量很大，但主路径并不神秘：

> **配置与进程组铺台子 → Spec 拼模型 → schedule 跑前后向 → DDP/DistOpt 收梯度与状态 → checkpoint 固化分片世界。**

把这条主路径读成肌肉记忆之后，MoE、CP、FP8、RL 都只是挂在同一骨架上的插件。

如果你把本专栏发到知乎，建议每篇保留：系列导航、对应源码路径、阅读的 commit/版本。API 会漂，注明版本是对读者的负责。

——专栏完。欢迎回到 `examples/run_simple_mcore_train_loop.py` 再跑一遍，看看此刻的日志是否比第一篇时「眼熟」许多。

感谢读到这里。若某篇写得不够清楚，欢迎在评论区指出具体文件与行号，我可以按文件继续开「单点精读」。
