# 知乎系列 10｜进阶路线：CP、推理、RL 与如何继续深挖

> 目标：在主路径（启动链 → 模型 → TP/PP/DP → 数据/优化器/ckpt → MoE）已经打通后，给你一张「下一站地图」，而不是再塞一篇同等厚度的专题。

---

## 你已经拥有的能力

如果前面 01–09 篇对应的源码你都跟过，那么你应当已经能：

1. 从 `pretrain_gpt.py` 指到 `train_step`  
2. 说明 Spec 如何换掉一层 Attention/MLP  
3. 手算一组 TP/PP/DP/CP 的组大小  
4. 指出 Column/Row、1F1B、DDP finalize、DistOpt、MoE dispatcher 的代码落点  

这已经超过「会跑脚本」的层次，进入「能改框架」的层次。

---

## 进阶专题 A：Context Parallel（CP）

**问题**：序列太长，激活与 attention 放不进单卡。  
**手段**：沿序列维切分，并在 attention 计算时交换必要的 KV/上下文信息。

推荐路径：

1. 文档：`docs/user-guide/features/context_parallel.md`  
2. 回顾：`parallel_state.initialize_model_parallel` 中 CP 组与 `dp-cp`  
3. 代码：`SelfAttention` / core attention 中与 CP 相关的分支；`get_batch_on_this_cp_rank`  
4. 动态 CP 等新特性：以 `dev` 分支与官方 blog 为准对照当前树  

**练习**：固定 `TP=1, PP=1, CP=2`，只观察 batch 的序列维如何被切开，以及通信出现在前向哪一步。

---

## 进阶专题 B：推理栈

目录：`megatron/core/inference/`（另有顶层 `megatron/inference/` 编排）

建议按 README 阅读：

- engines：生成循环如何驱动模型  
- contexts / KV：缓存与分页相关抽象  
- text_generation_controllers / server：对外服务  
- disaggregation / MoE inference：更前沿的拆分  

若你的目标是训练框架研发，推理可以「知其入口」；若目标是在线生成性能，则应把 inference 提升为与 PP schedule 同级的精读对象。

---

## 进阶专题 C：RL / RLHF

目录：`megatron/rl/`，示例：`examples/rl/`，入口脚本如 `train_rl.py`  
文档：`docs/user-guide/features/megatron_rl.md`

关注差异点：

- 与 `pretrain` 循环不同的数据/rollout 交互  
- 推理服务如何被训练进程调用  
- 策略模型与参考模型的并行配置是否共享  

建议在 dense SFT 路径非常熟之后再进入。

---

## 进阶专题 D：精度与算子融合

| 主题 | 入口 |
|------|------|
| FP8 | `megatron/core/fp8_utils.py`、TE 相关路径 |
| FP4 | `fp4_utils.py` |
| Fusion | `megatron/core/fusions/` |
| CUDA Graph | `full_cuda_graph.py`、`docs/user-guide/features/cuda_graph.md` |
| Activation recompute | `recompute.py`、TransformerConfig 相关字段 |

这些主题的共同特点是：**正确性与性能强绑定硬件/TE 版本**。阅读时务必记录 commit、CUDA、TE、PyTorch 版本。

---

## 进阶专题 E：模型多样性

| 方向 | 入口 |
|------|------|
| 多模态 | `examples/multimodal/`、`megatron/core/models/multimodal/`、`vision/` |
| Mamba / SSM | `pretrain_mamba.py`、`megatron/core/models/mamba/`、`ssm/` |
| Hybrid | `pretrain_hybrid.py`、`models/hybrid/` |
| 导出 | `megatron/core/export/`、`examples/export/` |

策略：先复用你对 `GPTModel` + Spec 的理解，再看各模型 builder 的差异，而不是从零重读训练循环。

---

## 进阶专题 F：弹性、Reshard、工程化

- `megatron/core/resharding/`：运行时重切分  
- `megatron/elastification/`：弹性相关  
- `megatron/core/distributed/fsdp/`：更激进的分片  
- `tests/` + `skills/mcore-testing/SKILL.md`：如何在本仓库跑测试  

若你要给 Megatron 贡献代码，请同时阅读：

- `docs/developer/contribute.md`  
- 根目录 `AGENTS.md`（含 ProcessGroupCollection 约定、PR draft、签名提交等）

---

## 把系列转化为长期习惯

### 1. 维护一张「个人调用图」

每遇到新旗标（如 `--xxx`），只更新图上的一个节点，避免重新从零探索。

### 2. 用单元测试当回归说明书

改 TP 通信？先找 `tests/unit_tests` 里 tensor parallel / schedules 相关用例。  
比「在 256 卡上试错」便宜得多。

### 3. 变更时遵守注入式进程组

新代码优先：

```text
外部创建 ProcessGroupCollection → 传入模块 / schedule
```

而不是：

```text
深层模块里 parallel_state.get_*_group()
```

### 4. 写博客/笔记时固定「配置三元组」

每篇笔记开头写清：

```text
world / TP / PP / DP / CP / EP
commit hash
是否 MoE / VPP / DistOpt
```

否则三个月后自己也看不懂当时的 shape 推导。

---

## 推荐的「第二轮」精读清单（任选 2 个做深）

1. `forward_backward_pipelining_with_interleaving` 全文 + 自己实现一个纸面 schedule simulator  
2. `MoEAlltoAllTokenDispatcher` + 一次 profiling（NCCL 时间线）  
3. `DistributedOptimizer.step` + 与 grad buffer 的字节级布局对照  
4. Dist checkpoint 换并行度加载的一次实操  
5. Inference engine 的 KV cache 路径  

---

## 系列回顾索引

| 篇 | 主题 | 核心文件 |
|----|------|----------|
| 01 | 地图与方法 | README、core/README |
| 02 | 启动链 | `pretrain_gpt.py`、`training.py` |
| 03 | 模型积木 | `gpt_model.py`、`transformer_layer.py`、specs |
| 04 | 进程组 | `parallel_state.py` |
| 05 | TP | `tensor_parallel/layers.py` |
| 06 | PP | `pipeline_parallel/schedules.py` |
| 07 | DP | `distributed_data_parallel.py`、`finalize_model_grads.py` |
| 08 | 数据/Optim/Ckpt | `gpt_dataset.py`、`distrib_optimizer.py`、`dist_checkpointing/` |
| 09 | MoE | `transformer/moe/` |
| 10 | 进阶 | CP / inference / RL / 精度… |

完整阅读计划见：`00-reading-plan.md`。

---

## 结束语

Megatron 源码的体量很大，但主路径并不神秘：**配置与进程组铺台子，Spec 拼模型，schedule 跑前后向，DDP/DistOpt 收梯度与状态，checkpoint 把分片世界固化下来**。  

把这条主路径读成「肌肉记忆」之后，MoE、CP、FP8、RL 都只是挂在同一骨架上的插件。  

如果你把本系列发布到知乎，建议每篇保留：

- 固定系列专栏名（如「Megatron 源码走读」）  
- 文首「上一篇 / 下一篇」链接  
- 文末「作业」与「对应源码路径」  
- 注明阅读的 commit / 版本，避免 API 漂移造成误导  

——系列完。欢迎从 `examples/run_simple_mcore_train_loop.py` 再跑一遍，看看此刻的日志是否比第一篇时「眼熟」许多。
