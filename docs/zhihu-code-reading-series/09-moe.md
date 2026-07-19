# 知乎系列 09｜MoE：Router、Token Dispatcher 与专家并行

> 目标：沿 `MoELayer` 前向路径，搞清「谁决定 token 去哪、如何跨卡搬运、本地 expert 怎么算」。

---

## MoE 在 Megatron 中的位置

对 dense GPT，`TransformerLayer` 的 MLP 侧是普通 `MLP`。  
对 MoE，这一侧变成 `MoELayer`（或 block spec 中的 MoE 层组合）。

关键路径：

- `megatron/core/transformer/moe/moe_layer.py`  
- `megatron/core/transformer/moe/router.py`  
- `megatron/core/transformer/moe/token_dispatcher.py`  
- `megatron/core/transformer/moe/experts.py`  
- `megatron/core/transformer/moe/moe_utils.py`  
- 文档：`docs/user-guide/features/moe.md`  
- 构建：`gpt_builders.py` → `get_gpt_decoder_block_spec`

建议先读官方 MoE 文档建立术语，再进源码。

---

## 前向路径（记住这一条链）

```text
hidden states
  → TopKRouter（选 expert、算 gating / aux loss）
  → Token Dispatcher（把 token 发到持有对应 expert 的卡）
  → Local Experts（TEGroupedMLP / SequentialMLP / …）
  → Dispatcher combine（还原到原 token 顺序与加权）
  → 输出 hidden
```

`BaseMoELayer` / `MoELayer` 把上述步骤编排在一起。

---

## 专家并行（EP）如何切专家

在 `BaseMoELayer` 初始化逻辑中，核心关系是：

```text
num_local_experts = num_moe_experts // expert_model_parallel_size
local_expert_indices 由 ep_rank 决定
```

因此有硬约束：**专家数必须能被 EP size 整除**。  

Expert 上的线性层还可能叠加 **expert-TP**（与 dense TP 配置可以不同）——进程组来自 `parallel_state` 的 expert 侧 getter / `ProcessGroupCollection` 对应字段。

回到第 04 篇：MoE 使用单独的 `RankGenerator` 路径创建 EP / expert-DP 等组；这是 dense 拓扑之外的第二套坐标。

---

## Router：Top-K 门控

文件：`router.py` 中的 `TopKRouter`（及基类）

关注点：

- logits 如何得到（线性层 / 可选 bias）  
- top-k 选择与概率归一  
- load balancing aux loss、sequence-level / global 统计  
- 与 token drop、expert capacity 相关的策略开关  

Router 的梯度往往要在特定 process group 上归约——`finalize_model_grads` 里与 router 相关的逻辑要和这里对照着看。

---

## Token Dispatcher：MoE 的「物流系统」

接口层：`MoETokenDispatcher`  
常见实现由 `moe_token_dispatcher_type` 选择：

| 类型 | 典型类 | 直觉 |
|------|--------|------|
| `allgather` | `MoEAllGatherTokenDispatcher` | 先汇聚再本地过滤 |
| `alltoall` | `MoEAlltoAllTokenDispatcher` | 按目的 expert 卡精确交换 |
| `flex` | `MoEFlexTokenDispatcher` | 对接 HybridEP / DeepEP / NCCL 等后端 |

阅读顺序建议：

1. 基类方法：`token_dispatch` / `token_combine`（名字以源码为准）  
2. 选你环境默认的一种实现精读  
3. 对照 `permute` / `unpermute`（`moe_utils.py`）理解 token 重排  

**性能与正确性 bug 大量出在 dispatcher**，而不是 expert MLP 本身。

---

## Experts：本地专家计算

文件：`experts.py`

- `TEGroupedMLP`：把多个 local expert 的 GEMM 更高效地组织起来（依赖 Transformer Engine）  
- `SequentialMLP`：更直观的逐 expert 计算，便于理解与兜底  

Expert 内部仍然是「类 MLP」结构，只是批量 token 来自 dispatcher 聚拢后的缓冲区。

---

## 与训练栈的交叉点

1. **构建**：`num_experts` 触发 decoder block spec，而不是普通 layer spec。  
2. **并行初始化**：EP、expert-TP、expert-DP 组必须在模型构建前就绪。  
3. **优化器**：expert 参数可能独立分组（学习率、DistOpt 行为）。  
4. **梯度收尾**：router / balancing loss 与 `finalize_model_grads`。  
5. **Checkpoint**：expert 参数的 sharding 元数据必须正确。  

---

## 建议阅读顺序

```text
docs/user-guide/features/moe.md
  → gpt_builder 的 num_experts 分支
  → BaseMoELayer / MoELayer
  → TopKRouter
  → TokenDispatcher 基类 + 一种实现
  → TEGroupedMLP 或 SequentialMLP
  → moe_utils.permute/unpermute
  → finalize_model_grads 中 MoE 相关段落
```

---

## 常见坑

1. **`num_moe_experts % ep_size != 0`**  
2. **dense DP 与 expert DP 假设不一致**（尤其自定义 order 时）  
3. **aux loss 未正确跨组平均** → 路由塌缩或数值不稳  
4. **A2A 与 AllGather 的 capacity/drop 行为不同** → 换 dispatcher 后指标突变  
5. **把 EP 当成 PP** —— 一个切专家，一个切层，通信模式完全不同  

---

## 本周作业

1. 写出 `num_experts=64, EP=8` 时每卡 `num_local_experts`。  
2. 在 `MoELayer.forward`（或等价）里标注 router / dispatch / expert / combine 四段。  
3. 记录你代码树中默认 `moe_token_dispatcher_type` 是什么，并打开对应类浏览。  

下一篇（终篇）给出 CP、推理、RL 等进阶地图，以及如何把本系列转化为长期的源码阅读习惯。
