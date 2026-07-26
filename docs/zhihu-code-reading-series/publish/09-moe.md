# 精读 Megatron 源码（9）：MoE——Router 决定去哪，Dispatcher 负责物流

> **专栏**：Megatron 源码精读 · 第 9 篇  
> **核心文件**：`transformer/moe/moe_layer.py`、`router.py`、`token_dispatcher.py`、`experts.py`  
> **官方文档**：`docs/user-guide/features/moe.md`

---

对 dense GPT，`TransformerLayer` 的 MLP 侧是普通 `MLP`。  
对 MoE，这一侧变成 `MoELayer`。

别被一堆 dispatcher 后端吓到。先记住一条前向链：

```text
hidden
  → TopKRouter          # 选 expert、算 gating / aux loss
  → Token Dispatcher    # 把 token 发到持有对应 expert 的卡
  → Local Experts       # TEGroupedMLP / SequentialMLP …
  → Dispatcher combine  # 还原顺序并加权
  → 输出 hidden
```

`BaseMoELayer` / `MoELayer` 把上述步骤编排在一起。建议先读官方 MoE 文档对齐术语，再进源码。

---

## 专家并行（EP）怎么切

在 `BaseMoELayer` 初始化逻辑中：

```text
num_local_experts = num_moe_experts // expert_model_parallel_size
local_expert_indices 由 ep_rank 决定
```

硬约束：**专家数必须能被 EP size 整除**。

Expert 上的线性层还可能叠加 **expert-TP**（与 dense TP 可以不同）。进程组来自 `parallel_state` 的 expert 侧 getter / `ProcessGroupCollection` 对应字段。

回想第 4 篇：MoE 使用单独的 RankGenerator 路径创建 EP / expert-DP——这是 dense 拓扑之外的第二套坐标。

---

## Router：Top-K 门控

文件：`router.py` 中的 `TopKRouter`

关注点：

- logits 如何得到  
- top-k 选择与概率归一  
- load balancing aux loss  
- token drop / expert capacity 相关策略  

Router 的梯度往往要在特定 process group 上归约——请对照 `finalize_model_grads` 里与 router 相关的逻辑。

---

## Token Dispatcher：MoE 的物流系统

接口：`MoETokenDispatcher`  
由 `moe_token_dispatcher_type` 选择实现：

| 类型 | 典型类 | 直觉 |
|------|--------|------|
| `allgather` | `MoEAllGatherTokenDispatcher` | 先汇聚再本地过滤 |
| `alltoall` | `MoEAlltoAllTokenDispatcher` | 按目的 expert 卡精确交换 |
| `flex` | `MoEFlexTokenDispatcher` | 对接 HybridEP / DeepEP / NCCL 等 |

精读顺序：

1. 基类：`token_dispatch` / `token_combine`（名字以源码为准）  
2. 选你环境默认的一种实现精读  
3. 对照 `moe_utils.permute` / `unpermute` 理解 token 重排  

**性能与正确性 bug 大量出在 dispatcher**，而不是 expert MLP 本身。

---

## Experts：本地专家计算

文件：`experts.py`

- `TEGroupedMLP`：把多个 local expert 的 GEMM 更高效地组织起来  
- `SequentialMLP`：逐 expert 计算，便于理解与兜底  

Expert 内部仍是「类 MLP」结构，只是 token 来自 dispatcher 聚拢后的缓冲区。

---

## 与训练栈的交叉点

1. **构建**：`num_experts` 触发 decoder block spec（见 `gpt_builder`）  
2. **并行初始化**：EP / expert-TP / expert-DP 须在模型构建前就绪  
3. **优化器**：expert 参数可能独立分组  
4. **梯度收尾**：router / balancing loss ↔ `finalize_model_grads`  
5. **Checkpoint**：expert 参数的 sharding 元数据必须正确  

---

## 建议精读顺序

```text
docs/user-guide/features/moe.md
  → gpt_builder 的 num_experts 分支
  → BaseMoELayer / MoELayer
  → TopKRouter
  → TokenDispatcher 基类 + 一种实现
  → TEGroupedMLP 或 SequentialMLP
  → finalize_model_grads 中 MoE 相关段落
```

---

## 常见坑

1. `num_moe_experts % ep_size != 0`  
2. dense DP 与 expert DP 假设不一致（尤其自定义 order）  
3. aux loss 未正确跨组平均 → 路由塌缩或数值不稳  
4. 换 dispatcher 后 capacity/drop 行为变了，指标突变  
5. 把 EP 当成 PP——一个切专家，一个切层，通信模式完全不同  

---

## 动手验证

1. 写出 `num_experts=64, EP=8` 时每卡 `num_local_experts`。  
2. 在 `MoELayer.forward` 里标注 router / dispatch / expert / combine 四段。  
3. 记录默认 `moe_token_dispatcher_type`，打开对应类浏览。  

下一篇是专栏收束：CP、推理、RL、精度与算子融合——以及如何把「读过一遍」变成长期能改框架的能力。
