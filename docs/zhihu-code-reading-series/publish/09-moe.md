# 精读 Megatron 源码（9）：MoE 完全精读——Router、Dispatcher、专家并行与负载均衡

> **专栏**：Megatron 源码精读 · 第 9 篇  
> **核心文件**：  
> - `megatron/core/transformer/moe/moe_layer.py`  
> - `megatron/core/transformer/moe/router.py`  
> - `megatron/core/transformer/moe/token_dispatcher.py`  
> - `megatron/core/transformer/moe/experts.py`  
> - `megatron/core/transformer/moe/moe_utils.py`  
> - `megatron/core/transformer/moe/shared_experts.py`  
> - `megatron/core/models/gpt/gpt_layer_specs.py`  
> - `megatron/core/distributed/finalize_model_grads.py`  
>
> **上一篇**：[精读 Megatron 源码（8）：数据管线、DistOpt、Dist Checkpoint](./08-data-optim-ckpt.md)  
> **EP 概念专题（建议先读）**：[精读 Megatron 源码（11）：专家并行 EP 完全指南](./11-expert-parallel.md)  
> **下一篇**：[精读 Megatron 源码（10）：从读通主路径到能改框架](./10-advanced.md)

---

Mixture of Experts（MoE）是当前大模型最热门的架构之一：用稀疏激活替代密集 MLP，实现"参数量增大、计算量不增大"。但 MoE 的分布式实现极其复杂——不同于 TP/PP/DP，它引入了全新的 **Expert Parallelism（EP）**，以及专门的通信原语（AllToAll vs AllGather）。

若你对「EP 到底切什么、和 DP / expert_DP 什么关系」仍模糊，请先读独立专题 **[第 11 篇](./11-expert-parallel.md)**，再回本篇看 Router / Dispatcher / 负载均衡。本篇在 EP 拓扑之上，完整精读 Megatron 的 MoE 算法与实现。

---

## 一、MoE 的位置：什么时候替换 MLP

### 1.1 TransformerLayer 的 MLP 选择

GPT 的每个 Transformer layer 都有一个 MLP（前馈网络）。当配置了 `num_moe_experts > 1` 时，这个 MLP 可以被替换成 `MoELayer`。**"替换"的逻辑在 layer spec 里**：

```python
# megatron/core/models/gpt/gpt_layer_specs.py（精简）
def get_gpt_decoder_block_spec(config, ...):
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0
            for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq

    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)    # MoE 层
        else:
            layer_specs.append(dense_layer_spec)  # 普通 MLP 层
```

**`moe_layer_freq` 的两种用法**：

| 用法 | 示例 | 效果 |
|------|------|------|
| 整数 N | `moe_layer_freq=2` | 每隔 2 层放一个 MoE 层：[1,0,1,0,...] |
| 列表 | `moe_layer_freq=[1,0,1,0,1,0,1,0]` | 完全自定义每层是否为 MoE |

Mixtral-8x7B 风格：`[1,0,1,0,1,0,1,0]`（偶数层 MoE）；DeepSeek-V3/R1 风格：第 1 层是 dense，其余 61 层都是 MoE（`moe_layer_freq` 配置为对应列表）。

---

## 二、BaseMoELayer：专家分配的基础逻辑

### 2.1 EP 分组与本地专家索引

```python
# moe_layer.py  BaseMoELayer.__init__（精简）
class BaseMoELayer(MegatronModule, ABC):
    def __init__(self, config, layer_number=None, pg_collection=None, ...):
        self.ep_group = pg_collection.ep
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)

        assert self.config.num_moe_experts % ep_size == 0   # 必须整除
        self.num_local_experts = self.config.num_moe_experts // ep_size

        local_expert_indices_offset = ep_rank * self.num_local_experts
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap
```

**直觉**：64 个全局专家，EP=8，则每个 EP rank 持有 `64/8=8` 个本地专家。EP rank 3 持有专家 [24, 25, 26, 27, 28, 29, 30, 31]。

### 2.2 worked example：64 experts，EP=8，ep_rank=3

```
num_moe_experts     = 64
ep_size             = 8
num_local_experts   = 64 // 8 = 8
offset              = 3 * 8   = 24
local_expert_indices = [24, 25, 26, 27, 28, 29, 30, 31]
```

**为什么要 assert 整除？**  
若 `num_moe_experts % ep_size != 0`，则无法均匀分配，必然有些 rank 持有更多专家，负载不均衡且代码逻辑复杂。框架直接 assert 强制用户配置合法参数。

---

## 三、MoELayer.forward：真实代码结构标注

真实的 `MoELayer.forward` 远比简化版复杂，这里完整标注每个分支的意义：

```python
# moe_layer.py  MoELayer.forward（第 598-720 行，关键路径标注）
def forward(
    self,
    hidden_states: torch.Tensor,
    intermediate_tensors=None,
    padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    # ① 训练时若有 TP + 未启用 SP，发出性能警告
    if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
        raise ValueError("...")

    # ② 推理/训练切换 dispatcher（推理用特化的 NCCL/NVLS dispatcher）
    if hasattr(self, "_inference_token_dispatcher"):
        if InferenceMode.is_active():
            self.token_dispatcher = self._inference_token_dispatcher
        else:
            self.token_dispatcher = self._training_token_dispatcher

    # ③ padding_mask 从 [bsz, seq] 转置为 [seq, bsz] 对齐 hidden_states
    if padding_mask is not None:
        padding_mask = padding_mask.transpose(0, 1).bool()

    def custom_forward(hidden_states, intermediate_tensors=None, padding_mask=None):
        if "route" in self.fwd_execution_map:
            # ④ Shared Expert 先算（非 overlap 模式），因为它不需要路由信息
            shared_expert_output = self.shared_experts_compute(hidden_states)
            # ⑤ Router：计算路由概率和 routing_map
            probs, routing_map = self.route(hidden_states, padding_mask)
            # ⑥ Preprocess：把 hidden_states 排列成 dispatcher 期望的顺序
            hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)

        if "expert_compute" in self.fwd_execution_map:
            # ⑦ Dispatch：实际通信（AllToAll / AllGather / Flex）
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            # ⑧ 本地专家计算（GroupedGEMM / Sequential MLP）
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs)
            # ⑨ Combine：逆通信，把专家输出汇聚回原始 token 位置
            output = self.combine(output)

        if "postprocess" in self.fwd_execution_map:
            # ⑩ Postprocess：加 shared expert 输出（如有）；latent 投影（如有）
            output = self.postprocess(output, shared_expert_output)

        return output, mlp_bias  # mlp_bias 通常为 None

    # ⑪ 可选：梯度重计算（selective recompute "moe" 模块）
    if self.moe_layer_recompute and self.training:
        outputs = tensor_parallel.checkpoint(custom_forward, False, hidden_states, ...)
    else:
        outputs = custom_forward(hidden_states, intermediate_tensors, padding_mask)

    return outputs
```

**`fwd_execution_map` 的作用**：CUDA Graph 局部捕获（`cuda_graph_impl='local'`）时，`fwd_execution_map` 控制在当前调用中执行哪些步骤，使路由/通信/专家计算可以分段捕获为不同的 CUDA graph，实现更精细的性能优化。

**Shared Expert 的位置**：注意 `shared_experts_compute` 在 `route` **之前**调用。这是因为 shared expert 对**所有** token 都执行（不需要路由决策），可以与后续路由/通信步骤并行或提前执行，减少串行延迟。

---

## 四、TopKRouter：路由的完整逻辑

### 4.1 路由类型（routing_type）

`TopKRouter` 支持多种路由类型，通过 `moe_router_load_balancing_type` 配置：

| 路由类型 | 配置值 | 负载均衡方式 | 特点 |
|---------|--------|-------------|------|
| Switch 辅助损失 | `"aux_loss"` | 软约束（梯度反传） | 最常用，Mixtral/LLaMA MoE |
| 序列级辅助损失 | `"seq_aux_loss"` | 序列级统计 | 长序列分布更精确 |
| 全局辅助损失 | `"global_aux_loss"` | 跨梯度累积步统计 | 更平滑，延迟更新 |
| Sinkhorn | `"sinkhorn"` | 最优传输 | 理论上最优，计算稍慢 |
| 量化均衡 (QB) | `"quantile_balancing"` | per-expert bias 更新 | 无须辅助损失，DeepSeek-V3 风格 |
| None | `None` | 无均衡 | 仅用于调试 |

**多种 aux_loss 同时使用**：`routing_type` 可以是列表，`moe_aux_loss_coeff` 对应也是列表：

```bash
--moe-router-load-balancing-type aux_loss seq_aux_loss
--moe-aux-loss-coeff 0.01 0.005
```

### 4.2 score_function：影响路由权重计算

`score_function`（配置 `moe_router_score_function`）决定 logits 如何转为专家权重 `probs`：

| score_function | 计算方式 | 特点 |
|---------------|---------|------|
| `"softmax"` | `softmax(logits)` | 权重归一化，和为 1，最常用 |
| `"sigmoid"` | `sigmoid(logits)` 后归一化 | 各专家权重独立，和不必为 1 |
| `"sqrtsoftplus"` | `sqrt(softplus(logits))` 后归一化 | 更平滑的梯度，TE >= 2.13 |

代码实现（`moe_utils.py topk_routing_with_score_function`）：

```python
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # 先 softmax 再选 topk（pre-softmax 风格）
    else:
        # 先选 topk 再 softmax（post-softmax 风格，默认）
        ...
elif score_function in ("sigmoid", "sqrtsoftplus"):
    scores = torch.sigmoid(logits.float())   # 或 sqrt(softplus)
    scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    # 归一化使权重可加
```

**`moe_router_pre_softmax` 的区别**：pre-softmax 先对全部专家 logits 做 softmax 再选 topk，权重包含了未选中专家的"压力"；post-softmax（默认）先选 topk 再对这 topk 归一，语义更纯粹。

### 4.3 TopKRouter 完整 forward 调用链

```python
# router.py  TopKRouter.routing（精简）
def routing(self, logits):
    if self.routing_type == "sinkhorn":
        scores, routing_map = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "quantile_balancing":
        scores, routing_map = self.quantile_balancing(logits)
    else:  # "aux_loss" / "seq_aux_loss" / "global_aux_loss" / None
        scores, routing_map = topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            score_function=self.score_function,
            expert_bias=self.expert_bias,   # QB bias（可选）
            fused=self.config.moe_router_fusion,
        )
        scores = self._apply_aux_loss(scores, scores_for_aux, routing_map)

    return scores, routing_map

def forward(self, input):
    # 1. 线性变换：hidden_states → logits [T, E]
    logits = self.gating(input)
    # 2. 可选：输入 jitter（随机噪声增强探索）
    if self.input_jitter:
        logits = self._add_noise(logits)
    # 3. routing 选择专家
    probs, routing_map = self.routing(logits)
    return probs, routing_map
```

---

## 五、Switch 辅助损失与 Z-loss

### 5.1 Switch 辅助损失直觉理解

辅助损失来源于 Switch Transformer 论文，目的是惩罚"部分专家收到太多 token"的不均衡情况：

```
loss = E * Σ_{i=1}^{E} (f_i * P_i)

其中：
  E = num_experts（专家数）
  f_i = 1/(T*topk) * Σ_{x∈Batch} routing_map(x, i)
      # f_i 是路由到专家 i 的 token 比例（离散、不可微）
  P_i = 1/T * Σ_{x∈Batch} probs(x, i)
      # P_i 是路由概率的均值（连续、可微）
```

直觉：`f_i` 大（专家 i 实际收到很多 token）且 `P_i` 大（路由器倾向于发给这个专家），乘积就大，损失就高，从而**惩罚路由器偏向某些专家**。注意 `f_i` 不可微，但 `P_i` 可微，梯度通过 `P_i` 反传到路由器 weight。

实际代码（`moe_utils.py switch_load_balancing_loss_func`）：

```python
def switch_load_balancing_loss_func(
    probs, tokens_per_expert, total_num_tokens,
    topk, num_experts, moe_aux_loss_coeff, ...
):
    # tokens_per_expert[i] = routing_map[:, i].sum()（已跨 TP/CP reduce）
    # probs 是 [T, E] 的 scores（topk 位置非零）
    aggregated_probs = probs.sum(dim=0)   # [E]：每个专家的 probs 之和
    scale = num_experts * moe_aux_loss_coeff / (topk * total_num_tokens ** 2)
    aux_loss = torch.dot(aggregated_probs, tokens_per_expert.float()) * scale
    return aux_loss
```

### 5.2 Z-loss：防止路由器 logits 过大

Z-loss 来自 ST-MoE 论文，防止路由器输出 logits 爆炸：

```python
# moe_utils.py  z_loss_func
def z_loss_func(logits, z_loss_coeff, padding_mask=None):
    """Encourages the router's logits to remain small to enhance stability.
    loss = z_loss_coeff * mean(log(sum(exp(logits)))^2)
    """
    z_loss = torch.mean(torch.log(torch.sum(torch.exp(logits), dim=1)) ** 2)
    return z_loss * z_loss_coeff
```

Z-loss 对每个 token 的 logsumexp 平方惩罚，使路由器不会输出极端大的 logits（导致某些专家 softmax 概率趋近 1.0，等价退化为非稀疏模型）。

**配置**：

```bash
--moe-z-loss-coeff 1e-3   # 典型值 1e-3 到 1e-4
```

### 5.3 MoEAuxLossAutoScaler：如何把辅助损失插入反向传播

```python
# moe_utils.py
class MoEAuxLossAutoScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, aux_loss):
        ctx.save_for_backward(aux_loss)
        return output   # forward 直接返回 output，不做任何变换

    @staticmethod
    def backward(ctx, grad_output):
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = ...
        scaled_aux_loss_grad = torch.ones_like(grad_output) * aux_loss_backward_scale
        return grad_output + scaled_aux_loss_grad, None
```

这是一个**"梯度注入"技巧**：`forward` 等价恒等变换，`backward` 把 aux_loss 的梯度注入到 `probs` 的梯度流中，不需要显式调用 `aux_loss.backward()`。这使得辅助损失对模型主 loss 的梯度影响可控（通过 `moe_aux_loss_coeff` 系数）。

---

## 六、Shared Expert（共享专家）路径

`moe_shared_expert_intermediate_size` 不为 None 时，`BaseMoELayer.__init__` 初始化 `use_shared_expert = True`，并建立 `self.shared_experts = SharedExpertMLP(...)`。

**共享专家 vs 路由专家对比**：

| 特性 | 路由专家（Routed Experts） | 共享专家（Shared Expert） |
|------|--------------------------|--------------------------|
| token 覆盖率 | topk/num_experts（稀疏） | 100%（所有 token） |
| EP 分布 | 各 rank 持有不同专家 | 每个 rank 都有完整副本 |
| 梯度同步 | EP 内部 ReduceScatter | 普通 DP AllReduce（全量） |
| 参数形状 | `[num_local_experts, ffn_size, hidden]` | `[ffn_size, hidden]`（单 MLP） |
| 代表模型 | Mixtral（无 shared） | DeepSeek-V2/V3（有 shared） |

**`shared_expert_overlap`**：若开启，shared expert 的计算**与通信重叠**。具体地，`shared_experts_compute` 在 route/dispatch 之前被调用，或者在推理模式下通过独立 CUDA stream 与 AllGather-V（NVLS）并行执行，减少端到端延迟。

**forward 中的位置**（见第三节代码标注 ④）：

```python
# 非 overlap 模式
shared_expert_output = self.shared_experts_compute(hidden_states)  # 先算
probs, routing_map = self.route(hidden_states, padding_mask)        # 再路由
...
output = self.postprocess(output, shared_expert_output)             # 最后加回来
```

---

## 七、Token Dispatcher：三种通信模式的选择

```
文件: megatron/core/transformer/moe/token_dispatcher.py
```

| 类名 | 配置 `moe_token_dispatcher_type` | 通信原语 | 适用场景 |
|------|----------------------------------|----------|---------|
| `MoEAllGatherTokenDispatcher` | `"allgather"` | AllGather（TP*EP 域） | 小 EP，token 数少 |
| `MoEAlltoAllTokenDispatcher` | `"alltoall"` | AllToAll（EP 域） | 大 EP，稀疏路由 |
| `MoEFlexTokenDispatcher` | `"flex"` | deepep/hybridep/ncclep | 超大规模 |

### 7.1 AllGather vs AllToAll 带宽对比

统一用「每 rank、单向、元素个数」比较；hidden 维为 `H`，总 token 数 `T`（均分到 `EP` 张卡，每卡 `T/EP`），每个 token 选 `topk` 个专家。

| 指标 | AllGather | AllToAll（dispatch） |
|------|-----------|----------------------|
| 每 rank 通信量 | `T × (EP-1)/EP × H` | `(T/EP) × topk × (EP-1)/EP × H` |
| 相对比例 | 1 | `topk / EP` |
| 形状是否固定 | ✅ 固定 | ⚠️ 动态（每 rank 发送量随路由变） |
| CUDA Graph 兼容 | ✅ 更容易 | ⚠️ 常需 capacity / pad 固定 shape |
| 实现复杂度 | 低 | 高（需要 `input/output_splits`） |
| 推荐 EP | 较小 EP | 较大 EP、稀疏路由 |

**数值推导**（`T` tokens，`hidden=H`，`topk=2`，`EP=4`，专家均匀分布）：

```text
AllGather（每 rank 接收）：
  本地已有 T/4 个 token，还需从其他卡收 3T/4 个
  元素数 = 3T/4 × H

AllToAll dispatch（每 rank 发送）：
  本地 T/4 个 token，每个复制 topk=2 份 → 共 T/2 个 token-expert 副本
  其中发往其他 3 个 EP rank 的比例约 (EP-1)/EP = 3/4
  元素数 = (T/4) × 2 × (3/4) × H = 3T/8 × H

比值：AllToAll / AllGather = (3T/8 × H) / (3T/4 × H) = topk/EP = 2/4 = 1/2
若 EP=16、topk=2：比值 = 2/16 = 1/8
```

Combine 阶段还有一次反向 AllToAll（量级同 dispatch）；AllGather 路径对应还有 ReduceScatter。上表先比「聚齐 token」这一侧，便于看清稀疏路由如何压通信量。

### 7.2 AllToAll Dispatcher step-by-step：T tokens，topk=2，EP=4 的完整推演

**初始配置**：
- 16 个 token（每 rank 4 个，T_local=4）
- 16 个专家（每 rank 4 个）
- topk=2（每 token 路由 2 个专家）
- EP=4（rank 0..3）

**阶段 0：路由决策（本地，无通信）**

```
rank0 tokens: [t0, t1, t2, t3]
路由结果：
  t0 → 专家 E1(rank0), E7(rank1)     # E0-E3 在 rank0, E4-E7 在 rank1
  t1 → 专家 E5(rank1), E12(rank3)
  t2 → 专家 E2(rank0), E9(rank2)
  t3 → 专家 E11(rank2), E15(rank3)
```

**阶段 1：计算 input_splits（本 rank 发给每个 rank 的 token-expert 对数量）**

```
rank0 需要发给 rank0（本地）：t0→E1, t2→E2  → 2 对
rank0 需要发给 rank1：t0→E7, t1→E5         → 2 对
rank0 需要发给 rank2：t2→E9, t3→E11        → 2 对
rank0 需要发给 rank3：t1→E12, t3→E15       → 2 对
input_splits = [2, 2, 2, 2]  # 共 8 对 = topk × T_local
```

**阶段 2：AllToAll 通信**

每个 rank 把"要发给 rank_i 的 token 表示"发出去，同时接收来自其他 rank 的 token。通信结果：

```
rank0 收到（output_splits）：
  来自 rank0：t0→E1, t2→E2（本地，不移动）
  来自 rank1 的 tokens 路由到 rank0 专家（E0-E3）的部分
  ...（每 rank 各异，取决于路由结果）
```

**阶段 3：本地专家计算**

```
rank0 的 E0 处理收到的所有分配给它的 tokens
rank0 的 E1 处理 t0（来自 rank0 本地）
rank0 的 E2 处理 t2（来自 rank0 本地）
rank0 的 E3 处理其他 rank 发来的 tokens
```

**阶段 4：逆 AllToAll（Combine 阶段）**

专家计算完成后，结果逆向通过 AllToAll 发回 token 所在的原始 rank，按 `probs` 加权求和（weighted combine）。

**数值总结（T=16, topk=2, EP=4，按每 rank 元素数）**：

```text
AllGather（单向）：     3T/4 × H = 12H
AllToAll dispatch：     3T/8 × H = 6H
AllToAll combine：      约再 6H
AllToAll 来回合计：     约 12H
```

同一套假设下，dispatch 单侧约为 AllGather 的 `topk/EP = 1/2`；来回两次后与「一次 AllGather」同量级。EP 更大、topk 仍小时（如 EP=16、topk=2），单侧比值降到 `1/8`，AllToAll 才明显更省。

### 7.3 AllGather Dispatcher

```
思路：把所有 rank 的 hidden_states 全部 AllGather 到本 rank，
然后本 rank 的专家从完整的 token 集合中挑选属于自己的 token 计算。
```

通信量：`T × H × (TP·EP - 1)` 个元素。

**优点**：实现简单，token 无需跨卡移动（本地选择），没有动态不平衡的通信量问题。  
**缺点**：通信量随 TP·EP 线性增长，大 EP 时通信成为瓶颈。

### 7.4 Flex Dispatcher：三种后端

`MoEFlexTokenDispatcher` 是一个统一前端，根据 `moe_flex_dispatcher_backend` 选择后端：

| 后端 | 说明 | 特点 |
|------|------|------|
| `deepep` | DeepSeek DeepEP 库 | 专门为 MoE EP 优化的 NCCL 扩展，低延迟 |
| `hybridep` | HybridEP（Megatron 内置）| 把排列与 AllToAll 融合，减少 kernel 启动开销 |
| `ncclep` | NCCL EP（Megatron 内置，需 SM100+ Blackwell 或支持 symm-mem）| 利用对称内存零拷贝 |

选择建议：
- 调试/中等规模：`alltoall`
- 大规模生产：根据硬件选 `flex` + `deepep` 或 `ncclep`
- 通信量分析实验：先跑 `allgather` 作基准，再对比 `alltoall`

### 7.5 Dispatcher 对比总结

| 对比维度 | AllGather | AllToAll | Flex (ncclep) |
|---------|-----------|---------|---------------|
| 通信量 | O(T·H·EP) | O(T·H·topk/num_experts) | O(T·H·topk/num_experts) |
| shape 是否静态 | ✅ | ⚠️（需 capacity） | ✅（symm-mem） |
| CUDA Graph 友好 | ✅ | ⚠️ | ✅ |
| 实现复杂度 | 低 | 中 | 高 |
| 推荐硬件 | 任意 | A100/H100 | Blackwell (SM100+) |
| 通信与计算重叠 | 部分 | 支持（deepep 后端） | 最优（零拷贝） |
| 最大 EP | ~8 | ~64 | ~512 |

---

## 八、专家计算：TEGroupedMLP vs SequentialMLP

```
文件: megatron/core/transformer/moe/experts.py
```

### 8.1 TEGroupedMLP（推荐，依赖 TransformerEngine）

```python
class TEGroupedMLP(MegatronModule):
    """使用 TransformerEngine 的 GroupedGEMM 批量计算多个专家的 MLP"""
```

`GroupedGEMM` 把 N 个专家的权重矩阵堆叠在一起，用单次 CUDA kernel 完成所有专家的 GEMM，显著减少 kernel 启动开销和内存访问碎片化。

**适用条件**：
- 安装了 TransformerEngine（`from transformer_engine.pytorch import ...`）
- `moe_grouped_gemm=True`

**sharded_state_dict 与 SequentialMLP 互换**：`TEGroupedMLP` 的 checkpoint key 格式与 `SequentialMLP` 完全兼容，因此两者训练的 checkpoint 可互相加载。

### 8.2 SequentialMLP（兼容性好）

```python
for expert_idx, expert in enumerate(self.local_experts):
    expert_input = dispatched_input[expert_start:expert_end]
    expert_output = expert(expert_input)
```

**适用场景**：不依赖 TE，适合调试和不支持 GroupedGEMM 的环境。性能低于 TEGroupedMLP，但输出语义完全相同。

### 8.3 FP8 MoE 的注意事项

当启用 FP8（`--fp8-format hybrid`）时，MoE 的专家计算走 `te_checkpoint` 而非普通 `tensor_parallel.checkpoint`：

```python
# moe_layer.py forward（FP8 分支）
if self.moe_layer_recompute and self.training:
    if self.config.fp8 or self.config.fp4:
        outputs = te_checkpoint(
            custom_forward, False,
            tensor_parallel.random.get_cuda_rng_tracker,
            self.tp_group, hidden_states, ...
        )
```

**已知注意事项**：
- `TEGroupedMLP` 的 FP8 支持依赖 TransformerEngine 版本（建议 TE >= 1.7）。
- 共享专家 (`shared_experts_recompute=True`) 同样需要走 `te_checkpoint`。
- FP8 与 CUDA graph 在 MoE 场景下有额外约束：`moe_ncclep_static_shape=True` 才能保证 static shape。
- AllToAll dispatcher 的 FP8 支持：token 在传输前/后需要反量化/量化，通信的仍是 BF16，仅专家内 GEMM 使用 FP8，不影响通信语义。

---

## 九、Expert Parallel 的并行维度

拓扑与公式以第 11 篇为准。下面只补 MoE 层里会碰到的三个旋钮。

### 9.1 EP 与其他并行维度的关系

```
全局 rank 布局（TP=2, EP=4, DP=2 为例，共 16 GPUs）：

         TP group (2 GPUs)
     ┌──────────────────────────┐
     │  rank0  rank1            │ ← EP rank 0，DP rank 0
     │  rank2  rank3            │ ← EP rank 1，DP rank 0
     │  rank4  rank5            │ ← EP rank 2，DP rank 0
     │  rank6  rank7            │ ← EP rank 3，DP rank 0
     └──────────────────────────┘
     ┌──────────────────────────┐
     │  rank8  rank9            │ ← EP rank 0，DP rank 1
     ...（DP 副本）
     └──────────────────────────┘
```

EP group 跨越不同的 TP 副本：rank 0、2、4、6 组成一个 EP group。

### 9.2 Expert-TP（专家内的 TP 切分）

对于每个专家自身，其 weight matrix 也可以被 TP 切分：

```bash
--expert-tensor-parallel-size 2   # 每个专家内部再 TP 切 2 份
```

在 `optimizer/__init__.py` 中，带 `expert_tp=True` 属性的参数会被放入专门的 param_group。

### 9.3 Expert DP（专家的数据并行）

EP ranks 之间实际上形成了专家级别的 DP：同一 EP rank 在不同 DP 副本中处理不同 batch，需要聚合梯度。这部分由 `finalize_model_grads` 中的 `_allreduce_router_grads` 处理：

```python
# finalize_model_grads.py
def _allreduce_router_grads(model, config):
    """All-reduce router grads（路由器参数的梯度需要在整个 DP 域 AllReduce）"""
    for module in model:
        for param in module.parameters():
            if getattr(param, 'allreduce', True) and param.grad is not None:
                ...
```

**路由器梯度为什么要单独处理？**  
路由器（TopKRouter）的 weight 存在于所有 EP rank 上（每个 rank 都有完整的 router），但 expert 参数只在特定 EP rank 上。`finalize_model_grads` 在 DDP 完成 expert 梯度同步之后，**额外做一次 router gradient AllReduce**，确保路由器参数更新的一致性。

---

## 十、容量与 token dropping

### 10.1 capacity_factor 的作用

```bash
--moe-expert-capacity-factor 1.2  # 每个专家最多接受 capacity 个 token
```

`capacity = capacity_factor × (total_tokens / num_experts / topk)`

当路由到某专家的 token 数超过 capacity 时，多余的 token 被 **drop**（不经过该专家，以零向量代替输出）。

**好处**：保证 AllToAll 通信大小固定（静态 shape），方便 CUDA graph 捕获。  
**坏处**：dropped token 的 loss 不正确（梯度消失），极端负载不均时质量下降。

---

## 十一、负载均衡监控：什么指标需要关注

训练 MoE 模型时，以下指标需要持续监控：

| 指标 | 计算方式 | 健康范围 | 异常原因 |
|------|---------|---------|---------|
| `tokens_per_expert` 方差 | `std(tokens_per_expert) / mean` | < 0.3 | 路由偏向某几个专家 |
| `tokens_per_expert` 最大值 | `max(tokens_per_expert) / mean` | < 2.0 | 单专家收到过多 token |
| `aux_loss / main_loss` | 辅助损失占总损失比例 | 1%~5% | > 10% 表示系数过大 |
| token dropping rate | `dropped / total × 100%` | < 3% | 负载极不均衡 + capacity_factor 过小 |
| expert 激活频率分布 | 每步各专家被激活次数 | 接近均匀 | 偏斜表示负载不均衡 |
| router weight 梯度范数 | AllReduce 后的 grad norm | 稳定 | 突然飙升 = router loss 过大 |

**如何在代码中添加监控**（`MoELayer.forward` 中 route 之后）：

```python
# 在 self.route() 之后添加（或在 TopKRouter.forward 末尾）
with torch.no_grad():
    tokens_per_expert = routing_map.float().sum(dim=0)  # [E]
    # 跨 EP AllReduce 汇总
    torch.distributed.all_reduce(tokens_per_expert, group=self.ep_group)
    # 记录到 TensorBoard/wandb
    if mpu.is_pipeline_last_stage():
        moe_metrics_tracker.log("tokens_per_expert", tokens_per_expert)
```

Megatron 已内置 `get_moe_metrics_tracker()`（`moe_logging.py`），可以订阅各层的负载均衡 metric，在 training loop 定期打印。

**训练过程中负载均衡的典型演化曲线**：

```
Step 0-500：
  tokens_per_expert 方差很大（某 1-2 个专家收到 60-80% 的 token）
  → 路由器还未收到足够梯度信号

Step 500-2000：
  方差逐渐降低（负载均衡辅助损失开始生效）
  → 各专家 token 分布趋于均匀

Step 2000+：
  方差稳定在较低水平（通常 std/mean < 0.3）
  → 达到稳定的均衡状态

异常情况：
  方差在训练后期再次上升 → aux_loss 系数过小，或学习率太高
  方差始终为 0（完全均匀）→ aux_loss 系数过大，路由器失去语义区分能力
```

---

## 十二、完整数据流 worked example

以一个简化场景完整演示 MoE layer 的前向过程：

**配置**：`num_moe_experts=4, ep_size=2, topk=1, hidden_size=4, batch_tokens=4`

```
初始 hidden_states（rank0 持有 token T0,T1；rank1 持有 T2,T3）：
  rank0: [[0.1, 0.2, 0.3, 0.4],   # T0
           [0.5, 0.6, 0.7, 0.8]]  # T1
  rank1: [[0.9, 1.0, 1.1, 1.2],   # T2
           [1.3, 1.4, 1.5, 1.6]]  # T3

route 输出（topk=1）：
  rank0: T0→E2, T1→E0
  rank1: T2→E3, T3→E1

local_expert_indices：
  rank0: [E0, E1]（ep_rank=0）
  rank1: [E2, E3]（ep_rank=1）

AllToAll 通信（token_dispatcher.token_dispatch）：
  rank0 需要发给 rank1：T0（要去 E2，在 rank1）
  rank0 保留：T1（要去 E0，在 rank0）
  rank1 需要发给 rank0：T3（要去 E1，在 rank0）
  rank1 保留：T2（要去 E3，在 rank1）

通信后各 rank 持有的 token：
  rank0: T1（给E0）, T3（给E1）
  rank1: T0（给E2）, T2（给E3）

专家计算：
  rank0: E0(T1) → output_T1, E1(T3) → output_T3
  rank1: E2(T0) → output_T0, E3(T2) → output_T2

逆 AllToAll（combine 阶段）：
  rank0: 取回 output_T0（从rank1），保留 output_T1 → rank0 输出 [output_T0, output_T1]
  rank1: 取回 output_T3（从rank0），保留 output_T2 → rank1 输出 [output_T2, output_T3]

probs（topk=1 时为 1.0）× output，weighted combine 完成。
```

---

## 十三、finalize_model_grads 的 MoE 特殊处理

```python
# finalize_model_grads.py（精简）
def finalize_model_grads(model, config):
    # 1. PP pipeline 梯度聚合（第 6 篇已讲）
    _allreduce_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)
    # 2. MoE 专属：router 参数梯度 AllReduce
    _allreduce_router_grads(model, config)
    # 3. 其余未参与 ReduceScatter 的参数（如 biases）AllReduce
    _allreduce_layernorm_grads(model, config)
```

关键点：router weight 不是 expert 参数（没有 `allreduce=False`），因此它在 DDP 里走普通 AllReduce 路径。但如果 EP > 1，router 参数的梯度可能只在 TP group 内汇聚了，还没有在跨 EP 的 DP group 上汇聚。`_allreduce_router_grads` 确保这一步完成。

---

## 十四、端到端 CLI 配置示例：最小 MoE 训练

以下是一个能在 8 GPU（TP=2, EP=4, DP=1）上运行的最小 MoE GPT 配置：

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
  # 模型结构
  --num-layers 8 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  # MoE 配置
  --num-moe-experts 8 \
  --moe-router-topk 2 \
  --moe-layer-freq 2 \           # 每 2 层放一个 MoE 层（4 个 MoE 层）
  --moe-token-dispatcher-type alltoall \
  --moe-aux-loss-coeff 0.01 \
  --moe-router-load-balancing-type aux_loss \
  # 并行配置
  --tensor-model-parallel-size 2 \
  --expert-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  # 数据
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --mock-data \
  # 精度
  --bf16 \
  # 优化器
  --use-distributed-optimizer \
  --lr 1e-4 \
  --train-iters 1000
```

**验证 MoE 是否正常工作的检查点**：
1. `tokens_per_expert` 各专家差异是否在 3 倍以内（初期）
2. `aux_loss` 是否约为主 loss 的 1%（`moe_aux_loss_coeff=0.01`）
3. 每步迭代速度是否稳定（AllToAll 不应有突变延迟）
4. 8 个 GPU 的内存使用是否均衡（EP 专家均匀分布）

---

## 十五、MoE 常见失败模式速查表

| 现象 | 最可能原因 | 排查方法 |
|------|------------|---------|
| 训练初期 loss 不降 | aux_loss 系数太大，淹没主 loss | 降低 `moe_aux_loss_coeff` 到 0.01 以下 |
| 某些专家从不被激活 | 路由器初始化导致偏向，负载均衡未生效 | 观察 `tokens_per_expert` 曲线，提高 aux_loss 或换 QB 路由 |
| AllToAll 通信 hang | EP 某 rank 通信超时 | 检查网络连接；增大 NCCL 超时；检查 `input_splits` 总和一致性 |
| GPU OOM 在 MoE 层 | Expert 激活内存 + 通信 buffer 超出显存 | 减小 `capacity_factor`；开启 `moe_layer_recompute`；减小 `micro_batch_size` |
| token dropping 率高 | `capacity_factor=1.0` + 路由极不均衡 | 提高 `capacity_factor` 到 1.5 或开启辅助损失 |
| EP 不整除报 assert | `num_moe_experts % ep_size != 0` | 调整 `num_moe_experts` 为 EP 的倍数 |
| checkpoint 中 expert 参数 shape 不对 | 修改了 `num_moe_experts` 后加载旧 checkpoint | expert weight key 包含专家数，不可直接 reshard |
| Shared expert 输出消失 | `moe_shared_expert_intermediate_size` 未设置 | 确认配置中设置了该参数 |
| FP8 + MoE + CUDA graph 不兼容 | AllToAll 动态 shape 无法捕获 | 启用 `moe_ncclep_static_shape=True` 或禁用 CUDA graph |
| Router 梯度为 None | expert 参数标记为 `allreduce=False` 污染了 router | 检查 router weight 的 `allreduce` 属性是否为 True |

---

## 十六、MoE 调试检查清单

```
路由相关：
  □ TopKRouter weight 的 shape 是 (num_experts, hidden_size)？
  □ routing_map.sum(dim=1) 是否等于 topk（每个 token 恰好路由 topk 个专家）？
  □ aux_loss 的量级是否合理（约为主 loss 的 1%~5%）？
  □ tokens_per_expert 的方差是否在训练中下降（负载均衡是否生效）？
  □ score_function 是否与 moe_router_pre_softmax 搭配合理？

通信相关：
  □ AllToAll 的 input_splits 之和 == output_splits 之和？
  □ ep_size 是否整除 num_moe_experts？
  □ token dropping 率是否 < 5%？
  □ 使用 FP8 时，CUDA graph 是否需要 static shape（ncclep 后端）？

专家相关：
  □ local_expert_indices 与 ep_rank 的对应关系是否正确？
  □ TEGroupedMLP 的 weight shape 是 (num_local_experts, ffn_hidden_size, hidden_size)？
  □ expert 梯度是否只在 ep_group 内聚合（而非全 DP group）？
  □ shared_expert_overlap 是否与推理 dispatcher 类型兼容？

性能相关：
  □ dispatcher 选择是否与硬件匹配（Blackwell → ncclep）？
  □ overlap_dispatch_backward_with_experts_wgrad 是否开启？
  □ GroupedGEMM 是否生效（检查 TE 版本 >= 1.7）？
```

---

## 十六、十二道 FAQ

**Q1：`num_moe_experts=16, topk=2` 时，每个 token 的计算量与 dense MLP 相比如何？**  
A：token 实际经过 2 个专家（topk=2），每个专家的 FFN 大小通常与 dense 相同。因此 FLOP 是 dense MLP 的 `topk/num_experts = 2/16 = 12.5%`。模型参数量是 dense 的 `(topk-1) * num_experts * ffn_size`，但计算量不变。这就是 MoE 的核心价值：参数量增大但 FLOP 不增大。

**Q2：`moe_layer_freq=2` 和 `moe_layer_freq=[1,0,1,0,...]` 效果一样吗？**  
A：不完全一样。`moe_layer_freq=2` 从层 0 开始，偶数层（0,2,4,...）是 MoE。`moe_layer_freq=[0,1,0,1,...]` 则是奇数层是 MoE。模式相同但起始层不同，对模型质量影响微小，但 checkpoint key 完全不同（一个模型的 0 层是 dense，另一个是 MoE）。

**Q3：Router weight 为什么是 FP32 而不是 BF16？**  
A：`Router.__init__` 中：`torch.empty(..., dtype=torch.float32)`，`reset_parameters` 中：`self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)`。初始化为 FP32，然后立即转为 `params_dtype`。但 Expert Bias（`expert_bias`）和 QB bias（`qb_beta`）保持 FP32，因为它们的更新需要高精度（routing 决策对数值极其敏感）。

**Q4：EP rank 的专家索引（`local_expert_indices`）和 global 专家 ID 的映射关系是什么？**  
A：固定映射：`local_expert_indices = [ep_rank * num_local + i for i in range(num_local)]`。EP rank 0 持有 [0..K-1]，EP rank 1 持有 [K..2K-1]，以此类推。这个映射在 token dispatch 中被用来过滤"哪些 token 属于本地专家"。

**Q5：`shared_expert_overlap` 在训练中如何实现通信与计算重叠？**  
A：训练模式下，`shared_experts_compute` 先于 `route` 和 `dispatch` 执行（第三节代码 ④）。如果 dispatching 是耗时的（AllToAll 通信），shared expert 的计算可以在通信等待期间完成。这是一种软重叠，不依赖 CUDA stream，适用于所有 dispatcher 类型。

**Q6：`moe_aux_loss_coeff=0.01` 和 `0.1` 有什么区别？**  
A：系数越大，负载均衡效果越强，但会过度干扰主 loss 梯度方向。过大（如 `0.1`）会导致 router 优化主要由 aux_loss 主导，忽略语言模型目标，表现为 perplexity 高但负载极均衡。典型值：`0.001`~`0.02`，需要根据模型规模和 expert 数量调整。

**Q7：Quantile Balancing (QB) 和 aux_loss 在路由上有什么本质区别？**  
A：aux_loss 通过梯度软约束 router weight，无法保证任意时刻的均衡；QB 通过直接更新 `expert_bias`（不经过梯度）强制均衡，类似"per-expert 价格调节"。QB 的均衡性更确定，但不兼容 aux_loss（两者不能同时使用）。

**Q8：AllToAll 中如果某个 EP rank 挂了，整个训练会怎样？**  
A：`torch.distributed.all_to_all` 是同步集合通信，某个 rank 挂了会导致所有 rank 永久等待（hang）。生产环境通常依赖 NCCL 超时检测 + 重启机制，或者使用支持 rank failure detection 的调度框架（如 Kubernetes + Fault Tolerant Training）。

**Q9：`tokens_per_expert` 的统计范围是全局还是本地？**  
A：取决于实现。`_apply_aux_loss` 中通过 `get_tokens_per_expert_and_token_count(reduce_group=self.tp_cp_group)` 对 TP+CP 组内 reduce，得到"这个 EP rank 视野内的 token 统计"。若要全局 EP 视野，需要额外跨 EP AllReduce，`tokens_per_expert` 监控 log 时通常已做此操作。

**Q10：DeepSeek-V3 风格的 QB 路由在代码中是否有实现？**  
A：有。`TopKRouter.quantile_balancing` 方法实现了 QB routing，`qb_dual_update`（`moe_utils.py`）实现了双坐标下降更新 `qb_beta`。在代码注释中可见"QB only picks the experts; reuse the shared score function for the probs"，与 DeepSeek-V3 论文描述的 auxiliary-loss-free 负载均衡思路一致。代码中的 `qb_beta` 对应论文的 bias 项。

**Q11：MoE 中的 Sequence Parallel（SP）如何与 EP 协作？**  
A：SP 把序列维度切分到 TP group，每个 TP rank 持有 `seq/TP` 的激活。进入 MoE layer 时，router 看到的是切分后的局部 token，`tp_cp_group` 上的 reduce 确保 aux_loss 统计全局 token 分布。SP 的激活节省（LN/Dropout）与 EP 的专家分布相互正交，可以同时开启。

**Q12：如何在不修改 Megatron 核心代码的情况下，为 MoE 添加自定义路由策略？**  
A：实现步骤：
1. 继承 `Router`（`router.py`），实现 `routing(logits)` 方法。
2. 在 `MoESubmodules(router=MyCustomRouter)` 中替换 router builder。
3. 在 layer spec 中使用新的 `MoESubmodules`。
4. 自定义 router 可以调用 `topk_routing_with_score_function` 复用现有的 topk 选择逻辑，只需替换负载均衡部分。

---

## 十七、五道进阶练习题

1. **架构理解**：如果 `num_layers=12`，`moe_layer_freq=3`，写出完整的 `moe_layer_pattern` 列表，并解释哪些层是 MoE 层。在此基础上，若 `moe_layer_freq=[1,0,0,1,0,0,1,0,0,1,0,0]` 等价吗？

2. **EP 计算**：在 `num_moe_experts=128, ep_size=16` 的配置下，ep_rank=7 持有哪些专家？写出 `local_expert_indices`。若 TP=4，EP=16，一共需要多少 GPU？

3. **辅助损失**：手工计算 `E=4, topk=1, T=8` 时，若 routing_map 为全部 token 都路由到 E0（极度不均衡），此时 `f_i` 和 `P_i` 各是多少？aux_loss 是多少（`moe_aux_loss_coeff=0.01`，忽略 TP/CP 缩放）？

4. **Dispatcher 选择**：你在一台 8-GPU A100 节点上训练 MoE，`num_moe_experts=8, ep_size=8`。应该选 AllGather 还是 AllToAll？为什么？如果节点变成 64 GPU（8 节点），建议切换到哪种 dispatcher？

5. **梯度流**：在 EP=4, DP=2 的配置下，画出一个 expert weight 的梯度从反向传播到 optimizer.step 的完整路径：经过哪些通信（ReduceScatter/AllReduce/AllGather），最终在哪些 rank 上更新？router weight 的梯度路径又是什么？

---

## 十八、本篇小结

| 模块 | 关键设计 | 记忆钩子 |
|------|----------|---------|
| moe_layer_freq | int N / list 控制哪些层是 MoE | "整数=间隔，列表=自定义" |
| BaseMoELayer | `num_local = total // ep_size`，assert 整除 | "均分专家，assert 就是文档" |
| MoELayer.forward | route→preprocess→dispatch→expert_compute→combine→postprocess | "六步链式，shared_expert 最先" |
| TopKRouter | logits → score_function → topk → probs + routing_map | "路由器 = 线性层 + topk 选择" |
| score_function | softmax / sigmoid / sqrtsoftplus | "默认 softmax，sigmoid 独立，sqrtsoftplus 平滑" |
| Switch 辅助损失 | `E * Σ f_i * P_i`；f 离散不可微，P 可微 | "f 是计数，P 是可微概率" |
| Z-loss | `mean(log(sum(exp(logits)))^2)` | "防止 logits 爆炸的稳定器" |
| QB routing | per-expert bias 双坐标下降，无 aux_loss | "QB = 自动调价，无需梯度惩罚" |
| Shared Expert | 所有 token 都经过，与路由专家并联 | "shared = 稠密底座，routed = 稀疏专家" |
| AllGather Dispatcher | 全量通信，本地选择 | "通信多，逻辑简单" |
| AllToAll Dispatcher | 稀疏通信，精确发送 | "通信少，实现复杂" |
| Flex Dispatcher | deepep/hybridep/ncclep 三种后端 | "flex = 硬件专属优化" |
| TEGroupedMLP | GroupedGEMM 批量专家计算 | "专家合批，一次 kernel" |
| finalize_model_grads | router grad 单独 AllReduce | "router 不是 expert，要额外聚合" |

---

> **上一篇**：[精读 Megatron 源码（8）：数据管线、DistOpt、Dist Checkpoint](./08-data-optim-ckpt.md)  
> **下一篇**：[精读 Megatron 源码（10）：从读通主路径到能改框架](./10-advanced.md)
