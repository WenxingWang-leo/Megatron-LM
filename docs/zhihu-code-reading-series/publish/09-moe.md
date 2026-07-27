# 精读 Megatron 源码（9）：MoE 完全精读——Router、Dispatcher、专家并行与负载均衡

> **专栏**：Megatron 源码精读 · 第 9 篇  
> **核心文件**：  
> - `megatron/core/transformer/moe/moe_layer.py`  
> - `megatron/core/transformer/moe/router.py`  
> - `megatron/core/transformer/moe/token_dispatcher.py`  
> - `megatron/core/transformer/moe/experts.py`  
> - `megatron/core/transformer/moe/moe_utils.py`  
> - `megatron/core/models/gpt/gpt_layer_specs.py`  
> - `megatron/core/distributed/finalize_model_grads.py`  
>
> **上一篇**：[精读 Megatron 源码（8）：数据管线、DistOpt、Dist Checkpoint](./08-data-optim-ckpt.md)  
> **下一篇**：[精读 Megatron 源码（10）：从读通主路径到能改框架](./10-advanced.md)

---

Mixture of Experts（MoE）是当前大模型最热门的架构之一：用稀疏激活替代密集 MLP，实现"参数量增大、计算量不增大"。但 MoE 的分布式实现极其复杂——不同于 TP/PP/DP，它引入了全新的 **Expert Parallelism（EP）**，以及专门的通信原语（AllToAll vs AllGather）。本篇完整精读 Megatron 的 MoE 实现。

---

## 一、MoE 的位置：什么时候替换 MLP

### 1.1 TransformerLayer 的 MLP 选择

GPT 的每个 Transformer layer 都有一个 MLP（前馈网络）。当配置了 `num_moe_experts > 1` 时，这个 MLP 可以被替换成 `MoELayer`。**"替换"的逻辑在 layer spec 里**：

```python
# megatron/core/models/gpt/gpt_layer_specs.py  第 563-670 行（精简）
def get_gpt_decoder_block_spec(config, ...):
    # ...
    # 解析 moe_layer_freq 决定哪些层是 MoE 层
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0
            for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq   # 直接用用户指定的列表

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

Mixtral-8x7B 风格：`[1,0,1,0,1,0,1,0]`（偶数层 MoE）；DeepSeek 风格：第 1 层是 dense，其余都是 MoE。

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

## 三、MoELayer.forward：六步链式调用

```python
# moe_layer.py  MoELayer（简化版，真实代码更复杂）
def forward(self, hidden_states, attention_mask=None):
    # ① 路由：计算每个 token 应去哪些专家
    probs, routing_map = self.route(hidden_states)
    # ② 预处理：为通信做准备（排列 token、计算 cu_seqlens）
    hidden_states, probs = self.preprocess(hidden_states, probs, routing_map)
    # ③ 分发：实际通信（AllGather / AllToAll / Flex）
    hidden_states, probs = self.dispatch(hidden_states, probs)
    # ④ 专家计算：本地专家做 MLP
    hidden_states, mlp_bias = self.routed_experts_compute(hidden_states, probs)
    # ⑤ 合并：逆通信，把专家输出汇聚回原始 token 位置
    output = self.combine(hidden_states, mlp_bias)
    # ⑥ 后处理：加 shared expert 输出、残差等
    output = self.postprocess(output, ...)
    return output, None  # None 是 bias（MoE 层通常无偏置）
```

**每一步的通信模式如下图**（以 AllToAll 模式为例）：

```
Token 所在 rank:   [rank0: T0,T1,T2]  [rank1: T3,T4,T5]  ...  [rankN: ...]
                        ↓ route（本地计算，无通信）
routing_map:       T0→E3, T1→E7, T2→E1, T3→E0, T4→E5, ...
                        ↓ preprocess（统计每个专家收到多少 token）
input_splits:      rank0 发 2 给 rank0，1 给 rank1，...
                        ↓ AllToAll（实际通信）
Expert 所在 rank:  [rank0: E0 收到 T3, E1 收到 T2, ...]  [rank1: E7 收到 T1, ...]
                        ↓ experts compute（本地 GEMM）
                        ↓ AllToAll（逆通信）
Token 所在 rank:   恢复 T0,T1,T2 的位置，加权求和（weighted combine）
```

---

## 四、TopKRouter：路由的完整逻辑

### 4.1 路由流程

```python
# router.py  TopKRouter（简化）
class TopKRouter(Router):
    def forward(self, hidden_states):
        # 1. 线性门控：hidden_states * weight^T → logits [T, E]
        logits = self.gating(hidden_states)    # [num_tokens, num_experts]

        # 2. topk 选择（附带 score function）
        # score_function 可以是 softmax / sigmoid / tanh 等
        probs, routing_map = topk_routing_with_score_function(
            logits, topk=self.topk, score_function=self.score_function
        )
        # probs: [T, topk]，每个 token 选中的 topk 专家的权重
        # routing_map: [T, E] bool，True 表示该 token 路由到该专家

        # 3. 可选：token dropping（capacity factor 限制）
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(probs, routing_map)

        # 4. 可选：计算辅助损失（负载均衡）
        if self.moe_aux_loss_func is not None:
            probs = MoEAuxLossAutoScaler.apply(probs, aux_loss)

        return probs, routing_map
```

### 4.2 Switch 辅助损失直觉理解

辅助损失来源于 Switch Transformer 论文，目的是惩罚"部分专家收到太多 token"的不均衡情况：

```
loss = E * Σ_{i=1}^{E} (f_i * P_i)

其中：
  E = num_experts（专家数）
  f_i = fraction of tokens dispatched to expert i
      = (1 / T*topk) * Σ_{x∈Batch} routing_map(x, i)
      # f_i 是路由到专家 i 的 token 比例（离散、不可微）
  P_i = average routing probability for expert i
      = (1 / T) * Σ_{x∈Batch} probs(x, i)
      # P_i 是路由概率的均值（连续、可微）
```

直觉：`f_i` 大（专家 i 实际收到很多 token）且 `P_i` 大（路由器倾向于发给这个专家），乘积就大，损失就高，从而**惩罚路由器偏向某些专家**。注意 `f_i` 不可微，但 `P_i` 可微，梯度通过 `P_i` 反传到路由器 weight。

在 `moe_utils.py` 中的实现：

```python
# moe_utils.py  switch_load_balancing_loss_func
aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
    num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
)
```

### 4.3 MoEAuxLossAutoScaler：如何把辅助损失插入反向传播

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
        # 在 output 的梯度上叠加 aux_loss 的梯度（scaled）
        aux_loss_backward_scale = ...
        scaled_aux_loss_grad = torch.ones_like(grad_output) * aux_loss_backward_scale
        return grad_output + scaled_aux_loss_grad, None
```

这是一个**"梯度注入"技巧**：`forward` 等价恒等变换，`backward` 把 aux_loss 的梯度注入到 `probs` 的梯度流中，不需要显式调用 `aux_loss.backward()`。这使得辅助损失对模型主 loss 的梯度影响可控（通过 `moe_aux_loss_coeff` 系数）。

---

## 五、Token Dispatcher：三种通信模式的选择

```
文件: megatron/core/transformer/moe/token_dispatcher.py
```

| 类名 | 配置 `moe_token_dispatcher_type` | 通信原语 | 适用场景 |
|------|----------------------------------|----------|---------|
| `MoEAllGatherTokenDispatcher` | `"allgather"` | AllGather（TP*EP 域） | 小 EP，token 数少，通信量可接受 |
| `MoEAlltoAllTokenDispatcher` | `"alltoall"` | AllToAll（EP 域） | 大 EP，稀疏路由，通信量小 |
| `MoEFlexTokenDispatcher` | `"flex"` | deepep/hybridep/ncclep | 超大规模，需要专门优化 |

### 5.1 AllGather Dispatcher

```
思路：把所有 rank 的 hidden_states 全部 AllGather 到本 rank，
然后本 rank 的专家从完整的 token 集合中挑选属于自己的 token 计算。
```

通信量：`T * H * (TP*EP - 1)` 个元素（T=tokens，H=hidden_size）。

**优点**：实现简单，token 无需跨卡移动（本地选择），没有动态不平衡的通信量问题。  
**缺点**：通信量随 TP*EP 线性增长，大 EP 时通信成为瓶颈。

```python
# token_dispatcher.py  MoEAllGatherTokenDispatcher（简化）
class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    def token_dispatch(self, hidden_states, probs):
        # AllGather hidden_states across TP*EP group
        global_hidden_states = all_gather_tensor(
            hidden_states, dim=0, group=self.tp_ep_group
        )
        # 本地专家从 global_hidden_states 中按 routing_map 提取
        ...
```

### 5.2 AllToAll Dispatcher

```
思路：每个 rank 只发送"本 rank 上路由到其他 rank 专家的 token"，
不需要发送所有 token。
```

通信量：约 `T * topk/EP * H`，远小于 AllGather。

**优点**：通信量与 EP 大小基本无关（稀疏路由下）。  
**缺点**：AllToAll 的输入/输出大小不定（动态路由），NCCL 的 Variable AllToAll 实现较复杂。

```python
# token_dispatcher.py  MoEAlltoAllTokenDispatcher（简化）
def token_dispatch(self, hidden_states, probs):
    # 计算 input_splits（本 rank 发多少给每个 ep_rank）
    # 计算 output_splits（本 rank 收多少来自每个 ep_rank）
    hidden_states = all_to_all(
        self.ep_group,
        hidden_states,
        output_split_sizes=self.output_splits,
        input_split_sizes=self.input_splits,
    )
    return hidden_states, probs
```

### 5.3 Flex Dispatcher：三种后端

`MoEFlexTokenDispatcher` 是一个统一前端，根据 `moe_flex_dispatcher_backend` 选择后端：

| 后端 | 说明 | 特点 |
|------|------|------|
| `deepep` | DeepSeek DeepEP 库（`github.com/deepseek-ai/deepep`） | 专门为 MoE EP 优化的 NCCL 扩展，低延迟 |
| `hybridep` | HybridEP（Megatron 内置）| 把排列与 AllToAll 融合，减少 kernel 启动开销 |
| `ncclep` | NCCL EP（Megatron 内置，需 SM100+ Blackwell 或支持 symm-mem）| 利用对称内存零拷贝，适合最新硬件 |

选择建议：
- 调试/中等规模：`alltoall`
- 大规模生产：根据硬件选 `flex` + `deepep` 或 `ncclep`
- 通信量分析实验：先跑 `allgather` 作基准，再对比 `alltoall`

---

## 六、专家计算：TEGroupedMLP vs SequentialMLP

```
文件: megatron/core/transformer/moe/experts.py
```

### 6.1 TEGroupedMLP（推荐，依赖 TransformerEngine）

```python
class TEGroupedMLP(MegatronModule):
    """使用 TransformerEngine 的 GroupedGEMM 批量计算多个专家的 MLP"""
```

`GroupedGEMM` 把 N 个专家的权重矩阵堆叠在一起，用单次 CUDA kernel 完成所有专家的 GEMM，显著减少 kernel 启动开销和内存访问碎片化。

**适用条件**：
- 安装了 TransformerEngine（`from transformer_engine.pytorch import ...`）
- `moe_grouped_gemm=True`

**sharded_state_dict 与 SequentialMLP 互换**：

```python
# experts.py  TEGroupedMLP
def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    # 生成的 ShardedTensor key 格式与 SequentialMLP 完全兼容
    # 因此 TEGroupedMLP 训练的 checkpoint 可以用 SequentialMLP 加载（反之亦然）
    ...
```

### 6.2 SequentialMLP（兼容性好）

```python
# SequentialMLP：把每个专家当作独立的 MLP，逐一 forward
for expert_idx, expert in enumerate(self.local_experts):
    expert_input = dispatched_input[expert_start:expert_end]
    expert_output = expert(expert_input)
```

**适用场景**：不依赖 TE，适合调试和不支持 GroupedGEMM 的环境。性能低于 TEGroupedMLP，但输出语义完全相同。

---

## 七、Expert Parallel 的并行维度

### 7.1 EP 与其他并行维度的关系

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

EP group 跨越不同的 TP 副本：rank 0、2、4、6 组成一个 EP group，它们各持有不同的专家。

### 7.2 Expert-TP（专家内的 TP 切分）

对于每个专家自身，其 weight matrix 也可以被 TP 切分（在 expert 的 hidden dimension 上切），称为 Expert-TP（`expert_tp`）。配置方式：

```bash
--expert-tensor-parallel-size 2   # 每个专家内部再 TP 切 2 份
```

在 `optimizer/__init__.py` 中，带 `expert_tp=True` 属性的参数会被放入专门的 param_group：

```python
# optimizer/__init__.py  第 783 行
if 'experts' in name and 'shared' not in name:
    param.expert_tp = True
```

### 7.3 Expert DP（专家的数据并行）

EP ranks 之间实际上形成了专家级别的 DP：同一 EP rank 在不同 DP 副本中处理不同 batch，需要聚合梯度。这部分由 `finalize_model_grads` 中的 `_allreduce_router_grads` 处理：

```python
# finalize_model_grads.py  第 278 行
def _allreduce_router_grads(model, config):
    """All-reduce router grads（路由器参数的梯度需要在整个 DP 域 AllReduce）"""
    for module in model:
        for param in module.parameters():
            if getattr(param, 'allreduce', True) and param.grad is not None:
                # 路由器 weight 是非 expert 参数，走正常 DP AllReduce
                ...
```

**路由器梯度为什么要单独处理？**  
路由器（TopKRouter）的 weight 存在于所有 EP rank 上（每个 rank 都有完整的 router），但 expert 参数只在特定 EP rank 上。`finalize_model_grads` 在 DDP 完成 expert 梯度同步之后，**额外做一次 router gradient AllReduce**，确保路由器参数更新的一致性。

---

## 八、容量与 token dropping

### 8.1 capacity_factor 的作用

```python
# 配置项
--moe-expert-capacity-factor 1.2  # 每个专家最多接受 capacity 个 token
```

`capacity = capacity_factor * (total_tokens / num_experts / topk)`

当路由到某专家的 token 数超过 capacity 时，多余的 token 被 **drop**（不经过该专家，以零向量代替输出）。

**好处**：保证 AllToAll 通信大小固定（静态 shape），方便 CUDA graph 捕获。  
**坏处**：dropped token 的 loss 不正确（梯度消失），极端负载不均时质量下降。

### 8.2 常见 pitfall

| 问题 | 原因 | 解决 |
|------|------|------|
| 切换 dispatcher 后指标变化 | allgather vs alltoall 通信范围不同，aggregation 逻辑略有差异 | 对齐 `tokens_per_expert` 计算域 |
| 辅助损失系数太大 | `moe_aux_loss_coeff=0.1` 过大，主 loss 被辅助损失淹没 | 通常取 `0.01` 到 `0.02` |
| EP 不整除报错 | `num_moe_experts=63, ep_size=8` | `63 % 8 != 0`，需调整 `num_moe_experts=64` |
| token dropping 率高 | capacity_factor=1.0 + 路由极不均衡 | 提高 capacity_factor 或加强辅助损失 |
| 梯度不一致 | EP 训练中某 rank 专家梯度未正确聚合 | 检查 `is_expert_parallel` 属性是否正确打标 |

---

## 九、完整数据流 worked example

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
  rank1: E2(T0) → output_T0, E2(T2) → output_T2

逆 AllToAll（combine 阶段）：
  rank0: 取回 output_T0（从rank1），保留 output_T1 → rank0 输出 [output_T0, output_T1]
  rank1: 取回 output_T3（从rank0），保留 output_T2 → rank1 输出 [output_T2, output_T3]

注意 probs（topk=1 时为 1.0）乘以输出，weighted combine 完成。
```

这就是 AllToAll dispatcher 的完整一轮数据流。

---

## 十、finalize_model_grads 的 MoE 特殊处理

```python
# finalize_model_grads.py  第 494-561 行（精简）
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

## 十一、MoE 调试检查清单

```
路由相关：
  □ TopKRouter weight 的 shape 是 (num_experts, hidden_size)？
  □ routing_map.sum(dim=1) 是否等于 topk（每个 token 恰好路由 topk 个专家）？
  □ aux_loss 的量级是否合理（约为主 loss 的 1%~5%）？
  □ tokens_per_expert 的方差是否在训练中下降（负载均衡是否生效）？

通信相关：
  □ AllToAll 的 input_splits 之和 == output_splits 之和？
  □ ep_size 是否整除 num_moe_experts？
  □ token dropping 率是否 < 5%？

专家相关：
  □ local_expert_indices 与 ep_rank 的对应关系是否正确？
  □ TEGroupedMLP 的 weight shape 是 (num_local_experts, ffn_hidden_size, hidden_size)？
  □ expert 梯度是否只在 ep_group 内聚合（而非 全 DP group）？

性能相关：
  □ dispatcher 选择是否与硬件匹配（Blackwell → ncclep）？
  □ overlap_dispatch_backward_with_experts_wgrad 是否开启？
  □ GroupedGEMM 是否生效（检查 TE 版本）？
```

---

## 十二、五道练习题

1. **架构理解**：如果 `num_layers=12`，`moe_layer_freq=3`，写出完整的 `moe_layer_pattern` 列表，并解释哪些层是 MoE 层。

2. **EP 计算**：在 `num_moe_experts=128, ep_size=16` 的配置下，ep_rank=7 持有哪些专家？写出 `local_expert_indices`。

3. **辅助损失**：手工计算 `E=4, topk=1, T=8` 时，若 routing_map 为全部 token 都路由到 E0（极度不均衡），此时 `f_i` 和 `P_i` 各是多少？aux_loss 是多少（忽略系数）？

4. **Dispatcher 选择**：你在一台 8-GPU A100 节点上训练 MoE，`num_moe_experts=8, ep_size=8`。应该选 AllGather 还是 AllToAll？为什么？如果节点变成 64 GPU（8 节点），建议切换到哪种 dispatcher？

5. **梯度流**：在 EP=4, DP=2 的配置下，画出一个 expert weight 的梯度从反向传播到 optimizer.step 的完整路径：经过哪些通信（ReduceScatter/AllReduce/AllGather），最终在哪些 rank 上更新？

---

## 十三、本篇小结

| 模块 | 关键设计 | 记忆钩子 |
|------|----------|---------|
| moe_layer_freq | int N / list 控制哪些层是 MoE | "整数=间隔，列表=自定义" |
| BaseMoELayer | `num_local = total // ep_size`，assert 整除 | "均分专家，assert 就是文档" |
| TopKRouter | logits → topk → probs + routing_map；aux_loss | "路由器 = 线性层 + topk 选择" |
| Switch 辅助损失 | `E * Σ f_i * P_i`；f 离散不可微，P 可微 | "f 是计数，P 是可微概率" |
| MoEAuxLossAutoScaler | 梯度注入技巧，无需显式 backward | "forward 恒等，backward 注入" |
| AllGather Dispatcher | 全量通信，本地选择 | "通信多，逻辑简单" |
| AllToAll Dispatcher | 稀疏通信，精确发送 | "通信少，实现复杂" |
| Flex Dispatcher | deepep/hybridep/ncclep 三种后端 | "flex = 硬件专属优化" |
| TEGroupedMLP | GroupedGEMM 批量专家计算 | "专家合批，一次 kernel" |
| finalize_model_grads | router grad 单独 AllReduce | "router 不是 expert，要额外聚合" |

---

> **上一篇**：[精读 Megatron 源码（8）：数据管线、DistOpt、Dist Checkpoint](./08-data-optim-ckpt.md)  
> **下一篇**：[精读 Megatron 源码（10）：从读通主路径到能改框架](./10-advanced.md)
