# 精读 Megatron 源码（11）：专家并行 EP 完全指南——切分对象、Token 迁移与进程组

> **专栏**：Megatron 源码精读 · 第 11 篇（独立专题）  
> **写给谁**：已经知道 TP/PP/DP 是什么，但一碰到 MoE 的 `EP` / `expert_DP` 就容易混的人  
> **核心文件**：  
> - `megatron/core/parallel_state.py`（`RankGenerator`、`expert_decoder_rank_generator`、`_EXPERT_*_GROUP`）  
> - `megatron/core/transformer/moe/moe_layer.py`（`BaseMoELayer` 本地专家划分）  
> - `megatron/core/transformer/moe/token_dispatcher.py`（AlltoAll / AllGather dispatch & combine）  
> - `docs/user-guide/parallelism-guide.md`（官方配置入口）  
>
> **相关篇**：  
> - [04 parallel_state](./04-parallel-state.md)（进程组与 `expert_DP`）  
> - [07 数据并行](./07-data-parallel.md)（dense DP vs expert DP 梯度）  
> - [09 MoE](./09-moe.md)（Router / 负载均衡 / Dispatcher 实现细节）  
> **下一篇思路**：读完本篇再回 [09](./09-moe.md)，会更容易把「拓扑」和「路由算法」分开。

---

## 0. 先给结论（读完可回来核对）

1. **EP（Expert Parallelism / 专家并行）切的是「专家列表」**：把 `num_moe_experts` 个专家权重分到 `EP` 张卡上，每张卡只常驻 `num_local_experts = num_experts / EP` 个专家。  
2. **EP 是模型并行的一种**，不是数据并行：跨 EP rank 的权重**不同**；跨 DP / expert_DP 的权重才相同。  
3. **Token 要搬家**：路由决定「这个 token 去哪个专家」后，必须用 **AlltoAll（或 AllGather）** 把 activation 送到持有该专家的 GPU，算完再搬回来（combine）。  
4. **`expert_DP` 不是 EP**：它是「同一批 local experts」的副本数，负责同步这些副本的梯度。  
5. 官方口诀 `world = TP × PP × CP × EP × DP` 里的 **DP，在 MoE 语境下多半指 expert 侧剩余数据并行**；Megatron Core 实际维护**两套**公式（dense 一套、expert 一套）。

如果你只记住一句话：

> **EP = 把不同专家放到不同卡；token 跟着专家走。**  
> **DP / expert_DP = 把同一份权重复制多份；梯度在副本间平均。**

---

## 1. 为什么需要 EP：从单卡 MoE 说起

### 1.1 没有并行时，MoE 在干什么

一个 MoE 层可以粗暴理解成：

```text
对每个 token x：
  1) Router 看 x，选出 top-k 个专家（比如 k=2）
  2) 只把 x 送给这 k 个专家的 MLP
  3) 用路由概率加权，把 k 路输出加回原位置
```

稀疏激活的好处是：**总参数可以很大，但每个 token 只激活一小部分参数**。  
Mixtral-8x7B 风格：8 个专家，每个 token 只进 2 个；DeepSeek-V3 风格：上百个专家，每个 token 仍只进很少几个。

### 1.2 为什么不能「每张卡都放全部专家」

假设一层有 **64 个专家**，每个专家是一个较大的 FFN。若每张训练卡都存完整 64 个专家：

- 显存随专家数线性涨，很快装不下；
- 多数专家对本地 batch 根本不会被用到，却仍占着权重内存；
- 想靠「加大卡数」扩展参数规模时，如果每卡仍是全量专家，卡数增加只复制了同样的巨型 MoE，**参数规模上不去**。

于是引入 EP：

```text
64 experts，EP=8
→ 每张 EP 卡只存 8 个 local experts
→ 专家参数显存大约降到 1/8
→ 想要更多专家，就加 EP（或加 num_experts 且 EP 跟着长）
```

这和 TP「切开矩阵」、PP「切开层」是同一类思路：**模型并行 = 权重不再每卡一份完整拷贝**。差别只是切分轴不同——EP 切的是 **expert 维**。

### 1.3 一图看懂：EP 切什么、不切什么

```text
Dense Transformer 一层：
  Attention 权重 ─── 通常走 TP / PP / DP（与 EP 无关或弱相关）
  MLP 权重       ─── 被 MoE 替换后：
                        Router（门控）   ：通常每卡都有（像 dense）
                        Routed Experts  ：按 EP 切开
                        Shared Expert   ：通常每卡完整副本（像 dense MLP）

EP 负责的是「Routed Experts 这块参数」如何分布在 GPU 上。
```

---

## 2. 五个易混概念：TP / PP / DP / EP / expert_DP

先用「切分对象 + 组内权重是否相同 + 典型通信」对照：

| 名字 | 切什么 | 同一进程组内，权重相同吗？ | 典型通信 | 一句话 |
|------|--------|---------------------------|----------|--------|
| **TP** | 单个矩阵的行/列 | 否（各持分片） | AllReduce / RS+AG | 同一层算子切开算 |
| **PP** | 不同层 / stage | 否（各持不同层） | P2P send/recv | 模型按深度切开 |
| **DP**（dense） | 训练数据 | **是**（Attention 等副本） | AllReduce / RS | 同权重、不同 batch |
| **EP** | 专家列表 | **否**（不同 local experts） | **AlltoAll**（token） | 同层、不同专家 |
| **expert_DP** | 专家侧数据副本 | **是**（同一批 local experts） | AllReduce / RS（梯度） | 专家权重的「DP」 |

### 2.1 最容易错的三组对照

**（1）EP vs DP**

| | EP | DP |
|--|----|----|
| 目的 | 放下更多专家 / 省专家显存 | 提高吞吐（更多数据） |
| 权重 | 跨 rank **不同** | 跨 rank **相同** |
| 移动的是 | **activation（token）** | 主要是 **梯度**（前向各自算） |
| 类比 | 「图书馆把不同书架分到不同分馆」 | 「同一本书印很多本，每人读不同章节作业」 |

**（2）EP vs TP**

| | EP | TP |
|--|----|----|
| 切分粒度 | 整颗专家（一个 MLP） | 矩阵的一半/几分之一 |
| 前向是否总要通信 | 几乎总要（token 路由） | 每层固定通信模式 |
| 通信对象 | 被路由到的 token | 部分激活或梯度归约 |
| 能否替代 | 不能互相替代 | 常组合：`expert_TP` + EP |

**（3）EP vs expert_DP**

| | EP | expert_DP |
|--|----|-----------|
| 角色 | **切开**专家 | **复制**已切开的 local-expert 分片 |
| 公式里的位置 | 分母上的并行度 | 推出来的「副本数」 |
| 组内专家 ID | 不同 | 相同 |
| 通信 | dispatch/combine（activation） | 梯度同步 |

记住口诀：

```text
EP 负责「专家怎么分家」
expert_DP 负责「同一家分片有几份副本、梯度怎么对齐」
```

---

## 3. 本地专家怎么分：`BaseMoELayer` 的硬规则

源码（精简）：

```python
# megatron/core/transformer/moe/moe_layer.py  BaseMoELayer.__init__
ep_size = utils.get_pg_size(self.ep_group)
ep_rank = utils.get_pg_rank(self.ep_group)

assert self.config.num_moe_experts % ep_size == 0
self.num_local_experts = self.config.num_moe_experts // ep_size
offset = ep_rank * self.num_local_experts
self.local_expert_indices = [offset + i for i in range(self.num_local_experts)]
```

### 3.1 数值例：8 experts，EP=4

```text
num_moe_experts = 8
EP = 4
num_local_experts = 2

ep_rank 0 → experts [0, 1]
ep_rank 1 → experts [2, 3]
ep_rank 2 → experts [4, 5]
ep_rank 3 → experts [6, 7]
```

### 3.2 数值例：64 experts，EP=8，看 ep_rank=3

```text
num_local_experts = 64/8 = 8
offset = 3*8 = 24
local = [24, 25, 26, 27, 28, 29, 30, 31]
```

### 3.3 为什么必须整除？

框架用**连续、均匀**的区间划分专家。若 `num_experts % EP != 0`：

- 有的卡多几个专家，负载与显存都偏斜；
- `local_expert_indices`、AlltoAll 的 split 元数据都会变复杂。

所以直接 `assert`，配置不合法就启动失败——这是好事。

### 3.4 Router 在哪？专家权重在哪？

| 模块 | 是否按 EP 切开 | 说明 |
|------|----------------|------|
| Router（gating 线性层） | 通常**不按 EP 切** | 每张卡都能本地算出「全局专家 ID」的 logits |
| Routed expert MLP | **按 EP 切** | 只实例化 `num_local_experts` |
| Shared expert | **不按 EP 切** | 每卡完整一份；梯度走 dense 路径 |

因此前向天然分成两段：

1. **本地路由**：我知道 token 想去全球专家 `#42`；  
2. **跨卡搬运**：`#42` 若不属于我，就把 token 发给拥有它的 EP rank。

---

## 4. Token 为什么必须「搬家」：dispatch / combine

### 4.1 核心矛盾

```text
Token 住在「数据并行意义下的本卡」上；
Expert 住在「EP 意义下的某张卡」上；
两者默认不在同一张卡 → 必须通信。
```

Megatron 把这段通信封装成 **Token Dispatcher**：

```text
route → preprocess(permute) → dispatch(通信) → 本地专家计算 → combine(逆通信) → unpermute
```

主流实现是 `MoEAlltoAllTokenDispatcher`（`--moe-token-dispatcher-type alltoall`）。

### 4.2 AlltoAll 在 EP 里到底交换什么？

AllReduce：「大家把同一形状的梯度加起来」。  
AllGather：「大家都拿到全量 token」。  
**AlltoAll**：「我把寄给你的那一段发给你；你把寄给我的那一段发给我」——每人发送量可以不同。

对 EP：

```text
每个 EP rank：
  发送：本地产生的、但目标专家在别人那里的 tokens
  接收：别人产生的、但目标专家在我这里的 tokens
```

### 4.3 手算例：EP=4，每卡 4 个 token，topk=1，8 experts

为了直觉，令 `num_experts=8`，`EP=4`，每卡 `num_local_experts=2`，且 **topk=1**（每个 token 只去 1 个专家）。  
再假设路由结果碰巧很均匀：

```text
全局专家归属：
  rank0: E0,E1 | rank1: E2,E3 | rank2: E4,E5 | rank3: E6,E7

某一步，四张卡上的 token 路由目标（专家 ID）：

rank0 的 4 tokens → 目标 [0, 2, 4, 6]
rank1 的 4 tokens → 目标 [1, 3, 5, 7]
rank2 的 4 tokens → 目标 [0, 2, 4, 6]
rank3 的 4 tokens → 目标 [1, 3, 5, 7]
```

对 **rank0** 来说：

| 本地 token | 目标专家 | 该专家在谁那 | 动作 |
|------------|----------|--------------|------|
| t0 | 0 | 自己 | 留下 |
| t1 | 2 | rank1 | 发给 rank1 |
| t2 | 4 | rank2 | 发给 rank2 |
| t3 | 6 | rank3 | 发给 rank3 |

对称地，rank0 会从 rank1/2/3 各收到一些「目标在 E0/E1」的 token。  
**Dispatch 之后**，rank0 手里的 token 集合变了：不再是「自己数据并行样本的 4 个 token」，而是「全世界路由到 E0/E1 的那些 token」。

然后：

```text
rank0 用本地 E0/E1 做 GEMM
→ Combine：按来时的逆路径 AlltoAll 回去
→ 每个原始 token 位置拿回自己的专家输出
```

### 4.4 topk>1 时发生什么？

若 `topk=2`，每个 token 会复制（或逻辑复制）成 2 份，分别发往两个专家所在 rank。  
Combine 时再按 `probs` **加权求和**回到原位置。  
通信量大致随 `topk` 上升，所以 AlltoAll 相对 AllGather 的优势在「专家很多、topk 很小」时更明显（详见第 9 篇 Dispatcher 带宽对比）。

### 4.5 和 AllGather dispatcher 的差别（只需建立直觉）

| | AlltoAll | AllGather |
|--|----------|-----------|
| 每卡最终看到的 token | 主要是「路由到我的 local experts」的那些 | 往往先聚齐更大范围的 token，再本地筛选 |
| 带宽 | 通常更省（稀疏） | EP 大时更贵 |
| 适用 | 大 EP、生产 MoE 默认方向 | 小规模 / 特定路径 |

源码类：`MoEAlltoAllTokenDispatcher` / `MoEAllGatherTokenDispatcher`（`token_dispatcher.py`）。

---

## 5. 进程组：EP 在 `parallel_state` 里如何出生

### 5.1 两套 RankGenerator

`initialize_model_parallel` 里有两条拓扑：

```python
# 精简自 parallel_state.initialize_model_parallel
decoder_rank_generator = RankGenerator(
    tp=tensor_model_parallel_size,
    ep=1,                      # dense 路径不把 EP 当维
    dp=data_parallel_size,
    pp=pipeline_model_parallel_size,
    cp=context_parallel_size,
    order=order,
)

expert_decoder_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size,   # expert_TP，默认可等于 TP
    ep=expert_model_parallel_size,    # ★ EP
    dp=expert_data_parallel_size,     # ★ expert_DP（算出来的）
    pp=pipeline_model_parallel_size,
    cp=1,                             # expert 路径强制 CP=1
    order=order,
)
```

关键断言：

- `ep==1 or cp==1`（同一 generator 内 EP 与 CP 不能同时 >1）  
- dense 与 expert 的 **PP ranks 必须一致**（流水线对齐）

### 5.2 公式：请把这两行背熟

```text
# Dense（Attention / 非专家 MLP 等）
DP = world_size / (TP × PP × CP)

# Expert（Routed experts）
expert_DP = world_size / (expert_TP × EP × PP)
# 注意：分母没有 CP（expert generator 里 cp 固定为 1）
```

CLI：

```bash
--expert-model-parallel-size 8          # EP
--expert-tensor-parallel-size 1         # expert_TP（细粒度 MoE 常用 1）
--num-experts 64                        # 必须能被 EP 整除
--moe-token-dispatcher-type alltoall
```

### 5.3 官方 `world = TP×PP×CP×EP×DP` 怎么理解才不拧巴？

官方 parallelism guide 写：

```text
Total GPUs = TP × PP × CP × EP × DP
```

在 **expert 拓扑**、且把这里的 `DP` 理解成 **`expert_DP`**、并把 `TP` 理解成 **`expert_TP`**、`CP=1` 时：

```text
world = expert_TP × PP × 1 × EP × expert_DP
```

这与 Core 源码一致。

但同时，**dense 拓扑**是：

```text
world = TP × PP × CP × dense_DP
```

两边 `world` 必须相等。于是当 `expert_TP = TP` 且 `CP = 1` 时，有漂亮关系：

```text
dense_DP = EP × expert_DP
```

含义：**EP 从「本来可以做数据并行的卡」里抽走一部分，改去做专家分片**。  
所以你会看到：开了 EP 之后，dense DP 往往仍较大，而 expert_DP 变小——同一份 local-expert 分片的副本变少了。

### 5.4 完整算例 A：`world=8, TP=1, EP=4, PP=1, CP=1`

```text
dense:   DP = 8/(1×1×1) = 8
expert:  expert_TP=1 → expert_DP = 8/(1×4×1) = 2

关系：dense_DP(8) = EP(4) × expert_DP(2)  ✓
```

`order="tp-cp-ep-dp-pp"` 下，expert generator 编组（手算/脚本验证）：

```text
EP 组（每组 4 卡，组内专家不同）:
  [0, 1, 2, 3]
  [4, 5, 6, 7]

expert_DP 组（每组 2 卡，组内专家相同）:
  [0, 4]   # 都持有 local experts 对应 ep_rank=0 的那批
  [1, 5]
  [2, 6]
  [3, 7]
```

若 `num_experts=8`：

```text
rank 0 与 rank 4：都持有 experts [0,1]  ← 它们是 expert_DP 副本
rank 1 与 rank 5：都持有 experts [2,3]
...
同一 EP 组 [0,1,2,3]：四张卡专家互斥，靠 AlltoAll 交换 token
```

### 5.5 完整算例 B：`world=8, expert_TP=2, EP=2, expert_DP=2, PP=1`

```text
EP 组:
  [0, 2], [1, 3], [4, 6], [5, 7]
expert_DP 组:
  [0, 4], [1, 5], [2, 6], [3, 7]
TP 组:
  [0, 1], [2, 3], [4, 5], [6, 7]
```

读法：

- 在 EP 组 `[0,2]` 内：专家切开；  
- rank0 与 rank1 是 TP 搭档，可能再把**单个专家权重**做张量并行（`expert_TP=2`）；  
- rank0 与 rank4 是 expert_DP 搭档，专家 ID 集合相同，训练不同数据，最后同步梯度。

### 5.6 算例 C：把 expert_TP 降为 1，把 EP 加大

细粒度 MoE 常见建议：`expert_TP=1`，把卡让给 EP。

```text
world=64, dense TP=4, PP=2, CP=1
→ dense DP = 64/(4×2×1) = 8

若 expert_TP=4, EP=4:
→ expert_DP = 64/(4×4×2) = 2

若改 expert_TP=1, EP=8:
→ expert_DP = 64/(1×8×2) = 4
```

直觉：专家不再做 TP 切分，改为更细的 EP；同时 expert 副本数从 2 升到 4。

---

## 6. 一张「数据流 + 梯度流」总图

```text
                    ┌──────────── dense 路径 ────────────┐
batch tokens ──► Attention / Router / SharedExpert ──► …
                    │ 权重副本：dense DP（×CP）组内相同
                    │ 梯度：intra_dp_cp AllReduce / RS
                    └────────────────────────────────────┘

                    ┌──────────── expert 路径 ───────────┐
Router 选出 expert id
   │
   ▼
AlltoAll dispatch  ──（EP 组内）──►  持有该 expert 的卡
   │
   ▼
本地 expert GEMM（仅 num_local_experts）
   │
   ▼
AlltoAll combine  ◄───────────────  输出搬回 token 原位
   │
   ▼
expert 权重梯度：只在 expert_DP 组内同步
（跨 EP 不同专家，绝不能拿去跟别人的专家梯度 AllReduce）
                    └────────────────────────────────────┘
```

### 6.1 前向：通信的是 activation

EP 前向的集体通信主要是 **token 搬运**，不是「把专家权重广播给大家」。  
专家权重尽量留在本卡；谁需要算，谁把 token 寄过来。

### 6.2 反向：通信仍要走 AlltoAll，外加 expert_DP 梯度同步

反向时：

1. combine/dispatch 的反传会再次触发 EP 组上的 AlltoAll 类通信（activation 梯度往回走）；  
2. 专家权重的梯度，只在 **持有同一 local-expert 分片** 的 ranks（`expert_DP` / `expt_dp`）之间 AllReduce 或 ReduceScatter。

这与第 7 篇「Expert DP vs Dense DP」、第 4 篇 §7.1.1 完全一致。

### 6.3 Shared Expert 别误判成 EP

Shared expert 对**所有 token**计算，通常每卡一份完整权重，**不参与 EP 分片**。  
它的梯度同步更接近 dense 参数，而不是 `expt_dp`。  
（DeepSeek 系模型常见「大量 routed + 少量 shared」。）

---

## 7. 源码路径：从配置到一次 MoE forward

### 7.1 配置入口

```bash
--num-experts 64
--expert-model-parallel-size 8
--expert-tensor-parallel-size 1
--moe-router-topk 2
--moe-token-dispatcher-type alltoall
--sequence-parallel          # 官方：TP 与 EP 同用时通常要开 SP
```

官方提醒（`parallelism-guide.md`）：**EP 与 TP 组合时必须开 Sequence Parallel**。

### 7.2 初始化顺序（概念级）

```text
initialize_model_parallel
  ├─ 建 dense 组：tp / pp / dp / cp / dp_cp ...
  └─ 建 expert 组：ep / expt_tp / expt_dp / ...

build GPTModel + MoE layer spec
  └─ MoELayer / BaseMoELayer
        ├─ 读 ep_group → 算 local_expert_indices
        ├─ 建 TopKRouter
        ├─ 建 local Experts（只含 num_local_experts）
        └─ 建 TokenDispatcher(ep_group, ...)
```

### 7.3 `MoELayer.forward` 里 EP 相关的几步

（完整标注见第 9 篇；这里只标 EP 视角）

```text
route(hidden)            # 本地：得到 probs, routing_map（全局专家 ID）
preprocess(...)          # 按目标专家重排 token，准备 splits
dispatch(...)            # ★ EP AlltoAll：token 到达专家所在卡
routed_experts_compute   # 只跑本卡 local experts
combine(...)             # ★ 逆 AlltoAll：输出回原 token 位置
postprocess(...)         # + shared expert 等
```

`MoEAlltoAllTokenDispatcher` 文档字符串里的七步流水，是实现级展开：

```text
(1) preprocess metadata + permute
(2) dispatch permute
(3) A2A(EP)          ← 专家并行通信本体
(4) AG(TP) + sort    ← 若 expert_TP>1，还要在 TP 维整理
(5) sort + RS(TP)
(6) A2A(EP) combine
(7) unpermute
```

所以：**EP 的「存在感」主要在第 (3)(6) 步；TP 是叠加上去的第二维。**

### 7.4 进程组 API（兼容层）

```python
parallel_state.get_expert_model_parallel_group()   # EP
parallel_state.get_expert_data_parallel_group()    # expert_DP
parallel_state.get_expert_tensor_parallel_group()  # expert_TP
```

新 `megatron/core` 代码更推荐经 `ProcessGroupCollection` 注入（`pg_collection.ep` / `expt_dp` 等），避免库代码直接读全局。

---

## 8. EP 与其它并行的组合规则

| 组合 | 是否常见 | 注意点 |
|------|----------|--------|
| EP + DP/expert_DP | 必须理解 | 见 §5：`dense_DP ≈ EP × expert_DP`（在 TP 对齐、CP=1 时） |
| EP + TP / expert_TP | 常见 | 开 SP；细粒度 MoE 常 `expert_TP=1` |
| EP + PP | 常见 | PP 组必须与 dense 一致；专家层落在某些 PP stage |
| EP + CP | **受限制** | 同一 RankGenerator 内禁止同时 >1；expert 路径 cp 强制 1 |
| EP + DistOpt | 常见 | 专家缓冲走 `expert_parallel_buffers` / `intra_expt_dp` |

### 8.1 「EP 会不会增加 world size？」

不会魔法变出新 GPU。  
`world_size` 固定时，设大 EP，意味着 **expert_DP（以及与 dense DP 的关系）被重新分配**。  
口语「加 EP」=「把更多卡用来切专家，而不是用来复制同一批专家」。

### 8.2 什么时候该加大 EP？

优先考虑加大 EP 的信号：

- `num_experts` 很大，单卡放不下全部专家；  
- 专家 FFN 很宽，专家权重显存是瓶颈；  
- 路由很稀疏（topk ≪ num_experts），AlltoAll 仍划算。

不必盲目加大 EP 的信号：

- 专家很少（例如 8）且卡也少——EP=8 时 `expert_DP` 可能变成 1，专家侧没有数据并行副本；  
- AlltoAll 延迟已经是瓶颈，再加大 EP 可能更碎、更抖；  
- 与 CP 长上下文需求冲突（需改整体策略）。

---

## 9. 常见误解 FAQ

**Q1：EP 是不是就是「MoE 版的 DP」？**  
不是。DP 复制相同权重；EP 让权重不同。MoE 里和 DP 对应的是 **expert_DP**。

**Q2：开了 EP 以后，是不是每张卡的 batch 也按 EP 切开？**  
数据仍按 dense 数据并行习惯进模型；到了 MoE 层，token 会按路由结果在 EP 组内重新分配。  
你本地 microbatch 的 token，算完 expert 后会回来，但中间短暂「住在别的卡上」。

**Q3：Router 要不要做 EP？**  
一般不做专家列表式的 EP 切分。Router 需要产出对**全局专家 ID** 的分数；专家权重本身才按 EP 分片。

**Q4：为什么我看到 `DP=8` 同时又看到 `expert_DP=2`？**  
正常。Attention 等有 8 份副本；某一组 local experts 可能只有 2 份副本（其余「DP 名额」被 EP 用来放不同专家了）。

**Q5：AlltoAll 失败/卡住，是不是 DP 组配错了？**  
先查 **EP 组**是否包含了正确的 ranks，以及 `num_experts % EP == 0`。  
其次查 dispatcher 的 `input_splits/output_splits` 是否与路由统计一致。  
DP 组配错更常表现为「能跑但 loss 不对 / 梯度爆炸」，而不一定是 AlltoAll hang。

**Q6：`param.allreduce=False` 的 expert 参数，是不是就不同步梯度了？**  
不是「不同步」，而是 **不走 dense DP 那条 AllReduce**。它们走 expert 缓冲与 `expt_dp`。

**Q7：EP=1 还算不算 MoE？**  
算。EP=1 表示所有专家都在每张（expert 拓扑下的）卡上；仍有 Router 与稀疏计算，只是没有跨卡专家分片。小实验常用 EP=1 先把功能跑通。

**Q8：Shared expert 参与 AlltoAll 吗？**  
通常不。它本地算完，在 postprocess 加回；通信主路径是 routed experts 的 dispatch/combine。

---

## 10. 排障决策树（EP 专题）

```text
MoE 训练异常
├─ 启动 assert: num_moe_experts % ep_size != 0
│    └─ 改 num_experts 或 EP，使整除
├─ 启动 assert: EP 与 CP 同时 >1
│    └─ 关掉一侧，或改整体并行方案
├─ AlltoAll hang / watchdog
│    ├─ 检查 EP 组 ranks 是否对称、是否混入错 PP stage
│    ├─ 检查是否有 rank 路由统计与 splits 不一致（drop/pad 配置）
│    └─ 先把 EP 降到 1 或 2 做对照
├─ loss 乱飞 / 不收敛
│    ├─ 是否误把 expert 梯度拿去 dense DP AllReduce
│    ├─ expert_DP 是否意外为 1 且全局 batch 对专家极不均衡
│    └─ 打开 tokens_per_expert 日志看负载
└─ OOM
     ├─ 专家权重 OOM → 加大 EP 或减小 num_experts / FFN
     ├─ activation OOM → 查 topk、capacity、是否 AllGather dispatcher
     └─ 与 TP 同用时确认 SP、以及 expert_TP 是否过大
```

---

## 11. 对照练习（建议手算）

**练习 1**  
`world=16, TP=2, PP=1, CP=1, EP=4, expert_TP=2`。  
求 dense `DP` 与 `expert_DP`。写出「同一 local experts」应该出现在哪些 rank 对上（可用 §5.5 同类方法推）。

**练习 2**  
`num_experts=32, EP=8`。写出 ep_rank=5 的 `local_expert_indices`。

**练习 3**  
用自己的话解释：为什么跨 EP rank 做 expert 权重 AllReduce 是错的？错了会出现什么现象？

**练习 4**  
画一张 4 卡图：`EP=4, expert_DP=1, topk=1`。假设 4 个 token 分别路由到 E0–E3，标注 AlltoAll 每条边传送几个 token。

**练习 5**  
阅读 `MoEAlltoAllTokenDispatcher` 类注释的 (1)–(7) 步，标出哪一步会调用 `all_to_all(..., group=ep_group)`。

**练习 6**  
配置 `expert_TP=1, EP=8` 与 `expert_TP=8, EP=1`（其它对齐）时，专家显存与通信模式有何本质不同？

---

## 12. 和本系列其它篇怎么接力读

| 你的问题 | 去哪篇 |
|----------|--------|
| 进程组怎么建、order 字符串、`expert_DP` 公式 | [04](./04-parallel-state.md) |
| 专家梯度如何进 DistOpt / finalize | [07](./07-data-parallel.md)、[08](./08-data-optim-ckpt.md) |
| Router、aux loss、drop/pad、Flex/DeepEP | [09](./09-moe.md) |
| 只想弄懂「EP 到底切啥」 | **本篇** |

推荐阅读顺序：

```text
04（拓扑）→ 本篇 11（EP 概念）→ 09（MoE 算法与实现细节）→ 07（梯度两条线）
```

---

## 13. 一页纸速查

```text
EP
  切分：num_experts → 每卡 num_experts/EP 个 local experts
  组内权重：不同
  前向通信：AlltoAll（token dispatch/combine）
  配置：--expert-model-parallel-size

expert_DP
  含义：同一 local-expert 分片的副本数
  公式：world/(expert_TP×EP×PP)
  通信：梯度 AllReduce/RS（不是 token AlltoAll）

dense DP
  公式：world/(TP×PP×CP)
  常见关系（expert_TP=TP, CP=1）：dense_DP = EP × expert_DP

不要做的事
  ✗ 把不同 EP rank 的专家梯度当 DP 去平均
  ✗ 把 EP 理解成「多卡各算各的 batch、专家还是全量」
  ✗ 忽略 num_experts 必须整除 EP
```

---

## 14. 结语

EP 难懂，通常不是因为数学复杂，而是因为 **Megatron 在同一 `world_size` 上叠了两套「谁和谁是副本」的说法**：

- 对 Attention：很多卡是副本（dense DP）；  
- 对 Experts：一部分卡变成「不同专家」，只剩更小的一组才是副本（expert_DP）。

把「切分」和「复制」分开以后，AlltoAll、本地专家索引、两条 DistOpt 缓冲，都会变得顺理成章。

下一站建议带着本篇的进程组图，打开 `token_dispatcher.py` 的 `MoEAlltoAllTokenDispatcher.dispatch_preprocess`，对着一次真实 `routing_map` 看 `input_splits` 如何生成——那是从「概念 EP」跨到「可调试 EP」的门槛。
