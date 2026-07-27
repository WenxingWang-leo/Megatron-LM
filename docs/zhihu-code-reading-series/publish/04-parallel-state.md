# 精读 Megatron 源码（4）：parallel_state 完全指南——进程组如何把 GPU 编成网格

> 源文件：`megatron/core/parallel_state.py`（2249 行）

---

## 1. 为什么进程组是 Megatron 的"骨架"

每块 GPU 都是一个独立的 CUDA 设备；要让多块 GPU 协同训练一个模型，必须先定义"谁和谁通信"。PyTorch 通过 `torch.distributed.ProcessGroup` 对象来描述一组需要互相通信的进程。进程组一旦错误——比如把本应做 DP AllReduce 的 rank 混进了 TP 组——梯度就会静默出错，模型收敛曲线毫无预兆地走歪。`parallel_state.py` 就是 Megatron 所有通信结构的"出生证明"：它在训练开始时构造出所有必要的进程组，并在整个训练生命周期内以全局变量的形式维护这些组的引用。

本文从"为什么需要多维并行"出发，逐步推导 rank 编号公式，然后用两个完整的 worked example（world=8 和 world=16）手工算出所有进程组，让你对 `parallel_state.py` 做到字面级理解。

---

## 2. 五种并行维度一览

| 并行类型 | 缩写 | 切分对象 | 通信原语 | 典型场景 |
|---|---|---|---|---|
| 张量并行 | TP | 权重矩阵（列/行） | AllReduce / ReduceScatter+AllGather | 单层 FFN / Attention |
| 流水线并行 | PP | 模型层（按 stage） | P2P send/recv | 超大模型 |
| 数据并行 | DP | 训练数据（mini-batch） | AllReduce / ReduceScatter | 标准多卡 |
| 上下文并行 | CP | 序列长度（tokens） | AllGather（KV） | 长上下文（>32k） |
| 专家并行 | EP | MoE expert 分片 | AlltoAll | MoE 模型 |

DP 的计算公式（不含 EP）为：

```
DP = world_size / (TP × PP × CP)
```

EP 不在这个公式里，因为 EP 复用 DP 分组（expert data parallel），并行度关系为：

```
expert_data_parallel_size = world_size / (expert_TP × EP × PP)
```

CP 复制权重：CP 内的 rank 持有完全相同的权重，不同的是它们处理不同的序列分片；因此 CP 组的梯度必须做 AllReduce，Megatron 把 CP 组"搭"在 DP 组上，合并为 `dp-cp` 联合组以方便 SHARP 优化（详见第 6 节）。

---

## 3. RankGenerator：rank 编号的唯一来源

```python
# megatron/core/parallel_state.py  第 446-521 行
class RankGenerator(object):
    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int,
        order: str, rank_offset: int = 0
    ) -> None:
        assert (
            ep == 1 or cp == 1
        ), "Both EP and CP > 1 is not allowed in one rank generator."
        ...
        self.world_size = tp * dp * pp * cp * ep
        self.order = order   # 默认 "tp-cp-ep-dp-pp"
```

**约束**：`ep==1 or cp==1`，即 EP 和 CP 不能同时大于 1。这来自底层通信的拓扑冲突：CP 需要跨序列块交换 KV，EP 需要跨 expert 做 AlltoAll，两者同时启用会让通信图难以调度。

### 3.1 order 的含义

`order="tp-cp-ep-dp-pp"` 是默认顺序，它决定了全局 rank 编号公式中各维度的步幅（stride）。公式如下（以 cp=ep=1 的简化情形为例）：

```
global_rank = tp_rank
           + dp_rank × TP
           + pp_rank × TP × DP
```

这与 C 语言中多维数组的"行优先"存储完全类似：最左边的维度（tp）变化最快，最右边的维度（pp）变化最慢。

### 3.2 get_ranks 的实现

```python
# megatron/core/parallel_state.py  第 505-521 行
def get_ranks(self, token):
    """token 示例: 'tp', 'dp', 'tp-dp', 'dp-cp' """
    mask = self.get_mask(self.order, token)
    ranks = generate_masked_orthogonal_rank_groups(
        self.world_size, self.ordered_size, mask
    )
    if self.rank_offset > 0:
        for rank_group in ranks:
            for i in range(len(rank_group)):
                rank_group[i] += self.rank_offset
    return ranks
```

`generate_masked_orthogonal_rank_groups` 是核心算法：给定一个 token 掩码，它枚举所有"固定非 token 维度、遍历 token 维度"的 rank 集合。

---

## 4. generate_masked_orthogonal_rank_groups 算法精读

```python
# megatron/core/parallel_state.py  第 250-356 行
def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
```

### 4.1 核心辅助函数

```python
def prefix_product(a, init=1):
    """计算前缀积，结果比输入长 1（第 0 项为 init）。
    例如: [2, 3, 4] → [1, 2, 6, 24]"""
    r = [init]
    for v in a:
        init = init * v
        r.append(init)
    return r

def decompose(index, shape, stride=None):
    """把标量 index 按 stride 分解为各维坐标。
    例如: index=5, shape=[2,3], stride=[1,2]
         → idx = [(5//1)%2, (5//2)%3] = [1, 2]"""
    if stride is None:
        stride = prefix_product(shape)
    idx = [(index // d) % s for s, d in zip(shape, stride)]
    return idx
```

### 4.2 小例子逐步执行

场景：world=8，order="tp-dp-pp"，parallel_size=[2,2,2]，token="dp"（mask=[False,True,False]）

```
masked_shape   = [2]    （仅 dp）
unmasked_shape = [2, 2] （tp, pp）
global_stride  = prefix_product([2,2,2]) = [1, 2, 4, 8]
masked_stride  = [2]    （dp 的步幅，即 tp 的 size=2）
unmasked_stride = [1, 4] （tp 和 pp 的步幅）

group_size = prefix_product([2])[-1] = 2
num_of_group = 8 // 2 = 4

group_index=0: decompose(0, [2,2]) → [0,0]  (tp_rank=0, pp_rank=0)
  rank_in_group=0: decompose(0, [2]) → [0]  (dp_rank=0)
    rank = 0*2 + 0*1 + 0*4 = 0
  rank_in_group=1: decompose(1, [2]) → [1]  (dp_rank=1)
    rank = 1*2 + 0*1 + 0*4 = 2
  → group[0] = [0, 2]  ✓ (tp=0, pp=0, dp=0/1)

group_index=1: decompose(1, [2,2]) → [1,0]  (tp_rank=1, pp_rank=0)
  → group[1] = [1, 3]  ✓ (tp=1, pp=0)

group_index=2: decompose(2, [2,2]) → [0,1]  (tp_rank=0, pp_rank=1)
  → group[2] = [4, 6]  ✓ (tp=0, pp=1)

group_index=3: decompose(3, [2,2]) → [1,1]  (tp_rank=1, pp_rank=1)
  → group[3] = [5, 7]  ✓ (tp=1, pp=1)
```

结果 `[[0,2],[1,3],[4,6],[5,7]]` 与第 4.2 节的 DP 组手工结果完全吻合。

---

## 5. 完整推导一：world=8，TP=2，PP=2，DP=2

参数：`world_size=8, TP=2, PP=2, DP=2, CP=1, EP=1`
顺序：`order = "tp-dp-pp"`（简化版，无 CP/EP）

### 5.1 rank 坐标映射表

按公式 `rank = tp_rank + dp_rank×2 + pp_rank×4`：

| global rank | tp_rank | dp_rank | pp_rank |
|:-----------:|:-------:|:-------:|:-------:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 1 | 0 |
| 3 | 1 | 1 | 0 |
| 4 | 0 | 0 | 1 |
| 5 | 1 | 0 | 1 |
| 6 | 0 | 1 | 1 |
| 7 | 1 | 1 | 1 |

### 5.2 TP 组（固定 dp_rank 和 pp_rank，遍历 tp_rank）

共 4 组，每组 2 个 rank：

```
TP groups（同一 PP stage、同一 DP replica 内共享激活）:
  [0, 1]   (dp=0, pp=0)
  [2, 3]   (dp=1, pp=0)
  [4, 5]   (dp=0, pp=1)
  [6, 7]   (dp=1, pp=1)
```

### 5.3 DP 组（固定 tp_rank 和 pp_rank，遍历 dp_rank）

共 4 组，每组 2 个 rank：

```
DP groups（相同 TP shard、相同 PP stage，跨 DP replica 同步梯度）:
  [0, 2]   (tp=0, pp=0)
  [1, 3]   (tp=1, pp=0)
  [4, 6]   (tp=0, pp=1)
  [5, 7]   (tp=1, pp=1)
```

### 5.4 PP 组（固定 tp_rank 和 dp_rank，遍历 pp_rank）

共 4 组，每组 2 个 rank：

```
PP groups（同一 pipeline，跨 stage 做 P2P 通信）:
  [0, 4]   (tp=0, dp=0)
  [1, 5]   (tp=1, dp=0)
  [2, 6]   (tp=0, dp=1)
  [3, 7]   (tp=1, dp=1)
```

---

## 6. 完整推导二：world=16，TP=4，PP=2，CP=1，EP=1

参数：`world_size=16, TP=4, PP=2, DP=2, CP=1, EP=1`
顺序：`order = "tp-dp-pp"`
公式：`rank = tp_rank + dp_rank×4 + pp_rank×8`

### 6.1 全量 rank 坐标表

| global rank | tp_rank | dp_rank | pp_rank |
|:-----------:|:-------:|:-------:|:-------:|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 2 | 2 | 0 | 0 |
| 3 | 3 | 0 | 0 |
| 4 | 0 | 1 | 0 |
| 5 | 1 | 1 | 0 |
| 6 | 2 | 1 | 0 |
| 7 | 3 | 1 | 0 |
| 8 | 0 | 0 | 1 |
| 9 | 1 | 0 | 1 |
| 10 | 2 | 0 | 1 |
| 11 | 3 | 0 | 1 |
| 12 | 0 | 1 | 1 |
| 13 | 1 | 1 | 1 |
| 14 | 2 | 1 | 1 |
| 15 | 3 | 1 | 1 |

### 6.2 所有 TP 组（固定 dp, pp，遍历 tp）

共 4 组，每组 4 个 rank：

```
TP groups:
  [0,  1,  2,  3]   (dp=0, pp=0)
  [4,  5,  6,  7]   (dp=1, pp=0)
  [8,  9, 10, 11]   (dp=0, pp=1)
  [12,13, 14, 15]   (dp=1, pp=1)
```

### 6.3 所有 DP 组（固定 tp, pp，遍历 dp）

共 8 组，每组 2 个 rank：

```
DP groups:
  [0,  4]   (tp=0, pp=0)
  [1,  5]   (tp=1, pp=0)
  [2,  6]   (tp=2, pp=0)
  [3,  7]   (tp=3, pp=0)
  [8,  12]  (tp=0, pp=1)
  [9,  13]  (tp=1, pp=1)
  [10, 14]  (tp=2, pp=1)
  [11, 15]  (tp=3, pp=1)
```

### 6.4 所有 PP 组（固定 tp, dp，遍历 pp）

共 8 组，每组 2 个 rank：

```
PP groups:
  [0,  8]   (tp=0, dp=0)
  [1,  9]   (tp=1, dp=0)
  [2,  10]  (tp=2, dp=0)
  [3,  11]  (tp=3, dp=0)
  [4,  12]  (tp=0, dp=1)
  [5,  13]  (tp=1, dp=1)
  [6,  14]  (tp=2, dp=1)
  [7,  15]  (tp=3, dp=1)
```

源码注释（第 683-697 行）与本结果完全吻合，可交叉验证。

---

## 7. Expert RankGenerator：EP>1 时的路径

当使用 MoE（`expert_model_parallel_size > 1`）时，Megatron 创建独立的 `expert_decoder_rank_generator`：

```python
# megatron/core/parallel_state.py  第 793-801 行
expert_decoder_rank_generator = RankGenerator(
    tp=expert_tensor_parallel_size,
    ep=expert_model_parallel_size,
    dp=expert_data_parallel_size,
    pp=pipeline_model_parallel_size,
    cp=1,           # EP 模式下 CP 强制为 1
    order=order,
    rank_offset=rank_offset,
)
```

### 7.1 expert_data_parallel_size 的计算

```python
# megatron/core/parallel_state.py  第 783-790 行
expert_tensor_model_pipeline_parallel_size = (
    expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
)
expert_data_parallel_size = world_size // expert_tensor_model_pipeline_parallel_size
```

公式：`expert_DP = world_size / (expert_TP × EP × PP)`

**注意** `expert_TP` 默认等于 `tensor_model_parallel_size`（非专家的 TP），但可以通过 `expert_tensor_parallel_size` 参数覆盖（例如 MoE 专家层用更小的 TP，以便用更多 GPU 做 EP）。

### 7.2 EP 约束：cp 必须为 1

RankGenerator 中有明确断言 `ep==1 or cp==1`。EP 组的 AlltoAll 通信需要在同一 PP stage 内的 EP rank 之间进行，而 CP 的 KV 交换也需要跨 rank 通信。两者同时启用会使通信拓扑图出现循环依赖，调度器无法安全地安排 kernel 执行顺序。因此代码强制：若 EP > 1，则 expert_decoder_rank_generator 中 cp=1。

### 7.3 EP vs 非 EP 组对比

| 属性 | 非专家（decoder_rank_generator） | 专家（expert_decoder_rank_generator） |
|------|------|------|
| CP | context_parallel_size（可 >1） | 强制 1 |
| EP | 强制 1 | expert_model_parallel_size |
| DP | world/(TP×PP×CP) | world/(expert_TP×EP×PP) |
| 通信原语 | AllReduce（DP）, AllGather（CP-KV） | AlltoAll（EP dispatch/combine） |

### 7.4 Assert：PP 组必须一致

```python
# megatron/core/parallel_state.py  第 809-812 行
assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp"), \
    "Pipeline parallel groups are expected to be the same for Non-Expert and Expert part"
```

这保证了 P2P 通信（PP 方向）在专家和非专家路径上使用同一套进程组，不会产生死锁。

---

## 8. initialize_model_parallel 中的建组顺序

建组顺序决定了哪个组可以用 NCCL COLLNET（SHARP）功能。NCCL 限制 SHARP 只能应用于第一个创建的通信子。Megatron 的解决方案是**把 dp-cp 组放在最前面**创建：

```python
# megatron/core/parallel_state.py  第 836-864 行
# Set NCCL_COLLNET_ENABLE to 1 to enable SHARP for the dp group.
if sharp_enabled_group == "dp":
    os.environ["NCCL_COLLNET_ENABLE"] = "1"

# dp-cp group 最先创建（为 SHARP 占位）
for ranks_with_cp in decoder_rank_generator.get_ranks('dp-cp'):
    group_with_cp = create_group(
        ranks_with_cp,
        timeout=timeout,
        pg_options=get_nccl_options("dp_cp", nccl_comm_cfgs),
        group_desc="DATA_PARALLEL_GROUP_WITH_CP",
    )
```

完整建组顺序如下：

| 步骤 | 组名 | 原因 |
|------|------|------|
| 1 | `dp-cp`（含 SHARP） | NCCL COLLNET 必须是第一个组 |
| 2 | `dp` | AllReduce 梯度（非 DistOpt 情况） |
| 3 | `cp` | 上下文并行 KV 交换 |
| 4 | `tp-pp`（model parallel） | 复合组，供 model parallel check 用 |
| 5 | `tp` | 张量并行 AllReduce/RS |
| 6 | `pp` + embedding + pos_embedding | P2P 通信 + embedding 对齐 |
| 7 | `tp-dp-cp`、`tp-dp` | FP8 及 MoE aux loss |
| 8 | Expert 相关组 | EP、expert DP、expert TP |
| 9 | Distributed Optimizer 分片组 | ReduceScatter sharding |

### 8.1 Gloo 组与 timeout 参数

`initialize_model_parallel` 同时为 DP 组创建 Gloo 后端进程组（`_DATA_PARALLEL_GROUP_GLOO`），用于需要 CPU 侧操作的场合（如 checkpoint save/load 时的广播）：

```python
if create_gloo_process_groups:
    group_with_cp_gloo = create_group(
        ranks_with_cp,
        timeout=timeout,
        backend="gloo",
        group_desc="DATA_PARALLEL_GROUP_WITH_CP_GLOO",
    )
```

`distributed_timeout_minutes`（默认 30 分钟）被转为 `timedelta` 传给所有组，控制集体通信的等待超时。若某个 rank 比其他 rank 慢超过 30 分钟，NCCL 会超时并抛出异常，避免整个作业无限 hang：

```python
# megatron/core/parallel_state.py  第 814 行
timeout = timedelta(minutes=distributed_timeout_minutes)
```

实践中 30 分钟对于大规模训练可能偏短（加载大型 checkpoint 可能超时），可以在启动时增大：

```python
initialize_model_parallel(..., distributed_timeout_minutes=120)
```

### 8.2 sharp_enabled_group 实用说明

`use_sharp=True` + `sharp_enabled_group="dp"` 是默认 SHARP 配置，让 DP AllReduce 走 SHARP 加速路径（需要 IB SHARP 硬件支持）。若设置 `sharp_enabled_group="dp_replica"`，则只有 `num_distributed_optimizer_instances > 1` 时的 intra_partial 组才用 SHARP，适合把 SHARP 资源集中给更频繁的小通信。

---

## 9. ProcessGroupCollection 重要字段

Megatron 0.9+ 引入了 `ProcessGroupCollection` 对象（`megatron/core/process_groups_config.py`），用于在 `finalize_model_grads`、pipeline schedule 等函数中显式传递进程组，而不再直接读取全局变量。

| 字段 | 类型 | 含义 |
|------|------|------|
| `tp` | `ProcessGroup` | 张量并行组 |
| `pp` | `ProcessGroup` | 流水线并行组 |
| `dp_cp` | `ProcessGroup` | DP+CP 联合组（梯度同步） |
| `embd` | `ProcessGroup` | embedding 对齐组（first+last stage） |
| `pos_embd` | `ProcessGroup` | 位置编码对齐组（first stage） |
| `cp` | `ProcessGroup` | 上下文并行组 |
| `tp_dp_cp` | `ProcessGroup` | TP+DP+CP 联合组（MoE expert bias） |

在 `schedules.py` 的 `forward_backward_pipelining_without_interleaving` 中，默认路径会从 `parallel_state` 读取并填充这个对象：

```python
# megatron/core/pipeline_parallel/schedules.py  第 2186-2197 行
pg_collection = ProcessGroupCollection()
pg_collection.tp    = parallel_state.get_tensor_model_parallel_group()
pg_collection.pp    = parallel_state.get_pipeline_model_parallel_group()
pg_collection.embd  = parallel_state.get_embedding_group(check_initialized=False)
pg_collection.dp_cp = parallel_state.get_data_parallel_group(
    with_context_parallel=True, partial_data_parallel=False
)
```

---

## 10. Embedding ranks 与 get_embedding_ranks

Word embedding 权重在 Megatron 里被设计成**首末两个 PP stage 共享**：stage 0 持有 input embedding，stage PP-1 持有 lm_head（output projection），两者指向同一份参数（`share_embeddings_and_output_weights=True`）。为了让梯度在这两处保持同步，需要建立一个 `embedding group`，它只包含 stage 0 和 stage PP-1 对应的 rank。

```python
# megatron/core/parallel_state.py  第 524-530 行
def default_embedding_ranks(pp_ranks):
    """Return the default ranks for word embeddings.
    For most models, these are the first and last pipeline stages."""
    if len(pp_ranks) == 1:
        return [pp_ranks[0]]
    else:
        return [pp_ranks[0], pp_ranks[-1]]
```

调用者可以通过 `get_embedding_ranks` 回调函数覆盖这个默认行为（例如 Diffusion 模型可能有不同的 embedding 分布策略）。位置编码（`position_embeddings`）则只在 stage 0：

```python
def default_position_embedding_ranks(pp_ranks):
    return [pp_ranks[0]]
```

---

## 11. VPP（Virtual Pipeline）约束

当启用 VPP（`virtual_pipeline_model_parallel_size > 1`）时，有以下约束：

1. `PP > 1` 是前提（源码第 740-744 行）。
2. `num_layers` 必须被 `PP × VPP` 整除；否则无法均匀切分。
3. VPP 下模型对象是一个 **list**，`model[v]` 对应第 v 个 virtual stage；每次 forward/backward 都按照 interleaved schedule 顺序调用不同的 `model[v]`。

```python
# megatron/core/parallel_state.py  第 740-748 行
if virtual_pipeline_model_parallel_size is not None:
    if not pipeline_model_parallel_size > 1:
        raise RuntimeError(
            "pipeline-model-parallel size should be greater than 1 "
            "with interleaved schedule"
        )
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size
```

VPP 把 PP stage 从 `[0..PP-1]` 扩展到 `[0..PP×V-1]`，每个物理 GPU 持有 V 个 virtual stage。以 16 层模型、PP=4、V=2 为例：

```
GPU 0: [layers 1-2]  [layers 9-10]
GPU 1: [layers 3-4]  [layers 11-12]
GPU 2: [layers 5-6]  [layers 13-14]
GPU 3: [layers 7-8]  [layers 15-16]
```

---

## 12. 常见 hang：world_size 乘积不整除

### 12.1 典型错误场景

用 12 张 GPU 跑 TP=4, PP=4（乘积=16）：

```
RuntimeError: world_size (12) is not divisible by 16
```

源码检查位置（第 735-736 行）：

```python
model_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
if world_size % model_size != 0:
    raise RuntimeError(f"world_size ({world_size}) is not divisible by {model_size}")
```

### 12.2 更隐蔽的 hang：进程组建立后挂起

若 `world_size` 整除成功但某个 rank 在建组时崩溃退出，其他 rank 会在 `create_group`（内部调用 `dist.new_group`）上永久等待，因为 NCCL 需要所有 rank 同时调用。这种 hang **没有错误信息**。

排查步骤：
1. 在 `initialize_model_parallel` 调用前添加 `print(f"rank {rank} about to init")`，确认所有 rank 都到达此处。
2. 在 `initialize_model_parallel` 调用后添加 `dist.barrier()`，确认所有 rank 都成功初始化进程组。
3. 检查日志中是否有某个 rank 的 CUDA OOM（GPU 内存不足时 NCCL group init 会静默失败）。

---

## 13. 如何用 print 验证进程组

在实际调试时，最直接的方式是在 `initialize_model_parallel` 之后立刻打印每个 rank 的坐标：

```python
import megatron.core.parallel_state as ps

# 确保在 initialize_model_parallel 之后调用
tp_rank  = ps.get_tensor_model_parallel_rank()
tp_size  = ps.get_tensor_model_parallel_world_size()
pp_rank  = ps.get_pipeline_model_parallel_rank()
pp_size  = ps.get_pipeline_model_parallel_world_size()
dp_rank  = ps.get_data_parallel_rank()
dp_size  = ps.get_data_parallel_world_size()
cp_rank  = ps.get_context_parallel_rank()
cp_size  = ps.get_context_parallel_world_size()

print(
    f"[rank {torch.distributed.get_rank()}] "
    f"tp={tp_rank}/{tp_size} "
    f"pp={pp_rank}/{pp_size} "
    f"dp={dp_rank}/{dp_size} "
    f"cp={cp_rank}/{cp_size}"
)
```

典型输出（world=8，TP=2，PP=2，DP=2）：

```
[rank 0] tp=0/2 pp=0/2 dp=0/2 cp=0/1
[rank 1] tp=1/2 pp=0/2 dp=0/2 cp=0/1
[rank 2] tp=0/2 pp=0/2 dp=1/2 cp=0/1
[rank 3] tp=1/2 pp=0/2 dp=1/2 cp=0/1
[rank 4] tp=0/2 pp=1/2 dp=0/2 cp=0/1
[rank 5] tp=1/2 pp=1/2 dp=0/2 cp=0/1
[rank 6] tp=0/2 pp=1/2 dp=1/2 cp=0/1
[rank 7] tp=1/2 pp=1/2 dp=1/2 cp=0/1
```

与第 5 节的坐标映射表完全一致。

还可以验证组成员：

```python
import torch.distributed as dist

# 打印当前 rank 的 TP 组成员
tp_group = ps.get_tensor_model_parallel_group()
tp_ranks = dist.get_process_group_ranks(tp_group)
print(f"[rank {dist.get_rank()}] TP group ranks: {tp_ranks}")

# 打印当前 rank 的 DP 组成员
dp_group = ps.get_data_parallel_group()
dp_ranks = dist.get_process_group_ranks(dp_group)
print(f"[rank {dist.get_rank()}] DP group ranks: {dp_ranks}")
```

---

## 14. NCCL 配置与 nccl_communicator_config_path

`initialize_model_parallel` 接受一个 `nccl_communicator_config_path` 参数，允许为不同的进程组指定不同的 NCCL 参数（`max_ctas`、`cga_cluster_size`、`min_ctas`、`net_name`）：

```yaml
# 示例 nccl_config.yaml
dp:
  max_ctas: 8
  cga_cluster_size: 2
tp:
  is_high_priority_stream: true
  max_ctas: 32
```

这对于在同一台机器上同时运行 TP 通信（InfiniBand，高带宽低延迟）和 DP 通信（Ethernet，较低带宽）的场景尤其重要：可以给 TP 组分配更多的 CUDA CTA 资源，保证 TP AllReduce 的延迟最小化。

`high_priority_stream_groups` 参数则允许指定哪些通信组使用 CUDA 高优先级流：

```python
initialize_model_parallel(
    ...,
    high_priority_stream_groups=["dp_cp", "tp"],
)
```

源码处理方式（第 765-768 行）：

```python
high_priority_stream_groups = high_priority_stream_groups or []
for pg_name in high_priority_stream_groups:
    overwrite_nccl_comm_cfgs(nccl_comm_cfgs, pg_name, ("is_high_priority_stream", True))
```

即把 `is_high_priority_stream: True` 注入到对应通信组的 NCCL 配置里，由 `get_nccl_options` 在建组时传给 NCCL。

---

## 15. Distributed Optimizer 与分片进程组

当启用 Distributed Optimizer（`use_distributed_optimizer=True`）时，Megatron 会把 DP 组的 AllReduce 替换为 ReduceScatter + AllGather，每个 rank 只持有 1/DP 的优化器状态，内存节省 ~3×。

为了支持多个 Distributed Optimizer 实例（`num_distributed_optimizer_instances > 1`），`parallel_state.py` 还会创建 `intra_partial_dp_group`，它是 `dp-cp` 组的一个子集：

```python
# megatron/core/parallel_state.py  第 866-893 行
if num_distributed_optimizer_instances > 1:
    for i in range(num_distributed_optimizer_instances):
        intra_partial_dp_ranks_with_cp = ranks_with_cp[
            i * intra_partial_data_parallel_size :
            (i + 1) * intra_partial_data_parallel_size
        ]
        intra_partial_dp_group_with_cp = create_group(
            intra_partial_dp_ranks_with_cp, ...
        )
```

这个特性用于在超大规模集群上把 ReduceScatter 的通信量进一步分片，减少单次通信的 rank 数量。

---

## 16. 全局变量生命周期与 destroy_model_parallel

`parallel_state.py` 中所有的 `_TENSOR_MODEL_PARALLEL_GROUP`、`_DATA_PARALLEL_GROUP` 等全局变量在 `initialize_model_parallel` 调用时设置，在 `destroy_model_parallel` 调用时清空。测试代码必须在每个测试用例之间正确调用 `destroy_model_parallel`，否则第二次 `initialize_model_parallel` 会触发 `assert ... is None` 的断言错误：

```python
# megatron/core/parallel_state.py  第 825 行
assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
```

标准的测试框架用法：

```python
def setup():
    torch.distributed.init_process_group(backend="nccl", ...)
    ps.initialize_model_parallel(tp_size, pp_size)

def teardown():
    ps.destroy_model_parallel()
    torch.distributed.destroy_process_group()
```

---

## 17. FAQ（进程组常见问题 12 条）

**Q1：为什么 TP 通信要用 NCCL 而 PP 通信有时用 UCC？**

TP AllReduce 消息大（整个激活 tensor），NCCL 环形 AllReduce 对大消息效率高。PP P2P 是点对点通信，UCC 的 zero-SM 特性（不占用 GPU SM，避免与 GEMM 争抢计算资源）使其在 PP stage 重叠 compute+comm 时更优。

**Q2：`get_data_parallel_group(with_context_parallel=True)` 和 `with_context_parallel=False` 返回什么不同？**

`with_context_parallel=True` 返回 `_DATA_PARALLEL_GROUP_WITH_CP`（大小=DP×CP），`False` 返回 `_DATA_PARALLEL_GROUP`（大小=DP）。前者用于梯度 AllReduce（需要覆盖 CP 副本），后者用于 Distributed Optimizer 的分片计算。

**Q3：TP=1 时 `sequence_parallel` 会自动关闭吗？**

是。`ColumnParallelLinear.__init__` 中有：
```python
if self.sequence_parallel and world_size <= 1:
    warnings.warn("... Disabling sequence parallel.")
    self.sequence_parallel = False
```
TP=1 时自动降级为非 SP 模式，不会报错。

**Q4：两次调用 `initialize_model_parallel` 会发生什么？**

第二次调用会触发 `AssertionError: data parallel group is already initialized`。必须先调用 `destroy_model_parallel()`。

**Q5：如何查看某个 rank 属于哪个 embedding group？**

```python
embd_group = ps.get_embedding_group(check_initialized=False)
if embd_group is not None:
    embd_ranks = dist.get_process_group_ranks(embd_group)
    print(f"[rank {rank}] embedding group: {embd_ranks}")
```
若返回 None，说明当前 rank 不在 embedding group 中（非 first/last stage）。

**Q6：VPP=2、PP=4 时总共有多少个 virtual stage？**

每个物理 GPU 有 V=2 个 virtual stage，PP=4 个物理 GPU，所以流水线共有 PP×V=8 个 virtual stage。每个 microbatch 在 forward 时依次经过所有 8 个 virtual stage。

**Q7：`order="tp-cp-ep-dp-pp"` 中 ep 和 cp 都出现了，但不是说不能同时 >1 吗？**

`order` 字符串是通用格式，支持描述所有维度。在实际建组时，`decoder_rank_generator` 固定 `ep=1`，`expert_decoder_rank_generator` 固定 `cp=1`。两个 generator 各自满足 `ep==1 or cp==1` 约束，互不影响。

**Q8：Distributed Optimizer 用的是哪个进程组做 ReduceScatter？**

用 `intra_partial_dp_group`（当 `num_distributed_optimizer_instances=1` 时等同于 `dp_cp_group`）。ReduceScatter 把梯度分散到组内每个 rank，每个 rank 持有 1/(DP×CP) 的梯度分片。

**Q9：PP 通信为什么不需要 Gloo 备用组？**

PP 通信是 P2P（send/recv），不是集体通信，不需要 Gloo 备选路径。Gloo 组主要用于需要 CPU 侧广播/归约的场景（如模型参数初始化广播、checkpoint 同步等），而这些操作通常只发生在 DP 组内。

**Q10：world_size 能否动态变化（弹性训练）？**

Megatron 原生不支持弹性 world_size。所有进程组在 `initialize_model_parallel` 时静态构建，rank 数量固定。弹性训练需要外部框架（如 PyTorch Elastic）在故障后重新启动所有进程并重新调用 `initialize_model_parallel`。

**Q11：`_EMBEDDING_GROUP` 和 `_POSITION_EMBEDDING_GROUP` 有什么区别？**

`_EMBEDDING_GROUP` 连接 first stage 和 last stage，用于同步 word embedding 梯度（当两处共享同一权重时）。`_POSITION_EMBEDDING_GROUP` 只包含 first stage 的 rank，用于同步位置编码权重（只有 first stage 需要位置编码，last stage 的 lm_head 不需要）。

**Q12：`parallel_state` 中的全局变量是进程级别的还是线程级别的？**

进程级别。每个 PyTorch 进程（即每块 GPU）有独立的全局变量副本。多线程（如 DataLoader worker）不会干扰这些变量，因为 NCCL 通信发生在主进程的 CUDA context 里。

---

## 18. 一次完整的进程组创建日志分析

使用 `NCCL_DEBUG=INFO` 运行时，每个进程组的建立都会输出一行日志，格式大致为：

```
[rank 0] NCCL INFO comm 0x... rank 0 nranks 2 cudaDev 0 busId ...  - Init COMPLETE
```

对于 world=8 TP=2 PP=2 DP=2 的配置，rank 0 会看到 9 条 Init COMPLETE（dp-cp/dp/cp/tp-pp/tp/pp/tp-dp-cp/tp-dp 各一条，加上 Gloo 备用组），rank 7 也是同样数量。

**如何确认建组顺序**：grep Init COMPLETE 并按时间戳排序，第一条必然是 dp-cp 组（对应 SHARP 占位）：

```bash
NCCL_DEBUG=INFO python train.py 2>&1 | grep "Init COMPLETE" | head -20
```

如果某个组的 Init COMPLETE 日志缺失，说明该组的建立 hang 了，应检查是否所有 rank 都调用了 `initialize_model_parallel`，以及是否有 rank 提前 OOM 退出。

---

## 19. 如何在单机 mock 环境中单元测试进程组逻辑

`parallel_state.py` 的所有建组逻辑都可以在 CPU 上用 `gloo` 后端模拟，无需真实 GPU，适合单元测试：

```python
import os
import torch
import torch.distributed as dist
import megatron.core.parallel_state as ps

def test_rank_groups_world8():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    # 模拟 rank 0，world_size=8，用 gloo 后端（CPU）
    dist.init_process_group(
        backend="gloo", rank=0, world_size=8,
        init_method="env://",
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            create_gloo_process_groups=False,
        )
        tp_rank = ps.get_tensor_model_parallel_rank()
        dp_rank = ps.get_data_parallel_rank()
        pp_rank = ps.get_pipeline_model_parallel_rank()
        assert tp_rank == 0
        assert dp_rank == 0
        assert pp_rank == 0
    finally:
        ps.destroy_model_parallel()
        dist.destroy_process_group()
```

注意：单机测试时需要用 `multiprocessing` 启动 8 个进程，每个进程有不同的 `rank`。Megatron 的测试套件在 `tests/unit_tests/dist_checkpointing/` 等目录下有大量此类范例。

---

## 18. "90 分钟精读 parallel_state.py" 导读清单

以下是一个有序的阅读路径，估计总时间 90 分钟：

**第 1-15 分钟：全局变量声明（1-120 行）**
- 快速浏览所有 `_XXX_GROUP = None` 声明，在脑海中建立"有哪些组"的清单
- 注意 Expert 相关变量的命名规律（`_EXPERT_MODEL_`、`_EXPERT_TENSOR_`、`_EXPERT_DATA_`）

**第 15-30 分钟：generate_masked_orthogonal_rank_groups（250-356 行）**
- 重点理解 `prefix_product`、`decompose`、`inner_product` 三个辅助函数
- 手工执行一遍 world=8、order="tp-dp-pp" 的例子（本文第 4.2 节）

**第 30-45 分钟：RankGenerator 类（446-521 行）**
- 理解 `__init__` 中 order 字符串的规范化流程（未出现的维度追加到末尾）
- 跟踪 `get_mask` 和 `get_ranks` 的调用链

**第 45-60 分钟：initialize_model_parallel 签名和前半部分（547-870 行）**
- 关注参数列表：哪些参数直接影响建组结果
- 重点看 SHARP 相关代码（836-864 行）和 dp-cp 组的创建

**第 60-75 分钟：initialize_model_parallel 后半部分（870-1200 行）**
- 依次看 tp、pp、embedding、expert 组的建组代码
- 找到每个 `if rank in ranks:` 赋值语句——这就是当前 rank 的组归属

**第 75-90 分钟：get/is 系列 API（1200 行后）**
- 快速浏览所有 `get_xxx_group()`、`get_xxx_rank()`、`get_xxx_world_size()` 函数
- 理解 `_MPU_xxx_OVERRIDE` 机制：可以在运行时临时覆盖 rank/size（用于测试）

**完成后做本文第 19 节的练习题作为自测。**

---

## 19. 实战：手动触发进程组错误并定位

以下是典型的进程组配置错误场景和诊断方法。

**场景 A**：用户误将 `tensor_model_parallel_size=4` 和 `pipeline_model_parallel_size=4` 用于 8 卡机器：

```
RuntimeError: world_size (8) is not divisible by 16
```

原因：`model_size = 4×4 = 16`，但 world_size=8，第 735-736 行抛出。

**场景 B**：用户启用了 `sequence_parallel=True` 但 TP=1（TP=1 时 SP 自动禁用，不报错但有 warning）：

```python
# megatron/core/tensor_parallel/layers.py  第 952-958 行
self.sequence_parallel = config.sequence_parallel
if self.sequence_parallel and world_size <= 1:
    warnings.warn(
        "`sequence_parallel` is set to `True`, but tensor model parallel size "
        f"is {world_size}. Disabling sequence parallel."
    )
    self.sequence_parallel = False
```

**场景 C**：进程组创建后 NCCL 通信 hang。排查步骤：

```bash
# 步骤 1：确认每个 rank 的进程组成员一致
NCCL_DEBUG=INFO python train.py 2>&1 | grep "NCCL INFO"

# 步骤 2：在 initialize_model_parallel 后立即做一次 barrier
torch.distributed.barrier()
torch.cuda.synchronize()
print(f"rank {rank} passed barrier")  # 如果 hang，说明组构造有问题

# 步骤 3：用 NCCL_DEBUG=WARN 查看是否有 "Timeout" 关键字
NCCL_DEBUG=WARN NCCL_DEBUG_SUBSYS=ALL python train.py 2>&1 | grep -i timeout
```

**场景 D**：CP>1 且 EP>1 同时设置，触发 `RankGenerator` 断言：

```
AssertionError: Both EP and CP > 1 in not allow in one rank generator. ...
```

修复方法：若模型同时需要 CP 和 MoE，必须确保 `context_parallel_size=1`（禁用 CP）或 `expert_model_parallel_size=1`（禁用 EP），两者不能共存。

---

## 20. 练习题

**题目 1**：world=16，TP=4，PP=2，DP=2，order="tp-dp-pp"。

1. 请写出所有 global rank 的 (tp, dp, pp) 坐标（参考第 6 节）。
2. 列出所有 DP 组（共 8 组）。
3. 列出所有 PP 组（共 8 组）。

**题目 2**：若在 world=16 的基础上加入 CP=2，即 TP=4，PP=2，CP=2，DP=1，请计算：

1. 新的 `dp-cp` 组的成员（有多少组，每组多少个 rank）。
2. `dp` 组和 `cp` 组分别是什么？

**题目 3**：阅读 `generate_masked_orthogonal_rank_groups`（第 250-356 行），用 Python 手动验证 world=8，TP=2，PP=2，DP=2，order="tp-dp-pp" 时 `get_ranks('dp')` 的输出（参考第 4 节的逐步执行）。

**题目 4**：设 world=32，expert_TP=2，EP=4，PP=2。计算 `expert_data_parallel_size`。若 world_size 除不尽，会触发哪行断言？

**题目 5**：参考本文第 4.2 节（generate_masked_orthogonal_rank_groups 逐步执行），对 world=8、order="tp-dp-pp"、mask=[True,False,True]（即 tp-pp 联合组）手工计算输出。每组应包含哪些 rank？

**题目 6**：在 TP=2、PP=2、DP=2（world=8）的配置下，若使用 `nccl_communicator_config_path` 给 `tp` 组设置 `is_high_priority_stream=true`，哪些 rank 的 CUDA 流会被设为高优先级？（提示：所有 rank 都属于某个 TP 组。）

---

## 20. 小结

`parallel_state.py` 的核心可以用三句话总结：

1. **`RankGenerator`** 把 `(TP, DP, PP, CP, EP)` 映射为全局 rank 编号，顺序由 `order` 字符串决定，默认 `"tp-cp-ep-dp-pp"`，约束 `ep==1 or cp==1`。EP>1 时创建独立的 `expert_decoder_rank_generator`（cp 强制为 1），expert_DP 按 `world/(expert_TP×EP×PP)` 计算。
2. **`initialize_model_parallel`** 按固定顺序（dp-cp 优先）构造所有进程组，确保 SHARP 等硬件特性落在正确的通信子上；建组总顺序：dp-cp → dp → cp → tp-pp → tp → pp → tp-dp-cp → 专家组 → DistOpt 分片组。Gloo 备用组与 NCCL 组并行创建，供 CPU 侧操作使用；`distributed_timeout_minutes` 控制所有组的超时上限。
3. **全局变量 + `get_*_group()` API** 让代码的任意位置都能低开销地拿到所属进程组，但新 `megatron/core` 生产代码应优先使用 `ProcessGroupCollection` 显式传递进程组，避免在库代码中隐式依赖全局状态。

---

## 22. 附录：parallel_state.py 全局变量速查表

以下是 `parallel_state.py` 中最常用的全局变量及其含义，供快速查阅：

| 变量名 | 类型 | 含义 | 访问函数 |
|--------|------|------|---------|
| `_TENSOR_MODEL_PARALLEL_GROUP` | ProcessGroup | TP 组 | `get_tensor_model_parallel_group()` |
| `_PIPELINE_MODEL_PARALLEL_GROUP` | ProcessGroup | PP 组 | `get_pipeline_model_parallel_group()` |
| `_DATA_PARALLEL_GROUP` | ProcessGroup | DP 组（不含CP） | `get_data_parallel_group()` |
| `_DATA_PARALLEL_GROUP_WITH_CP` | ProcessGroup | DP+CP 联合组 | `get_data_parallel_group(with_context_parallel=True)` |
| `_CONTEXT_PARALLEL_GROUP` | ProcessGroup | CP 组 | `get_context_parallel_group()` |
| `_EMBEDDING_GROUP` | ProcessGroup | first+last stage embedding 组 | `get_embedding_group()` |
| `_POSITION_EMBEDDING_GROUP` | ProcessGroup | first stage 位置编码组 | `get_position_embedding_group()` |
| `_EXPERT_MODEL_PARALLEL_GROUP` | ProcessGroup | EP 组 | `get_expert_model_parallel_group()` |
| `_EXPERT_DATA_PARALLEL_GROUP` | ProcessGroup | Expert DP 组 | `get_expert_data_parallel_group()` |
| `_TENSOR_AND_DATA_PARALLEL_GROUP` | ProcessGroup | TP+DP 联合组（FP8/MoE） | `get_tensor_and_data_parallel_group()` |
| `_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK` | int | 当前 virtual stage 编号 | `get_virtual_pipeline_model_parallel_rank()` |
| `_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE` | int | VPP 数量 | `get_virtual_pipeline_model_parallel_world_size()` |

下一篇（05）将在进程组的基础上，深入剖析张量并行的通信原语（mappings.py）和 `ColumnParallelLinear` / `RowParallelLinear` 的完整实现。
