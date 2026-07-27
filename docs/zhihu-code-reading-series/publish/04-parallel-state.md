# 精读 Megatron 源码（4）：parallel_state 完全指南——进程组如何把 GPU 编成网格

> 源文件：`megatron/core/parallel_state.py`（2249 行）

---

## 1. 为什么进程组是 Megatron 的"骨架"

每块 GPU 都是一个独立的 CUDA 设备；要让多块 GPU 协同训练一个模型，必须先定义"谁和谁通信"。PyTorch 通过 `torch.distributed.ProcessGroup` 对象来描述一组需要互相通信的进程。进程组一旦错误——比如把本应做 DP AllReduce 的 rank 混进了 TP 组——梯度就会静默出错，模型收敛曲线毫无预兆地走歪。`parallel_state.py` 就是 Megatron 所有通信结构的"出生证明"：它在训练开始时构造出所有必要的进程组，并在整个训练生命周期内以全局变量的形式维护这些组的引用。

本文从"为什么需要多维并行"出发，逐步推导 rank 编号公式，最后把 world=8 TP=2 PP=2 DP=2 的完整进程组表格"手工算"一遍，让你对 `parallel_state.py` 做到字面级理解。

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

CP 复制权重：CP 内的 rank 持有完全相同的权重，不同的是它们处理不同的序列分片；因此 CP 组的梯度必须做 AllReduce，Megatron 把 CP 组"搭"在 DP 组上，合并为 `dp-cp` 联合组以方便 SHARP 优化（详见第 5 节）。

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

## 4. 完整推导：world=8，TP=2，PP=2，DP=2

参数：`world_size=8, TP=2, PP=2, DP=2, CP=1, EP=1`  
顺序：`order = "tp-dp-pp"`（简化版，无 CP/EP）

### 4.1 rank 坐标映射表

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

### 4.2 TP 组（固定 dp_rank 和 pp_rank，遍历 tp_rank）

共 4 组，每组 2 个 rank：

```
TP groups（同一 PP stage、同一 DP replica 内共享激活）:
  [0, 1]   (dp=0, pp=0)
  [2, 3]   (dp=1, pp=0)
  [4, 5]   (dp=0, pp=1)
  [6, 7]   (dp=1, pp=1)
```

### 4.3 DP 组（固定 tp_rank 和 pp_rank，遍历 dp_rank）

共 4 组，每组 2 个 rank：

```
DP groups（相同 TP shard、相同 PP stage，跨 DP replica 同步梯度）:
  [0, 2]   (tp=0, pp=0)
  [1, 3]   (tp=1, pp=0)
  [4, 6]   (tp=0, pp=1)
  [5, 7]   (tp=1, pp=1)
```

### 4.4 PP 组（固定 tp_rank 和 dp_rank，遍历 pp_rank）

共 4 组，每组 2 个 rank：

```
PP groups（同一 pipeline，跨 stage 做 P2P 通信）:
  [0, 4]   (tp=0, dp=0)
  [1, 5]   (tp=1, dp=0)
  [2, 6]   (tp=0, dp=1)
  [3, 7]   (tp=1, dp=1)
```

### 4.5 源码注释印证

`parallel_state.py` 第 683-697 行用 16 卡 TP=2 PP=4 的例子给出了类似的推导，可以作为交叉验证：

```python
# Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
# use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
# the model pipeline. The present function will create 8 tensor
# model-parallel groups, 4 pipeline model-parallel groups and 8 data-
# parallel groups as:
#     8 data_parallel groups:
#         [g0, g2], [g1, g3], [g4, g6], ...
#     8 tensor model-parallel groups:
#         [g0, g1], [g2, g3], [g4, g5], ...
#     4 pipeline model-parallel groups:
#         [g0, g4, g8, g12], [g1, g5, g9, g13], ...
```

---

## 5. initialize_model_parallel 中的建组顺序

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

---

## 6. ProcessGroupCollection 重要字段

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

## 7. Embedding ranks 与 get_embedding_ranks

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

## 8. VPP（Virtual Pipeline）约束

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

## 9. 如何用 print 验证进程组

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

与第 4 节的坐标映射表完全一致。

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

## 10. 进阶细节：专家并行（EP）组的构造

当使用 MoE 时，Megatron 会创建一个独立的 `expert_decoder_rank_generator`，其中 `ep > 1, cp = 1`：

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

EP 组的 rank 覆盖范围：在同一个 PP stage 内，固定 TP 坐标和 DP 坐标，遍历 EP 维度。Expert DP 则是：固定 EP 坐标，遍历 DP 维度。这意味着 EP 下的 AlltoAll 通信只在同一 PP stage 的不同 expert rank 之间进行，与 PP 的 P2P 通信互不干扰。

---

## 11. 练习题

**题目 1**：world=16，TP=4，PP=2，DP=2，order="tp-dp-pp"。

1. 请写出所有 global rank 的 (tp, dp, pp) 坐标。
2. 列出所有 TP 组（共 4 组）。
3. 列出所有 DP 组（共 8 组）。
4. 列出所有 PP 组（共 8 组）。

提示：按公式 `rank = tp + dp×4 + pp×8`。

**题目 2**：若在 world=16 的基础上加入 CP=2，即 TP=4，PP=2，CP=2，DP=1，请计算：

1. 新的 `dp-cp` 组的成员（有多少组，每组多少个 rank）。
2. `dp` 组和 `cp` 组分别是什么？

**题目 3**：阅读 `generate_masked_orthogonal_rank_groups`（第 250-356 行），用 Python 手动验证 world=8，TP=2，PP=2，DP=2，order="tp-dp-pp" 时 `get_ranks('dp')` 的输出。

---

## 12. 全局变量生命周期与 destroy_model_parallel

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

## 13. NCCL 配置与 nccl_communicator_config_path

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

---

## 14. Distributed Optimizer 与分片进程组

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

## 15. 实战：手动触发进程组错误并定位

以下是一个典型的进程组配置错误场景和诊断方法：

**场景**：用户误将 `tensor_model_parallel_size=4` 和 `pipeline_model_parallel_size=4` 用于 8 卡机器，导致 `world_size (8) < TP×PP (16)`。

```
RuntimeError: world_size (8) is not divisible by 16
```

**场景**：用户启用了 `sequence_parallel=True` 但 TP=1。

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

**场景**：进程组创建后 NCCL 通信 hang。排查步骤：

```bash
# 步骤 1：确认每个 rank 的进程组成员一致
NCCL_DEBUG=INFO python train.py 2>&1 | grep "NCCL INFO"

# 步骤 2：在 initialize_model_parallel 后立即做一次 barrier
torch.distributed.barrier()
torch.cuda.synchronize()
print(f"rank {rank} passed barrier")  # 如果 hang，说明组构造有问题

# 步骤 3：检查 _global_process_group_list 的长度
# 在测试代码中暴露这个私有变量
```

---

## 16. 小结

`parallel_state.py` 的核心可以用三句话总结：

1. **`RankGenerator`** 把 `(TP, DP, PP, CP, EP)` 映射为全局 rank 编号，顺序由 `order` 字符串决定，默认 `"tp-cp-ep-dp-pp"`，约束 `ep==1 or cp==1`。
2. **`initialize_model_parallel`** 按固定顺序（dp-cp 优先）构造所有进程组，确保 SHARP 等硬件特性落在正确的通信子上；建组总顺序：dp-cp → dp → cp → tp-pp → tp → pp → tp-dp-cp → 专家组 → DistOpt 分片组。
3. **全局变量 + `get_*_group()` API** 让代码的任意位置都能低开销地拿到所属进程组，但新 `megatron/core` 生产代码应优先使用 `ProcessGroupCollection` 显式传递进程组，避免在库代码中隐式依赖全局状态。

下一篇（05）将在进程组的基础上，深入剖析张量并行的通信原语（mappings.py）和 `ColumnParallelLinear` / `RowParallelLinear` 的完整实现。
