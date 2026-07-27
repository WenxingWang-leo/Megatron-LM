# 精读 Megatron 源码（5）：张量并行完全精读——mappings、Column/Row 与 Attention/MLP

> 源文件：`megatron/core/tensor_parallel/mappings.py`、`layers.py`

---

## 1. 张量并行的核心思想

张量并行（Tensor Parallelism，TP）把单个权重矩阵沿某个维度切开，分散到多个 GPU 上，每个 GPU 只做部分 GEMM，最后用通信原语把结果拼回来。它的优势是显存占用随 TP 线性下降，代价是每层都需要额外的集体通信。

最重要的数学事实：矩阵乘法 `Y = XA` 有两种等价切分方式：

- **按列切分 A**（Column Parallel）：`Y = X [A_1 | A_2 | ... | A_p]` → 每个 GPU 算 `Y_i = X A_i`，结果需要 AllGather。
- **按行切分 A**（Row Parallel）：`Y = [X_1, X_2, ..., X_p] [A_1; A_2; ...; A_p]^T` → 每个 GPU 算 `Y_i = X_i A_i^T`，结果需要 AllReduce。

Megatron 的设计是**把 ColumnParallelLinear 和 RowParallelLinear 串联**，中间无需 AllGather/AllReduce，从而把通信次数从每层 2 次降到每层 1 次。

---

## 2. Y=XA 切分图解（ASCII）

### 2.1 Column Parallel：A 沿列切分

```
输入 X: [B, S, H]    权重 A: [H, 4H]

           ┌─ A_0 [H, H] ─┐
X ──── TP  ├─ A_1 [H, H] ─┤  ──→  [Y_0 | Y_1 | ... | Y_p]
           └─ A_p [H, H] ─┘

每个 GPU 算: Y_i = X @ A_i,   shape: [B, S, H]
全局结果:  Y = [Y_0 | Y_1 ... Y_p],  shape: [B, S, 4H]

通信: gather_output=True 时做 AllGather，否则保持分片输出
```

### 2.2 Row Parallel：A 沿行切分（兼 X 沿列切分）

```
输入 X (已分片): [B, S, H]   权重 A_i: [H, 4H/p]

GPU_i 算: Y_i = X_i @ A_i,   shape: [B, S, 4H]
                                      ↓
                               AllReduce（或 ReduceScatter）
                                      ↓
                               Y = Σ Y_i,  shape: [B, S, 4H]
```

关键：Row Parallel 的输入 `X` 必须已经是分片的（即 Column Parallel 的输出），这就是为什么 Column + Row 串联后只需要 **一次** 通信而不是两次。

---

## 3. mappings.py：六种通信原语

```
文件: megatron/core/tensor_parallel/mappings.py
```

| 函数名 | 前向传播 | 反向传播 | 用途 |
|--------|---------|---------|------|
| `copy_to_tensor_model_parallel_region` (CopyTo) | 恒等（广播） | AllReduce | Column Parallel 非 SP 情况下的输入 |
| `reduce_from_tensor_model_parallel_region` (ReduceFrom) | AllReduce | 恒等 | Row Parallel 非 SP 情况下的输出 |
| `scatter_to_tensor_model_parallel_region` | 沿最后维切片 | 沿最后维 AllGather | Vocabulary 并行输出 |
| `gather_from_tensor_model_parallel_region` | 沿最后维 AllGather | 沿最后维切片 | Vocab 并行输入 |
| `scatter_to_sequence_parallel_region` (SP) | 沿第一维切片（seq） | 沿第一维 AllGather | SP 下 Row→layernorm 衔接 |
| `reduce_scatter_to_sequence_parallel_region` (SP) | ReduceScatter（第一维） | AllGather | Row Parallel SP 模式下输出 |
| `gather_from_sequence_parallel_region` (SP) | AllGather（第一维） | ReduceScatter | Column Parallel SP 模式下输入 |

"前向恒等 + 反向 AllReduce"就是 CopyTo 的设计：前向传播时每个 GPU 都有完整输入，不需要通信；反向传播时输入梯度需要从所有 TP rank 汇总。

源码摘录 `_CopyToModelParallelRegion`：

```python
# megatron/core/tensor_parallel/mappings.py  第 201-218 行
class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_               # 前向：恒等，什么都不做

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None   # 反向：AllReduce
```

---

## 4. ColumnParallelLinear 详解

### 4.1 权重形状

```
weight shape: [output_size / TP,  input_size]
             = [out_per_rank,      in]

# megatron/core/tensor_parallel/layers.py  第 870 行
self.output_size_per_partition = divide(output_size, world_size)
```

注意 PyTorch linear 实际计算的是 `XW^T`，所以存储的是转置形式 `[out, in]`，但概念上是把"输出维"按 TP 切分。

### 4.2 前向步骤（非 SP 模式）

```
输入 X: [B, S, H]   （每个 GPU 上相同）

Step 1: copy_to_tensor_model_parallel_region(X)
        → 前向恒等，反向 AllReduce

Step 2: Y_i = X @ weight_i.T + bias_i
        weight_i shape: [H/TP, H]
        Y_i shape: [B, S, H/TP]

Step 3 (gather_output=True):
        AllGather → Y: [B, S, H]

Step 3 (gather_output=False):
        输出 Y_i: [B, S, H/TP]   （下一层 RowParallel 直接消费）
```

代码关键路径：

```python
# megatron/core/tensor_parallel/layers.py  第 1039-1097 行
if (self.allreduce_dgrad or self.sequence_parallel
        or self.explicit_expert_comm or self.disable_grad_reduce):
    input_parallel = input_
else:
    # 非 SP 模式：前向恒等，反向 AllReduce
    input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

output_parallel = self._forward_impl(
    input=input_parallel, weight=weight, ...
)

if gather_output:
    output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
else:
    output = output_parallel
```

### 4.3 前向步骤（SP 模式，sequence_parallel=True）

SP 模式下输入 X 是按序列维度分片的（每个 GPU 只有 `[B, S/TP, H]`）：

```
Step 1: gather_from_sequence_parallel_region(X)
        AllGather 序列维 → X_full: [B, S, H]

Step 2: Y_i = X_full @ weight_i.T
        Y_i shape: [B, S, H/TP]

Step 3: 输出 Y_i（不做 AllGather，直接传给 RowParallel）
```

---

## 5. RowParallelLinear 详解

### 5.1 权重形状

```
weight shape: [output_size,  input_size / TP]
             = [out,          in_per_rank]

# megatron/core/tensor_parallel/layers.py  第 1221 行
self.input_size_per_partition = divide(input_size, world_size)
```

行切分：把"输入维"按 TP 切分，每个 GPU 持有完整的输出维。

### 5.2 关键约束

```python
# megatron/core/tensor_parallel/layers.py  第 1209-1210 行
if self.sequence_parallel and not self.input_is_parallel:
    raise RuntimeError(
        "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
    )
```

SP 下 Row Parallel 的输入必须是分片的（来自 Column Parallel 的 SP 输出），否则直接报 `RuntimeError`。

### 5.3 前向步骤（非 SP 模式）

```
输入 X_i: [B, S, H/TP]    （来自 Column Parallel 的分片输出）

Step 1: Y_i = X_i @ weight_i.T
        Y_i shape: [B, S, H_out]

Step 2: AllReduce → Y = Σ Y_i: [B, S, H_out]
```

### 5.4 前向步骤（SP 模式）

```
Step 1: Y_i = X_i @ weight_i.T
        Y_i shape: [B, S, H_out]

Step 2: ReduceScatter（序列维）
        → Y_local: [B, S/TP, H_out]

输出送给 LayerNorm（它也是序列分片的）
```

SP 模式下整个 Transformer 的序列维度一直保持分片，只在 Column Parallel 的 AllGather 前后短暂"聚合"。这让 LayerNorm、Dropout 等算子也运行在更小的张量上，进一步节省激活内存。

---

## 6. 完整工作示例：h=1024，FFN=4096，TP=4

### 6.1 非 SP 模式

```
FC1 (ColumnParallelLinear): input_size=1024, output_size=4096
  每个 GPU 权重: [4096/4, 1024] = [1024, 1024]
  输入: [B, S, 1024] (每个 GPU 相同)
  输出: [B, S, 1024] (分片，不做 AllGather)

FC2 (RowParallelLinear): input_size=4096, output_size=1024
  每个 GPU 权重: [1024, 4096/4] = [1024, 1024]
  输入: [B, S, 1024] (分片，来自 FC1)
  局部输出: [B, S, 1024]
  AllReduce → 最终输出: [B, S, 1024]

通信开销: 1次 AllReduce（反向传播时 Column Parallel 再来 1次 AllReduce）
```

### 6.2 SP 模式

```
LayerNorm 输入: [B, S/4, 1024]   (序列分片)

FC1 (ColumnParallelLinear, SP):
  gather_from_sequence_parallel_region → [B, S, 1024]
  每个 GPU 权重: [1024, 1024]
  输出: [B, S, 1024]   (分片但序列维度未分片)

FC2 (RowParallelLinear, SP):
  每个 GPU 权重: [1024, 1024]
  局部输出: [B, S, 1024]
  reduce_scatter_to_sequence_parallel_region → [B, S/4, 1024]

LayerNorm 输入: [B, S/4, 1024]   (再次回到序列分片)

通信开销: 1次 AllGather + 1次 ReduceScatter
         （等效带宽与 1次 AllReduce 相同，但激活内存减少 TP 倍）
```

---

## 7. SelfAttention：QKV Column + Out Row

Megatron 的 Self-Attention 实现（`megatron/core/transformer/attention.py`）使用：

- **QKV projection**：`ColumnParallelLinear(hidden, 3*hidden/TP_heads, gather_output=False)`
- **Output projection**：`RowParallelLinear(hidden/TP_heads, hidden, input_is_parallel=True)`

每个 GPU 负责 `num_heads / TP` 个 attention head 的完整计算。

### 7.1 GQA 示例：h=4096，heads=32，kv_heads=8，TP=4

GQA（Grouped Query Attention）下 Q 和 KV 的 head 数不同：

```
Q heads:  32 → 每个 GPU: 32/4 = 8 个 Q head
KV heads:  8 → 每个 GPU:  8/4 = 2 个 KV head

QKV projection 的输出维度（每个 GPU）:
  Q: 8 heads × head_dim = 8 × 128 = 1024
  K: 2 heads × head_dim = 2 × 128 =  256
  V: 2 heads × head_dim = 2 × 128 =  256
  合计: 1024 + 256 + 256 = 1536

ColumnParallelLinear: input_size=4096, output_size=4096+1024+1024=6144
  每个 GPU 权重: [6144/4, 4096] = [1536, 4096]
  输出: [B, S, 1536]   (分片，不 AllGather)
```

这个数字就是注释和文档里经常出现的 `[4096, 1536]` 权重形状的来源。

---

## 8. VocabParallelEmbedding 与 vocab_parallel_cross_entropy

词表并行（Vocabulary Parallel）是一种特殊的 ColumnParallel：

```python
# megatron/core/tensor_parallel/layers.py  第 198 行开始
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension."""
```

每个 GPU 只持有 `vocab_size / TP` 个词的 embedding 向量。查表时：

1. 先判断哪些 token ID 落在本 GPU 的词表范围内。
2. 本 GPU 范围内的 token 做正常嵌入，范围外的置零。
3. AllReduce 把所有 GPU 的结果相加（因为只有一个 GPU 非零，结果等效于广播）。

Loss 计算同样需要并行化：`vocab_parallel_cross_entropy`（`megatron/core/tensor_parallel/cross_entropy.py`）让每个 GPU 只计算自己词表范围内 logit 的 softmax，再通过通信合并。

原因：如果先 AllGather 得到完整 logit 再计算 cross entropy，通信量与 `vocab_size` 成正比，而 vocab_size 通常很大（32k-128k）；直接在分片 logit 上做并行 softmax 把通信量降低到 `TP` 倍。

---

## 9. RNG 状态追踪：CudaRNGStatesTracker

Dropout 在 TP 下需要特别处理：TP 组内的每个 GPU 独立执行 dropout，为了保证模型正确（各 GPU 上的 dropout mask 不需要一致），Megatron 为 TP 组创建了独立的 RNG 状态。

```python
# megatron/core/tensor_parallel/random.py
class CudaRNGStatesTracker:
    """Tracker for the CUDA RNG states."""

    def fork(self, name=None):
        """Fork the CUDA RNG state."""
        # 保存当前 RNG 状态，切换到 TP-local 的独立状态
        ...
```

在 `_initialize_affine_weight_gpu` 中：

```python
# megatron/core/tensor_parallel/layers.py  第 144-148 行
if not is_expert:
    with get_cuda_rng_tracker().fork():
        init_method(weight)
else:
    with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
        init_method(weight)
```

`fork()` 确保每个 TP rank 上的权重初始化使用独立的随机种子，避免各分片权重完全相同（那样 TP 就退化为无效的计算冗余）。

---

## 10. Debug 速查：Shape 不匹配排查清单

实践中张量并行最常见的问题是 shape mismatch。以下是系统排查流程：

### 问题 1：`RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**成因**：ColumnParallelLinear 的 `input_size` 与实际输入的最后一维不匹配。

**排查**：

```python
print(f"input shape: {input_.shape}")
print(f"weight shape: {self.weight.shape}")
print(f"expected input_size: {self.input_size}")
```

**常见原因**：上游层没有正确设置 `gather_output=False` 或 `input_is_parallel=True`，导致 shape 翻倍。

### 问题 2：`AssertionError: First dimension of the tensor should be divisible by tensor parallel size`

**成因**：序列长度 S 不能被 TP 整除（SP 模式下 `scatter_to_sequence_parallel_region` 要求 S % TP == 0）。

**排查**：

```python
assert seq_length % tp_size == 0, \
    f"seq_length ({seq_length}) must be divisible by TP ({tp_size}) for SP"
```

### 问题 3：SP 下 Row Parallel 报 `RuntimeError: To enable sequence_parallel, input_is_parallel must be True`

**解决**：确保 `RowParallelLinear(input_is_parallel=True)` 与 `ColumnParallelLinear(gather_output=False)` 配对使用。

### 问题 4：梯度正确但值异常（TP=1 正常，TP>1 发散）

**成因**：某处使用了 `gather_output=True` 但反向传播时没有正确 scatter 梯度，或者 AllReduce 了不该 AllReduce 的梯度。

**排查**：用 TP=1 和 TP=2 分别运行一个 iteration，比较每个参数的 `.grad` 范数，找到第一个不一致的参数。

---

## 11. 完整 TP+SP 数据流图

以一个 Transformer block 为例，展示在 TP=4、SP=True 情况下每个算子的张量形状变化：

```
输入 (每 GPU): [B, S/4, H]   ← 序列分片

LayerNorm:     [B, S/4, H]   ← 序列分片操作
               ↓
Column Parallel (FC1 / Q,K,V proj):
  gather_from_sequence_parallel_region → [B, S, H]   (AllGather)
  GEMM → [B, S, H/4]                                 (分片输出)
               ↓
Attention Core:  [B, S, H/4]  ← 每 GPU 处理 heads/4 个 head
               ↓
Row Parallel (Out proj):
  GEMM → [B, S, H]            (局部结果)
  reduce_scatter_to_SP_region → [B, S/4, H]   (ReduceScatter)
               ↓
LayerNorm:     [B, S/4, H]   ← 序列分片操作
               ↓
Column Parallel (FC1 of FFN):
  AllGather → [B, S, H]
  GEMM → [B, S, FFN/4]
               ↓
Row Parallel (FC2 of FFN):
  GEMM → [B, S, H]
  ReduceScatter → [B, S/4, H]
               ↓
输出 (每 GPU): [B, S/4, H]   ← 序列分片
```

整个 block 只有 2 次 AllGather + 2 次 ReduceScatter，等效通信量与 2 次 AllReduce 相同，但激活内存减少了约 TP 倍（因为大多数算子都工作在 `[B, S/4, H]` 上）。

---

## 12. 练习题

**题目 1**：对于 `h=2048, ffn=8192, TP=8`，分别计算非 SP 和 SP 模式下：
1. FC1、FC2 的权重 shape（每个 GPU）
2. FC1 输出的 shape（每个 GPU，在 AllGather/ReduceScatter 之后）
3. 每个 Transformer block 的通信量（以 B×S×H 为单位）

**题目 2**：阅读 `_ReduceFromModelParallelRegion` 和 `_ReduceScatterToSequenceParallelRegion` 的源码，解释：
1. 为什么前向 AllReduce 对应反向恒等，而不是反向 scatter？
2. 在求和分解的意义下，两种通信方式（AllReduce vs ReduceScatter）如何保证梯度正确？

**题目 3**：在 GQA 场景（heads=64, kv_heads=8）下，如果 TP=8，每个 GPU 持有几个 Q head 和几个 KV head？当 TP 增大到 16 时会发生什么问题？

---

## 13. 参数 allreduce_dgrad 与 gradient_accumulation_fusion

`ColumnParallelLinear` 有两个不太直觉的配置项值得特别说明。

**`allreduce_dgrad`**：控制输入梯度（`grad_input`，即 `dL/dX`）的 AllReduce 是否异步化。

```python
# megatron/core/tensor_parallel/layers.py  第 960-962 行
self.allreduce_dgrad = (
    world_size > 1 and not self.sequence_parallel and not self.disable_grad_reduce
)
```

当 `allreduce_dgrad=True` 时，反向传播中对 `grad_input` 的 AllReduce 会与权重梯度（`grad_weight`）的计算**异步并行**执行，需要 `CUDA_DEVICE_MAX_CONNECTIONS=1` 来保证正确调度顺序（通信 kernel 先于计算 kernel 调度）：

```python
# megatron/core/tensor_parallel/layers.py  第 563-565 行
if ctx.allreduce_dgrad:
    # Asynchronous all-reduce
    handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # all-reduce is scheduled before the weight gradient computation
```

**`gradient_accumulation_fusion`**：把权重梯度的计算与 FP16→FP32 精度转换合并为一个 CUDA kernel（`fused_weight_gradient_mlp_cuda`），减少一次读写：

```python
# megatron/core/tensor_parallel/layers.py  第 606-607 行
fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
    total_input, grad_output, weight.main_grad
)
```

该特性需要安装 APEX（`--cpp_ext --cuda_ext`）。开启后每次反向传播直接把梯度累加到 `weight.main_grad`（FP32），不再创建额外的 FP16 梯度缓冲区。

---

## 14. 小结

张量并行的实现由三层组成：

1. **`mappings.py`**：六个 `torch.autograd.Function` 子类，封装了前向/反向完全对称的通信原语（CopyTo、ReduceFrom、Scatter、Gather 及其 SP 变体）。
2. **`ColumnParallelLinear` / `RowParallelLinear`**（`layers.py`）：把权重矩阵按输出维/输入维切分，通过 `gather_output` 和 `input_is_parallel` 控制与相邻层的衔接；SP 模式下把序列维也一并切分，进一步节省激活内存。
3. **VocabParallelEmbedding + vocab_parallel_cross_entropy**：词表维度的并行，避免在大词表上做全量 AllGather。

SP 是一个重要的优化：它以相同的通信量换来了 TP 倍的激活内存节省，但要求序列长度 S 能整除 TP，且所有算子都必须正确标注 `sequence_parallel=True`。

下一篇（06）将深入流水线并行：1F1B 调度、气泡分析、P2P 通信 API，以及 VPP 的交错执行。
