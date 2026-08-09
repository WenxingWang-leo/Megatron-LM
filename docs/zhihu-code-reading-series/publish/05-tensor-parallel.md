# 精读 Megatron 源码（5）：张量并行完全精读——mappings、Column/Row 与 Attention/MLP

> 源文件：`megatron/core/tensor_parallel/mappings.py`、`layers.py`

---

## 1. 张量并行的核心思想

张量并行（Tensor Parallelism，TP）把单个权重矩阵沿某个维度切开，分散到多个 GPU 上，每个 GPU 只做部分 GEMM，最后用通信原语把结果拼回来。它的优势是显存占用随 TP 线性下降，代价是每层都需要额外的集体通信。

最重要的数学事实：矩阵乘法 `Y = XA`（此处 A 为数学布局 `[in, out]`；PyTorch `nn.Linear` 存的是 `[out, in]`，算的是 `X @ W.T`）有两种等价切分方式：

- **按列切分 A**（Column Parallel）：`A = [A_1 | A_2 | ... | A_p]`，`Y_i = X A_i`，再在最后一维拼接（或 AllGather）。
- **按行切分 A**（Row Parallel）：`A = [A_1; A_2; ...; A_p]`（沿 **第一维 / 输入维** 切开），同时 `X = [X_1 | X_2 | ... | X_p]`（沿最后一维切开），则  
  `Y = X A = Σ_i X_i A_i`。每个 GPU 算 `Y_i = X_i A_i`，再 **AllReduce 求和**（不是拼接）。

> 注意：源码注释里的 `A = transpose([A_1 .. A_p])` 是在兼顾 PyTorch 权重转置存储；数学上就是上面的「A 按行块堆叠、X 按列块切开、局部 GEMM 再求和」。**不是** `X_i A_i^T`。

Megatron 的设计是**把 ColumnParallelLinear 和 RowParallelLinear 串联**，中间无需 AllGather/AllReduce，从而把通信次数从每层 2 次降到每层 1 次。

---

## 2. Y=XA 切分图解（ASCII）

下面用 FFN 的一对线性层举例（与后文 §8 数值例一致）：先 Column（`H → 4H`），再 Row（`4H → H`）。设 TP=`p`。

### 2.1 Column Parallel：A 沿列切分（FFN 第一层）

```
输入 X: [B, S, H]     数学权重 A: [H, 4H]
                      每卡 A_i:   [H, 4H/p]

           ┌─ A_0 [H, 4H/p] ─┐
X ──── TP  ├─ A_1 [H, 4H/p] ─┤  ──→  Y_i = X @ A_i
           └─ A_{p-1} ...   ─┘

每个 GPU 算: Y_i = X @ A_i,   shape: [B, S, 4H/p]
若 gather_output=True:  AllGather 最后一维 → [B, S, 4H]
若 gather_output=False: 保持分片 [B, S, 4H/p]，直接交给下一层 RowParallel
```

（PyTorch 存储：每卡 `weight` 为 `[4H/p, H]`，即 `[out_per_rank, in]`。）

### 2.2 Row Parallel：A 沿行切分（兼 X 沿最后一维切分）（FFN 第二层）

```
接上一步 gather_output=False 的输出：
  输入 X_i (已分片): [B, S, 4H/p]
  数学权重 A:        [4H, H]，沿行切开
  每卡 A_i:          [4H/p, H]

GPU_i 算: Y_i = X_i @ A_i,   shape: [B, S, H]     ← 每卡都是完整 out 维，但是「部分和」
                                      ↓
                               AllReduce（求和；SP 下为 ReduceScatter）
                                      ↓
                               Y = Σ_i Y_i,       shape: [B, S, H]
```

（PyTorch 存储：每卡 `weight` 为 `[H, 4H/p]`，即 `[out, in_per_rank]`，与 `RowParallelLinear` 一致。）

对照关系：

| | Column（§2.1） | Row（§2.2） |
|--|----------------|-------------|
| 切 A 的轴 | 输出维（列） | 输入维（行） |
| 输入 X | 通常完整 `[..., H]` | 必须已按最后一维切开 `[..., 4H/p]` |
| 局部结果 | 输出维的一块 | 完整输出维上的部分和 |
| 聚合方式 | 拼接 / AllGather | **求和** / AllReduce |

关键：Row Parallel 的输入 `X` 必须已经是分片的（即上一层 Column Parallel `gather_output=False` 的输出），这就是为什么 Column + Row 串联后只需要 **一次** 通信而不是两次。

---

## 3. mappings.py：通信原语怎么数？

```
文件: megatron/core/tensor_parallel/mappings.py
```

旧版标题写「六种」容易对不上号：表里常列出 7 行，源码里 `autograd.Function` 还更多。  
更干净的数法是：**经典 TP 四件套（2 对对偶）+ Sequence Parallel 三件套**。

### 3.0 先认两对对偶（经典 4 种）

非 SP 时，Megatron 论文/早期代码的核心就是这 **4** 个封装——两两互为前反向对偶：

| # | 函数名 | 前向 | 反向 | 典型用途 |
|---|--------|------|------|----------|
| 1 | `copy_to_tensor_model_parallel_region` | 恒等 | AllReduce | Column 入口（非 SP）：把输入标成 TP region |
| 2 | `reduce_from_tensor_model_parallel_region` | AllReduce | 恒等 | Row 出口（非 SP）：把各卡部分和加总 |
| 3 | `scatter_to_tensor_model_parallel_region` | 最后一维切分 | 最后一维 AllGather | 把完整向量拆进各 TP rank |
| 4 | `gather_from_tensor_model_parallel_region` | 最后一维 AllGather | 最后一维切分 | 把各 TP 分片拼回完整向量 |

记忆口诀：

```text
CopyTo  ↔  ReduceFrom     （「进 region / 出 region」）
Scatter ↔  Gather         （「切开 / 拼回」，默认沿最后一维）
```

### 3.1 再加上 SP 三件套（切的是序列维）

开 Sequence Parallel 后，通信改走**第一维（seq）**，又多 **3** 个常用封装：

| # | 函数名 | 前向 | 反向 | 典型用途 |
|---|--------|------|------|----------|
| 5 | `scatter_to_sequence_parallel_region` | 第一维切分 | 第一维 AllGather | 进入序列分片布局 |
| 6 | `gather_from_sequence_parallel_region` | 第一维 AllGather | ReduceScatter | Column（SP）算 GEMM 前聚齐序列 |
| 7 | `reduce_scatter_to_sequence_parallel_region` | ReduceScatter | AllGather | Row（SP）出口：求和并回到 S/TP |

因此：**读 Column/Row 主路径，心里记「4 + 3 = 7 个常用 wrapper」**；说「六种」是历史含混说法，本文不再用。

### 3.2 源码里还有、主路径较少提的

同文件还有例如：

- `all_gather_last_dim_from_tensor_parallel_region` / `reduce_scatter_last_dim_to_tensor_parallel_region`
- `all_to_all`，以及 `all_to_all_sp2hp` / `all_to_all_hp2sp`（SP 与 hidden-parallel 布局互换）

MoE dispatcher、部分融合路径会用到它们；先把上面 4+3 吃透即可。

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

## 4. ColumnParallelLinear 前向逐行注释

### 4.1 权重形状

```
weight shape: [output_size / TP,  input_size]
             = [out_per_rank,      in]

# megatron/core/tensor_parallel/layers.py  第 870 行
self.output_size_per_partition = divide(output_size, world_size)
```

注意 PyTorch linear 实际计算的是 `XW^T`，所以存储的是转置形式 `[out, in]`，但概念上是把"输出维"按 TP 切分。

### 4.2 前向步骤（非 SP 模式）：关键代码路径

```python
# megatron/core/tensor_parallel/layers.py  第 1033-1101 行
def forward(self, input_, weight=None, runtime_gather_output=None):
    # --- Step 1: 路由输入 ---
    if (
        self.allreduce_dgrad          # 非SP模式：反向会做dgrad AllReduce
        or self.sequence_parallel     # SP模式：见4.3节
        or self.explicit_expert_comm  # MoE expert 路径
        or self.disable_grad_reduce   # 手动管理梯度
    ):
        input_parallel = input_       # 直接使用，不插入CopyTo
    else:
        # 非SP、非expert的普通模式：
        # 前向恒等，反向AllReduce（由_CopyToModelParallelRegion保证）
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    # --- Step 2: GEMM ---
    output_parallel = self._forward_impl(
        input=input_parallel,
        weight=weight,
        bias=bias,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        allreduce_dgrad=allreduce_dgrad,
        sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
        tp_group=self.tp_group,
    )
    # output_parallel shape: [B, S, out_per_rank]

    # --- Step 3: 可选的 AllGather ---
    gather_output = self.gather_output
    if runtime_gather_output is not None:
        gather_output = runtime_gather_output  # 推理时可动态覆盖

    if gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        # output shape: [B, S, output_size]（全量，每GPU一份）
    else:
        output = output_parallel
        # output shape: [B, S, out_per_rank]（分片，传给RowParallel）

    return output, output_bias
```

### 4.3 前向步骤（SP 模式，sequence_parallel=True）

SP 模式下输入 X 是按序列维度分片的（每个 GPU 只有 `[B, S/TP, H]`）：

```python
# SP模式的 _forward_impl 内部（linear_with_grad_accumulation_and_async_allreduce）
if ctx.sequence_parallel:
    # 反向时做 ReduceScatter（把 dX 从全序列散布回 S/TP）
    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * tp_group.size()
    all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
    dist_all_gather_func(all_gather_buffer, input, group=tp_group)
    total_input = all_gather_buffer   # [B, S, H] 已拼合
```

完整路径：

```
输入 [B, S/TP, H]
  ↓ AllGather（gather_from_sequence_parallel_region）
[B, S, H]
  ↓ GEMM（每GPU持有 output_size/TP 的权重列）
[B, S, output_size/TP]
  ↓ 输出分片（不做AllGather，直接传给RowParallel）
[B, S, output_size/TP]
```

---

## 5. RowParallelLinear 前向逐行注释

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

### 5.3 前向代码路径

```python
# megatron/core/tensor_parallel/layers.py  第 1318-1364 行
def forward(self, input_):
    # --- Step 1: 确认输入是分片的 ---
    if self.input_is_parallel:
        input_parallel = input_    # 来自ColumnParallel的分片输出
    else:
        assert not self.sequence_parallel
        input_parallel = scatter_to_tensor_model_parallel_region(input_, group=self.tp_group)
        # 非分片输入时，主动切分（scatter）

    # --- Step 2: GEMM ---
    output_parallel = self._forward_impl(
        input=input_parallel,      # [B, S, in_per_rank]
        weight=self.weight,        # [out, in_per_rank]
        bias=None,
        gradient_accumulation_fusion=self.gradient_accumulation_fusion,
        allreduce_dgrad=False,     # Row Parallel 不做 dgrad AllReduce
        sequence_parallel=False,
        tp_group=None,
    )
    # output_parallel shape: [B, S, out]（局部求和，需要跨TP汇总）

    # --- Step 3: 归约 ---
    if self.explicit_expert_comm:
        output_ = output_parallel  # expert路径：不做通信，由调用方处理
    elif self.sequence_parallel:
        output_ = reduce_scatter_to_sequence_parallel_region(
            output_parallel, group=self.tp_group
        )   # ReduceScatter: [B, S, out] → [B, S/TP, out]（序列维分片）
    else:
        output_ = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        # AllReduce: [B, S, out]（对各GPU的局部结果求和）

    return output_, output_bias
```

---

## 6. 梯度路径：Column vs Row 的反向传播

### 6.1 ColumnParallelLinear 的梯度路径

反向传播时，`linear_with_grad_accumulation_and_async_allreduce` 的 `backward` 方法负责：

1. **dgrad（输入梯度）**：`grad_input = grad_output @ weight`，shape `[B, S, H]`。由于前向时输入是完整的（非SP），dgrad AllReduce 是通过 `_CopyToModelParallelRegion` 的反向实现的——等效于对所有TP rank的dgrad做AllReduce。

   当 `allreduce_dgrad=True` 时（非SP普通模式），AllReduce 与权重梯度计算**异步并行**：
   ```python
   if ctx.allreduce_dgrad:
       handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
       # 此处依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 确保 AllReduce kernel
       # 先于 wgrad GEMM 提交，实现通信计算重叠
   ```

2. **wgrad（权重梯度）**：`grad_weight = grad_output.T @ input`。开启 `gradient_accumulation_fusion` 时，使用 APEX 的 `fused_weight_gradient_mlp_cuda`，直接把梯度累加到 `weight.main_grad`（FP32），避免额外的内存分配：
   ```python
   if weight.main_grad.dtype == torch.float32:
       fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
           total_input, grad_output, weight.main_grad
       )
   ```

### 6.2 RowParallelLinear 的梯度路径

Row Parallel 的前向是 AllReduce（ReduceScatter in SP），反向是其对偶：

- **非SP模式**：前向 AllReduce，反向恒等（`_ReduceFromModelParallelRegion.backward` 直接 pass，dgrad 已经是完整的）
- **SP模式**：前向 ReduceScatter，反向 AllGather（`_ReduceScatterToSequenceParallelRegion.backward`）

权重梯度 `wgrad = grad_output.T @ input_parallel`，由于 `input_parallel` 已经是分片的，所以 wgrad 也是分片的——Row Parallel 无需额外的 wgrad AllReduce，这正是其高效性的来源。

---

## 7. SP 模式的 AllToAll 辅助函数

除了 ReduceScatter/AllGather，Megatron 还在 `mappings.py` 里提供了两个 AllToAll 辅助函数，用于从 SP 格式（序列并行，按序列维度分片）和 HP 格式（head/hidden 并行，按隐藏维切分）之间互换：

```python
# megatron/core/tensor_parallel/mappings.py  第 566-621 行
def all_to_all_sp2hp(input_, group=None):
    """[num_tokens/TP, H] → [num_tokens, H/TP]（SP→HP）
    
    用于从序列并行（SP）切换到头并行（HP）的场景，
    例如在某些 attention 变体中需要对不同 token 做不同操作时。
    """
    group = get_tensor_model_parallel_group_if_none(group)
    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])
    # 把 H 维切成 TP 份，然后拼成 [num_tokens*TP, H/TP]
    split_tensors = torch.split(input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(group, concat_tensor)   # AllToAll: 重分配 token
    return output

def all_to_all_hp2sp(input_, group=None):
    """[num_tokens, H/TP] → [num_tokens/TP, H]（HP→SP）
    逆操作
    """
    ...
```

这两个函数的应用场景：当模型某些算子（如 cross-attention 中的 key-value 重分配）需要改变并行维度时，可以用 AllToAll 代替 AllGather+ReduceScatter 的两步操作，通信量相同但只需要一次内核调用。

---

## 8. 完整工作示例一：h=1024，FFN=4096，TP=4

### 8.1 非 SP 模式

```
FC1 (ColumnParallelLinear): input_size=1024, output_size=4096
  每个 GPU 权重: [4096/4, 1024] = [1024, 1024]   → 存储为 [out_per_rank, in]
  输入: [B, S, 1024] (每个 GPU 相同)
  前向: copy_to_region（恒等）→ GEMM → [B, S, 1024]（分片，不做 AllGather）

FC2 (RowParallelLinear): input_size=4096, output_size=1024
  每个 GPU 权重: [1024, 4096/4] = [1024, 1024]   → 存储为 [out, in_per_rank]
  输入: [B, S, 1024] (分片，来自 FC1)
  前向: GEMM → [B, S, 1024]（局部） → AllReduce → [B, S, 1024]（完整）

通信开销（前向）: 1次 AllReduce，数据量 B×S×1024×2 bytes（BF16）
通信开销（反向）: 1次 AllReduce（Column dgrad），总计 2次 AllReduce
等效 AllReduce 带宽: 2 × (TP-1)/TP × B×S×1024 × 2 bytes
                    = 2 × 3/4 × B×S×2048 bytes（TP=4时）
```

### 8.2 SP 模式

```
LayerNorm 输入: [B, S/4, 1024]   (序列分片)

FC1 (ColumnParallelLinear, SP):
  AllGather（序列维）: [B, S/4, 1024] → [B, S, 1024]
  GEMM → [B, S, 1024]   (分片输出)

FC2 (RowParallelLinear, SP):
  GEMM → [B, S, 1024]
  ReduceScatter（序列维）: [B, S, 1024] → [B, S/4, 1024]

LayerNorm 输入: [B, S/4, 1024]   (再次回到序列分片)

通信开销: 1次 AllGather + 1次 ReduceScatter
         （等效带宽与 1次 AllReduce 相同，但激活内存减少 TP 倍）
```

SP 模式下整个 Transformer 的序列维度一直保持分片，只在 Column Parallel 的 AllGather 前后短暂"聚合"。这让 LayerNorm、Dropout 等算子也运行在更小的张量上，进一步节省激活内存。

---

## 9. SelfAttention：QKV Column + Out Row

Megatron 的 Self-Attention 实现（`megatron/core/transformer/attention.py`）使用：

- **QKV projection**：`ColumnParallelLinear(hidden, 3*hidden/TP_heads, gather_output=False)`
- **Output projection**：`RowParallelLinear(hidden/TP_heads, hidden, input_is_parallel=True)`

每个 GPU 负责 `num_heads / TP` 个 attention head 的完整计算。

### 9.1 GQA + TP=8 完整形状推导

场景：h=8192，Q-heads=64，KV-heads=8，head_dim=128，TP=8

```
每个 GPU 分配：
  Q heads:  64/8 = 8 个 Q head
  KV heads:  8/8 = 1 个 KV head

每个 GPU 的 QKV 输出维度：
  Q: 8 × 128 = 1024
  K: 1 × 128 = 128
  V: 1 × 128 = 128
  合计: 1024 + 128 + 128 = 1280

ColumnParallelLinear 参数（QKV projection）：
  input_size  = 8192
  output_size = 64×128 + 8×128 + 8×128 = 8192 + 1024 + 1024 = 10240
  每GPU权重: [10240/8, 8192] = [1280, 8192]

注意力计算（每 GPU）：
  Q: [B, S, 8, 128]，K: [B, S, 1, 128]，V: [B, S, 1, 128]
  每个 Q head 对应同 GPU 上的唯一 KV head（MHA: 1Q↔1KV，GQA: 8Q↔1KV）

RowParallelLinear 参数（Out projection）：
  input_size  = 64×128 = 8192，每GPU: 8192/8 = 1024
  output_size = 8192
  每GPU权重: [8192, 1024]
```

当 TP 增大到 16 时：KV heads=8/16=0.5，无法整除，需要 KV heads 至少等于 TP。这是 GQA + 大 TP 的典型约束，通常要求 `kv_heads >= TP`。

---

## 10. SwiGLU / Gated MLP 与 Column 输出大小翻倍

SwiGLU 激活函数的 FFN 层形如：

```python
FFN(x) = W2 * SiLU(W1 * x) ⊙ (W3 * x)
```

其中 W1 和 W3 的输出维度都是 `ffn_hidden_size`（通常 = 8/3 × hidden，如 Llama 中），W2 的输入维度是 `ffn_hidden_size`。

在 Megatron 的实现中，W1 和 W3 被合并为一个 `ColumnParallelLinear`，输出大小翻倍：

```python
# megatron/core/transformer/mlp.py（概念示意）
self.linear_fc1 = ColumnParallelLinear(
    config.hidden_size,
    config.ffn_hidden_size * 2,   # ← 翻倍！包含 gate 和 up projection
    gather_output=False,
)
self.linear_fc2 = RowParallelLinear(
    config.ffn_hidden_size,
    config.hidden_size,
    input_is_parallel=True,
)
```

每个 GPU 的 FC1 权重：`[ffn_hidden_size*2/TP, hidden]`，输出 `[B, S, ffn_hidden_size*2/TP]`，在激活函数中拆成两半：前一半是 up projection，后一半是 gate。经过 `SiLU(gate) * up` 后输出 `[B, S, ffn_hidden_size/TP]`，正好是 FC2 的 `input_size_per_partition`。

---

## 11. VocabParallelEmbedding 与 vocab_parallel_cross_entropy

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

## 12. TE fused vs 本地 ColumnParallelLinear 的差异

TransformerEngine（TE）提供了融合版本的 Column/RowParallelLinear，在以下方面与本地实现有差异：

| 特性 | 本地实现（layers.py） | TE 融合实现 |
|------|------|------|
| FP8 支持 | 否 | 是（e4m3/e5m2） |
| 分层 bias fusion | 否 | 是（bias 直接在 GEMM 中 fuse） |
| activation checkpointing | 手动（`recompute_activation_function`） | 自动（TE recompute） |
| 权重更新格式 | BF16 main_grad | FP8/BF16 混合 |
| 后缀 | `ColumnParallelLinear` | `TEColumnParallelLinear` |

读者在阅读基于 TE 的 Megatron 代码时，会看到大量 `te_modules.py` 中的 TE 包装类。其通信逻辑与本地实现相同，主要差异在于 GEMM 内部的精度控制。

---

## 13. RNG 状态追踪：CudaRNGStatesTracker

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

## 14. 通信量估计（量级分析）

对于一个 Transformer block，TP 通信量估算（以字节为单位，BF16）：

| 操作 | 通信模式 | 数据量（单向） |
|------|------|------|
| Attention QKV（Column，SP） | AllGather | B × S × H × 2 |
| Attention Out（Row，SP） | ReduceScatter | B × S × H × 2 |
| FFN FC1（Column，SP） | AllGather | B × S × H × 2 |
| FFN FC2（Row，SP） | ReduceScatter | B × S × H × 2 |
| **合计（SP模式）** | 2×AllGather + 2×ReduceScatter | 4 × B×S×H×2 |

以 LLaMA-7B（H=4096，S=2048，B=1，TP=4）为例：

```
单向通信量 = 4 × 1 × 2048 × 4096 × 2 = 67 MB
双向（前向+反向）= 134 MB  per Transformer block
32 层合计      = 4.3 GB  per step（仅 TP 通信）
```

这个数字说明 TP 通信对 NVLink 带宽（H100 为 900 GB/s）的压力是可控的——在 NVLink 速度下，67 MB 的延迟约为 0.07 ms，远小于一层 forward 的计算时间（通常 > 1 ms）。

---

## 15. 完整 TP+SP 数据流图

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

## 16. Debug 速查：Shape 不匹配排查清单（扩展版）

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

### 问题 5：`gradient_accumulation_fusion` 报 `module not found`

**成因**：没有安装带 APEX cuda extensions 的版本。

```
RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set
to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found.
```

**解决**：安装 `apex --cpp_ext --cuda_ext`，或者设置 `config.gradient_accumulation_fusion = False`。

### 问题 6：GQA 场景 KV head 数无法被 TP 整除

**成因**：如 kv_heads=6，TP=8，6 < 8，无法切分。

**解决**：调整 TP 使 `kv_heads % TP == 0`，或使用 TP=1（不做 TP）。

---

## 17. FAQ（张量并行常见问题 10 条）

**Q1：Column + Row 串联为什么只需要一次通信而不是两次？**

Column Parallel（`gather_output=False`）输出已经是分片的，正好是 Row Parallel 需要的分片输入（`input_is_parallel=True`）。两者衔接处不需要任何通信，因此整个 Column+Row 对只需要 Row 的一次 AllReduce（或 ReduceScatter）。

**Q2：`allreduce_dgrad=True` 和 `sequence_parallel=True` 不能共存的原因？**

两者都是 Column Parallel 的 dgrad 处理方式，但路径不同：前者用 AllReduce，后者用 ReduceScatter（SP 模式下dgrad 要回到序列分片格式）。共存会导致 dgrad 被归约两次，梯度值翻倍。源码第 976-979 行有明确断言。

**Q3：TE 模式下 Column/Row 还是同样的通信吗？**

通信逻辑相同，但 TE 把 GEMM 和通信融合得更紧密（FP8 GEMM + async AllGather 可以 pipeline），理论上比本地实现延迟更低。

**Q4：词表并行的 embedding AllReduce 通信量是多少？**

每次前向传播，VocabParallelEmbedding 做 AllReduce，数据量为 `B × S × H × 2`（BF16）。以 B=1，S=2048，H=4096 为例约 16 MB，属于可接受范围。

**Q5：SP 模式下 `S % TP != 0` 时能否 padding？**

技术上可以在输入时 padding 到 TP 的倍数，但 Megatron 不自动 padding，需要调用方保证。训练时通常固定 seq_length 为 TP 的倍数即可。

**Q6：ColumnParallelLinear 的 `skip_bias_add` 参数有什么用？**

`skip_bias_add=True` 时，bias 不在 forward 里直接加上，而是以 `output_bias` 的形式返回，由调用方（例如 LayerNorm fusion）在 fused kernel 里一并完成加法，节省一次内存读写。

**Q7：`gradient_accumulation_fusion` 开启后对显存有何影响？**

开启后不再为 wgrad 创建独立的 BF16 缓冲区，而是直接累加到 `param.main_grad`（FP32）。这减少了约 `2N bytes`（N 为参数量）的临时 BF16 梯度缓冲区，但要求 APEX 编译。

**Q8：non-SP 模式下 TP 通信是否与反向计算重叠？**

是，当 `allreduce_dgrad=True` 且 `CUDA_DEVICE_MAX_CONNECTIONS=1` 时，dgrad AllReduce 与 wgrad GEMM 并行执行。这是 Megatron 文档中反复强调设置 `CUDA_DEVICE_MAX_CONNECTIONS=1` 的原因。

**Q9：VocabParallelEmbedding 在 PP last stage 有特殊处理吗？**

有。lm_head 通常是 `ColumnParallelLinear`（与 embedding 共享权重），last stage 需要把输出 logit（分片）传给 `vocab_parallel_cross_entropy` 而不是先 AllGather，避免通信 logit 的大 tensor。

**Q10：TE 的 `TEColumnParallelLinear` 和本地 `ColumnParallelLinear` 参数是否兼容（checkpoint 可互换）？**

是的，Megatron 的 `sharded_state_dict` 接口保证了分片语义相同（axis 0 for Column，axis 1 for Row），checkpoint 可以在 TE 和非 TE 实现之间无缝迁移。

---

## 18. 参数 allreduce_dgrad 与 gradient_accumulation_fusion

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

## 21. 附录：关键 autograd Function 速查表

| 类名 | 前向 | 反向 | 调用入口 |
|------|------|------|---------|
| `_CopyToModelParallelRegion` | 恒等 | AllReduce | `copy_to_tensor_model_parallel_region` |
| `_ReduceFromModelParallelRegion` | AllReduce | 恒等 | `reduce_from_tensor_model_parallel_region` |
| `_ScatterToModelParallelRegion` | 沿最后维 Scatter | 沿最后维 AllGather | `scatter_to_tensor_model_parallel_region` |
| `_GatherFromModelParallelRegion` | 沿最后维 AllGather | 沿最后维 Scatter | `gather_from_tensor_model_parallel_region` |
| `_ScatterToSequenceParallelRegion` | 沿第 0 维 Scatter | 沿第 0 维 AllGather | `scatter_to_sequence_parallel_region` |
| `_ReduceScatterToSequenceParallelRegion` | ReduceScatter（第 0 维） | AllGather（第 0 维） | `reduce_scatter_to_sequence_parallel_region` |
| `_GatherFromSequenceParallelRegion` | AllGather（第 0 维） | ReduceScatter（第 0 维） | `gather_from_sequence_parallel_region` |
| `_AllToAll` | AllToAll | AllToAll（转置split） | `all_to_all` |

所有 Function 的设计原则：**前向和反向互为对偶**——若前向是 AllGather，则反向是 ReduceScatter；若前向是恒等，则反向是 AllReduce。这保证了自动微分的正确性。

---

## 22. 非 SP 与 SP 模式对比总结表

| 维度 | 非 SP 模式（`sequence_parallel=False`） | SP 模式（`sequence_parallel=True`） |
|------|------|------|
| 输入激活形状（每GPU） | `[B, S, H]`（全量） | `[B, S/TP, H]`（序列分片） |
| Column 前向通信 | 前向恒等（`CopyTo`），反向 AllReduce | 前向 AllGather（`S/TP→S`） |
| Row 后向通信 | AllReduce | ReduceScatter（`S→S/TP`） |
| LayerNorm 输入 | `[B, S, H]` | `[B, S/TP, H]` |
| 激活内存节省 | 无 | 约 TP 倍（大部分激活 `S/TP`） |
| 要求 | 无 | `S % TP == 0` |
| 通信量 | 每层 1×AllReduce（双向计 2×） | 每层 1×AllGather+1×ReduceScatter（等效带宽同） |
| 适用场景 | TP < 4 或短序列 | TP ≥ 4 且长序列（>4k tokens） |

---

## 23. 完整工作示例二：attention block 的 SP 模式通信时序

以 TP=4，S=4096，H=4096，B=1 为例，逐步追踪 attention block 的所有通信事件：

```
t=0  LayerNorm:  input [1, 1024, 4096]  (S/TP=1024)

t=1  AllGather (Column, QKV):
     [1, 1024, 4096] × 4 GPU → [1, 4096, 4096]   共 32 MB 传输

t=2  QKV GEMM (每GPU):
     [1, 4096, 4096] × W_qkv[qkv_per_rank, 4096] → [1, 4096, qkv_per_rank]

t=3  FlashAttention / SDPA (每GPU, 在自己负责的 heads 上运行)

t=4  ReduceScatter (Row, Out proj):
     [1, 4096, 4096] → [1, 1024, 4096]   共 32 MB 传输

t=5  LayerNorm:  input [1, 1024, 4096]  回到 S/TP 状态

总通信时间（NVLink，900 GB/s）: (32+32) MB × 2方向 / 900 GB/s ≈ 0.14 ms
总计算时间（H100，估算 2 个 GEMM）: 约 1-2 ms
通信/计算比: ~7-14%，可重叠
```

这证明了 TP 通信在 NVLink 环境下是计算绑定，不是通信绑定。

---

## 24. 实战：如何在自定义层中正确使用 TP 通信原语

当你为 Megatron 添加自定义层时，需要选择合适的通信包装：

```python
from megatron.core.tensor_parallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

class MyColumnParallelLayer(torch.nn.Module):
    """自定义列并行层示例"""
    def __init__(self, input_size, output_size, config, tp_group=None):
        super().__init__()
        world_size = tp_group.size() if tp_group else ps.get_tensor_model_parallel_world_size()
        self.out_per_rank = output_size // world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, input_size))
        self.sequence_parallel = config.sequence_parallel
        self.tp_group = tp_group

    def forward(self, x):
        if self.sequence_parallel:
            # SP模式：输入是 [B, S/TP, H]，先 AllGather
            x_full = gather_from_sequence_parallel_region(x, group=self.tp_group)
        else:
            # 非SP模式：前向恒等，反向 AllReduce
            x_full = copy_to_tensor_model_parallel_region(x, group=self.tp_group)

        # GEMM：[B, S, H] × [out_per_rank, H]^T → [B, S, out_per_rank]
        out = x_full @ self.weight.t()
        return out  # 返回分片输出，不 AllGather
```

关键点：
1. SP 模式用 `gather_from_sequence_parallel_region`，非 SP 用 `copy_to_tensor_model_parallel_region`。
2. 返回值保持分片（`gather_output=False`），让配对的 Row 层消费。
3. 若需要返回全量输出（如最后一层），调用 `gather_from_tensor_model_parallel_region`。

---

## 19. 练习题

**题目 1**：对于 `h=2048, ffn=8192, TP=8`，分别计算非 SP 和 SP 模式下：
1. FC1、FC2 的权重 shape（每个 GPU）
2. FC1 输出的 shape（每个 GPU，在 AllGather/ReduceScatter 之后）
3. 每个 Transformer block 的通信量（以 B×S×H 为单位）

**题目 2**：阅读 `_ReduceFromModelParallelRegion` 和 `_ReduceScatterToSequenceParallelRegion` 的源码，解释：
1. 为什么前向 AllReduce 对应反向恒等，而不是反向 scatter？
2. 在求和分解的意义下，两种通信方式（AllReduce vs ReduceScatter）如何保证梯度正确？

**题目 3**：在 GQA 场景（heads=64, kv_heads=8）下，如果 TP=8，每个 GPU 持有几个 Q head 和几个 KV head？当 TP 增大到 16 时会发生什么问题？

**题目 4**：SwiGLU FFN 中 FC1 的 `output_size` 为什么是 `ffn_hidden_size * 2`？切分后每 GPU 的 FC1 输出是多少，FC2 的 `input_size_per_partition` 是多少？

**题目 5**：当 `gather_output=True`（Column Parallel 输出全量）时，反向传播中 dgrad 的形状和通信模式与 `gather_output=False` 有何不同？（提示：AllGather 的反向是 Scatter/ReduceScatter。）

**题目 6**：设 TP=4，SP 模式，序列长度 S=8192。请计算每个 GPU 在 LayerNorm、GEMM、AllGather 各阶段占用的峰值激活内存（以 BF16，H=4096，B=1 计）。与非 SP 模式相比节省了多少内存？

---

## 25. 不同并行模式下参数量与激活内存对比

以 LLaMA-7B（32 层，H=4096，FFN=11008，heads=32，kv_heads=32）为例：

| 配置 | 每GPU参数（BF16）| 每GPU激活（B=1,S=4096,BF16）| 通信量/步 |
|------|------|------|------|
| TP=1 | 13 GB | 6.4 GB | 0 |
| TP=2（非SP） | 6.5 GB | 6.4 GB | 2× AllReduce/层 |
| TP=4（非SP） | 3.3 GB | 6.4 GB | 2× AllReduce/层 |
| TP=4（SP） | 3.3 GB | 1.6 GB（↓4×）| 2× AllGather+2×RS/层 |
| TP=8（SP） | 1.65 GB | 0.8 GB（↓8×）| 2× AllGather+2×RS/层 |

SP 模式在 TP=4 时将激活内存从 6.4 GB 降到 1.6 GB，是训练长序列（S>8192）时 TP 必须配合 SP 使用的原因。参数量随 TP 线性减少（embedding 和 LayerNorm 不切分，但占比小），激活内存在 SP 模式下也随 TP 线性减少。

---

## 26. "阅读 layers.py 的 60 分钟导图"

**第 0-10 分钟：** 阅读 `VocabParallelEmbedding`（198-350 行），理解词表并行的基本逻辑和查表置零机制。

**第 10-25 分钟：** 阅读 `linear_with_grad_accumulation_and_async_allreduce`（400-650 行）的前向和反向，重点关注 `sequence_parallel` 和 `allreduce_dgrad` 两个分支，理解异步通信的调度依赖。

**第 25-40 分钟：** 阅读 `ColumnParallelLinear.__init__`（800-987 行）和 `forward`（994-1101 行），对照本文第 4 节的注释确认每个分支。

**第 40-55 分钟：** 阅读 `RowParallelLinear.__init__`（1150-1300 行）和 `forward`（1302-1365 行），对照本文第 5 节的注释，特别注意 SP 下的 ReduceScatter 路径。

**第 55-60 分钟：** 查看 `sharded_state_dict`（Column 按 axis=0，Row 按 axis=1），理解 checkpoint 分片与 TP 切分轴的对应关系。

---

## 27. 小结（增补：梯度路径速查）

前文 Section 20 已给出架构总结。以下是梯度路径的快速提醒：

**Column Parallel 梯度：**
- dX（input gradient）：AllReduce（非SP）或通过 ReduceScatter 回 S/TP 格式（SP）。
- dW（weight gradient）：局部 GEMM，无通信（每 GPU 只负责自己那列的 wgrad）。

**Row Parallel 梯度：**
- dX：AllGather（SP模式前向 ReduceScatter 的反向）或恒等（非SP模式前向 AllReduce 的反向）。
- dW：局部 GEMM，无通信（每 GPU 只负责自己那行的 wgrad）。

**VocabParallelEmbedding 梯度：**
- 词表范围外的 token 梯度为零（前向置零），词表范围内的梯度正常反传。
- AllReduce 的反向就是"每 GPU 只取自己范围内的梯度，其余丢弃"——对应 Scatter 语义。

记住这张表，就能在调试梯度异常时快速定位是哪种通信原语出了问题。

---

## 28. TP 下的参数命名与 sharded_state_dict

在 Megatron 的 checkpoint 中，TP 切分的参数会带有特殊的 `tensor_parallel` 元信息：

```python
# ColumnParallelLinear.sharded_state_dict
# weight 按 axis=0（输出维）切分
return make_sharded_tensors_for_checkpoint(
    state_dict, prefix,
    {"weight": 0, "bias": 0},   # axis=0
    sharded_offsets,
    tp_group=self.tp_group,
)

# RowParallelLinear.sharded_state_dict
# weight 按 axis=1（输入维）切分
return make_sharded_tensors_for_checkpoint(
    state_dict, prefix,
    {"weight": 1},              # axis=1
    sharded_offsets,
    tp_group=self.tp_group,
)
```

这意味着 TP=4 时，rank 0 的 Column Parallel 权重是全局权重的 `[0:out/4, :]` 部分，rank 1 是 `[out/4:out/2, :]` 部分，以此类推。Megatron 的 checkpoint 工具在 `dist_checkpointing` 模块中处理这种分片元信息，允许在不同 TP 大小之间转换 checkpoint（例如从 TP=8 训练的 checkpoint 加载到 TP=4 的推理环境）。

---

## 29. 常见配置的通信模式速查

| 场景 | 推荐配置 | 原因 |
|------|------|------|
| 单机 8 卡，S=2048 | TP=4，SP=True | 节省激活内存，NVLink 带宽充足 |
| 单机 8 卡，S=512 | TP=2 or 4，SP=False | 短序列激活内存小，SP 收益有限 |
| 多机 TP，S=8192 | TP=8，SP=True | 长序列必须 SP；多机 TP 需要 IB |
| 推理（无反向） | TP=8，`gather_output=True` | 推理不需要 SP，最后层 gather 后接 lm_head |
| MoE + TP | TP for attention，EP for experts | attention 用 TP，expert 层用 EP，两者不叠加 |



---

## 20. 小结

张量并行的实现由三层组成：

1. **`mappings.py`**：经典 **4** 个 TP 原语（CopyTo↔ReduceFrom、Scatter↔Gather）+ SP **3** 件套；另有 AllToAll / last-dim RS-AG 等扩展。AllToAll 辅助函数（`all_to_all_sp2hp`/`hp2sp`）用于 SP 和 HP 格式互换。
2. **`ColumnParallelLinear` / `RowParallelLinear`**（`layers.py`）：把权重矩阵按输出维/输入维切分，通过 `gather_output` 和 `input_is_parallel` 控制与相邻层的衔接；SP 模式下把序列维也一并切分，进一步节省激活内存。梯度路径：Column dgrad 做 AllReduce（异步与 wgrad GEMM 并行），Row wgrad 无需额外通信。
3. **VocabParallelEmbedding + vocab_parallel_cross_entropy**：词表维度的并行，避免在大词表上做全量 AllGather。SwiGLU FFN 中 FC1 输出大小翻倍（含 gate），FC2 输入大小减半（激活后）。

SP 是一个重要的优化：它以相同的通信量换来了 TP 倍的激活内存节省，但要求序列长度 S 能整除 TP，且所有算子都必须正确标注 `sequence_parallel=True`。

下一篇（06）将深入流水线并行：1F1B 调度、气泡分析、P2P 通信 API，以及 VPP 的交错执行。
