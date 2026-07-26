# 精读 Megatron 源码（5）：张量并行——Column / Row 这一对为什么必须配对

> **专栏**：Megatron 源码精读 · 第 5 篇  
> **核心文件**：`megatron/core/tensor_parallel/mappings.py`、`layers.py`

---

张量并行（TP）一句话：

> **把单层内的大型 GEMM 按卡切开，用集合通信在前向/反向把数学结果拼正确。**

Megatron 的经典实现就是 Column Parallel + Row Parallel。别先背公式，先认通信原语，再看两个 Linear 类如何调用它们。

---

## 先读「TP 汇编」：`mappings.py`

在碰 `layers.py` 之前，先认准这些操作：

| 原语 | 典型用途 |
|------|----------|
| `copy_to_tensor_model_parallel_region` | 进入 TP 区域（反向对应 all-reduce） |
| `reduce_from_tensor_model_parallel_region` | 前向 all-reduce（Row 输出） |
| `scatter_to_tensor_model_parallel_region` | 按最后一维切开输入 |
| `gather_from_tensor_model_parallel_region` | 把各卡输出 gather 回来 |
| Sequence Parallel 变体 | all-gather / reduce-scatter |

上层 Linear 只是在合适时机调用这些「指令」。精读时建议点进对应的 `torch.autograd.Function`，看反向做了什么——TP 的坑一半出在反向通信不对称。

---

## `ColumnParallelLinear`：按输出列切

源码开篇注释写得很清楚：

```778:793:megatron/core/tensor_parallel/layers.py
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    ...
        gather_output:
            If true, call all-gather on output and make Y available to all GPUs,
            otherwise, every GPU will have its output which is Y_i = XA_i
```

对 \(Y = XA + b\)，权重按**输出维**切：

```text
每卡 weight ≈ (output_size / TP, input_size)
```

前向直觉：

1. 把输入放到正确的 TP/SP 布局  
2. 本地 matmul  
3. `gather_output=True` 则 all-gather；否则保持分区输出给下一层  

典型挂载：Attention 的 QKV、MLP 的 `fc1`。

读 `forward` 时盯住：`gather_output`、`sequence_parallel`、以及异步通信+梯度融合路径（如 `LinearWithGradAccumulationAndAsyncCommunication`）。

---

## `RowParallelLinear`：按输入行切

权重按**输入维**切：

```text
每卡 weight ≈ (output_size, input_size / TP)
```

前向直觉：

1. `input_is_parallel=False` 时先 scatter  
2. 本地 matmul  
3. 输出 all-reduce；开 Sequence Parallel 时常常是 reduce-scatter  

源码里还有硬约束（开 SP 时）：

```1209:1210:megatron/core/tensor_parallel/layers.py
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")
```

典型挂载：Attention 输出投影、MLP 的 `fc2`。

---

## 为什么必须「一列一行」配对

以 MLP 为例（忽略 SwiGLU 细节）：

```text
X  --ColumnParallel--> 激活（已按 TP 分区）--RowParallel--> Y
```

Column 切完后，中间激活天然是分区的；Row 期望 `input_is_parallel=True`，本地算完再在 TP 组上还原。

Attention 同理：

```text
QKV(Column) → 各卡算各自分头 → Out(Row)
```

这就是层内模型并行能与数学等价的几何结构。**上下游的 `gather_output` / `input_is_parallel` 必须匹配**，否则不是 shape 错就是数值错。

---

## Sequence Parallel：别当成第五种并行

开启 SP 后，**激活在序列维也被切开**，省激活显存；代价是进入需要完整序列的算子前 all-gather，出去再 reduce-scatter。

要点：

- 通常要求 `TP > 1`  
- LayerNorm 放置会影响通信点  
- 它更像 TP 的激活布局优化，不是独立并行维  

---

## Vocab 并行与 RNG

同目录还有：

- `VocabParallelEmbedding`：词表按 TP 切  
- `vocab_parallel_cross_entropy`：在切分词表上算 loss，避免 gather 巨大 logits  

这与 `GPTModel(parallel_output=True)` 一脉相承。

另外 `tensor_parallel/random.py` 的 `CudaRNGStatesTracker` 保证 dropout / 初始化在「各 TP 卡一致 vs 独立」时行为正确——排随机性相关 bug 时别忘了这条线。

---

## 建议精读顺序

```text
mappings.py
  → ColumnParallelLinear.__init__ / forward
  → RowParallelLinear.__init__ / forward
  → 回到 SelfAttention / MLP 看如何实例化二者
  → VocabParallelEmbedding + vocab_parallel_cross_entropy
```

---

## 强烈建议的对照实验

设 `TP=2`，打印：

1. `fc1`（Column）的 `weight.shape`  
2. `fc2`（Row）的 `weight.shape`  

再和 `TP=1` 对比。你会立刻看到「输出维减半 / 输入维减半」的实物证据。

手推一题：`hidden=1024, ffn=4096, TP=4` 时，每卡 `fc1` weight 应是 `(1024, 1024)` 还是别的？（注意 Megatron 里 weight 布局约定，以源码 `out_features/in_features` 为准。）

---

## 常见坑

1. Row 的 `input_is_parallel` 与上游 Column 的 `gather_output` 不匹配  
2. SP 开启但下游仍按非并行输入处理  
3. Expert 线性层可能走 expert TP 组（`is_expert`）  
4. 忘记 `set_tensor_model_parallel_attributes` → checkpoint / DistOpt 元数据错  

---

下一篇上到「层间」：流水线并行的 1F1B、microbatch，以及 `get_forward_backward_func` 如何选型。TP 解决一层怎么切开，PP 解决一层层怎么排进流水线。
