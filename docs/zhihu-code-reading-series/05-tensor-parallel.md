# 知乎系列 05｜张量并行：Column / Row Parallel Linear

> 目标：从 `mappings.py` 的通信原语，读到 `ColumnParallelLinear` / `RowParallelLinear` 如何拼出一层 Attention/MLP。

---

## TP 一句话

**张量并行把单层内的大型 GEMM 按卡切开**，用集合通信在前向/反向把数学结果拼正确。  

Megatron 经典做法来自「Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism」那一套 Column / Row 切分；代码落点主要在：

- `megatron/core/tensor_parallel/mappings.py`  
- `megatron/core/tensor_parallel/layers.py`  
- `megatron/core/tensor_parallel/random.py`  
- `megatron/core/tensor_parallel/cross_entropy.py`

---

## 先读通信原语：`mappings.py`

在读 Linear 之前，先认准几类操作（名字以源码为准）：

| 方向 | 典型用途 |
|------|----------|
| `copy_to_tensor_model_parallel_region` | 进入 TP 区域（反向对应 all-reduce） |
| `reduce_from_tensor_model_parallel_region` | 前向 all-reduce（Row 输出） |
| `scatter_to_tensor_model_parallel_region` | 把输入按最后一维切到各 TP rank |
| `gather_from_tensor_model_parallel_region` | 把各卡输出 gather 回来 |
| Sequence Parallel 变体 | all-gather / reduce-scatter 与 SP 配合 |

把这些函数当成「TP 汇编指令」：上层 Linear 只是在合适的时机调用它们。

---

## `ColumnParallelLinear`：按输出列切

文件：`layers.py` 中 `ColumnParallelLinear`

对 \(Y = X A + b\)，权重 \(A\) 按**输出维度**切到各 TP 卡：

```text
每卡 weight shape ≈ (output_size / TP, input_size)
```

前向直觉：

1. （必要时）让输入处于正确的 TP/SP 布局  
2. 本地 matmul  
3. 若 `gather_output=True`，all-gather 拼回完整输出；否则保持分区输出给下一层  

典型挂载位置：

- Attention 的 QKV 投影  
- MLP 的第一层（`fc1`）

**读 `forward` 时盯住**：`gather_output`、`sequence_parallel`、`async` 通信与 grad 融合路径（如 `LinearWithGradAccumulationAndAsyncCommunication`）。

---

## `RowParallelLinear`：按输入行切

同一文件中的 `RowParallelLinear`

权重按**输入维度**切：

```text
每卡 weight shape ≈ (output_size, input_size / TP)
```

前向直觉：

1. 若 `input_is_parallel=False`，先 scatter 输入  
2. 本地 matmul  
3. 对输出做 all-reduce；若开启 Sequence Parallel，则常常是 reduce-scatter  

典型挂载位置：

- Attention 输出投影  
- MLP 第二层（`fc2`）

Bias 通常不切分（每卡完整 bias，或按实现约定处理）——读构造函数注释比猜更准。

---

## 为什么 Attention/MLP 要「一列一行」配对

以 MLP 为例（忽略融合与 SwiGLU 细节）：

```text
X  --ColumnParallel--> 激活  --RowParallel--> Y
```

Column 切完后，中间激活在 TP 维上天然是分区的；Row 期望 `input_is_parallel=True`，本地算完再在 TP 组上还原。  

Attention 同理：QKV（Column）→ 各卡算各自分头 → Out（Row）。  

这就是「层内模型并行」能跟数学等价的关键几何结构。

---

## Sequence Parallel（SP）你要建立的直觉

开启 SP 后，**激活在序列维也被切开**，以降低激活显存；代价是在进入需要完整序列的算子前做 all-gather，出去时再 reduce-scatter。  

阅读时注意：

- SP 与 TP 强绑定（通常 `SP` 开启要求 `TP>1`）  
- LayerNorm 等与 SP 的放置会影响通信点  
- `RowParallelLinear` 在 SP 下的输出通信形态会变化  

不要把 SP 理解成第五种「独立并行维」——它更像 TP 的激活布局优化。

---

## Vocab 并行：Embedding 与 Cross Entropy

同目录还有：

- `VocabParallelEmbedding`：词表按 TP 切  
- `vocab_parallel_cross_entropy`：在切分词表上算 loss，避免 gather 巨大 logits  

这与 `GPTModel` 的 `parallel_output=True` 一脉相承：尽量让 logits 保持分区，直到 loss 需要时再以高效方式归约。

---

## RNG：为什么 TP 需要特殊种子

文件：`tensor_parallel/random.py`

`model_parallel_cuda_manual_seed` / `CudaRNGStatesTracker` 保证：

- Dropout 等需要「各 TP 卡一致」或「各卡独立」的随机性时，行为可复现且正确  
- 与 data parallel 副本之间的种子策略不打架  

读到 dropout / 初始化相关 bug 时，别忘了这条线。

---

## 建议阅读顺序

```text
mappings.py（认齐原语）
  → ColumnParallelLinear.__init__ / forward
  → RowParallelLinear.__init__ / forward
  → 回到 SelfAttention / MLP 看它们如何实例化这两个类
  → VocabParallelEmbedding + vocab_parallel_cross_entropy
  → random.py（需要排随机性时再精读）
```

---

## 对照实验（强烈推荐）

配置 `TP=2`，在下面两处打印 `weight.shape`：

1. `ColumnParallelLinear`（例如 `fc1`）  
2. `RowParallelLinear`（例如 `fc2`）  

再与 `TP=1` 对比。你会立刻看到「输出维减半 / 输入维减半」的实物证据。

---

## 常见坑

1. **Row 的 `input_is_parallel` 与上游 Column 的 `gather_output` 不匹配** → shape 或数值错误。  
2. **SP 开启但下游仍按非并行输入处理** → 通信与 shape 双重灾难。  
3. **Expert 线性层** 可能走 expert TP 组（`is_expert`）——不要假设永远是 dense TP group。  
4. **忘记 TP 属性标记**（`set_tensor_model_parallel_attributes`）→ checkpoint / DistOpt 分片元数据可能错。  

---

## 本周作业

1. 手推 `hidden=1024, TP=4` 时，`fc1`（Column，`ffn=4096`）每卡 weight 的 shape。  
2. 在 `mappings.py` 找到 forward all-reduce 对应的函数，写下它在 autograd 反向会做什么（读 `torch.autograd.Function` 实现）。  
3. 指出 `SelfAttention` 里哪个子模块是 Column、哪个是 Row。  

下一篇上到「层间」：流水线并行的 1F1B 与 microbatch 调度。
