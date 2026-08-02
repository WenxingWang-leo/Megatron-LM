# 精读 Megatron 源码（3）：GPTModel 全拆解——从 Config、Spec 到每一层计算

> **前置阅读**：本文假设你读过第 1 篇（地图）和第 2 篇（训练循环）。目标：深入 `GPTModel` 内部，读懂 `TransformerConfig` 设计哲学、`ModuleSpec` 插件机制，以及一个 token 从 embedding 到 logits 的完整计算路径。

---

## 0. 为什么要单独讲 GPTModel？

在第 2 篇，我们到达了 `GPTModel.forward()` 的大门，但没有进去。这里面有三个让初学者最困惑的设计：

1. **`TransformerConfig` 有 400+ 个字段**，怎么读？哪些字段最重要？
2. **`ModuleSpec` 是什么？** 为什么不直接 `nn.Linear` 而是要用 Spec？
3. **PP 分段之后，同一个 `GPTModel` 类在不同 GPU 上结构是不一样的**，这是怎么实现的？

读完本文，这三个问题都会有清晰的答案。

---

## 1. GPTModel 的整体结构

文件：`/workspace/megatron/core/models/gpt/gpt_model.py`

`GPTModel` 继承自 `LanguageModule`，后者继承自 `MegatronModule`，后者继承自 `torch.nn.Module`。

### 1.1 `__init__` 中的条件性组件

```python
class GPTModel(LanguageModule):
    def __init__(self, config, transformer_layer_spec, vocab_size,
                 max_sequence_length, pre_process=True, post_process=True,
                 mtp_block_spec=None, ...):

        # ── 位置编码类型（rope / yarn / mrope / learned_absolute / none）──
        self.position_embedding_type = config.position_embedding_type

        # ── MTP 判断：是否在此 rank 处理 Multi-Token Prediction ──
        self.mtp_process = mtp_block_spec is not None and mtp_on_this_rank(
            layout=config.pipeline_model_parallel_layout,
            mtp_num_layers=config.mtp_num_layers,
            vp_stage=vp_stage,
        )

        # ── 仅 pre_process=True 或 mtp_process=True 的 stage ──
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )

        # ── 位置编码（各类 RoPE 变体）──
        if position_embedding_type == 'rope' and not config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
                rope_scaling=rope_scaling,
                ...
            )
        elif position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(...)
        elif position_embedding_type == 'mrope':
            self.rotary_pos_emb = MultimodalRotaryEmbedding(...)
        # 'learned_absolute' 或 'none'：位置信息由 LanguageModelEmbedding 内部处理

        # ── 必有的组件（所有 PP stage 都有）──
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self.pg_collection,
            vp_stage=vp_stage,
        )

        # ── Multi-Token Prediction Block（如果 mtp_process=True）──
        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(
                config=config,
                spec=self.mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=self.pg_collection,
            )

        # ── 仅 post_process=True 的 stage（PP 最后 rank）──
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                config=config,
                bias=False,
                gather_output=not self.parallel_output,  # True 时收集 TP 分片
                ...
            )
```

### 1.2 PP=4 时各 stage 的实际组件表

| PP rank | `pre_process` | `post_process` | 拥有的组件 |
|---------|---------------|----------------|-----------|
| 0 | **True** | False | `embedding` + `rotary_pos_emb` + `decoder`（layers 0-7） |
| 1 | False | False | `decoder`（layers 8-15） |
| 2 | False | False | `decoder`（layers 16-23） |
| 3 | False | **True** | `decoder`（layers 24-31） + `output_layer` |

**关键洞察**：`TransformerBlock`（即 `self.decoder`）在每个 PP stage 上都存在，但里面包含的层数不同（通过 `get_num_layers_to_build` 计算）。`embedding` 和 `output_layer` 是条件性的。

---

## 2. `_preprocess` 的详细逻辑：三条路径

文件：`/workspace/megatron/core/models/gpt/gpt_model.py`，第 305 行。

`_preprocess` 有三条执行路径，由 `decoder_input` 是否为 None 和 `pre_process` 标志决定：

```python
def _preprocess(self, input_ids, position_ids, decoder_input=None, ...):

    # ── 路径 A：中间 PP stage（由 set_input_tensor 填入激活）──
    if decoder_input is not None:
        pass  # decoder_input 已由 schedules.py 的 set_input_tensor() 填好

    # ── 路径 B：PP stage 0（有 embedding）──
    elif self.pre_process:
        decoder_input = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids
        )
        # 如果开启 SP 且 embedding 未内部 scatter，则在这里 scatter
        if self.config.sequence_parallel and not self.embedding.scatter_to_sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input, group=self.pg_collection.tp
            )

    # ── 路径 C：理论上不会到达的分支 ──
    else:
        decoder_input = None  # 会由 set_input_tensor() 在之后填入

    # ── RoPE 计算（所有非 MLA 模型）──
    if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, self.decoder, decoder_input, self.config, packed_seq_params
        )
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, ...)

    return (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin,
            sequence_len_offset, padding_mask)
```

**`set_input_tensor` 的工作原理**：

当流水线并行调度（`schedules.py`）通过 P2P 从上一个 stage 接收到激活张量时，会调用：

```python
model.set_input_tensor(recv_tensor)
# 这会调用:
# self.decoder.set_input_tensor(recv_tensor)
# 在 TransformerBlock.forward 时使用 self.input_tensor 而非传入的 hidden_states=None
```

因此，对于中间 PP stage，`_preprocess` 接收到的 `decoder_input` 已经是上一个 stage 传来的激活，而 `input_ids` 和 `position_ids` 对它无意义（它们是 None）。

### 2.1 SP 散列的时机细节

Embedding 层可以在两个地方触发 SP scatter：
1. **`LanguageModelEmbedding` 内部**：当 `reduce_scatter_embeddings=True` 时（仅在 RoPE 且无 position embedding 时启用），embedding 会直接输出 `[S/TP, B, H]` 格式的张量。
2. **`_preprocess` 中**：当 `scatter_to_sequence_parallel=True` 但 embedding 没有内部 scatter 时，在这里手动散列。

默认情况下（`scatter_embedding_sequence_parallel=True`），embedding 会在 `LanguageModelEmbedding.forward` 内部完成 scatter，`_preprocess` 中的 if 分支不会被触发。

---

## 3. `_postprocess` 的关键分支

文件：`/workspace/megatron/core/models/gpt/gpt_model.py`，第 609 行。

```python
def _postprocess(self, hidden_states, labels, mtp_in_postprocess=None, ...):

    # ── 分支 A：非最后 PP stage，直接返回 hidden states ──
    if not self.post_process:
        return hidden_states   # [S/TP, B, H] (SP 模式) 或 [S, B, H]

    # ── 分支 B：MTP 处理（仅当 mtp_process=True 且 不在推理模式）──
    if mtp_in_postprocess and not in_inference_mode:
        # mtp 模块计算额外 D 个 token 的损失
        hidden_states = process_mtp_loss(
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            output_layer=self.output_layer,
            output_weight=output_weight,
            ...
        )

    # ── 分支 C：计算 logits（所有 post_process=True 的 stage）──
    logits, _ = self.output_layer(
        hidden_states,
        weight=output_weight,  # 若 share_embeddings，使用共享权重
        runtime_gather_output=runtime_gather_output,
    )
    # 注意：parallel_output=True（默认训练模式）时不 gather，
    # logits 保持 [S, B, vocab/TP] 分片，由 vocab_parallel_cross_entropy 处理

    if labels is None:
        # 推理模式：返回 [B, S, V] 格式的 logits
        return logits.transpose(0, 1).contiguous()

    # ── 分支 D：训练模式，计算 per-token 交叉熵 loss ──
    loss = self.compute_language_model_loss(labels, logits)
    return loss   # [B, S]，每个位置的 CE loss
```

**`parallel_output=True` 的意义**：

在训练时，`output_layer` 是 `ColumnParallelLinear`，默认 `gather_output=False`（即 `parallel_output=True`）。这意味着 logits 保持在 `[S, B, V/TP]` 的分片状态，不做 AllGather。`vocab_parallel_cross_entropy` 可以直接在这个分片上计算交叉熵，避免了将完整 vocab 的 logits 收集到每张卡上，节省显存和通信。

---

## 4. RoPE vs 学习型绝对位置编码：深度对比

### 4.1 学习型绝对位置编码（`learned_absolute`）

这是原始 GPT/BERT 的方案：

```python
# LanguageModelEmbedding.__init__
if position_embedding_type == 'learned_absolute':
    self.position_embeddings = torch.nn.Embedding(
        max_sequence_length, config.hidden_size
    )
```

```python
# LanguageModelEmbedding.forward
# 词嵌入 + 位置嵌入直接相加
embeddings = word_embeddings + position_embeddings(position_ids)
```

**优点**：简单直接，位置信息在 embedding 阶段融入。  
**缺点**：无法外推到超过 `max_sequence_length` 的长度；位置嵌入是独立参数，不直接利用相对位置信息。

### 4.2 旋转位置编码（RoPE）

RoPE 不在 embedding 阶段处理，而是在**每个注意力层**中，对 Q 和 K 分别施加旋转变换：

```python
# 在 SelfAttention.forward 中
q, k, v = linear_qkv(hidden_states)  # 分离 QKV
q = apply_rotary_pos_emb(q, rotary_pos_emb)  # 旋转 Q
k = apply_rotary_pos_emb(k, rotary_pos_emb)  # 旋转 K
# V 不旋转
```

旋转变换公式（对每对维度 `(2i, 2i+1)`）：

```
q_rotated[2i]   =  q[2i] * cos(m*θ_i) - q[2i+1] * sin(m*θ_i)
q_rotated[2i+1] =  q[2i] * sin(m*θ_i) + q[2i+1] * cos(m*θ_i)
```

其中 `m` 是 token 的位置，`θ_i = rotary_base^(-2i/d)`（`d` 是 kv_channels）。

**对比表**：

| 特性 | learned_absolute | RoPE | YaRN |
|------|-----------------|------|------|
| 位置信息注入层 | embedding | 每层 attention | 每层 attention |
| 长度外推能力 | 无（截断到 max_seq_len） | 一定程度有（频率外推） | 显著增强（专为长度外推设计） |
| 额外参数量 | `max_seq_len × H` | 0（无参数，纯计算） | 0（无参数） |
| 相对位置感知 | 弱（绝对位置相减不等于相对位置）| 强（内积保持相对位置信息） | 强 |
| 代表模型 | 原始 GPT-2、BERT | LLaMA、Mistral | Code LLaMA 长序列版本 |
| Megatron 配置 | `position_embedding_type='learned_absolute'` | `'rope'` | `'yarn'` |

**YaRN 的关键参数**（在 `TransformerConfig` 中）：

```python
yarn_rotary_scaling_factor: float  # 缩放因子（通常 8 或 16，对应 64K/128K 上下文）
yarn_original_max_position_embeddings: int  # 原始训练的最大长度（如 4096）
yarn_beta_fast: int = 32   # 高频部分的动态缩放
yarn_beta_slow: int = 1    # 低频部分的动态缩放
```

### 4.3 `position_embedding_type='none'` 的用途

设置为 `'none'` 时，既不加学习型位置嵌入，也不应用 RoPE。主要用于：
- 使用 ALiBi（Attention with Linear Biases）等其他位置编码方案
- 某些不需要位置编码的实验性模型

---

## 5. `share_embeddings_and_output_weights`：跨 PP 共享权重

### 5.1 什么是权重共享

当 `share_embeddings_and_output_weights=True` 时，词嵌入矩阵（shape `[vocab_size, H]`）和 output_layer 的权重矩阵（shape `[H, vocab_size]` 的转置）指向**同一份参数**。

这在语言模型中是常见做法（"input-output weight tying"），理论依据是：词嵌入空间和 logit 空间应该是对偶的。GPT-2 使用了这个设计。

### 5.2 PP 场景下的挑战

问题在于：`embedding` 在 PP stage 0（`pre_process=True`），而 `output_layer` 在 PP stage 的最后（`post_process=True`）。它们在**不同的 GPU** 上！

Megatron 的解决方案（来自 `setup_embeddings_and_output_layer`）：

1. **初始化**：在最后 stage 创建一个 `output_layer`，其权重初始化为 0，并标记为 `shared=True`
2. **首步 AllReduce**：训练开始前，通过 `torch.distributed.all_reduce`（在 embedding 所在 rank 和 output_layer 所在 rank 之间）将 embedding 权重同步到 output_layer
3. **梯度同步**：每次参数更新后，embedding 梯度和 output_layer 梯度通过 AllReduce 合并，确保两端权重保持一致

```python
# language_module.py（精简）
def setup_embeddings_and_output_layer(self):
    if not self.share_embeddings_and_output_weights:
        return

    if self.config.pipeline_model_parallel_size == 1:
        # 单 PP stage：embedding 和 output_layer 在同一张卡，直接共享即可
        self.shared_embedding_or_output_weight().zero_out_wgrad = True
        return

    # 多 PP stage：最后 stage 的 output_layer.weight 初始化为 0
    if self.post_process and not self.pre_process:
        weight = self.shared_embedding_or_output_weight()
        weight.data.fill_(0)
        weight.shared = True
        weight.shared_embedding = True
```

### 5.3 实际效果

| 配置 | 参数量变化 | 显存节省 | 训练影响 |
|------|-----------|---------|----------|
| `share_embeddings=False`（默认） | 完整 `vocab_size × H` 在 stage 0 和最后 stage 各一份 | 无 | 无 |
| `share_embeddings=True` | 逻辑上共享，但仍需在两端各保留一份副本（PP 场景） | PP=1 时节省 `vocab_size × H`，PP>1 时实际没有节省参数 | 需要额外的跨 stage AllReduce |

---

## 6. 激活重计算：选择性 vs 全量

文件：`/workspace/megatron/core/transformer/transformer_config.py`，第 508 行。

### 6.1 核心配置字段

```python
recompute_granularity: Optional[Literal['full', 'selective']] = None
# None = 不重计算，保存所有激活（最快，最耗显存）
# 'selective' = 只重计算指定子模块
# 'full' = 重计算整个 Transformer 层

recompute_method: Optional[Literal['uniform', 'block']] = None
# 仅 full 模式有效
# 'uniform' = 均匀分配：每 recompute_num_layers 层为一个重计算单元
# 'block' = 块分配：前 recompute_num_layers 层重计算，其余不重计算

recompute_num_layers: Optional[int] = None
# uniform: 每个重计算单元的层数
# block: 重计算的层数上限

recompute_modules: Optional[List[str]] = None
# selective 模式：重计算哪些子模块
# 默认: ["core_attn"]
# 可选: "core_attn", "mlp", "moe", "layernorm", "mla_up_proj", ...
```

### 6.2 各配置的激活保存情况（同一套数字算到底）

先固定一个 **Llama-2-70B 风格** 假设（后文所有 GB 都只用这组数，避免前后矛盾）：

| 符号 | 取值 | 含义 |
|------|------|------|
| \(B\) | 1 | micro batch |
| \(S\) | 4096 | sequence length |
| \(H\) | 8192 | hidden size |
| \(L\) | 80 | 层数（70B 是 ~80 层，不是 32） |
| \(N_h\) | 64 | attention heads |
| \(d\) | 128 | head dim（\(H/N_h\)） |
| dtype | fp16 = 2 bytes | |

**基本积木**（后面反复用）：

```text
一个 [S, B, H] 激活 = 4096 × 1 × 8192 × 2 B
                    = 67,108,864 B ≈ 64 MiB
```

**若物化全注意力矩阵**（非 FlashAttention 的朴素路径）：

```text
[B, N_h, S, S] = 1 × 64 × 4096 × 4096 × 2 B
               = 2,147,483,648 B = 2.0 GiB / 层
```

注意：旧文曾写「单层约 2GB」又写「注意力矩阵约 4GB」，还用了 \(N_h=32\)、\(L=32\)，三处互不一致。  
正确对照是：**注意力矩阵本身就是约 2.0 GiB/层**（上式）；「整层其它激活」是另一笔账。

#### 不重计算（`recompute_granularity=None`）时，一层大概存什么？

| 类别 | 代表 shape | 每层约占用（fp16） |
|------|------------|-------------------|
| 残差 / LN / 投影等「线性尺寸」激活 | 多个 `[S,B,H]` 及 FFN 中间态 | 约 **0.8–1.5 GiB**（随是否 SwiGLU、具体 checkpoint 边界略变） |
| 注意力分数矩阵（若物化） | `[B,N_h,S,S]` | **2.0 GiB** |
| **单层合计（朴素 attn）** | | **约 3 GiB 量级** |
| **80 层合计（仅激活，粗估）** | | **约 200+ GiB 量级**（远超单卡；还需叠加参数/梯度/优化器状态） |

直觉：长序列下 **S² 项（注意力矩阵）往往是单层激活里最大的一块**；  
「线性尺寸」激活随 \(S\) 一次方涨，注意力矩阵随 \(S^2\) 涨。

> 现代 TE / FlashAttention 路径常常**根本不落盘完整 `[B,N_h,S,S]`**。下面谈 selective 时，把它理解为「避开注意力核心里最吃显存的中间状态」；数字上仍用上面 2.0 GiB 作为「若物化 S² 矩阵」的对照上限，方便建立数量级。

#### `selective` + `recompute_modules=['core_attn']`（推荐）

行为：

- **保留**：QKV 投影输出等，作为重算输入  
- **不长期保存 / 反向时重算**：`core_attn`（`QK^T → softmax → ×V`）里的大块中间态  

显存账（同一套假设）：

```text
省下的主项 ≈ 注意力 S² 工作区
            ≈ 2.0 GiB / 层   （不是 4 GiB，也不是「整层只有 2 GiB」那种含糊说法）

相对「朴素整层 ~3 GiB」：
  约省 2.0 / 3 ≈ 60% 的单层激活
  （若底层已是 FlashAttention、本来就没存满 S²，实际节省会小于这个上限）
```

80 层粗估：仅这一项就可能对应 **O(100 GiB)** 量级的差额（\(80 × 2.0\)），所以 selective 对长序列特别划算。

#### `full` + `recompute_method='uniform'`

每个重计算单元只永久保存单元入口激活，单元内部前向在反传时重跑。

- 激活峰值可再到「只留入口」的量级（常再砍一大截）  
- 代价：这部分前向算两遍，训练时间常见 +30%–40% 量级（视实现/硬件而定）

#### 实战建议

```text
显存充裕：recompute_granularity=None（最快）
显存紧张：selective + core_attn（默认推荐平衡点）
极限显存：full + uniform（最省激活，最慢）
```

**读数口诀**（避免再混）：

1. 先写出配置：\(B,S,H,N_h,L\)、dtype  
2. 线性激活用 \(S·B·H·bytes\)（上例 64 MiB / 张）  
3. 注意力矩阵用 \(B·N_h·S·S·bytes\)（上例 **2.0 GiB / 层**）  
4. 整层 ≈ 线性类之和 +（是否物化的）注意力矩阵；不要用「\(S×B×H×4$」这种未定义系数去「约等于 2GB」

---

## 7. TransformerConfig：字段分组详解

文件：`/workspace/megatron/core/transformer/transformer_config.py`

`TransformerConfig` 有超过 400 个字段，继承自 `ModelParallelConfig`（包含并行配置）。下面按功能分组介绍最重要的字段：

### 7.1 模型架构核心字段

| 字段名 | 类型 | 默认值 | 含义 |
|--------|------|--------|------|
| `num_layers` | int | 0 | Transformer 层数（整个模型，非单 PP stage） |
| `hidden_size` | int | 0 | 隐藏层维度 H |
| `num_attention_heads` | int | 0 | 注意力头数 NH |
| `num_query_groups` | Optional[int] | None→NH | GQA 的 KV 头组数（若 < NH 即 GQA） |
| `ffn_hidden_size` | Optional[int] | None→4H | FFN 中间层维度（SwiGLU 时通常为 8H/3） |
| `kv_channels` | Optional[int] | None→H/NH | 每个注意力头的维度 |

**自动填充逻辑**（在 `__post_init__` 中）：

```python
# megatron/core/transformer/transformer_config.py, 第 1309-1316 行
if self.ffn_hidden_size is None:
    self.ffn_hidden_size = 4 * self.hidden_size

if self.kv_channels is None:
    self.kv_channels = self.hidden_size // self.num_attention_heads

if self.num_query_groups is None:
    self.num_query_groups = self.num_attention_heads
```

### 7.2 并行相关字段（继承自 ModelParallelConfig）

| 字段名 | 含义 |
|--------|------|
| `tensor_model_parallel_size` | TP 并行度（`--tensor-model-parallel-size`） |
| `pipeline_model_parallel_size` | PP 并行度 |
| `virtual_pipeline_model_parallel_size` | VPP 虚拟流水线并行度 |
| `sequence_parallel` | 是否开启 SP（需要 `tensor_model_parallel_size > 1`） |
| `context_parallel_size` | CP 并行度 |

### 7.3 激活重计算（内存 vs 速度 tradeoff）

| 字段名 | 含义 |
|--------|------|
| `fp16`/`bf16`/`fp8` | 数值精度（bf16 推荐，fp8 需 TE） |
| `fp32_residual_connection` | 残差连接用 fp32（提升稳定性） |
| `recompute_granularity` | `'full'`：重计算整层；`'selective'`：只重计算特定子模块 |
| `recompute_method` | `'uniform'`：均匀分配；`'block'`：前 N 层重计算 |
| `recompute_num_layers` | 每个 recompute 单元的层数 |
| `recompute_modules` | selective 模式：重计算哪些子模块（默认 `['core_attn']`） |

`recompute_granularity='full'` 在反向传播时重新执行前向，以时间换空间。70B 模型光激活就可能占数十 GB，重计算可以将峰值显存减少 60-80%，代价是增加约 30-40% 训练时间。

### 7.4 TransformerConfig 字段分组全表

```
TransformerConfig
├── 模型架构: num_layers, hidden_size, num_attention_heads, num_query_groups,
│            ffn_hidden_size, kv_channels, gated_linear_unit, add_bias_linear
├── 并行策略: tensor/pipeline/virtual_pipeline_model_parallel_size,
│            sequence_parallel, context_parallel_size
├── 数值精度: fp16, bf16, fp8, fp4, fp32_residual_connection, layernorm_epsilon
├── 激活重计算: recompute_granularity, recompute_method, recompute_num_layers,
│             recompute_modules
├── MTP 配置: mtp_num_layers, mtp_loss_scaling_factor
├── MoE 专家: num_moe_experts, moe_router_topk, moe_ffn_hidden_size
├── 位置编码: position_embedding_type, rotary_base, rope_scaling
└── 推理优化: flash_decode, cuda_graph_impl, inference_fuse_tp_communication
```

---

## 8. ModuleSpec：插件化装配机制

文件：`/workspace/megatron/core/transformer/spec_utils.py`

### 8.1 ModuleSpec 的定义

```python
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]   # 模块类（或 (module_path, ClassName) 元组）
    params: dict = {}             # 构造时的额外参数
    submodules: object = None     # 子模块的 spec（嵌套 Spec）
    metainfo: dict = {}           # 元信息（不用于构造）
```

### 8.2 `build_module` 算法

```python
def build_module(spec_or_module, *args, **kwargs):
    # 情况 1：传入的是函数，直接返回
    if isinstance(spec_or_module, types.FunctionType):
        return spec_or_module

    # 情况 2：传入的是已导入的类
    if isinstance(spec_or_module, type):
        module = spec_or_module
        return module(*args, **kwargs)

    # 情况 3：传入的是 ModuleSpec 实例
    if isinstance(spec_or_module, ModuleSpec):
        # 获取模块类（可能需要动态 import）
        module = get_module(spec_or_module)
        # 合并 spec 中的 params 和调用时的 kwargs
        combined_kwargs = {**spec_or_module.params, **kwargs}
        # 如果有 submodules，也传进去
        if spec_or_module.submodules is not None:
            combined_kwargs['submodules'] = spec_or_module.submodules
        return module(*args, **combined_kwargs)
```

**为什么需要 ModuleSpec？**

直接 `import` 再 `nn.Linear()` 是最简单的，但 Megatron 需要支持多个后端：

- **Local backend**：纯 PyTorch 实现（`ColumnParallelLinear`）
- **Transformer Engine backend**：NVIDIA 优化的 FP8/Flash Attention（`TEColumnParallelLinear`）
- **Inference backend**：进一步优化的推理专用实现

通过 `ModuleSpec`，调用方不需要知道具体用哪个类，只需要调用 `build_module(spec, ...)`，装配逻辑由 spec 决定。这是一个标准的**策略模式（Strategy Pattern）**。

### 8.3 Local backend 与 TE backend 的具体类名对比

| 模块角色 | Local backend（`LocalSpecProvider`） | TE backend（`TESpecProvider`） |
|---------|--------------------------------------|-------------------------------|
| QKV 投影 | `ColumnParallelLinear` | `TEColumnParallelLinear` |
| 输出投影 | `RowParallelLinear` | `TERowParallelLinear` |
| FFN 上投影 | `ColumnParallelLinear` | `TEColumnParallelLinear` （或 `TELayerNormColumnParallelLinear`，融合 LN） |
| FFN 下投影 | `RowParallelLinear` | `TERowParallelLinear` |
| 核心注意力 | `DotProductAttention` | `TEDotProductAttention`（支持 Flash Attention） |
| LayerNorm | `RMSNorm` / `WrappedTorchNorm` | `TENorm`（融合 CUDA kernel） |
| MLP（整体） | `MLP` | `TELayerNormMLP` / `TEFusedMLP`（融合 LN+Linear+GELU） |

**如何选择**：
- 开发/调试：使用 `get_gpt_layer_local_spec()`，不需要安装 TE，代码更透明
- 生产训练：使用 `get_gpt_layer_with_transformer_engine_spec()`，性能更好

```python
# 来自 gpt_layer_specs.py
def get_gpt_layer_local_spec(...) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=get_gpt_layer_local_submodules(...)
    )
```

### 8.4 ModuleSpec 的嵌套结构举例（Local backend）

```python
# 注意力子模块 spec
attention_submodules = SelfAttentionSubmodules(
    linear_qkv  = ModuleSpec(module=ColumnParallelLinear),  # Q/K/V 合并投影
    core_attention = ModuleSpec(module=DotProductAttention),  # 核心注意力计算
    linear_proj = ModuleSpec(module=RowParallelLinear),      # 输出投影
)

# 单层 Transformer 的完整 spec
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm = ModuleSpec(module=RMSNorm),
        self_attention  = ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=attention_submodules,
        ),
        self_attn_bda   = get_bias_dropout_add,   # 函数，不是类
        pre_mlp_layernorm = ModuleSpec(module=RMSNorm),
        mlp = ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1 = ModuleSpec(module=ColumnParallelLinear),
                linear_fc2 = ModuleSpec(module=RowParallelLinear),
            )
        ),
        mlp_bda = get_bias_dropout_add,
    )
)
```

**视觉化**：

```
TransformerLayer
├── input_layernorm: RMSNorm
├── self_attention: SelfAttention
│   ├── linear_qkv: ColumnParallelLinear     ← QKV 合并，按列切分
│   ├── core_attention: DotProductAttention   ← Q×K^T softmax V
│   └── linear_proj: RowParallelLinear        ← 输出投影，按行切分
├── self_attn_bda: bias_dropout_add
├── pre_mlp_layernorm: RMSNorm
├── mlp: MLP
│   ├── linear_fc1: ColumnParallelLinear      ← FFN 上投影，按列切分
│   └── linear_fc2: RowParallelLinear         ← FFN 下投影，按行切分
└── mlp_bda: bias_dropout_add
```

---

## 9. TransformerLayer 的单步计算拆解

文件：`/workspace/megatron/core/transformer/transformer_layer.py`

### 9.1 `_forward_attention` 的步骤列表

```
输入: hidden_states [S, B, H]

步骤 1: input_layernorm(hidden_states)
  → input_layernorm_output [S, B, H]
  → residual = hidden_states（保存用于残差连接）

步骤 2: self_attention(input_layernorm_output, ...)
  内部:
    2a. linear_qkv(input_layernorm_output)
        [S, B, H] → [S, B, (NH + 2*nKV) * head_dim]  (合并 Q/K/V)
        ColumnParallel: 每个 TP rank 只持有 1/TP 份
    2b. 分离 Q, K, V 并 reshape
        Q: [S, B, NH/TP, head_dim]
        K: [S, B, nKV/TP, head_dim]
        V: [S, B, nKV/TP, head_dim]
    2c. RoPE 应用（如果 position_embedding_type='rope'）
    2d. core_attention(Q, K, V, ...)
        → [S, B, NH/TP * head_dim]
    2e. linear_proj(context_layer)
        RowParallel: [S, B, NH/TP * head_dim] → [S, B, H]
        AllReduce（或 RS in SP mode）
  → attention_output [S, B, H]

步骤 3: bias_dropout_add
  hidden_states = attention_output + residual（残差连接）
  → hidden_states [S, B, H]
```

### 9.2 `_forward_mlp` 的步骤列表

```
输入: hidden_states [S, B, H]（来自步骤 3）

步骤 4: pre_mlp_layernorm(hidden_states)
  → pre_mlp_layernorm_output [S, B, H]
  → residual = hidden_states（保存用于残差连接）

步骤 5: linear_fc1(pre_mlp_layernorm_output)
  ColumnParallel: [S, B, H] → [S, B, FFN/TP]
  如果 gated_linear_unit=True (SwiGLU):
    同时计算 gate 和 value，然后 gate × SiLU(value)

步骤 6: 激活函数 (GELU 或 SiLU)

步骤 7: linear_fc2(act_output)
  RowParallel: [S, B, FFN/TP] → [S, B, H]
  AllReduce（或 RS in SP mode）
  → mlp_output [S, B, H]

步骤 8: bias_dropout_add
  hidden_states = mlp_output + residual（残差连接）
  → hidden_states [S, B, H]  ← 单层输出
```

---

## 10. 数值例子 A：seq=4096, mbs=2, TP=4, SP=on

### 配置

| 参数 | 数值 |
|------|------|
| `hidden_size` (H) | 4096 |
| `num_attention_heads` (NH) | 32 |
| `num_query_groups` (nKV) | 8 |
| `kv_channels` (head_dim) | 128 |
| `ffn_hidden_size` (FFN) | 16384 |
| `tensor_model_parallel_size` (TP) | 4 |
| `batch_size` (B) | 2 |
| `seq_length` (S) | 4096 |

### 10.1 Embedding 阶段（PP stage 0）

```
输入 tokens: [B, S] = [2, 4096]
↓ LanguageModelEmbedding（词嵌入查表，VocabParallelEmbedding）
词嵌入: [S, B, H] = [4096, 2, 4096]     ← S-first 格式
↓ scatter_to_sequence_parallel（SP=True，TP=4）
SP 散列后: [S/TP, B, H] = [1024, 2, 4096]
```

每个 TP rank 持有 1024 个序列位置（整个序列的 1/4）。

### 10.2 注意力层形状变化（TP=4，SP=on）

```
输入（SP 模式）: [S/TP, B, H] = [1024, 2, 4096]

SP gather（注意力前先 gather 完整序列）:
→ [S, B, H] = [4096, 2, 4096]

── linear_qkv（ColumnParallel）──
weight 形状: [H, (NH/TP + 2*nKV/TP) * head_dim]
           = [4096, (32/4 + 2*8/4) * 128]
           = [4096, (8 + 4) * 128]
           = [4096, 1536]
输出 qkv: [S, B, 1536] = [4096, 2, 1536]

分离后:
  Q: [S, B, NH/TP, head_dim] = [4096, 2, 8, 128]   ← 8 个 Q 头/TP rank
  K: [S, B, nKV/TP, head_dim] = [4096, 2, 2, 128]  ← 2 个 KV 头/TP rank
  V: [S, B, nKV/TP, head_dim] = [4096, 2, 2, 128]

── core_attention（Flash Attention / DotProductAttention）──
QK^T: 矩阵乘 [B, 8, S, head_dim] × [B, 2, head_dim, S]
     → 通过 GQA broadcasting（8 个 Q 头 → 2 个 KV 组，每组 4 头）
     → [B, 8, S, S] = [2, 8, 4096, 4096]（causal mask）
softmax + ×V: [B, 8, S, S] × [B, 2, S, head_dim]
     → [B, 8, S, head_dim] → reshape
context: [S, B, NH/TP * head_dim] = [4096, 2, 1024]

── linear_proj（RowParallel）──
weight 形状: [NH/TP * head_dim, H] = [1024, 4096]
输出: [S, B, H] = [4096, 2, 4096]
ReduceScatter（SP 模式下，代替 AllReduce）
→ [S/TP, B, H] = [1024, 2, 4096]   ← 回到 SP 分片状态
```

### 10.3 FFN 层形状变化（TP=4，SP=on，SwiGLU）

```
输入（SP 模式）: [S/TP, B, H] = [1024, 2, 4096]

SP gather（FFN 前先 gather，或等效操作）:
→ [S, B, H] = [4096, 2, 4096]

── linear_fc1（ColumnParallel，SwiGLU）──
weight 形状: [H, 2 * FFN/TP] = [4096, 2 * 16384/4] = [4096, 8192]
（SwiGLU 需要 2× FFN 维度）
输出 gate_and_value: [S, B, 8192] = [4096, 2, 8192]

分离 + SwiGLU：gate × SiLU(value)
→ [S, B, FFN/TP] = [4096, 2, 4096]

── linear_fc2（RowParallel）──
weight 形状: [FFN/TP, H] = [4096, 4096]
输出: [S, B, H] = [4096, 2, 4096]
ReduceScatter（SP 模式）
→ [S/TP, B, H] = [1024, 2, 4096]
```

---

## 11. 数值例子 B：seq=2048, mbs=2, TP=2, SP=on

这个更小的例子便于心算验证。

**配置**：H=2048, NH=16, nKV=4, head_dim=128, FFN=8192, TP=2, B=2, S=2048

```
Embedding 输出（SP 散列后）: [S/TP, B, H] = [1024, 2, 2048]

── linear_qkv weight ──
(NH/TP + 2*nKV/TP) * head_dim = (8 + 4) * 128 = 1536
weight: [2048, 1536]

── core_attention 每 TP rank ──
Q: [2048, 2, 8, 128]    K/V: [2048, 2, 2, 128]
(8 Q 头 / 2 KV 头 per TP rank = GQA ratio 4:1)
QK^T: [2, 8, 2048, 2048]  → 注意这是全序列，总内存约 2×8×2048²×2B = 128MB/rank

── output_layer（最后 PP stage，gather 后）──
hidden: [2048, 2, 2048]
weight: [2048, V/TP]  (V=32000, V/TP=16000)
logits: [2048, 2, 16000]
vocab_parallel_cross_entropy 在分片 logits 上计算
```

---

## 12. VPP 层编号：全局层号 vs 局部层号

文件：`/workspace/megatron/core/transformer/transformer_block.py`，第 332 行。

### 12.1 非 VPP 时的层编号

32 层 / PP=4 / VPP=1（无 VPP）：

| PP rank | 局部层号（在 TransformerBlock 内） | 全局层号 |
|---------|----------------------------------|---------|
| 0 | 0, 1, 2, 3, 4, 5, 6, 7 | 0-7 |
| 1 | 0, 1, 2, 3, 4, 5, 6, 7 | 8-15 |
| 2 | 0, 1, 2, 3, 4, 5, 6, 7 | 16-23 |
| 3 | 0, 1, 2, 3, 4, 5, 6, 7 | 24-31 |

全局层号 = `get_transformer_layer_offset(config, vp_stage=None, pp_rank) + 局部层号`

### 12.2 VPP=2 时的层编号

32 层 / PP=4 / VPP=2：每个 PP rank 有 2 个 chunk（vp_stage=0 和 vp_stage=1），每个 chunk 4 层：

| PP rank | vp_stage | 局部层号 | 全局层号 |
|---------|----------|---------|---------|
| 0 | 0 | 0,1,2,3 | **0-3** |
| 1 | 0 | 0,1,2,3 | **4-7** |
| 2 | 0 | 0,1,2,3 | **8-11** |
| 3 | 0 | 0,1,2,3 | **12-15** |
| 0 | 1 | 0,1,2,3 | **16-19** |
| 1 | 1 | 0,1,2,3 | **20-23** |
| 2 | 1 | 0,1,2,3 | **24-27** |
| 3 | 1 | 0,1,2,3 | **28-31** |

**VPP 的好处**：微批次 1 可以先在 PP0-chunk0 → PP1-chunk0 → ... → PP3-chunk0 流动，同时微批次 2 在 PP0-chunk0 上开始。bubble 比例从 `(PP-1)/N` 降低到 `(PP-1)/(N*VPP)` 左右。

**代码中的层偏移计算**：

```python
# megatron/core/transformer/transformer_block.py, 第 332-334 行
global_layer_number = layer_number + get_transformer_layer_offset(
    self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
)
```

---

## 13. Multi-Token Prediction（MTP）简介

文件：`/workspace/megatron/core/transformer/multi_token_prediction.py`

### 13.1 MTP 的原理

MTP（Multi-Token Prediction）是 DeepSeek-V3 引入的技术，将语言模型训练目标从"预测下一个 token"扩展到"同时预测接下来 D 个 token"。其优势是：
1. 以更少的训练步数达到相同的模型质量（每步有效 token 数 × D）
2. 推理时可以与 speculative decoding 结合，加速生成

### 13.2 Megatron 的 MTP 实现

配置字段：
```python
mtp_num_layers: Optional[int] = None   # MTP 预测的额外 token 数（即 D）
mtp_loss_scaling_factor: float = 0.1   # MTP 损失的权重系数
```

`MultiTokenPredictionBlock` 是一个包含 D 个 MTP 模块的容器。每个 MTP 模块由以下组件构成：

```
MTP 模块 k（预测第 k+2 个 token）:
├── 共享 embedding（与主模型 embedding 共享权重）
├── 线性投影（将 hidden_state 与 embedding 拼接后投影）
├── Transformer 层（通常只有 1 层）
└── 共享 output head（与主模型 output_layer 共享权重）
```

### 13.3 MTP 在 GPTModel 中的位置

MTP Block 挂载在 PP pipeline 的 `mtp_process` 阶段，这通常与 `pre_process` 阶段对应（共享 embedding）：

```
forward 路径:
  _preprocess() → decoder(主 Transformer Block) → _postprocess()
                                                       ↓
                                              mtp(hidden_states)   ← 在 postprocess 之前
                                                       ↓
                                              output_layer(hidden_states) → loss
```

MTP 的损失被加权（`mtp_loss_scaling_factor`）后与主损失相加：

```
total_loss = main_lm_loss + mtp_loss_scaling_factor × avg(mtp_losses[0..D-1])
```

---

## 14. gpt_builder 分支树

文件：`/workspace/gpt_builders.py`（或各后端的 spec provider）

`gpt_builder` 函数根据配置参数选择合适的 layer spec：

```
gpt_builder(args)
├── 确定后端
│   ├── args.spec == 'local'      → get_gpt_layer_local_spec()
│   ├── args.spec == 'te' (默认)  → get_gpt_layer_with_transformer_engine_spec()
│   └── args.spec == 'inference_optimized'
│                                  → get_gpt_layer_with_inference_submodules()
│
├── 注意力类型
│   ├── multi_latent_attention=False（MHA/GQA，默认）
│   │   └── SelfAttention + SelfAttentionSubmodules
│   └── multi_latent_attention=True（MLA，DeepSeek 风格）
│       └── MLASelfAttention + MLASelfAttentionSubmodules
│
├── MoE 配置
│   ├── num_experts=None（Dense FFN，默认）
│   │   └── MLP
│   └── num_experts > 0（MoE）
│       └── MoELayer（包含 Router + N个 Expert）
│
└── 归一化
    ├── normalization='RMSNorm'（LLaMA 风格）
    └── normalization='LayerNorm'（原始 BERT/GPT 风格）
```

---

## 15. 连接图：从 Config 到每一次矩阵乘法

```
TransformerConfig
  │  (传给)
  ▼
GPTModel.__init__()
  │  (创建)
  ├─► LanguageModelEmbedding  (if pre_process)
  │     ├─ word_embeddings: VocabParallelEmbedding [vocab_size/TP, H]
  │     └─ position_embeddings: Embedding [max_seq_len, H]（仅 learned_absolute）
  │
  ├─► RotaryEmbedding / YarnRotaryEmbedding  (if rope/yarn)
  │     └─ 预计算 cos/sin 表（无参数，每步根据 seq_len 动态计算）
  │
  ├─► TransformerBlock  (所有 PP stage 都有)
  │     │  (按 get_num_layers_to_build 创建 N 层)
  │     ▼
  │     TransformerLayer × N
  │       ├─ input_layernorm: RMSNorm / TENorm
  │       ├─ SelfAttention
  │       │   ├─ linear_qkv: ColumnParallelLinear [H → (Q+K+V)/TP]
  │       │   │   权重: [H, (NH+2*nKV)*head_dim/TP]
  │       │   ├─ core_attention: DotProductAttention / TEDotProductAttention
  │       │   └─ linear_proj: RowParallelLinear [NH*head_dim/TP → H]
  │       │       权重: [NH*head_dim/TP, H]
  │       ├─ pre_mlp_layernorm: RMSNorm / TENorm
  │       └─ MLP
  │           ├─ linear_fc1: ColumnParallelLinear [H → FFN/TP]
  │           │   权重: [H, FFN/TP] (SwiGLU: [H, 2*FFN/TP])
  │           └─ linear_fc2: RowParallelLinear [FFN/TP → H]
  │               权重: [FFN/TP, H]
  │
  ├─► MultiTokenPredictionBlock  (if mtp_process)
  │     └─ MTP 层 × mtp_num_layers（每层含 1 个 Transformer 层）
  │
  └─► output_layer: ColumnParallelLinear [H → vocab_size/TP]  (if post_process)
        权重: [H, vocab_size/TP]（若 share_embeddings，与 word_embeddings 共享）
```

---

## 16. Debug 断点表

| 断点位置 | 文件 | 目的 |
|---------|------|------|
| `GPTModel.__init__` 末尾 | `gpt_model.py` | 确认 embedding/output_layer 存在性，打印 `pre_process`/`post_process` |
| `_preprocess` 的 SP scatter | `gpt_model.py` | 验证 SP 散列后的形状 `[S/TP, B, H]` |
| `TransformerLayer._forward_attention` 的 linear_qkv 输出 | `transformer_layer.py` | 验证 QKV 形状与公式一致 |
| `linear_proj` 后（AllReduce/RS 后） | `attention.py` | 验证 TP 聚合后恢复 `[S, B, H]` |
| `_postprocess` 的 logits 计算 | `gpt_model.py` | 验证 logits 形状 `[S, B, vocab/TP]` |

**快速验证脚本**：

```python
# 在 model_provider 调用后加入
print(f"[rank {torch.distributed.get_rank()}]")
print(f"  pre_process={model.pre_process}, post_process={model.post_process}")
print(f"  has embedding: {hasattr(model, 'embedding')}")
print(f"  position_embedding_type: {model.position_embedding_type}")
print(f"  num decoder layers: {len(model.decoder.layers)}")
for name, p in model.named_parameters():
    if 'linear_qkv.weight' in name:
        print(f"  {name}: {p.shape}")
    if 'output_layer.weight' in name:
        print(f"  {name}: {p.shape}")
```

---

## 17. 课后练习

**练习 1（形状计算）**：
给定配置：`hidden_size=8192`, `num_attention_heads=64`, `num_query_groups=8`, `TP=8`
计算：
- 每个 TP rank 的 Q 头数
- 每个 TP rank 的 KV 头数
- `linear_qkv.weight` 的形状
- `linear_proj.weight` 的形状

**参考答案**：
- Q 头数/TP rank = 64/8 = 8
- KV 头数/TP rank = 8/8 = 1（GQA 8:1）
- `linear_qkv.weight`: `[8192, (8+1+1)*128]` = `[8192, 1280]`（head_dim = 8192/64 = 128）
- `linear_proj.weight`: `[8*128, 8192]` = `[1024, 8192]`

**练习 2（追踪代码）**：
在 `get_gpt_layer_local_submodules` 中，当 `multi_latent_attention=False, num_experts=None` 时，找到 `linear_fc1` 对应的类是什么？追踪 `build_module` 的调用链，直到找到实际的 `nn.Linear`。

**练习 3（理解 VPP）**：
32 层模型，PP=4, VPP=4。计算每个 PP rank 的 chunk 数、每个 chunk 的层数、PP rank=2, vp_stage=1 对应哪些全局层？（提示：全局层偏移 = `vp_stage × PP + pp_rank` × layers_per_chunk）

**练习 4（实验题）**：
修改 `get_gpt_layer_local_submodules` 中的 `linear_fc1`，替换为打印输入形状的包装类，验证 FFN 输入的实际形状。

**练习 5（思考题）**：
为什么 `linear_qkv` 使用 `ColumnParallelLinear` 而 `linear_proj` 使用 `RowParallelLinear`？如果反过来会发生什么？（提示：考虑矩阵乘法中哪个维度被切分，以及结果如何合并。）

---

## 18. FAQ：10 个模型架构问题

**Q1：GQA（Grouped Query Attention）是怎么在代码里实现的？**

A：在 `SelfAttention` 内部，通过 `expand_qkv` 或 `repeat_kv` 将 K/V 的头数从 `nKV` 广播到 `NH`：
```python
# 伪代码
q: [B, NH, S, head_dim]
k: [B, nKV, S, head_dim]
k = k.repeat_interleave(NH // nKV, dim=1)  # 广播 K
v = v.repeat_interleave(NH // nKV, dim=1)  # 广播 V
attn = q @ k.transpose(-2, -1)  # [B, NH, S, S]
```

**Q2：`parallel_output=True` 和 `parallel_output=False` 什么时候用哪个？**

A：训练时用 `True`（默认），logits 保持分片，配合 `vocab_parallel_cross_entropy` 计算；推理时用 `False`（或通过 `runtime_gather_output=True` 临时覆盖），需要完整 vocab logits 来 argmax 采样。

**Q3：`share_embeddings_and_output_weights` 是否影响 PP > 1 时的参数量？**

A：参数量不变（PP > 1 时两端各有一份 embedding），但**参数一致性**由 AllReduce 保证，**梯度**也会被合并。实际上，PP > 1 时 `share_embeddings_and_output_weights` 的主要意义是梯度合并，而非节省参数。

**Q4：RoPE 的 `rotary_base=10000` 代表什么？**

A：`rotary_base` 控制旋转频率的基础周期：`θ_i = 10000^(-2i/d)`。较大的 base 使得高维度的旋转频率更慢，从而能表示更长的相对位置信息。LLaMA-2 使用 10000，LLaMA-3 使用 500000（更适合长序列）。

**Q5：`position_embedding_type='none'` 会怎样？**

A：`LanguageModelEmbedding` 不会创建 `position_embeddings`，embedding forward 只有词嵌入：`output = word_embeddings(input_ids)`。这种模式通常与自定义位置编码（如 ALiBi 通过注意力 mask 施加）配合使用。

**Q6：`mtp_num_layers=1` 和 `mtp_num_layers=3` 的区别是什么？**

A：`mtp_num_layers` 决定了额外预测的 token 数 D。`=1` 表示额外预测下一个 token（共预测 2 个）；`=3` 则额外预测 3 个 token（共 4 个），损失是所有深度 MTP 损失的平均。DeepSeek-V3 使用了 `mtp_num_layers=1`。

**Q7：SP（Sequence Parallel）模式下，注意力计算是在完整序列还是部分序列上进行的？**

A：完整序列。SP 仅在 LayerNorm 和 Dropout 阶段保持序列切分；进入注意力和 FFN 计算前，会先 AllGather 完整序列（通过 `gather_from_sequence_parallel_region`）；计算完后再 ReduceScatter 回分片状态。这是 SP 不需要 Ring Attention 的原因——注意力仍然是全局的。

**Q8：如果 `num_query_groups` 不能被 TP 整除会怎样？**

A：构建模型时会报错，因为每个 TP rank 必须有完整数量的 KV 头（不能有 0.5 个 KV 头）。规则：`num_query_groups % TP == 0`。例如 nKV=4, TP=8 是不合法的（4 < 8）。这时应考虑降低 TP 或增加 nKV。

**Q9：`output_layer` 的 `bias=False` 有什么含义？**

A：语言模型的输出层通常不加 bias（来自 Press & Wolf 2017 的实验表明 bias 对困惑度影响很小，且会增加每步更新的通信量）。大部分现代 LLM（LLaMA, GPT-3, Falcon 等）都不使用 output bias。

**Q10：什么是 `skip_weight_param_allocation`，在什么时候触发？**

A：当 `pre_process and share_embeddings_and_output_weights` 时，`output_layer` 的权重指向 `embedding.word_embeddings.weight`，不需要再分配一块新内存，所以 `skip_weight_param_allocation=True`。只有当 PP stage 同时持有 embedding 和 output_layer（即 PP=1 或 pre_process=post_process=True）时，这个参数才为 True。

---

## 19. 本文小结

通过本文，你应该能够：

1. **画出** `GPTModel` 在不同 PP stage 上的组件差异（embedding/output_layer 是条件性的）
2. **解释** `TransformerConfig` 的 5 个字段组，以及关键字段的自动填充逻辑
3. **理解** `ModuleSpec` 的 `build_module` 算法，以及为什么需要这个抽象层
4. **追踪** 一个 token 从 embedding 到 logits 的完整形状变化（两个数值例子）
5. **计算** GQA + TP 下的 QKV weight 形状
6. **理解** VPP 的层编号逻辑
7. **对比** RoPE、YaRN 和学习型绝对位置编码的差异
8. **理解** `share_embeddings_and_output_weights` 在 PP > 1 时的工作机制
9. **选择** `recompute_granularity` 的合适级别（None / selective / full）
10. **识别** MTP Block 在 GPTModel 中的挂载位置

至此，系列的 Phase 1（单卡单步）已完成。Phase 2 将深入并行原理：从 `parallel_state` 的进程组初始化，到张量并行的矩阵切分，再到流水线并行的 1F1B 调度。

---

*上一篇：[精读 Megatron 源码（2）：完整拆解一次训练](./02-pretrain-loop.md)*
*下一篇：[精读 Megatron 源码（4）：parallel_state——所有并行的地基](./04-parallel-state.md)*
