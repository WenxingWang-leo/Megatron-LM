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
                 max_sequence_length, pre_process=True, post_process=True, ...):

        # ── 必有的组件（所有 PP stage 都有）──
        self.decoder = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

        # ── 仅 pre_process=True 的 stage（PP rank 0）──
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=config,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        # ── 仅 post_process=True 的 stage（PP 最后 rank）──
        if self.post_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                vocab_size,
                ...
            )

        # ── 位置编码（所有 stage，但实际只在 pre_process stage 用）──
        if position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(...)
        elif position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(...)
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

## 2. GPTModel.forward() 的三段结构

文件：`/workspace/megatron/core/models/gpt/gpt_model.py`，第 508 行。

```python
def forward(self, input_ids, position_ids, attention_mask,
            decoder_input=None, labels=None, ...):

    # ── 第一段：预处理（embedding + RoPE 计算）──
    preproc_output = self._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        packed_seq_params=packed_seq_params,
    )
    (decoder_input, rotary_pos_emb, ...) = preproc_output[:6]

    # ── 第二段：TransformerBlock（所有 Transformer 层）──
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        ...
    )

    # ── 第三段：后处理（输出层 + loss 或 logits）──
    return self._postprocess(
        hidden_states=hidden_states,
        labels=labels,
        ...
    )
```

### 2.1 `_preprocess` 的详细逻辑

```python
def _preprocess(self, input_ids, position_ids, decoder_input, ...):
    if decoder_input is not None:
        pass  # 中间 PP stage：从上一个 stage 接收，直接用
    elif self.pre_process:
        # PP stage 0：从 tokens 计算 embedding
        decoder_input = self.embedding(input_ids=input_ids,
                                        position_ids=position_ids)

        # Sequence Parallel 散列（如果开启 SP）
        if self.config.sequence_parallel and not self.embedding.scatter_to_sequence_parallel:
            decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                decoder_input, group=self.pg_collection.tp
            )
    else:
        # 中间 PP stage（没有 decoder_input 传入时）
        decoder_input = None  # 会由 set_input_tensor() 填入

    # 计算 RoPE（所有 stage 都计算，因为每层 attention 都需要）
    if self.position_embedding_type == 'rope':
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(...)
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, ...)

    return (decoder_input, rotary_pos_emb, rotary_pos_cos,
            rotary_pos_sin, sequence_len_offset, padding_mask)
```

**SP 散列的时机**：当开启 Sequence Parallel（`--sequence-parallel`），embedding 的输出形状是 `[S, B, H]`，经过 `scatter_to_sequence_parallel_region` 后变为 `[S/TP, B, H]`。每个 TP rank 只持有序列的一段，这样后续的 LayerNorm 和 Dropout 的计算量都除以 TP。

### 2.2 `_postprocess` 的关键分支

```python
def _postprocess(self, hidden_states, labels, ...):
    if not self.post_process:
        return hidden_states  # 中间 PP stage：直接返回隐藏状态

    # post_process=True（最后 PP stage）：
    # 1. output_layer：[S/TP, B, H] × [H, V/TP] = [S/TP, B, V/TP]
    logits, _ = self.output_layer(hidden_states, weight=output_weight,
                                   runtime_gather_output=runtime_gather_output)

    if labels is None:
        # 推理模式：返回 logits，形状 [B, S, V]
        return logits.transpose(0, 1).contiguous()

    # 训练模式：计算 cross-entropy loss
    loss = self.compute_language_model_loss(labels, logits)
    return loss  # 形状 [B, S]，每个 token 的 CE loss
```

---

## 3. TransformerConfig：字段分组详解

文件：`/workspace/megatron/core/transformer/transformer_config.py`

`TransformerConfig` 有超过 400 个字段，继承自 `ModelParallelConfig`（包含并行配置）。下面按功能分组介绍最重要的字段：

### 3.1 模型架构核心字段

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

### 3.2 并行相关字段（继承自 ModelParallelConfig）

| 字段名 | 含义 |
|--------|------|
| `tensor_model_parallel_size` | TP 并行度（`--tensor-model-parallel-size`） |
| `pipeline_model_parallel_size` | PP 并行度 |
| `virtual_pipeline_model_parallel_size` | VPP 虚拟流水线并行度 |
| `sequence_parallel` | 是否开启 SP（需要 `tensor_model_parallel_size > 1`） |
| `context_parallel_size` | CP 并行度 |

### 3.3 激活重计算（内存 vs 速度 tradeoff）

| 字段名 | 含义 |
|--------|------|
| `fp16`/`bf16`/`fp8` | 数值精度（bf16 推荐，fp8 需 TE） |
| `fp32_residual_connection` | 残差连接用 fp32（提升稳定性） |
| `recompute_granularity` | `'full'`：重计算整层；`'selective'`：只重计算特定子模块 |
| `recompute_method` | `'uniform'`：均匀分配；`'block'`：前 N 层重计算 |
| `recompute_num_layers` | 每个 recompute 单元的层数 |
| `recompute_modules` | selective 模式：重计算哪些子模块（如 `['core_attn']`） |

`recompute_granularity='full'` 在反向传播时重新执行前向，以时间换空间。70B 模型光激活就可能占数十 GB，重计算可以将峰值显存减少 60-80%，代价是增加约 30-40% 训练时间。

### 3.4 TransformerConfig 字段分组全表

```
TransformerConfig
├── 模型架构: num_layers, hidden_size, num_attention_heads, num_query_groups,
│            ffn_hidden_size, kv_channels, gated_linear_unit, add_bias_linear
├── 并行策略: tensor/pipeline/virtual_pipeline_model_parallel_size,
│            sequence_parallel, context_parallel_size
├── 数值精度: fp16, bf16, fp8, fp4, fp32_residual_connection, layernorm_epsilon
├── 激活重计算: recompute_granularity, recompute_method, recompute_num_layers
├── MoE 专家: num_moe_experts, moe_router_topk, moe_ffn_hidden_size
├── 位置编码: position_embedding_type, rotary_base, rope_scaling
└── 推理优化: flash_decode, cuda_graph_impl, inference_fuse_tp_communication
```

---

## 4. ModuleSpec：插件化装配机制

文件：`/workspace/megatron/core/transformer/spec_utils.py`

### 4.1 ModuleSpec 的定义

```python
@dataclass
class ModuleSpec:
    module: Union[Tuple, type]   # 模块类（或 (module_path, ClassName) 元组）
    params: dict = {}             # 构造时的额外参数
    submodules: object = None     # 子模块的 spec（嵌套 Spec）
    metainfo: dict = {}           # 元信息（不用于构造）
```

### 4.2 `build_module` 算法

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

### 4.3 ModuleSpec 的嵌套结构举例

```python
# 来自 gpt_layer_specs.py 的 local backend 配置（精简版）

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

## 5. TransformerLayer 的单步计算拆解

文件：`/workspace/megatron/core/transformer/transformer_layer.py`

### 5.1 `_forward_attention` 的步骤列表

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

### 5.2 `_forward_mlp` 的步骤列表

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

## 6. 数值例子：追踪一个 token 的形状变化

### 配置

| 参数 | 数值 | 说明 |
|------|------|------|
| `hidden_size` (H) | 4096 | 隐藏层维度 |
| `num_attention_heads` (NH) | 32 | 注意力头数 |
| `num_query_groups` (nKV) | 8 | GQA 的 KV 头数（GQA 4:1） |
| `kv_channels` (head_dim) | 4096/32 = 128 | 每头维度 |
| `ffn_hidden_size` (FFN) | 16384 | FFN 中间维度 |
| `tensor_model_parallel_size` (TP) | 4 | 张量并行度 |
| `batch_size` (B) | 2 | 批次大小 |
| `seq_length` (S) | 4096 | 序列长度 |

### 6.1 Embedding 阶段（PP stage 0）

```
输入 tokens: [B, S] = [2, 4096]
↓ LanguageModelEmbedding（词嵌入查表）
词嵌入: [S, B, H] = [4096, 2, 4096]     ← S-first 格式
↓ scatter_to_sequence_parallel（SP=True，TP=4）
SP 散列后: [S/TP, B, H] = [1024, 2, 4096]
```

**注意**：散列后每个 TP rank 持有 1024 个序列位置，而不是所有 4096 个。

### 6.2 注意力层的形状变化（TP=4）

```
输入: [S/TP, B, H] = [1024, 2, 4096]  （SP 模式下）

SP gather（attention 前先 gather）:
[S, B, H] = [4096, 2, 4096]

── linear_qkv（ColumnParallel）──
weight 形状: [H, (NH + 2*nKV) * head_dim / TP]
           = [4096, (32 + 2*8) * 128 / 4]
           = [4096, 48 * 128 / 4]
           = [4096, 1536]
输出 qkv: [S, B, (NH/TP + 2*nKV/TP) * head_dim]
        = [4096, 2, 48*128/4]
        = [4096, 2, 1536]

分离后:
  Q: [S, B, NH/TP, head_dim] = [4096, 2, 8, 128]   (32/4=8 头)
  K: [S, B, nKV/TP, head_dim] = [4096, 2, 2, 128]  (8/4=2 头)
  V: [S, B, nKV/TP, head_dim] = [4096, 2, 2, 128]

── core_attention ──
Q×K^T: [B, NH/TP, S, S] = [2, 8, 4096, 4096]  (causal mask)
softmax + V: [B, NH/TP, S, head_dim] → reshape
context: [S, B, NH/TP * head_dim] = [4096, 2, 1024]

── linear_proj（RowParallel）──
weight 形状: [NH/TP * head_dim, H] = [1024, 4096]
输出: [S, B, H] = [4096, 2, 4096]
AllReduce/ReduceScatter（TP 组内求和）

SP scatter（attention 后再 scatter）:
[S/TP, B, H] = [1024, 2, 4096]
```

**GQA 的关键细节**：每个 TP rank 有 8 个 Q 头和 2 个 KV 头。2 个 KV 头被所有 8 个 Q 头共享（在 core_attention 内部广播）。这就是 GQA（Grouped Query Attention）的含义——多个 Q 头共享一对 KV。

### 6.3 FFN 层的形状变化（TP=4）

```
输入: [S/TP, B, H] = [1024, 2, 4096]

SP gather（FFN 前）:
[S, B, H] = [4096, 2, 4096]

── linear_fc1（ColumnParallel，SwiGLU 模式）──
weight 形状: [H, 2 * FFN/TP] = [4096, 2 * 16384/4] = [4096, 8192]
（SwiGLU 需要 2× FFN 维度：一份 value，一份 gate）
输出 gate_and_value: [S, B, 2 * FFN/TP] = [4096, 2, 8192]

分离 gate 和 value，计算 gate × SiLU(value):
[S, B, FFN/TP] = [4096, 2, 4096]

── linear_fc2（RowParallel）──
weight 形状: [FFN/TP, H] = [4096, 4096]
输出: [S, B, H] = [4096, 2, 4096]
AllReduce/ReduceScatter

SP scatter（FFN 后）:
[S/TP, B, H] = [1024, 2, 4096]
```

### 6.4 输出层的形状变化（post_process stage）

```
最后一层 Transformer 输出: [S/TP, B, H] = [1024, 2, 4096]

SP gather（输出层前）:
[S, B, H] = [4096, 2, 4096]

── output_layer（ColumnParallel）──
weight 形状: [H, vocab_size/TP] = [4096, 32000/4] = [4096, 8000]
输出 logits: [S, B, vocab_size/TP] = [4096, 2, 8000]

（parallel_output=True 时不 gather，由 loss 函数在 TP 分片上计算）

── compute_language_model_loss ──
输入 labels: [B, S] = [2, 4096]
交叉熵 loss（vocab 维度在 TP 分片上，使用 vocab_parallel_cross_entropy）:
output: [B, S] = [2, 4096]，每个位置的 CE loss
```

---

## 7. VPP 层编号：全局层号 vs 局部层号

文件：`/workspace/megatron/core/transformer/transformer_block.py`，第 332 行。

### 7.1 非 VPP 时的层编号

32 层 / PP=4 / VPP=1（无 VPP）：

| PP rank | 局部层号（在 TransformerBlock 内） | 全局层号 |
|---------|----------------------------------|---------|
| 0 | 0, 1, 2, 3, 4, 5, 6, 7 | 0-7 |
| 1 | 0, 1, 2, 3, 4, 5, 6, 7 | 8-15 |
| 2 | 0, 1, 2, 3, 4, 5, 6, 7 | 16-23 |
| 3 | 0, 1, 2, 3, 4, 5, 6, 7 | 24-31 |

全局层号 = `get_transformer_layer_offset(config, vp_stage=None, pp_rank) + 局部层号`

### 7.2 VPP=2 时的层编号

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

**VPP 的好处**：微批次 1 可以先在 PP0-chunk0 → PP1-chunk0 → ... → PP3-chunk0 流动，同时微批次 2 在 PP0-chunk0 上开始。这使得流水线更密集，bubble 比例从 `(PP-1)/N` 降低到 `(PP-1)/(N*VPP)` 左右。

**代码中的层偏移计算**：

```python
# megatron/core/transformer/transformer_block.py, 第 332-334 行
global_layer_number = layer_number + get_transformer_layer_offset(
    self.config, self.vp_stage, get_pg_rank(self.pg_collection.pp)
)
```

`get_transformer_layer_offset` 返回当前 `(vp_stage, pp_rank)` 组合对应的全局层偏移。

---

## 8. gpt_builder 分支树

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

## 9. 连接图：从 Config 到每一次矩阵乘法

```
TransformerConfig
  │  (传给)
  ▼
GPTModel.__init__()
  │  (创建)
  ├─► LanguageModelEmbedding  (if pre_process)
  │     └─ word_embeddings: [vocab_size, H]（按 TP 切分词表）
  │
  ├─► RotaryEmbedding  (if rope)
  │     └─ precompute cos/sin tables
  │
  └─► TransformerBlock  (所有 PP stage 都有)
        │  (按 get_num_layers_to_build 创建 N 层)
        ▼
        TransformerLayer × N
          ├─ input_layernorm
          ├─ SelfAttention
          │   ├─ linear_qkv: ColumnParallelLinear [H → (Q+K+V)/TP]
          │   │   (权重: H × (NH+2*nKV)*head_dim/TP)
          │   ├─ core_attention: Flash/DotProduct
          │   └─ linear_proj: RowParallelLinear [H/TP → H]
          │       (权重: NH*head_dim/TP × H)
          ├─ pre_mlp_layernorm
          └─ MLP
              ├─ linear_fc1: ColumnParallelLinear [H → FFN/TP]
              │   (权重: H × FFN/TP)
              └─ linear_fc2: RowParallelLinear [FFN/TP → H]
                  (权重: FFN/TP × H)
```

---

## 10. Debug 断点表

| 断点位置 | 文件 | 目的 |
|---------|------|------|
| `GPTModel.__init__` 末尾 | `gpt_model.py` | 确认 embedding/output_layer 存在性，打印 `pre_process`/`post_process` |
| `_preprocess` 的 SP scatter | `gpt_model.py` | 验证 SP 散列后的形状 `[S/TP, B, H]` |
| `TransformerLayer._forward_attention` 的 linear_qkv 输出 | `transformer_layer.py` | 验证 QKV 形状与公式一致 |
| `linear_proj` 后（AllReduce 后） | `attention.py` | 验证 TP 聚合后恢复 `[S, B, H]` |
| `_postprocess` 的 logits 计算 | `gpt_model.py` | 验证 logits 形状 `[S, B, vocab/TP]` |

**快速验证脚本**：

```python
# 在 model_provider 调用后加入
print(f"[rank {torch.distributed.get_rank()}]")
print(f"  pre_process={model.pre_process}, post_process={model.post_process}")
print(f"  has embedding: {hasattr(model, 'embedding')}")
print(f"  num decoder layers: {len(model.decoder.layers)}")
for name, p in model.named_parameters():
    if 'linear_qkv.weight' in name:
        print(f"  {name}: {p.shape}")
```

---

## 12. 课后练习

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

## 13. 本文小结

通过本文，你应该能够：

1. **画出** `GPTModel` 在不同 PP stage 上的组件差异（embedding/output_layer 是条件性的）
2. **解释** `TransformerConfig` 的 5 个字段组，以及关键字段的自动填充逻辑
3. **理解** `ModuleSpec` 的 `build_module` 算法，以及为什么需要这个抽象层
4. **追踪** 一个 token 从 embedding 到 logits 的完整形状变化
5. **计算** GQA + TP 下的 QKV weight 形状
6. **理解** VPP 的层编号逻辑

至此，系列的 Phase 1（单卡单步）已完成。Phase 2 将深入并行原理：从 `parallel_state` 的进程组初始化，到张量并行的矩阵切分，再到流水线并行的 1F1B 调度。

---

*上一篇：[精读 Megatron 源码（2）：完整拆解一次训练](./02-pretrain-loop.md)*
*下一篇：[精读 Megatron 源码（4）：parallel_state——所有并行的地基](./04-parallel-state.md)*
