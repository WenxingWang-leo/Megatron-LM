# 知乎系列 03｜`GPTModel`：Spec 驱动的 Transformer 积木

> 目标：搞清「配置 + Spec → 具体层实现」这条线，之后换 TE/local/MoE 不再靠猜。

---

## 先看骨架：`GPTModel` 装了什么

文件：`megatron/core/models/gpt/gpt_model.py`

`GPTModel` 继承 `LanguageModule`，`__init__` 按流水线阶段有选择地构建：

```text
[pre_process=True]  embedding (+ 位置/RoPE 相关)
        ↓
     decoder: TransformerBlock   ← 多层 TransformerLayer
        ↓
[post_process=True] output_layer（logits / loss）
        + 可选 MTP（Multi-Token Prediction）块
```

`forward` 的主路径可以记成：

```text
_preprocess → self.decoder → _postprocess
```

- `_preprocess`：token embedding、RoPE 等  
- `decoder`：真正的层堆叠  
- `_postprocess`：输出投影、共享 embedding 权重、可选 LM loss / MTP  

**和 Hugging Face `GPT2LMHeadModel` 的心智差异**：Megatron 从第一天就为 PP 切分预留了 `pre_process/post_process`；不是事后再切。

---

## 配置中枢：`TransformerConfig`

文件：`megatron/core/transformer/transformer_config.py`

`TransformerConfig` 继承并行相关配置（如 `ModelParallelConfig`），集中存放：

- 结构：`num_layers`、`hidden_size`、`num_attention_heads`、`ffn_hidden_size`…  
- 并行与布局：PP/VPP、layout、sequence parallel…  
- MoE / MTP / 精度 / recompute 等大量训练特性开关  

训练侧常见转换函数：`core_transformer_config_from_args(args)`（在 `megatron/training/arguments.py` / `argument_utils.py` 一带）。  
阅读建议：**先扫字段名分组，再深入某一组默认值与校验**。

---

## 关键设计：`ModuleSpec` 为什么重要

文件：

- `megatron/core/transformer/spec_utils.py`（`ModuleSpec`、`build_module`）  
- `megatron/core/models/gpt/gpt_layer_specs.py`

Megatron 没有把 Attention/MLP 写死在 `TransformerLayer` 里，而是用 **Spec** 声明：

- 用哪个类（TE 的 `DotProductAttention`？local 实现？）  
- 子模块如何嵌套  
- MoE 时整块 decoder 的 layer 列表长什么样  

`gpt_builder` 根据 args 选择 spec，例如：

| 条件 | 典型函数 |
|------|----------|
| TE dense | `get_gpt_layer_with_transformer_engine_spec` |
| local dense | `get_gpt_layer_local_spec` |
| MoE | `get_gpt_decoder_block_spec` |
| 异构层 | `get_gpt_heterogeneous_layer_spec` |

`TransformerLayer` 在构建时调用 `build_module(spec.xxx)` 把声明变成真实 `nn.Module`。

**工程含义**：框架核心调度与并行代码可以稳定演进；模型变体更多是「换积木」而不是「fork 一份训练循环」。

---

## 单层：`TransformerLayer` 在算什么

文件：`megatron/core/transformer/transformer_layer.py`

标准路径（概念上）：

```text
x → input_layernorm → SelfAttention → residual
  → pre_mlp_layernorm → MLP/MoE → residual
```

对应方法常拆成 `_forward_attention` 与 `_forward_mlp`（便于 PP、recompute、CUDA Graph 等插入点）。

阅读时同时打开：

- `attention.py`：`SelfAttention`（QKV、core attention、输出投影；GQA、CP 等）  
- `mlp.py`：`MLP`（`linear_fc1` → 激活 → `linear_fc2`；SwiGLU 等）  
- MoE 时则是 `transformer/moe/moe_layer.py` 替换 MLP 侧  

**和 TP 的衔接点**：QKV、`fc1` 多为 Column Parallel；attn out、`fc2` 多为 Row Parallel。下一篇并行系列会从进程组讲到这些层。

---

## 堆叠：`TransformerBlock`

文件：`megatron/core/transformer/transformer_block.py`

`TransformerBlock` 负责：

- 按配置构建 N 个 `TransformerLayer`（考虑 PP 本地层数、VPP chunk、MoE block spec）  
- 可选 final layernorm  
- 与 CUDA Graph / recompute 等执行策略协作  

读 `GPTModel.forward` 时，把 `self.decoder(...)` 当成「本地持有的那一段层」。

---

## 建议阅读顺序（模型专题）

```text
gpt_builder
  → TransformerConfig 关键字段
  → GPTModel.__init__ / forward
  → gpt_layer_specs 选一条 dense TE spec 读透
  → TransformerLayer.forward
  → SelfAttention.forward
  → MLP.forward
  → （可选）MoELayer 对比 MLP
```

每读一个类，记三件事：**输入 shape 约定、依赖哪些 process group、输出是否已经是并行切分后的状态**。

---

## 一个具体对照实验

在脑内或真实小实验中对比：

1. `transformer_impl=transformer_engine` vs `local`  
2. 只看 `gpt_builder` 选中的 spec 函数名是否变化  
3. 再看 `TransformerLayer` 里 `self.self_attention` 的实际类型  

你会直观感受到：**训练循环几乎不动，变的是 spec 指向的实现类**。

---

## 常见坑

1. **在 `GPTModel` 里找全部 MoE 逻辑** —— 路由与 dispatcher 在 `transformer/moe/`。  
2. **忽略 `vp_stage`** —— Virtual PP 下同一进程持有多个 model chunk，spec/层数按 stage 变化。  
3. **以为 `parallel_output=True` 可随意改** —— 它与 vocab parallel cross entropy、是否在 TP 组内聚合 logits 强相关。  
4. **共享 embedding 与 PP** —— `share_embeddings_and_output_weights` 会引出跨 PP stage 的梯度同步（见 `finalize_model_grads`）。

---

## 本周作业

1. 画出你当前配置下 `GPTModel` 子模块树（embedding / N layers / output）。  
2. 在 `gpt_layer_specs.py` 里找到 TE dense layer spec，列出 attention 与 mlp 指向的类名。  
3. 阅读 `TransformerLayer.forward`，标出两处 residual 相加的位置。  

下一篇离开「算什么」，进入「切开算」的根基：`parallel_state` 与进程组拓扑。
