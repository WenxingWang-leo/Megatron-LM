# 精读 Megatron 源码（3）：`GPTModel` 为什么能「换积木不换训练循环」

> **专栏**：Megatron 源码精读 · 第 3 篇  
> **核心文件**：`megatron/core/models/gpt/gpt_model.py`、`gpt_builders.py`、`gpt_layer_specs.py`、`transformer_layer.py`

---

如果你用过 Hugging Face，会直觉地以为模型类里写死了 Attention 和 MLP。Megatron 不是这样。

它的关键抽象是：

```text
TransformerConfig  +  ModuleSpec  →  具体层实现
```

训练循环几乎不动；你换的是 Spec 指向的类。搞清这条线，后面看 MoE、TE、local 实现才不会觉得「又 fork 了一份框架」。

---

## `GPTModel` 骨架长什么样

`GPTModel` 继承 `LanguageModule`，按流水线阶段有选择地构建：

```text
[pre_process=True]   embedding（+ RoPE 等）
          ↓
      decoder: TransformerBlock     ← 多层 TransformerLayer
          ↓
[post_process=True]  output_layer（logits / loss）
          + 可选 MTP 块
```

`forward` 主路径可以记成三步：

```text
_preprocess → self.decoder → _postprocess
```

和 HF 的关键差异：**从第一天就为 PP 预留了 `pre_process/post_process`**，不是事后硬切。首 stage 负责 embedding，尾 stage 负责输出；中间 stage 只有 decoder 层。

---

## 配置中枢：`TransformerConfig`

文件：`megatron/core/transformer/transformer_config.py`

它继承并行相关配置，集中存放：

- 结构：`num_layers`、`hidden_size`、`num_attention_heads`、`ffn_hidden_size`…  
- 并行与布局：PP/VPP、sequence parallel…  
- MoE / MTP / 精度 / recompute 等开关  

训练侧常见转换：`core_transformer_config_from_args(args)`。精读时先扫字段分组，再钻某一组默认值与校验——不要试图一次背完。

---

## Spec：真正的「插件点」

关键文件：

- `transformer/spec_utils.py`：`ModuleSpec`、`build_module`  
- `models/gpt/gpt_layer_specs.py`：各类 `get_gpt_layer_*_spec`

`TransformerLayer` 并不写死子模块类型，而是在构建时 `build_module(spec.xxx)`。  
`gpt_builder` 根据 args 选 Spec，例如：

| 条件 | 典型入口 |
|------|----------|
| TE dense | `get_gpt_layer_with_transformer_engine_spec` |
| local dense | `get_gpt_layer_local_spec` |
| MoE | `get_gpt_decoder_block_spec` |
| 异构层 | `get_gpt_heterogeneous_layer_spec` |

看一眼 `gpt_builders.py` 的分支就明白「配置如何变成模型」：

```25:56:gpt_builders.py
def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building GPT model ...')
    if config is None:
        # ...
        config = core_transformer_config_from_args(args)
    # ...
        if args.num_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(...)
        # ...
        else:
            transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
```

**工程含义**：并行调度与训练编排可以稳定演进；模型变体更多是换积木。

---

## 单层在算什么：`TransformerLayer`

标准路径（概念上）：

```text
x → input_layernorm → SelfAttention → residual
  → pre_mlp_layernorm → MLP / MoE → residual
```

同时打开：

- `attention.py`：`SelfAttention`（QKV、core attn、输出投影；GQA、CP…）  
- `mlp.py`：`MLP`（`fc1` → 激活 → `fc2`）  
- MoE 时：`transformer/moe/moe_layer.py` 替换 MLP 侧  

和张量并行的衔接点（下下篇展开）：QKV / `fc1` 多为 Column Parallel；attn out / `fc2` 多为 Row Parallel。

`TransformerBlock` 则负责把本地持有的那一段层堆起来（考虑 PP 本地层数、VPP chunk、MoE block spec）。

---

## 建议精读顺序

```text
gpt_builder
  → TransformerConfig 关键字段
  → GPTModel.__init__ / forward
  → 选一条 dense TE spec 读透
  → TransformerLayer.forward
  → SelfAttention / MLP
  →（可选）MoELayer 对比
```

每读一个类，只记三件事：**输入 shape 约定、依赖哪些 process group、输出是否已是并行切分后的状态**。

---

## 一个对照实验

1. 对比 `transformer_impl=transformer_engine` vs `local`  
2. 看 `gpt_builder` 选中的 spec 函数名是否变化  
3. 再看 `TransformerLayer` 里 `self.self_attention` 的实际类型  

你会直观感到：训练循环几乎不动，变的是 Spec 指向的实现类。

---

## 常见坑

1. 在 `GPTModel` 里找全部 MoE 逻辑 —— 路由在 `transformer/moe/`。  
2. 忽略 `vp_stage` —— Virtual PP 下同一进程持有多个 model chunk。  
3. 乱改 `parallel_output` —— 与 vocab parallel cross entropy 强相关。  
4. 共享 embedding + PP —— 会引出跨 stage 梯度同步（见 `finalize_model_grads`）。

---

下一篇离开「算什么」，进入「切开算」的根基：`parallel_state` 如何把 `world_size` 切成 TP/PP/DP/CP/EP 进程组。读并行篇之前，务必先把这篇的 Spec 思路稳住。
