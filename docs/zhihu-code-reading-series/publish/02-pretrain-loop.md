# 精读 Megatron 源码（2）：从 `pretrain_gpt.py` 拆开一次训练步

> **专栏**：Megatron 源码精读 · 第 2 篇  
> **上一篇**：代码地图与阅读方法  
> **核心文件**：`pretrain_gpt.py`、`megatron/training/training.py`、`gpt_builders.py`

---

打开 `pretrain_gpt.py`，很多人会失望：这里居然没有 `for step in range(...)`。

这恰恰是 Megatron-LM 最重要的设计——**脚本不写训练循环，只注入三样东西**：

| 你提供什么 | 职责 |
|------------|------|
| `train_valid_test_datasets_provider` | 怎么建 train/valid/test |
| `forward_step` | 一个 microbatch 如何取数、前向、构造 loss |
| `gpt_builder`（经 `model_provider`） | 如何实例化 `GPTModel` |

框架负责编排；领域逻辑留在脚本里。抓住这层契约，后面读 `training.py` 才不会晕。

---

## 入口长什么样

`__main__` 的真实逻辑非常直白：

```495:530:pretrain_gpt.py
if __name__ == "__main__":
    # ...
    args = parse_and_validate_args(
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
    # ...
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
```

翻译成人话：

1. 解析并校验 CLI（`parse_and_validate_args`）  
2. 把 args 收成配置容器  
3. 把数据集 provider、`forward_step` 交给 `pretrain(...)`

训练 for-loop 在 `megatron/training/training.py` 里，不在这个脚本里。

---

## 一条调用链记到肌肉里

```text
pretrain_gpt.__main__
  → parse_and_validate_args
  → pretrain(...)                         # training.py
       → initialize_megatron(...)         # 分布式 / 种子 / 并行
       → setup_model_and_optimizer(...)
       → 构建 data iterators
       → train(...)
            → train_step(...)
                 → forward_backward_func(...)   # pipeline_parallel/schedules
                      → forward_step(...)       # 你在脚本里写的
```

建议：对 `pretrain` 和 `train_step` 各下一处断点，用极小模型跑一轮。比纯阅读快得多。

---

## 阶段 A：参数与全局状态

关键文件：`megatron/training/arguments.py`、`global_vars.py`

典型流程：`parse_args` →（可选从 ckpt 恢复）→ `validate_args` → `set_global_variables`。之后满仓库的 `get_args()` 都从这里取。

**精读提示**：别试图读完上千个 argument。搜你关心的旗标（如 `tensor_model_parallel_size`、`num_experts`），再看 `validate_args` 里相关断言。

---

## 阶段 B：先有进程组，再有模型

`initialize_megatron`（`megatron/training/initialize.py`）会完成：

- `torch.distributed` 初始化  
- 落到 Core 的 `parallel_state.initialize_model_parallel`  
- 随机种子（含 TP 独立 RNG）

记住一句话：**模型构建之前，进程组拓扑必须就绪**。因为 `ColumnParallelLinear` 在 `__init__` 里就要知道自己的 TP size / rank。

---

## 阶段 C：模型怎么被建出来

```text
model_provider(...)
  → gpt_builder(args, pre_process, post_process, ...)
       → TransformerConfig
       → 选择 ModuleSpec（TE / local / MoE …）
       → GPTModel(...)
```

`gpt_builders.py` 里值得对照的分支：

- dense → `get_gpt_layer_with_transformer_engine_spec` / local spec  
- MoE → `get_gpt_decoder_block_spec`  
- 异构层 / 实验 attention / MTP → 各自 spec  

`pre_process` / `post_process` 由流水线阶段决定：只有首 PP stage 做 embedding，尾 stage 做 output。这是 PP「按深度切模型」的前提——第 3、6 篇还会回来。

---

## 阶段 D：`forward_step` 是真正的接口

`forward_step` 大致三步：

1. `get_batch(data_iterator)`：取 tokens/labels/mask，并处理 TP broadcast、CP 切分等  
2. `model(...)` 前向  
3. 返回 `(output_tensor, partial(loss_func, ...))`

注意：第二个返回值是 **loss 函数**，不是已经算好的标量。

为什么？因为只有最后一个 PP stage 看得见 logits，loss 必须在正确的 stage、由 pipeline schedule 调用。这是很多人的「啊哈」时刻。

---

## 阶段 E：`train_step` 里发生了什么

简化骨架：

1. `optimizer.zero_grad`  
2. 取得 `forward_backward_func`（由 PP/VPP 决定）  
3. 跑完本 step 所有 microbatch 的前向+反向  
4. 梯度同步 / `finalize_model_grads`  
5. `optimizer.step` + LR scheduler  
6. 日志 /（按间隔）checkpoint  

选型逻辑在源码里写得很直白：

```156:163:megatron/core/pipeline_parallel/schedules.py
    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func
```

`train_step` 自己不手写流水线，它只是调用 schedule。第 6 篇专门拆 1F1B。

---

## 职责边界一张图

```text
┌────────────── pretrain_gpt.py ──────────────┐
│  datasets_provider / forward_step / builder │
└────────────────────┬────────────────────────┘
                     │ 注入
                     ▼
┌────────────── megatron/training ────────────┐
│  pretrain → train → train_step              │
└────────────────────┬────────────────────────┘
                     │ 调用
                     ▼
┌────────────── megatron/core ────────────────┐
│  parallel_state / schedules / GPTModel ...  │
└─────────────────────────────────────────────┘
```

---

## 常见误读

1. 以为 `pretrain_gpt.py` 里有完整训练循环 —— 在 `training.py`。  
2. 以为 `forward_step` 必须返回 loss 标量 —— 返回的是 `(output, loss_func)`。  
3. 忽略 `pre_process/post_process` —— 多 PP 时中间 stage 没有 embedding/output。  
4. 一上来通读 `arguments.py` —— 性价比极低。

---

## 动手验证（可选）

1. 从 `__main__` 到手画一张不超过 10 步的调用序列。  
2. 在源码里定位：`forward_backward_func` 是在哪个函数里被取到的。  
3. 打开 `gpt_builder`，列出 dense / MoE 两条建模型分支的入口函数名。  

下一篇进入模型本体：`GPTModel` 如何用 `ModuleSpec` 拼出一层 Transformer——这也是 Megatron 能「换积木不换训练循环」的关键。
