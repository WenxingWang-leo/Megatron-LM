# 知乎系列 02｜从 `pretrain_gpt.py` 走进训练主循环

> 目标：看完这篇，你能从命令行入口一路指到一次 `train_step` 的前后向，并说清「模型、数据、损失」三个回调分别由谁提供。

---

## 入口长什么样

`pretrain_gpt.py` 的 `__main__` 非常直白（逻辑摘要）：

1. 打印 PyTorch / Megatron-Core / Transformer Engine 版本  
2. `parse_and_validate_args(...)` 解析并校验 CLI  
3. `gpt_config_from_args` / `pretrain_cfg_container_from_args` 组装配置  
4. 调用 `pretrain(full_config, datasets_provider, ModelType, forward_step, ...)`

也就是说：**脚本不自己写 for-loop**，它把三样东西交给训练框架：

| 回调 / 对象 | 职责 |
|-------------|------|
| `train_valid_test_datasets_provider` | 怎么建 train/valid/test 数据集 |
| `forward_step` | 一个 microbatch 如何取数、前向、构造 loss |
| `gpt_builder`（经 `model_provider`） | 如何实例化 `GPTModel` |

这是读 Megatron-LM 最重要的「契约」：框架负责编排，用户脚本负责领域逻辑。

---

## 调用链总览

```text
pretrain_gpt.__main__
  → parse_and_validate_args
  → pretrain(...)                         # megatron/training/training.py
       → initialize_megatron(...)         # 分布式、种子、并行
       → setup_model_and_optimizer(...)   # 模型 + 优化器 + scheduler
       → 构建 data iterators
       → train(...)
            → train_step(...)
                 → forward_backward_func(...)   # 来自 pipeline_parallel/schedules
                      → forward_step(data_iterator, model)  # 你写的
```

建议你在 IDE 里对 `pretrain`、`train_step` 各下一个断点，跑一次极小配置，比纯阅读快一个数量级。

---

## 阶段 A：参数与全局状态

关键文件：

- `megatron/training/arguments.py`：`parse_and_validate_args`
- `megatron/training/global_vars.py`：`get_args()` 等

典型流程：

1. `parse_args`：把上千个训练开关收进 `args`  
2. （可选）从 checkpoint 恢复部分 args  
3. `validate_args`：检查并行度乘积、互斥选项等  
4. `set_global_variables`：之后各处通过 `get_args()` 取配置  

**阅读提示**：不要试图一次读完所有 argument 定义。先搜你关心的旗标（如 `tensor_model_parallel_size`、`num_experts`），再看 `validate_args` 里相关校验。

---

## 阶段 B：`initialize_megatron`

文件：`megatron/training/initialize.py`

这里会完成：

- 日志与 rerun / fault 相关状态机  
- `_initialize_distributed`：`torch.distributed` init  
- 调用 Core 的模型并行初始化（最终落到 `parallel_state.initialize_model_parallel`）  
- 随机种子（含 TP 所需的独立 RNG，详见后续 TP 篇）

读到这里时，你只需要记住一句话：**在模型构建之前，进程组拓扑必须已经就绪**，因为 `ColumnParallelLinear` 等模块在 `__init__` 里就要知道自己的 TP size / rank。

---

## 阶段 C：模型如何被「建出来」

链路：

```text
model_provider(...)
  → gpt_builder(args, pre_process, post_process, ...)
       → core_transformer_config_from_args(args)  # → TransformerConfig
       → 选择 ModuleSpec（TE / local / MoE / …）
       → GPTModel(config, transformer_layer_spec, ...)
```

`gpt_builders.py` 里的分支很值得对照着读：

- 普通 dense：`get_gpt_layer_with_transformer_engine_spec` 或 local spec  
- MoE：`get_gpt_decoder_block_spec`  
- 异构层 / 实验 attention / MTP：各自走不同 spec  

`pre_process` / `post_process` 由流水线阶段决定：只有第一个 PP stage 做 embedding，最后一个做 output/logits。这是 PP 能「按深度切模型」的前提。

---

## 阶段 D：`forward_step` —— 脚本与框架的接口

在 `pretrain_gpt.py` 中，`forward_step` 大致做三件事：

1. `get_batch(data_iterator)`：取出 tokens、labels、mask，并处理 TP broadcast、CP 切分等  
2. 调用 `model(...)` 做前向  
3. 返回 `(output_tensor, partial(loss_func, ...))`

注意第二个返回值是 **loss 函数**（或 partial），不是已经算好的标量 loss。真正的 loss 计算发生在 pipeline schedule 内部——因为只有最后一个 PP stage 才看得到 logits，需要在正确的 stage 上调用 loss。

这是很多新人第一次读代码时的「啊哈」时刻。

`get_batch` 本身也值得单开笔记：它连接了 dataset、并行拓扑与 `PackedSeqParams`（变长/打包序列）。后续第 08 篇会回到数据侧。

---

## 阶段 E：`train_step` 里一步训练做了什么

文件：`megatron/training/training.py` 中的 `train_step`

逻辑骨架（简化）：

1. `optimizer.zero_grad`（及必要的 grad buffer 复位）  
2. 取得 `forward_backward_func`（由 PP/VPP 配置决定）  
3. 执行前向+反向，累积各 microbatch 的梯度  
4. 梯度同步 / finalize（与 DDP、SP、embedding 跨 PP 等相关）  
5. `optimizer.step` + LR scheduler  
6. 日志、定时器、（按间隔）checkpoint  

`forward_backward_func` 的选择在 `megatron/core/pipeline_parallel/schedules.py` 的 `get_forward_backward_func`：

- `pp == 1`：无流水线  
- `pp > 1` 且无 virtual PP：经典 1F1B（`without_interleaving`）  
- `pp > 1` 且有 VPP：交错调度（`with_interleaving`）  

第 06 篇会专门拆调度；这里你只要把「`train_step` 不手写 PP，而是调用 schedule」记牢。

---

## 一张图记住职责边界

```text
┌──────────────── pretrain_gpt.py ────────────────┐
│  datasets_provider / forward_step / gpt_builder │
└───────────────────────┬─────────────────────────┘
                        │ 注入
                        ▼
┌──────────────── megatron/training ──────────────┐
│  pretrain → train → train_step                  │
│  args / initialize / checkpointing / logging    │
└───────────────────────┬─────────────────────────┘
                        │ 调用
                        ▼
┌──────────────── megatron/core ──────────────────┐
│  parallel_state / schedules / GPTModel / DistOpt│
└─────────────────────────────────────────────────┘
```

---

## 调试时建议打断点的位置

| 断点 | 你想确认的事 |
|------|----------------|
| `parse_and_validate_args` 返回后 | 并行度、batch、seqlen 是否符合预期 |
| `initialize_model_parallel` 末尾 | 各组 world size / rank |
| `gpt_builder` 里 `GPTModel(...)` 前 | spec 选了哪条分支 |
| `forward_step` | batch 各字段 shape |
| `train_step` 调用 `forward_backward_func` 前后 | 是否真正进入 PP schedule |

---

## 常见误读

1. **以为 `pretrain_gpt.py` 里有完整训练循环** —— 循环在 `training.py`。  
2. **以为 `forward_step` 必须返回 loss 标量** —— 返回的是 `(output, loss_func)`。  
3. **忽略 `pre_process/post_process`** —— 多 PP stage 时会表现为「中间 stage 没有 embedding/output」。  
4. **一上来读完 `arguments.py`** —— 性价比极低，按需检索。

---

## 本周作业

1. 从 `__main__` 到手写一张 10 步以内的调用序列（允许省略日志/timer）。  
2. 在源码里定位：`forward_backward_func` 是在哪个函数里被取到的。  
3. 打开 `gpt_builder`，列出「dense / MoE / heterogeneous」三条建模型分支的入口函数名。  

下一篇进入模型本体：`GPTModel` 如何用 `ModuleSpec` 拼出一层 Transformer。
