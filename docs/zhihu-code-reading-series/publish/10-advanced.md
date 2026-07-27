# 精读 Megatron 源码（10）：从「读通主路径」到「能改框架」——CP、推理、RL、精度与贡献指南

> **专栏**：Megatron 源码精读 · 第 10 篇（终篇）  
> **上一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)

---

你已经读完了前九篇。这不是一篇"总结大纲"，而是把那些**前九篇只点到为止、但值得深挖**的主题彻底展开：Context Parallel 的实现原理、推理栈的组织方式、RL 训练与预训练的差异、精度优化的代码位置、以及如何真正参与 Megatron-LM 的贡献。最后用一份完整的系列索引和个人知识系统方法收尾。

---

## 一、掌握度自测：不看代码能讲清楚吗？

下面 15 个问题，每道题你都应该能"不看代码、口头讲清楚"。如果卡壳，回去重读对应章节。

| # | 问题 | 对应章节 |
|---|------|---------|
| 1 | Megatron 的 `parallel_state` 里保存了哪些进程组？它们的含义和相互关系是什么？ | 第 4 篇 |
| 2 | TP 切分 Attention 时，Q/K/V 是列切还是行切？为什么 output projection 反过来？ | 第 5 篇 |
| 3 | PP 的 1F1B schedule 相比 all-forward-then-backward 有什么优势？bubble rate 公式是什么？ | 第 6 篇 |
| 4 | VPP（Virtual Pipeline Parallel）引入了什么新通信？为什么能降低 bubble？ | 第 6 篇 |
| 5 | DDP 的 bucket 是按什么顺序填充的？为什么是逆序？ | 第 7 篇 |
| 6 | `overlap_grad_reduce=True` 时，为什么只有 PP rank 0 的 bucket 启用异步通信？ | 第 7 篇 |
| 7 | `finalize_model_grads` 的三件事是什么？ | 第 7 篇 |
| 8 | `add_extra_token_to_sequence=True` 是什么意思？如果为 False 会怎样？ | 第 8 篇 |
| 9 | DistributedOptimizer 的三阶段是什么？内存收益的公式是什么？ | 第 8 篇 |
| 10 | `ShardedTensor` 的六个字段分别代表什么？从 TP=2 到 TP=4 加载时哪些字段会变？ | 第 8 篇 |
| 11 | Switch Transformer 辅助损失公式是什么？`f_i` 和 `P_i` 哪个可微？ | 第 9 篇 |
| 12 | AllGather Dispatcher 和 AllToAll Dispatcher 通信量各是多少？何时选哪个？ | 第 9 篇 |
| 13 | MoE 中 router weight 的梯度为什么需要在 `finalize_model_grads` 里单独 AllReduce？ | 第 9 篇 |
| 14 | `GPTModel` 的 spec 里 `get_gpt_decoder_block_spec` 是怎么把 MoE 层插进去的？ | 第 3 篇 + 第 9 篇 |
| 15 | `pretrain` 函数的训练主循环最少需要几个 callback？它们的签名是什么？ | 第 2 篇 |

如果 12 道以上能自信作答，你已经进入了"能读懂任意 PR"的阶段。

---

## 二、Context Parallel（CP）深度精读

### 2.1 为什么需要 CP

TP/PP/DP 都不解决"单条序列超长"的问题：

- TP 切的是 hidden 维度，序列长度是全量的。
- PP 切的是层数，序列长度仍是全量的。
- 序列长度 128K 时，单个 QKV 矩阵的 `[B, seq_len, H]` 激活本身就 OOM。

CP（Context Parallel）的思路：**把序列维度切 N 份，每个 CP rank 持有 `seq_len/N` 的 token**，从而把注意力计算和激活内存也切分。

### 2.2 序列切分：Zigzag 负载均衡

直接把序列前 `seq/N` 给 rank 0、后 `seq/N` 给 rank N-1 会导致**注意力不均衡**（rank 0 处理的都是序列前部，causal mask 下看不到后续 token，计算量少；rank N-1 计算量多）。

Megatron 采用 **zigzag 策略**：

```
原始序列：[0, 1, 2, 3, 4, 5, 6, 7]（8 个 token，CP=2）

rank 0 分配：[0, 1, 6, 7]   # 前半的前两个 + 后半的后两个
rank 1 分配：[2, 3, 4, 5]   # 前半的后两个 + 后半的前两个
```

这种"头尾配对"的分配方式，使得 causal mask 下两个 rank 的实际注意力计算量趋于一致。

在代码中，`get_batch_on_this_cp_rank` 调用 `_get_batch_on_this_cp_rank_per_sequence_balancing`，该函数内部通过 `cp_rank` 和 `cp_size` 计算出本 rank 应取的 index，然后 **gather** 相应位置的 token。

### 2.3 get_batch_on_this_cp_rank 的调用链

```python
# megatron/core/utils.py  第 2532 行
def get_batch_on_this_cp_rank(batch, is_hybrid_cp, cp_group=None, ...):
    if use_per_sequence_balancing or batch.get("cu_seqlens") is None:
        batch = _get_batch_on_this_cp_rank_per_sequence_balancing(batch, cp_group)
    elif is_hybrid_cp:
        # Hybrid CP：对 cp_size 内先做一次 per-sequence，再做 per-document
        hybrid_cp_group = hybrid_cp_group_func(group_size=batch['local_cp_size'].item())
        batch = _get_batch_on_this_cp_rank_per_sequence_balancing(batch, hybrid_cp_group)
    else:
        # 普通 packed sequence 模式：per-document zigzag
        batch = _get_batch_on_this_cp_rank_per_document_balancing(batch, cp_group)
    return batch
```

### 2.4 CP 注意力的 Ring Attention 直觉

切分序列后，每个 CP rank 只有 Q/K/V 的一部分。自注意力要求每个 Q 能看到所有 K/V（causal 下看到之前的所有 K/V），这就需要跨 CP rank 的 K/V 通信。

Ring Attention 方案：
```
CP rank 0 拿着自己的 Q，先用本地 K/V 算一部分注意力，
然后把 K/V 发给下一个 rank（同时从上一个 rank 接收），
轮 CP 次后，每个 rank 的 Q 都看到了所有 K/V。
```

这个"环形传递"使得通信和计算可以重叠（overlap），额外通信量是 `O(seq_len * H * CP)` 而非 AllGather 的 `O(seq_len * H * CP^2)`。

### 2.5 DP-CP 联合梯度同步

CP 对梯度的影响：CP 组内不同 rank 的梯度来自同一条序列的不同段，它们**不是副本**，梯度之间不需要 AllReduce（每个 rank 负责自己那段的梯度）。

但 DP 副本之间的梯度需要 AllReduce。因此 Megatron 维护 `dp_cp_group`（DP 与 CP 的联合组），DDP 的 AllReduce 在这个组上执行，保证跨 DP 副本、同 CP rank 的梯度被正确聚合。

```python
# distributed_data_parallel.py 中，AllReduce 使用 dp_cp_group
self.ddp_config.data_parallel_group  # 实际是 dp_cp_group
```

### 2.6 worked example：CP=2，seq_len=8

```
原始 tokens: [t0, t1, t2, t3, t4, t5, t6, t7]
DataLoader 输出（TP broadcast 后，全量）：tokens shape [B=1, 8]

get_batch_on_this_cp_rank（CP=2, zigzag）：
  cp_rank=0: 取 index [0, 1, 6, 7] → tokens [t0, t1, t6, t7]，shape [B=1, 4]
  cp_rank=1: 取 index [2, 3, 4, 5] → tokens [t2, t3, t4, t5]，shape [B=1, 4]

Attention（Ring Attention, causal）：
  cp_rank=0 先用 [t0,t1,t6,t7] 自身 Q/K/V 计算局部注意力
  然后接收 cp_rank=1 的 K/V [t2,t3,t4,t5]，更新 t6,t7 的注意力输出
  （t0,t1 是序列前部，causal 下不看 t2+ 的 K/V）

内存收益：
  没有 CP：BF16 激活 [1, 8, H] = 8H 个元素
  CP=2：每个 rank [1, 4, H] = 4H 个元素，内存减半
```

官方文档：`docs/user-guide/context-parallelism.md`

---

## 三、推理栈：megatron/core/inference/ 地图

Megatron 的推理代码在 `megatron/core/inference/` 下，分为几个子系统：

### 3.1 推理栈层次

```
megatron/core/inference/
├── apis/                    # 用户入口
│   ├── llm.py               # LLM 类：阻塞式生成接口
│   └── async_llm.py         # AsyncLLM：异步流式生成
├── engines/                 # 推理引擎核心
│   ├── abstract_engine.py   # 接口定义
│   ├── dynamic_engine.py    # 动态 batch、KV 缓存管理
│   └── mcore_engine.py      # 封装 model.generate
├── contexts/                # KV 缓存上下文管理
│   ├── static_context.py    # 固定 batch size（prefill+decode 统一）
│   ├── dynamic_context.py   # 动态 KV block 分配（类 PagedAttention）
│   └── kv_block_allocator.py# KV 缓存块分配器
├── config.py                # 推理配置（beam_size, top_k, 等）
└── common_inference_params.py # 推理时传递的参数对象
```

### 3.2 每个子系统的职责

**`engines/`**：控制推理的"外循环"——接收请求、调度 prefill/decode、处理并发。`dynamic_engine.py` 是最完整的实现，支持动态 batch 和 continuous batching。

**`contexts/`**：管理 KV 缓存的分配和释放。`static_context` 适合固定序列长度的基准测试；`dynamic_context` 实现了类似 vLLM 的 PagedAttention 思路，把 KV 缓存分成固定大小的 block，按需分配。

**`apis/`**：面向用户的高层接口。`LLM.generate(prompts)` 是最简单的入口；`AsyncLLM` 支持异步流式返回，适合服务化场景。

### 3.3 什么时候需要读推理代码

- 给 Megatron 部署推理服务时
- 需要修改 KV 缓存策略（如支持更长上下文）
- 集成 FlashInfer / Triton kernel 替换注意力计算
- 实现 speculative decoding

**阅读建议**：从 `LLM.generate` → `engine.generate` → `_forward_step_helper` → 模型 `forward` 的路径入手，理解调用链后再深入 context/allocator。

---

## 四、RL 训练：megatron/rl 与 train_rl.py

### 4.1 RL 训练与预训练的核心差异

```
预训练（pretrain.py）：
  固定数据集 → get_batch → forward → CE loss → backward → step

RL 训练（train_rl.py）：
  Rollout（推理模型生成样本） → 奖励模型打分 → 计算策略梯度损失 → backward → step
```

`train_rl.py` 复用了 `pretrain()` 函数作为训练主循环（传入自定义的 `forward_step`），但 `forward_step` 的实现与预训练差异显著：

```python
# train_rl.py  forward_step（精简）
def forward_step(data_iterator, model):
    runtime_state = get_rl_runtime_state()   # 访问 RL 特有的 runtime 状态
    # 读取 batch（包含 prompt + 生成的 response + rewards）
    ...
    # 计算 log_probs（当前策略）和 reference log_probs
    logprobs = get_logprobs(output, tokens, loss_mask)
    # 计算 GRPO loss（含 KL 散度惩罚）
    loss = calculate_grpo_loss(logprobs, ref_logprobs, rewards, ...)
    return loss, ...
```

### 4.2 RL 训练的关键文件

| 文件 | 职责 |
|------|------|
| `train_rl.py` | 训练入口，定义 `forward_step`、模型提供者 |
| `megatron/rl/rl_utils.py` | `calculate_grpo_loss`、`get_logprobs` 实现 |
| `megatron/rl/agent/` | 与环境/奖励模型交互的 agent 接口 |
| `megatron/rl/inference/` | RL 中的推理接口（生成 rollout 样本） |
| `megatron/rl/server/` | Agent server，处理奖励请求 |

### 4.3 与预训练的内存/并行差异

1. **需要同时持有两份模型**（policy + reference），内存压力倍增。
2. **推理和训练交替**（rollout → train），需要在 `training=False` 模式下生成样本，然后切回 `training=True` 更新参数。
3. **`recompute_granularity`** 的设置更重要：RL 的序列通常更长（prompt+response），激活内存压力大。

---

## 五、精度优化：FP8/FP4/Fusions/CUDA Graph/Recompute

### 5.1 FP8 训练

```python
# 配置项
--fp8-format hybrid     # 前向 E4M3，反向 E5M2
--fp8-interval 1        # 每次 step 都更新量化 scale
```

代码位置：
- `megatron/core/transformer/transformer_config.py`：`fp8` / `fp8_format` 字段
- `megatron/core/extensions/transformer_engine.py`：TE 的 FP8 Attention 和 Linear 集成
- `megatron/core/fp8_utils.py`：FP8 tensor 的量化/反量化工具函数

FP8 的核心原理：使用 8-bit 浮点数（E4M3 或 E5M2 格式）存储激活和权重，减少显存和带宽。TransformerEngine 自动管理量化 scale，通过 `amax history` 动态调整每层的 scale 因子。

**版本钉注意**：FP8 功能对 TransformerEngine 版本要求严格。升级 TE 版本时务必检查 `docker/.ngc_version.dev` 中钉住的镜像版本，FP8 API 在不同 TE 版本间有 breaking change。

### 5.2 FP4 训练（更新：NVIDIA Blackwell）

```python
--fp4   # 启用 FP4 量化（需要 SM100+ 硬件）
```

代码位置：
- `megatron/core/fp4_utils.py`：FP4 tensor 量化工具
- `megatron/core/optimizer/distrib_optimizer.py` 中 `quantize_nvfp4_param_shard`

FP4 是 NVIDIA Blackwell 架构引入的特性，只在最新硬件上可用。代码中有大量 `if config.fp4: ...` 的条件分支。

### 5.3 Fused Kernels

```python
# 配置项
--apply-query-key-layer-scaling   # QK scaling 融合进 attention kernel
--bias-dropout-fusion             # bias + dropout 融合
--masked-softmax-fusion           # masked softmax 融合
```

这些 fusions 通过减少 kernel 启动次数和中间 tensor 的内存读写来提速。TransformerEngine 的 `TELinear` 会自动判断是否使用 cuBLAS 或 cuDNN 的 fused op。

### 5.4 Activation Recompute（梯度检查点）

```python
# 配置项（对应 TransformerConfig 字段）
recompute_granularity: Literal['full', 'selective']
recompute_method: Literal['uniform', 'block']
recompute_num_layers: int
```

- `'full'`：反向传播时重算整个 Transformer layer 的激活，显存最省但计算量翻倍。
- `'selective'`：只重算特定子模块（通过 `recompute_modules` 列表配置）：
  - `"core_attn"`：重算注意力核心（最常用，节省大量 QK/AV 激活）
  - `"moe_act"`：重算 MoE MLP 的激活函数（MoE 场景有用）
  - `"mlp"`：重算整个 MLP 子模块

代码位置：
- `megatron/core/transformer/transformer_config.py`：所有 recompute 相关字段
- `megatron/core/transformer/transformer_layer.py`：`te_checkpoint` 调用点

### 5.5 CUDA Graph

CUDA Graph 把 GPU kernel 调用序列录制成一个图，之后每次 step 直接重放（replay）而无需 CPU side 的逐个 kernel 调度，消除 CPU-GPU 同步开销。

```python
# 配置项
--cuda-graph-modules "attention,mlp"  # 哪些模块使用 CUDA graph
```

代码位置：
- `megatron/core/transformer/cuda_graph_config.py`：CUDA graph 配置
- `megatron/core/optimizer/optimizer_cuda_graph.py`：优化器的 CUDA graph 支持

**注意**：CUDA graph 与动态 shape（如 variable sequence length、MoE 动态路由）不兼容。`ncclep` dispatcher 支持 static shape 模式 (`moe_ncclep_static_shape=True`) 以配合 CUDA graph。

---

## 六、多模态、Mamba 与 Hybrid 模型入口

### 6.1 多模态（LLaVA 风格）

```
megatron/core/models/multimodal/
├── llava_model.py       # ViT + LLM 的拼接模型
├── llava_spec.py        # 各子模块的 ModuleSpec
└── context_parallel.py  # 多模态的 CP 特殊处理（vision token 不参与 CP）
```

入口：`llava_model.py` 的 `LlavaModel.forward`，核心是 vision encoder 输出的 token 被插入到文本序列中，然后统一过 Transformer layers。

### 6.2 Mamba（SSM 模型）

```
megatron/core/models/mamba/
├── mamba_model.py       # 纯 SSM 模型
└── mamba_layer_specs.py # SSM layer 的 ModuleSpec
```

Mamba 不使用 Attention，而是 Selective State Space Model（S6 算子）。代码结构与 GPT 模型一致，只是 layer spec 里把 Attention 替换成了 SSM block。

### 6.3 Hybrid 模型（Attention + SSM 混合）

```
megatron/core/models/hybrid/
├── hybrid_model.py         # Attention + Mamba/RWKV 混合
├── hybrid_layer_specs.py   # 混合层的 spec
└── hybrid_layer_allocation.py  # 控制哪些层是 Attention，哪些是 SSM
```

Hybrid 模型是 Megatron 近期重点方向（对应 NVIDIA Hymba / Falcon-H1 等模型）。`hybrid_builders.py` 是其构建入口，`hybrid_layer_allocation.py` 控制 Attention:SSM 的比例（如 1:7 的配置）。

---

## 七、如何向 Megatron-LM 贡献代码

### 7.1 ProcessGroupCollection 注入规则

这是 `CLAUDE.md` 明确的核心规范，**贡献者必须遵守**：

> 在 `megatron/core` 生产代码中，**避免直接读取全局进程组**（如 `parallel_state.get_tensor_model_parallel_group()`）。改为从调用方接受 `ProcessGroupCollection` 或显式的 `torch.distributed.ProcessGroup`。

**违规示例**（不要这样写）：

```python
# megatron/core/transformer/my_module.py  ← 错误
from megatron.core import parallel_state

class MyModule:
    def forward(self, x):
        group = parallel_state.get_tensor_model_parallel_group()  # 直接读全局状态
        ...
```

**合规示例**（应该这样写）：

```python
# megatron/core/transformer/my_module.py  ← 正确
from megatron.core.process_groups_config import ProcessGroupCollection

class MyModule:
    def __init__(self, ..., pg_collection: ProcessGroupCollection = None):
        self.tp_group = pg_collection.tp if pg_collection else None

    def forward(self, x):
        group = self.tp_group   # 从注入的 pg_collection 取，而非全局状态
        ...
```

**允许的例外**：`parallel_state.py` 本身、测试代码、`process_groups_config.py` 的初始化代码、以及带有显式注释说明的迁移兼容路径。

### 7.2 PR 提交规范

根据 `CLAUDE.md` 和 `docs/developer/contribute.md`：

```bash
# 1. Fork 仓库，在 fork 上工作，不要直接 push 到 NVIDIA/Megatron-LM
git remote add origin https://github.com/<your-username>/Megatron-LM
git checkout -b feature/my-feature

# 2. 修改代码后运行 isort 修复 import 顺序
uv run isort megatron/core/transformer/my_module.py

# 3. 运行 linting
bash tools/autoformat.sh   # 或参考 skills/mcore-linting-and-formatting/SKILL.md

# 4. 提交时必须同时加 -s（Signed-off-by）和 -S（GPG 签名）
git commit -s -S -m "feat: add ProcessGroupCollection to MyModule"

# 5. 创建草稿 PR（必须是 draft）
gh pr create --draft --title "feat: ..." --body "..."
```

### 7.3 测试在哪里

```
tests/
├── unit_tests/          # 单元测试（pytest，无需 GPU）
│   ├── transformer/     # Transformer 层测试
│   ├── dist_checkpointing/  # checkpoint 测试
│   └── models/          # 模型测试
└── functional_tests/    # 功能测试（需要 GPU，CI 跑）
    └── test_recipes/    # YAML recipe 文件，描述训练配置+golden values
```

**添加单元测试**：在 `tests/unit_tests/` 下对应目录添加 `test_my_module.py`，用 `pytest` 框架。无需多卡，可以 mock 进程组（`initialize_model_parallel(tensor_model_parallel_size=1, ...)`）。

**添加功能测试**：在 `tests/functional_tests/test_recipes/` 下添加 YAML recipe，跑完后更新 golden values（见 `skills/mcore-testing/SKILL.md`）。

---

## 八、五个深入实践项目

这五个项目都有明确的"完成标准"，适合在读完全系列后动手实践：

### 项目 1：给一个简单模块实现 ProcessGroupCollection 注入

**目标**：找一个 `megatron/core` 里还在直接调用 `parallel_state.get_*_group()` 的模块，按规范改为接受 `pg_collection` 注入。  
**成功标准**：对应的单元测试通过；PR 描述清楚 why/what/how；CI 通过。  
**参考**：`megatron/core/transformer/moe/moe_utils.py` 中 `get_default_pg_collection` 函数的注释。

### 项目 2：实现一个自定义的 loss_mask 策略

**目标**：修改 `GPTDataset.__getitem__`，实现"多文档打包时文档与文档之间不互相预测"（即 inter-document masking），在 `loss_mask` 和 `attention_mask` 上体现。  
**成功标准**：`MockGPTDataset` 可以演示；写单元测试验证边界情况（单文档、多文档、EOD 在序列首尾）。  
**提示**：注意 `cu_seqlens` 字段的含义（packed sequence 的边界）。

### 项目 3：理解并复现一次 ShardedTensor Resharding

**目标**：用 `MockGPTDataset` 训练一个极小的模型（TP=2，2 层）保存 checkpoint，然后用 TP=4 加载。  
**成功标准**：加载成功，loss 曲线连续（不跳变），无 shape 错误。  
**难点**：需要本地起 4 GPU 进程，或在 CI 环境中。可以先用纸笔模拟 resharding 逻辑验证理解。

### 项目 4：为 MoE 层添加 per-expert 统计 logging

**目标**：在每次 forward 时记录每个 expert 实际收到的 token 数（`tokens_per_expert`），在 TensorBoard/wandb 中展示负载均衡曲线。  
**成功标准**：训练时可以看到各 expert 的 token 分布随 step 变化（初期不均衡→逐渐均衡）。  
**提示**：`MoELayer.forward` 里 `route` 之后就有 `routing_map`；用 `torch.distributed.all_reduce` 跨 EP 汇总，再在 rank 0 打 log。

### 项目 5：实现一个最小的 CP=2 单机测试

**目标**：在 2 GPU 单机上（或用 `torchrun --nproc_per_node=2`）起一个 CP=2 的训练，对比 CP=1 的损失曲线（应完全一致）。  
**成功标准**：CP=2 和 CP=1 产生相同的 loss（允许 FP 误差），验证 CP 实现正确性。  
**提示**：用 `MockGPTDataset`；设置 `--context-parallel-size 2`；对比的核心是 RNG seed 和 data shuffle 完全一致。

---

## 九、个人知识系统：如何维护 Megatron 理解

读完系列只是起点。大型框架的代码会持续演进，维护理解比第一次学更难。推荐以下方法：

### 9.1 配置三元组笔记

每次遇到一个新的训练配置，在笔记里记录：

```
配置名: TP=4, PP=2, EP=8, CP=1 MoE 训练
---
进程组：
  tp_group: 4个rank (rank 0-3, 4-7, ...)
  pp_group: 2个stage
  ep_group: 8个rank (跨pp)
  dp_group: 总进程数 / (TP*PP*EP)

关键 shape（以 hidden=4096, seq=8192 为例）：
  tokens per rank: [B, 8192]（CP=1 时不切）
  TP0 attention weight: [4096/4, 4096] = [1024, 4096]
  每 EP rank local experts: total/EP

通信模式：
  TP: AllReduce / AllGather (inside TP group)
  EP: AllToAll (inside EP group)
  DP: ReduceScatter + AllGather (DDP bucket)
```

### 9.2 维护调用图

在项目/PR 阅读过程中，用 Mermaid 或简单文本维护核心调用链：

```
pretrain() 
  → train_step()
    → forward_backward_func()      # 调度 microbatch
      → forward_step()             # 单次前向
        → model.forward()          # GPTModel
          → TransformerLayer.forward()
            → SelfAttention.forward()
            → MoELayer.forward()   # 若 MoE 层
      → backward_step()
    → finalize_model_grads()       # PP 残差梯度
    → optimizer.step()             # DistOpt 三阶段
```

每次读 PR 或新功能时，更新这张图，把新增的分支标注清楚。

### 9.3 黄金测试集

维护一组自己写的最小测试，覆盖你认为"最容易出错"的组合：

```
test_tp2_pp2_dp2.py       # 基础并行组合
test_moe_ep4_topk2.py     # MoE 路由和辅助损失
test_distopt_zero1.py     # DistributedOptimizer 分片
test_checkpoint_reshard.py # checkpoint resharding
```

每次框架升级后，跑这组测试，快速发现 regression。

---

## 十、完整系列索引与一行精华

| 篇 | 标题 | 一行精华 |
|----|------|---------|
| 1 | 全局地图 | 目录结构 = 功能边界；`megatron/core` 是可移植核心库 |
| 2 | pretrain 启动链 | `pretrain()` 是唯一入口；`initialize_megatron → model → optimizer → train_step` |
| 3 | GPTModel 与 ModuleSpec | `ModuleSpec` 是"配方"，`build_module` 是"烹饪"；VPP 通过 chunk_list 组织 |
| 4 | parallel_state | 进程组是并行的"路由表"；`tp_rank * pp_size * dp_size = world_size` |
| 5 | Tensor Parallel | 列切 → 行切保证 AllReduce 只做一次；Sequence Parallel 节省 LN 激活 |
| 6 | Pipeline Parallel | 1F1B 把 bubble 降到 `(P-1)/(M+P-1)`；VPP 进一步降低 bubble |
| 7 | Data Parallel | Bucket 逆序填充 + ReduceScatter 支持 DistOpt；`finalize_model_grads` 处理残余梯度 |
| 8 | 数据/DistOpt/Checkpoint | 多取一 token shift 得 labels；ZeRO-1 三阶段；ShardedTensor = 全局坐标系 |
| 9 | MoE | Router TopK + Switch 辅助损失；AllToAll Dispatcher；finalize 单独 AllReduce router grad |
| 10 | 进阶与贡献 | CP = zigzag seq 切分 + Ring Attention；pg_collection 注入；贡献必须 draft PR |

---

## 十一、尾声：重新跑一遍最简训练，看你注意到什么

在系列开始时，你可能这样启动训练：

```bash
python pretrain_gpt.py \
  --num-layers 4 --hidden-size 512 --num-attention-heads 8 \
  --seq-length 1024 --max-position-embeddings 1024 \
  --micro-batch-size 2 --global-batch-size 16 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --mock-data ...
```

现在重新跑这个命令，这次你应该注意到：

1. **启动时**：`parallel_state.initialize_model_parallel` 建立了哪些进程组（只有 `world_size=1` 时所有组都是单 rank）。
2. **第一次 DataLoader**：`BlendedMegatronDatasetBuilder` 正在构建 shuffle 索引，或读取缓存。
3. **get_batch**：TP rank 0（此时也是唯一 rank）加载数据，没有 broadcast。CP=1，没有序列切分。PP=1，没有 None 填充。
4. **forward**：单个 `TransformerLayer` 的 SelfAttention + MLP，没有 AllReduce（TP=1）。
5. **backward**：`param_and_grad_buffer` 里只有一个 bucket，`_all_reduce` 对单 rank 是 no-op。
6. **optimizer.step**：`MixedPrecisionOptimizer` 做 BF16→FP32 unscale，FP32 Adam 更新，再拷贝回 BF16。因为 `use_distributed_optimizer=False`（单卡不需要），没有 ReduceScatter/AllGather。
7. **loss 曲线**：使用 `MockGPTDataset` 时 loss 应该从约 `log(vocab_size) ≈ 9.2`（ln(10000)）开始下降。

从单 GPU 完整跑通，再逐步增加 TP/PP/DP/MoE，每次只改一个维度，验证 loss 等价，这是理解分布式框架最扎实的路径。

欢迎你加入 Megatron-LM 的贡献者行列。

---

> **系列全部文章**：[精读 Megatron 源码（1-10）系列索引](./README.md)  
> **上一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)
