# 精读 Megatron 源码（1）：从零建立心智模型——地图、术语表与阅读路线

> **系列导言**：这是一个面向完全初学者的 Megatron-LM 源码精读系列，共 10 篇。你不需要事先了解分布式训练，但需要会 Python 和基本的 PyTorch。每篇文章都会带你从"看不懂"到"能改代码"。

---

## 0. 为什么要读 Megatron-LM？

2020 年，GPT-3 的训练让世界意识到：大模型时代来临了。随之而来的工程挑战是——如何在数百乃至数千块 GPU 上高效地训练一个千亿参数的语言模型？Megatron-LM 就是 NVIDIA 给出的答案。

它不仅是一套训练框架，更是一部"大规模分布式 Transformer 训练"的工程教科书。LLaMA、Falcon、Mistral、Nemotron 等知名模型的预训练都曾使用过它或受其启发。如果你想真正理解：

- 为什么 GPU 之间需要 AllReduce 而不是简单的平均梯度？
- 流水线并行（Pipeline Parallelism）的 bubble 怎么来的，又怎么减少？
- 为什么 Transformer 的注意力头必须能被张量并行度整除？

那么，读懂 Megatron 就是最好的答案。

本文是这个系列的**第一篇**，我们先画一张地图，建立全局观。

---

## 1. 仓库全局结构：一次鸟瞰

克隆仓库之后，你会看到以下顶层目录：

```
Megatron-LM/
├── megatron/               ← 所有核心代码
│   ├── core/               ← Megatron Core（通用库）
│   └── training/           ← 训练驱动层（绑定 GPT 等任务）
├── pretrain_gpt.py         ← GPT 预训练入口
├── pretrain_hybrid.py      ← Hybrid（Mamba+Transformer）入口
├── pretrain_vlm.py         ← 视觉语言模型入口
├── gpt_builders.py         ← GPTModel 的构建工厂函数
├── model_provider.py       ← 模型提供函数（供 pretrain 调用）
├── tests/                  ← 单元测试 & 功能测试
├── docs/                   ← 文档
└── examples/               ← 示例脚本（启动命令参考）
```

**最关键的规律**：`megatron/core/` 是"纯粹的库"，`megatron/training/` 是"任务相关的胶水"，`pretrain_gpt.py` 是"最终用户的入口"。

---

## 2. megatron/core 目录树（精简标注版）

```
megatron/core/
├── parallel_state.py           ← ✦ 进程组注册表（TP/PP/DP/CP/EP 的 group 在此）
├── model_parallel_config.py    ← 并行配置 dataclass
├── process_groups_config.py    ← ProcessGroupCollection（显式 PG 携带者）
├── tensor_parallel/            ← ✦ 张量并行（Column/Row Linear, Vocab Embedding）
├── pipeline_parallel/          ← ✦ 流水线并行（schedules, p2p_communication）
├── distributed/                ← ✦ 数据并行（DDP, DistributedOptimizer, FSDP）
├── transformer/                ← ✦ Transformer 组件库
│   ├── transformer_config.py   ←   TransformerConfig（400+ 配置字段）
│   ├── transformer_block.py    ←   TransformerBlock（所有层的容器）
│   ├── transformer_layer.py    ←   单层逻辑（Attn + MLP + LayerNorm）
│   ├── attention.py            ←   注意力实现（MHA/GQA/MLA）
│   ├── mlp.py                  ←   MLP（Dense & MoE）
│   ├── spec_utils.py           ←   ModuleSpec（插件化装配机制）
│   └── moe/                    ←   MoE（路由、expert、all-to-all）
├── models/
│   ├── gpt/
│   │   ├── gpt_model.py        ← ✦ GPTModel 主类
│   │   └── gpt_layer_specs.py  ←   各后端的 layer spec 定义
│   ├── common/
│   │   ├── embeddings/         ←   词嵌入、RoPE、YaRN 等位置编码
│   │   └── language_module/    ←   LanguageModule 基类（loss、shared weights）
│   └── bert/ hybrid/ mamba/ … ←   其他模型族
├── optimizer/                  ← ✦ 分布式优化器（DistributedOptimizer）
├── datasets/                   ←   数据集（GPTDataset, BlendedDataset）
├── dist_checkpointing/         ←   分布式 checkpoint（ShardedTensor）
├── num_microbatches_calculator.py ← 全局 batch / micro batch 计算器
└── utils.py                    ←   通用工具（StragglerDetector 等）
```

**✦ 标记**表示"核心中的核心"，初学者应优先阅读。

---

## 3. megatron/training 目录树

```
megatron/training/
├── training.py         ← ✦ pretrain() 函数、train_step()
├── initialize.py       ← initialize_megatron()、_initialize_distributed()
├── arguments.py        ← ✦ 所有命令行参数定义与 validate_args()
├── checkpointing.py    ←   保存 / 加载 checkpoint
├── global_vars.py      ←   get_args()、get_timers() 等全局访问器
└── utils.py            ←   print_rank_0 等工具
```

---

## 4. 完整术语表（中英对照）

读 Megatron 代码，你会反复遇到这些缩写。下面逐一解释，附上代码中的实际变量名。

### 4.1 并行维度

| 缩写 | 全名 | 中文解释 | 代码变量 |
|------|------|----------|----------|
| **TP** | Tensor Parallelism | **张量并行**：将单个矩阵运算（GEMM）切分到多张 GPU。例如，一个 `[H, 4H]` 的 weight 矩阵按列切分成 TP 份，每张卡只算 `[H, 4H/TP]`。 | `args.tensor_model_parallel_size` |
| **PP** | Pipeline Parallelism | **流水线并行**：把模型的若干 Transformer 层分配到不同 GPU（stage），用流水线方式执行。 | `args.pipeline_model_parallel_size` |
| **DP** | Data Parallelism | **数据并行**：多个 GPU 各持一份完整（或分片）的模型副本，处理不同数据，梯度汇总后同步更新。 | `args.data_parallel_size`（通常由计算得到） |
| **CP** | Context Parallelism | **上下文并行**：将序列长度维度切分到多张 GPU，用于超长上下文（如 128K+）。 | `args.context_parallel_size` |
| **EP** | Expert Parallelism | **专家并行**：MoE 模型中，不同 Expert 分布在不同 GPU 上。 | `args.expert_model_parallel_size` |
| **SP** | Sequence Parallelism | **序列并行**：与 TP 配合使用，将 LayerNorm 和 Dropout 的激活值按序列维度切分，节省显存。 | `args.sequence_parallel`（布尔值） |
| **VPP** | Virtual Pipeline Parallelism | **虚拟流水线并行**：每个 PP stage 持有不止一个"chunk"的层，减少流水线 bubble。 | `args.virtual_pipeline_model_parallel_size` |

### 4.2 批次与计算单位

| 术语 | 中文解释 | 公式 / 说明 |
|------|----------|-------------|
| **microbatch** (mbs) | 微批次 | 单次前向传播处理的样本数，`--micro-batch-size` |
| **global batch** (gbs) | 全局批次 | 每次参数更新消耗的总样本数，`--global-batch-size` |
| **microbatches per step** | 每步微批次数 | `num_microbatches = gbs / (mbs × DP)` |
| **gradient accumulation** | 梯度累积 | 在 PP 场景下，`num_microbatches` 个微批次跑完才做一次 `optimizer.step` |
| **MFU** | Model FLOPs Utilization | 实际 FLOPs / 硬件峰值 FLOPs，衡量 GPU 利用率 |
| **HFU** | Hardware FLOPs Utilization | 考虑了重计算（recompute）开销的 MFU 变体 |

### 4.3 优化器与内存

| 缩写 | 全名 | 中文解释 |
|------|------|----------|
| **DistOpt** | Distributed Optimizer | 分布式优化器：将 Adam 优化器状态（m/v/fp32 param）按 DP shard 切分，大幅降低显存 |
| **Recompute** | Activation Recomputation | 激活重计算（梯度检查点）：前向不保存中间激活，反向时重新计算，用时间换显存 |
| **Spec** | Speculative Decoding | 投机推理：用小模型先猜若干 token，再用大模型并行验证 |
| **ZeRO** | Zero Redundancy Optimizer | FSDP 类优化，Megatron 的 DistributedOptimizer 对应 ZeRO-1/2 |

### 4.4 模型结构相关

| 术语 | 中文解释 |
|------|----------|
| **pre_process** | 该 PP stage 是否持有 Embedding 层（仅 PP stage 0 为 True） |
| **post_process** | 该 PP stage 是否持有 output_layer 和 loss 计算（仅最后 PP stage 为 True） |
| **GQA** | Grouped Query Attention：多个 Q 头共享一对 KV 头，节省 KV cache |
| **MLA** | Multi-head Latent Attention：DeepSeek 提出的 KV cache 压缩方案 |
| **MoE** | Mixture of Experts：将 FFN 替换为多个 expert + 路由器 |
| **RoPE** | Rotary Positional Embedding：旋转位置编码（LLaMA 系列采用） |
| **MTP** | Multi-Token Prediction：同时预测多个未来 token 的训练目标 |

---

## 5. 一次训练步的"一句话故事"

> 数据从 `DataIterator` 取出一个 microbatch → PP stage 0 的 GPU 做 Embedding + 前几层 Transformer（计算中 TP 组内的 GPU 协作完成矩阵乘法）→ 激活值通过 P2P 通信传给 stage 1 → 依次流向最后一个 PP stage → 最后 stage 计算 softmax loss → 反向传播逆向流回 stage 0 → 每个 DP replica 在组内做梯度 AllReduce（或 ReduceScatter）→ DistributedOptimizer 在 DP shard 内更新参数 → 一步结束。

这个故事中有三个并行维度同时工作：**TP 切分单层计算**，**PP 切分层间流水**，**DP 切分数据副本**。CP 和 EP 是可选的扩展。

---

## 6. 核心代码路径的连接图

```
pretrain_gpt.py  (__main__)
    │
    ├─ parse_and_validate_args()          ← megatron/training/arguments.py
    ├─ gpt_config_from_args()             ← megatron/training/argument_utils.py
    └─ pretrain(full_config, ...)         ← megatron/training/training.py
           │
           ├─ initialize_megatron()       ← megatron/training/initialize.py
           │      └─ _initialize_distributed()
           │             └─ mpu.initialize_model_parallel()
           │                    └─ megatron/core/parallel_state.py
           │
           ├─ setup_model_and_optimizer()
           │      ├─ model_provider()     ← model_provider.py / gpt_builders.py
           │      │      └─ GPTModel()   ← megatron/core/models/gpt/gpt_model.py
           │      └─ get_megatron_optimizer()
           │             └─ megatron/core/optimizer/
           │
           └─ train()
                  └─ train_step()        ← megatron/training/training.py
                         ├─ zero_grad_buffer()
                         ├─ optimizer.zero_grad()
                         ├─ forward_backward_func()
                         │      └─ megatron/core/pipeline_parallel/schedules.py
                         │             └─ forward_step() (from pretrain_gpt.py)
                         │                    └─ GPTModel.forward()
                         ├─ optimizer.step()
                         └─ opt_param_scheduler.step()
```

---

## 7. 推荐阅读顺序及理由

### Phase 1：理解单卡单步（第 1-3 篇）

1. **本文（地图篇）**：建立全局概念，不要跳过
2. **第 2 篇（训练循环）**：从 `__main__` 到 `optimizer.step`，读懂一次完整训练步
3. **第 3 篇（GPTModel 全拆）**：深入模型内部，理解 Config/Spec/Layer 体系

**理由**：先看"外"再看"内"。先理解训练流程的骨架，再填入模型细节，否则一上来就陷入 `TransformerLayer` 会迷失方向。

### Phase 2：理解并行（第 4-7 篇）

4. **第 4 篇（parallel_state）**：所有并行的"地基"——进程组是怎么初始化的
5. **第 5 篇（张量并行）**：ColumnParallelLinear / RowParallelLinear 的原理和代码
6. **第 6 篇（流水线并行）**：1F1B schedule、VPP、P2P 通信
7. **第 7 篇（数据并行）**：DDP + DistributedOptimizer，梯度如何汇总

**理由**：并行之间有依赖：TP 假设 PP 已完成进程组划分；DP 依赖 TP+PP 决定 DP rank。按此顺序读，依赖关系自然清晰。

### Phase 3：进阶专题（第 8–11 篇）

8. **第 8 篇（优化器与 Checkpoint）**：DistributedOptimizer 内部、ShardedTensor 格式
9. **第 11 篇（EP 专题，建议先于 09）**：专家并行切什么、Token AlltoAll、与 DP/expert_DP
10. **第 9 篇（MoE）**：Expert 路由、Dispatcher 实现、负载均衡
11. **第 10 篇（前沿扩展）**：CP、FP8、speculative decoding、multi-token prediction

---

## 8. 如何搭建阅读环境

### 8.1 克隆与依赖

```bash
# 克隆仓库
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# 安装最小依赖（CPU 模式，仅用于阅读和单元测试）
pip install torch transformers sentencepiece
```

如果想真正运行，需要 NVIDIA GPU + CUDA + `transformer_engine` (TE)。但**阅读代码不需要**，大部分核心逻辑在没有 TE 的情况下仍可追踪。

### 8.2 快速跑通一个最小示例

Megatron 提供了 `MockGPTDataset`，可以绕过真实数据：

```bash
torchrun --nproc_per_node=1 pretrain_gpt.py \
    --num-layers 4 \
    --hidden-size 256 \
    --num-attention-heads 4 \
    --micro-batch-size 2 \
    --global-batch-size 4 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --train-iters 10 \
    --mock-data \
    --no-pipeline-model-parallel \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /path/to/gpt2-vocab.json \
    --merge-file /path/to/gpt2-merges.txt \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2
```

### 8.3 用 VSCode 配置跳转

在 `.vscode/settings.json` 添加：

```json
{
  "python.analysis.extraPaths": ["./"],
  "python.defaultInterpreterPath": "/your/venv/bin/python"
}
```

这样 `Ctrl+Click` 跳转就可以顺畅地在 `pretrain_gpt.py` → `training.py` → `gpt_model.py` 之间穿梭。

### 8.4 添加 print 调试

在初读时，最有效的方法是加 `print` 语句。例如，在 `megatron/training/training.py` 的 `train_step` 开头加：

```python
import torch.distributed as dist
print(f"[rank {dist.get_rank()}] train_step iteration={iteration}")
```

这样运行时就能看到每张卡的执行轨迹。

### 8.5 版本与 commit 固定建议

Megatron-LM 的 `main` 分支更新频繁，接口可能在两次读代码之间已变更。建议：

1. **记录你读代码时的 commit SHA**：`git log --oneline -1`
2. **固定 transformer_engine 版本**：TE 的接口与 Megatron 紧密耦合。查看 `docker/.ngc_version.dev` 中的 PyTorch 镜像版本，对应的 TE 版本也记录下来。
3. **阅读 CHANGELOG 和 release notes**：每个大版本的 `CHANGELOG.md` 会列出接口变更。

```bash
# 固定到某个稳定 commit 阅读
git checkout <commit-sha>

# 查看当前使用的 TE 要求
grep -r "transformer_engine" requirements.txt pyproject.toml 2>/dev/null | head -5
```

**TE 依赖说明**：`megatron/core/extensions/transformer_engine.py` 通过 `try: from transformer_engine import ...` 导入 TE，并设置 `HAVE_TE = True/False`。如果没有安装 TE，代码会回退到纯 PyTorch 实现（`local` backend），功能完整但缺少 FP8 和部分 Flash Attention 加速。

---

## 9. 初学者常见陷阱（避坑指南）

### 陷阱 1：混淆 world_size 和 DP size

很多人以为 `world_size == DP size`，其实：

```
DP = world_size / (TP × PP × CP)
```

一个 256 卡的集群，若 TP=8, PP=4, CP=1，则 DP=8，而不是 256。

### 陷阱 2：以为每个 rank 都执行 loss 计算

实际上，**只有最后一个 PP stage**（`post_process=True` 的 rank）才计算 loss。中间 PP stage 的 `forward_step` 返回的是 `None` 的 loss。

### 陷阱 3：搞不清 `micro_batch_size` 和 `global_batch_size`

- `--micro-batch-size 2`：每个 rank 单次前向处理 2 条数据
- `--global-batch-size 128`：每次参数更新消耗 128 条数据
- 因此 `num_microbatches = 128 / (2 × DP_size)`

如果这个数字不是整数，初始化就会报错。

### 陷阱 4：期望 TP=4 就能训练 head_dim 任意的模型

TP 要求 `num_attention_heads % TP == 0`（对 GQA 还要求 `num_key_value_heads % TP == 0`）。如果头数不能整除，代码会在 `validate_args` 阶段报错。

### 陷阱 5：忘记 PP 需要 `virtual_pipeline_model_parallel_size` 整除 `num_layers/PP`

VPP 要求 `num_layers / (PP × VPP)` 必须是整数。32 层 / PP=4 / VPP=2 = 4 层/chunk，合法。33 层则不合法。

### 陷阱 6：以为 `megatron/core` 和 `megatron/training` 可以互换

`megatron/core` 是平台无关的纯库，不依赖 `megatron/training`。`megatron/training` 依赖 `megatron/core`。你可以把 `megatron/core` 单独发布为 `megatron-core` 包（实际上 NVIDIA 正是这样做的）。

### 陷阱 7：直接 `import parallel_state` 调用全局 group

在 `megatron/core` 内部的生产代码中，应避免直接调用 `parallel_state.get_tensor_model_parallel_group()`，而应通过 `ProcessGroupCollection` 传递 process group。这是代码风格的重要约定（详见 `CLAUDE.md`）。

---

## 10. 全系列目录（共 10 篇）

| 篇号 | 标题 | 核心内容 |
|------|------|----------|
| **1** | 地图、术语表与阅读路线 | 本文 |
| **2** | 完整拆解一次训练——从 `__main__` 到 `optimizer.step` | 训练循环全流程 |
| **3** | GPTModel 全拆解——Config、Spec 与每一层计算 | 模型架构深度分析 |
| **4** | parallel_state：所有并行的地基 | 进程组初始化 |
| **5** | 张量并行：矩阵如何被切开 | ColumnParallel/RowParallel |
| **6** | 流水线并行：1F1B 与 VPP | 调度算法、P2P 通信 |
| **7** | 数据并行：DDP 与分布式优化器 | 梯度同步、ZeRO |
| **8** | 优化器与 Checkpoint | DistOpt、ShardedTensor |
| **9** | MoE：专家如何路由和并行 | 路由、All-to-All、EP |
| **10** | 前沿扩展：CP/FP8/Spec/MTP | 最新特性 |

---

## 11. `examples/run_simple_mcore_train_loop.py` 全注解

Megatron-LM 在 `examples/` 下提供了一个极简的训练示例 `run_simple_mcore_train_loop.py`，它只用 `megatron/core`，**完全绕过 `megatron/training`**。这是理解 Megatron Core 独立能力的最佳入口。

### 11.1 它证明了什么

这个文件证明：你**不需要**使用 `pretrain_gpt.py`、`megatron/training/arguments.py` 等庞大的驱动层，就能用 Megatron Core 组装一个可以训练的模型。换句话说：

- `megatron/core` 是一个独立的、可移植的库
- 进程组初始化 + 模型构建 + 数据加载 + 前向/反向 + 优化 = 完整训练，只需 ~280 行代码

### 11.2 关键函数逐行注解

```python
# ============================================================
# initialize_distributed()  —— 第 29-54 行
# ============================================================
def initialize_distributed(tensor_model_parallel_size=1,
                            pipeline_model_parallel_size=1):
    # 第 1 步：销毁任何已存在的模型并行状态（方便多次调用）
    parallel_state.destroy_model_parallel()

    # 从环境变量读取 rank/world_size（torchrun 注入）
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 标准的 PyTorch NCCL 初始化
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    # 第 2 步：Megatron Core 进程组初始化
    # 内部会创建 TP/PP/DP/CP 等所有进程组
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )
```

关键点：`parallel_state.initialize_model_parallel()` 是唯一需要调用的 Megatron 初始化函数，等价于 `megatron/training/initialize.py` 里 `_initialize_distributed()` 的核心逻辑。

```python
# ============================================================
# model_provider()  —— 第 57-79 行
# ============================================================
def model_provider():
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,   # 在 CPU 初始化，避免 GPU OOM
        pipeline_dtype=torch.float32,  # PP 通信时的数据类型
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),  # local backend
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
        # pre_process=True, post_process=True 是默认值（单 PP stage）
    )
    return gpt_model
```

注意 `get_gpt_layer_local_spec()`：这是"local backend"，即纯 PyTorch 实现，不依赖 TE。对于读代码，local backend 远比 TE backend 更易追踪，因为没有 TE 包装层。

```python
# ============================================================
# 主训练循环  —— 第 249-271 行
# ============================================================
forward_backward_func = get_forward_backward_func()
# 根据 PP size 自动选择调度算法：
#   PP=1 → forward_backward_no_pipelining()
#   PP>1 → forward_backward_pipelining_with_interleaving() 或
#           forward_backward_pipelining_without_interleaving()

for iteration in range(5):
    optim.zero_grad()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=train_iterator,
        model=gpt_model,             # 这里传的是 DDP 包装后的 model
        num_microbatches=1,          # 简化示例：每步只有 1 个 microbatch
        seq_length=_SEQUENCE_LENGTH,
        micro_batch_size=8,
        decoder_seq_length=_SEQUENCE_LENGTH,
        forward_only=False,          # False = 前向 + 反向
    )

    # 关键！finalize_model_grads 做两件事：
    # 1. 非 TP 并行参数（如 LayerNorm）跨 TP rank 做 AllReduce
    # 2. 所有参数跨 DP rank 做梯度汇总
    finalize_model_grads([gpt_model])

    optim.step()
```

`finalize_model_grads` 是这个示例最容易被忽视的调用——它等价于 `pretrain_gpt.py` 中由 `DistributedOptimizer` 内部自动完成的梯度同步。

### 11.3 此示例与完整 `pretrain_gpt.py` 的差异

| 特性 | `run_simple_mcore_train_loop.py` | `pretrain_gpt.py` |
|------|---------------------------------|-------------------|
| 优化器 | `torch.optim.Adam`（普通 Adam） | `MegatronOptimizer` / `DistributedOptimizer` |
| 参数更新 | `optim.step()` 直接调用 | 有 gradient clipping、overflow 检测、bf16 精度转换 |
| 梯度同步 | `finalize_model_grads` 手动调用 | 自动由 DDP/DistOpt 内部处理 |
| 进程组 | `parallel_state` 全局 | 通过 `ProcessGroupCollection` 显式传递 |
| 数据 | `MockGPTDataset` + 简单 DataLoader | `BlendedMegatronDataset` + 分布式 Sampler |
| checkpoint | `dist_checkpointing.save/load` | 完整的 `save_checkpoint` / `load_checkpoint` |

**结论**：如果你想做科研实验或快速迭代，从这个示例出发是最快路径。如果你需要生产级训练（FP8、梯度裁剪、分布式 ckpt），则必须使用完整的驱动层。

---

## 12. HuggingFace Trainer vs Megatron 心智模型对比

很多读者来自 HuggingFace 生态，习惯了 `Trainer` 的使用方式。下面做一个直接对比，帮助建立"翻译"关系：

| 概念 | HuggingFace Trainer | Megatron 等价物 |
|------|---------------------|-----------------|
| 模型 | `AutoModelForCausalLM.from_pretrained(...)` | `GPTModel(config, transformer_layer_spec, ...)` |
| 训练配置 | `TrainingArguments` | `TransformerConfig` + `args`（命令行参数） |
| 一步训练 | `trainer.train()` 内部的 `training_step()` | `train_step()` in `training.py` |
| 梯度累积 | `gradient_accumulation_steps=N` | `num_microbatches = gbs / (mbs × DP)` |
| 数据并行 | `DataParallel` / `DistributedDataParallel` | `DistributedDataParallel` in `megatron/core/distributed/` |
| 混合精度 | `fp16=True` / `bf16=True` in TrainingArguments | `config.bf16=True` + `DistributedOptimizer` 的 fp32 master params |
| checkpoint | `trainer.save_model()` | `save_checkpoint()` → `dist_checkpointing.save()` |
| 评估循环 | `trainer.evaluate()` | `evaluate_and_print_results()` |
| 分布式启动 | `accelerate launch` / `torchrun` | `torchrun` / `srun` + `pretrain_gpt.py` |
| 张量并行 | 不支持（单卡单模型） | `ColumnParallelLinear` + `RowParallelLinear` |
| 流水线并行 | 不支持 | `schedules.py` 的 1F1B 调度 |

**最根本的差异**：HuggingFace Trainer 是"单机多卡"的抽象，梯度同步由 `torch.nn.parallel.DistributedDataParallel` 完全自动处理。Megatron 则需要**显式管理三个并行维度的进程组**——这是它强大也是它复杂的原因。

**什么时候用哪个**：
- 模型 < 7B，单机 8 卡可以放下 → HuggingFace Trainer 更简单
- 模型 > 13B 或需要 TP/PP → Megatron 是首选
- 需要 FP8 训练 + 自定义并行策略 → Megatron + TE

---

## 13. 并行组合食谱：5 种真实配置详解

以下 5 种配置覆盖了从研究到生产的典型场景。每种配置给出 DP 计算和适用场景。

### 配置 A：研究调试（8 卡单机，7B 模型）

```
world_size=8, TP=1, PP=1, CP=1
DP = 8 / (1×1×1) = 8
num_microbatches = gbs / (mbs × 8)
```

**特点**：全 DP，无模型并行。每张卡持有完整模型副本，梯度全量 AllReduce。适合 ≤7B 模型（bf16 下约 14GB）。
**何时使用**：快速实验、消融研究、超参搜索。

### 配置 B：显存紧张（8 卡单机，13B 模型）

```
world_size=8, TP=2, PP=1, CP=1
DP = 8 / (2×1×1) = 4
num_microbatches = gbs / (mbs × 4)
```

**特点**：TP=2 将每层切分到 2 张卡（NVLink 互连），DP=4 提供 4× 数据吞吐。13B 模型 bf16 约 26GB，单卡放不下，TP=2 后每卡约 13GB。
**何时使用**：单机 A100/H100 8 卡，但模型刚好超过单卡显存。

### 配置 C：中等规模（64 卡，30B 模型）

```
world_size=64, TP=4, PP=2, CP=1
DP = 64 / (4×2×1) = 8
num_microbatches = gbs / (mbs × 8)
```

**特点**：TP=4 在单机内（4 卡 NVLink），PP=2 跨两台机器，DP=8 提供数据并行。30B 模型 bf16 约 60GB，TP=4 后每卡约 15GB。PP=2 的 bubble 开销约 `(2-1)/num_microbatches`。
**何时使用**：8×8 卡集群，30-40B 量级模型。

### 配置 D：大规模生产（256 卡，70B 模型）

```
world_size=256, TP=8, PP=4, CP=1
DP = 256 / (8×4×1) = 8
num_microbatches = gbs / (mbs × 8)
```

**特点**：TP=8 充分利用单机 8 卡 NVLink，PP=4 跨 4 台机器，DP=8 提供适度的数据并行。70B bf16 约 140GB，TP=8 后每卡约 17.5GB。bubble 约 `(4-1)/num_microbatches`，需要 num_microbatches≥12 才能将 bubble 控制在 25% 以内。
**何时使用**：NVIDIA DGX H100 集群，Llama-70B 规模训练。

### 配置 E：超长序列（512 卡，70B 模型，128K 上下文）

```
world_size=512, TP=8, PP=4, CP=2
DP = 512 / (8×4×2) = 8
num_microbatches = gbs / (mbs × 8)
```

**特点**：在配置 D 的基础上，CP=2 将 128K 的序列切成两段，每段 64K，分布在 2 张卡上。这使得注意力矩阵的内存开销也减半。CP 使用 Ring AllReduce 通信注意力 KV，通信量随 CP 增大而增大。
**何时使用**：序列长度 > 32K 的训练，如长文档、代码库上下文。

### 一句话规律

```
TP ≤ 单机 GPU 数（NVLink 域内）
PP = 机器数的因数（跨机器通信可接受）
CP > 1 仅在序列长度 > 32K 时考虑
DP = 剩余卡数（越大越好，但受 gbs/mbs 整除性限制）
```

---

## 14. "如何阅读 2000 行文件"——以 schedules.py 为例

`megatron/core/pipeline_parallel/schedules.py` 是 Megatron 中最复杂的单个文件之一，约 2000 行，包含多种流水线调度算法。对于初学者，直接从头读是灾难——正确的方法是**分层剥洋葱**。

### 第 1 层：先读顶层函数签名

```bash
# 用 grep 找出所有 def 函数
grep "^def " megatron/core/pipeline_parallel/schedules.py
```

输出类似：
```
def get_forward_backward_func()
def forward_backward_no_pipelining(...)
def forward_backward_pipelining_without_interleaving(...)
def forward_backward_pipelining_with_interleaving(...)
```

这 4 个函数就是文件的全部公开 API。其中 `get_forward_backward_func()` 是入口，它根据 PP size 返回合适的调度函数。

### 第 2 层：从最简单的路径开始

`forward_backward_no_pipelining`（PP=1 时使用）是最简单的，只有约 50 行。先把它读透：它做什么？在什么条件下被选中？返回什么？

### 第 3 层：对比阅读

读完 `no_pipelining` 后，再读 `without_interleaving`（1F1B，无 VPP）。重点找"这两个函数有什么不同"：
- `without_interleaving` 多了什么循环？
- P2P 通信在哪里发生？
- microbatch 的顺序是怎样的？

### 第 4 层：最后读 `with_interleaving`（1F1B + VPP）

这是最复杂的路径，但有了前面的对比基础，你能识别"哪些是新增的 VPP 特有逻辑"。

### 关键洞察：调度文件的状态机模式

所有调度函数都遵循同一个模式：

```python
# 伪代码展示调度函数的骨架
def forward_backward_pipelining_without_interleaving(...):
    # 阶段 1：warm-up（流水线填满）
    for step in range(warmup_steps):
        run_forward()     # 前向 microbatch
        send_forward()    # P2P 发送激活给下一 stage

    # 阶段 2：稳态 1F1B（每步一进一出）
    for step in range(steady_state_steps):
        run_forward()
        send_forward()
        receive_backward()
        run_backward()
        send_backward()

    # 阶段 3：cooldown（流水线排空）
    for step in range(cooldown_steps):
        receive_backward()
        run_backward()
        send_backward()
```

一旦你认出这个骨架，文件中 90% 的代码都是在处理细节（VPP、CP、错误处理、性能 profiling）。

---

## 15. 本文小结

现在你手里有了：

1. **目录结构地图**：知道每个文件大概做什么
2. **完整术语表**：再也不会被 TP/PP/DP/CP/EP/SP/VPP/DistOpt/MFU 搞混
3. **一句话故事**：能向别人解释"Megatron 一步训练发生了什么"
4. **推荐阅读路线**：Phase 1 → 2 → 3，有先后依赖
5. **环境搭建方法**：能运行最小示例；版本固定建议
6. **避坑清单**：7 个常见错误，提前规避
7. **`run_simple_mcore_train_loop.py` 注解**：理解 Megatron Core 最小用法
8. **HuggingFace vs Megatron**：翻译关系，帮助已有 HF 经验的读者快速上手
9. **5 种并行配置食谱**：覆盖 7B-180B 规模的典型场景
10. **阅读 2000 行文件的方法论**：以 schedules.py 为例

下一篇，我们将用"探针法"从 `pretrain_gpt.py` 的第一行开始，逐行追踪到 `optimizer.step`，让整个训练循环在脑子里留下清晰的印记。

---

## 课后练习

**练习 1**：打开 `/workspace/pretrain_gpt.py`，找到 `BATCH_KEYS` 列表，回答：这个列表的顺序有什么特殊之处？（提示：看注释）

**练习 2**：运行以下命令，观察输出中 `DP` 是多少：

```bash
torchrun --nproc_per_node=2 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --num-layers 4 --hidden-size 256 --num-attention-heads 4 \
    --micro-batch-size 1 --global-batch-size 2 \
    --seq-length 64 --max-position-embeddings 64 \
    --train-iters 1 --mock-data \
    --tokenizer-type NullTokenizer --vocab-size 1000 \
    --lr 0.0001 --min-lr 0.0001 --lr-decay-style cosine
```

**练习 3**：在 `megatron/core/` 目录下，找一个同时被 `megatron/training/` 和 `megatron/core/models/` 导入的文件，说明它属于哪个"层"。

**练习 4（思考题）**：假设你有 16 张 A100，想训练一个 13B 参数的模型，你会如何设置 TP/PP/DP？请写出你的理由（没有唯一正确答案）。

---

## 附录 A：并行维度组合的快速计算器

当你拿到一个训练配置时，第一件事是验证并行维度是否合法：

```python
def check_parallel_config(world_size, tp, pp, cp, dp_expected=None):
    """验证并行配置合法性"""
    total_model_parallel = tp * pp * cp
    assert world_size % total_model_parallel == 0, (
        f"world_size={world_size} 不能整除 tp*pp*cp={total_model_parallel}"
    )
    dp = world_size // total_model_parallel
    if dp_expected is not None:
        assert dp == dp_expected, f"期望 DP={dp_expected}，实际 DP={dp}"
    print(f"合法配置: TP={tp}, PP={pp}, CP={cp}, DP={dp}")
    print(f"  模型并行总大小: {total_model_parallel}")
    print(f"  每个 DP group 有 {dp} 个 replica")
    return dp

# 示例：256 卡集群
check_parallel_config(world_size=256, tp=8, pp=4, cp=1)
# 输出: 合法配置: TP=8, PP=4, CP=1, DP=8
```

### 常见配置参考表

| 总卡数 | TP | PP | CP | DP | 适合场景 |
|--------|----|----|----|----|----------|
| 8 | 1 | 1 | 1 | 8 | 7B 以下模型，单机 |
| 8 | 2 | 1 | 1 | 4 | 13B，显存紧张 |
| 64 | 4 | 2 | 1 | 8 | 30B，8×8 卡集群 |
| 256 | 8 | 4 | 1 | 8 | 70B，大集群 |
| 512 | 8 | 4 | 2 | 8 | 70B + 长序列 |
| 1024 | 8 | 8 | 1 | 16 | 180B+ |

**规律**：
- TP 不要超过单机 GPU 数（NVLink 带宽限制）
- PP 跨机器，bubble 开销随 PP 增大
- CP 用于序列长度 > 32K 的场景
- DP 决定有效 batch size 的扩展性

---

## 附录 B：Megatron Core 与 megatron-core 包的关系

NVIDIA 将 `megatron/core/` 单独发布为 PyPI 包 `megatron-core`，这意味着：

```bash
# 可以单独安装
pip install megatron-core

# 也可以从源码使用
git clone https://github.com/NVIDIA/Megatron-LM.git
# 直接 import megatron.core
```

这个设计允许其他框架（如 NeMo）直接依赖 `megatron-core` 而不需要整个 Megatron-LM。在阅读代码时，要注意：

- `megatron/core/` 中的代码是"库代码"，有严格的 API 稳定性要求
- `megatron/training/` 中的代码是"应用代码"，变动更频繁
- `pretrain_gpt.py` 等顶层脚本是"脚本代码"，根据需求自由修改

---

## 附录 C：关键文件快速索引

下面是"当你遇到问题 X，应该去看哪个文件"的索引：

| 问题 | 文件 |
|------|------|
| `--tensor-model-parallel-size` 是怎么传进去的？ | `megatron/training/arguments.py` |
| DP size 是怎么计算的？ | `megatron/training/arguments.py` 的 `validate_args` |
| `torch.distributed.init_process_group` 在哪里调用？ | `megatron/training/initialize.py` 的 `_initialize_distributed` |
| TP 的进程组是怎么建的？ | `megatron/core/parallel_state.py` |
| `get_args()` 是全局变量还是函数？ | `megatron/training/global_vars.py`（全局单例） |
| GPTModel 的 forward 在哪里？ | `megatron/core/models/gpt/gpt_model.py` |
| Transformer 单层（self-attention + MLP）在哪里？ | `megatron/core/transformer/transformer_layer.py` |
| ColumnParallelLinear 在哪里？ | `megatron/core/tensor_parallel/layers.py` |
| 1F1B 流水线调度在哪里？ | `megatron/core/pipeline_parallel/schedules.py` |
| 分布式优化器在哪里？ | `megatron/core/optimizer/distrib_optimizer.py` |
| checkpoint 保存逻辑？ | `megatron/training/checkpointing.py` |
| num_microbatches 怎么算的？ | `megatron/core/num_microbatches_calculator.py` |

---

## 附录 D：代码中常见的 `mpu` 别名

你会在代码里频繁看到 `mpu`，它是 `megatron/core/__init__.py` 中导出的 `parallel_state` 的别名：

```python
# megatron/core/__init__.py
from megatron.core import parallel_state as mpu
```

常见用法：

```python
from megatron.core import mpu

# 获取当前 rank 在各个并行维度的位置
tp_rank = mpu.get_tensor_model_parallel_rank()
pp_rank = mpu.get_pipeline_model_parallel_rank()
dp_rank = mpu.get_data_parallel_rank()

# 判断是否在特殊位置
mpu.is_pipeline_first_stage()   # pre_process
mpu.is_pipeline_last_stage()    # post_process

# 获取进程组
tp_group = mpu.get_tensor_model_parallel_group()
dp_group = mpu.get_data_parallel_group()
```

在 `megatron/core` 内部的新代码中，应避免直接调用 `mpu.get_*_group()`，而应通过 `ProcessGroupCollection` 传递（见 `CLAUDE.md`）。但在 `megatron/training` 和阅读时，`mpu` 是理解代码的关键入口。

---

## 附录 E：扩展 FAQ（10 个初学者高频问题）

**Q1：为什么 Megatron 使用 `[S, B, H]`（Sequence-first）而不是 PyTorch 惯用的 `[B, S, H]`（Batch-first）？**

A：历史原因。Sequence-first 格式在张量并行下对内存访问更友好——矩阵乘法时，序列维度是"外层循环"，batch 内部的并行计算可以更好地利用 CUDA 的 warp 布局。现代版本已开始支持 `[B, S, H]`（`qkv_format` 参数），但默认仍是 `[S, B, H]`。

**Q2：`--no-pipeline-model-parallel` 这个参数存在吗？**

A：不存在这个参数名。设置 `--pipeline-model-parallel-size 1`（默认值）即相当于不使用流水线并行。`get_forward_backward_func()` 会自动选择 `forward_backward_no_pipelining`。

**Q3：Megatron 的 `DistributedDataParallel` 和 PyTorch 的 `torch.nn.parallel.DistributedDataParallel` 有什么区别？**

A：Megatron 的 DDP（`megatron/core/distributed/`）做了以下定制：
- 将所有参数的梯度聚合到一个**连续内存 buffer**，便于一次性 AllReduce/ReduceScatter
- 支持梯度桶（bucket）的重叠通信（`overlap_grad_reduce=True`）
- 与 `DistributedOptimizer` 深度集成，支持 ZeRO-style 的 optimizer state sharding

**Q4：`--sequence-parallel` 到底省了多少显存？**

A：理论上，LayerNorm 和 Dropout 的激活值按 TP 切分，省了 `(1 - 1/TP)` 的比例。对于 TP=8，这意味着这部分激活省了 87.5%。在 70B 模型的完整 activation recompute 场景下，SP 带来的显存节省相对有限；但在不启用 recompute 时，它是显存优化的重要手段。

**Q5：可以在 CPU 上运行 Megatron 做单元测试吗？**

A：可以，通过 `use_cpu_initialization=True` 在 CPU 上初始化模型，并使用 `gloo` backend 代替 `nccl`：
```python
torch.distributed.init_process_group(backend="gloo", ...)
TransformerConfig(use_cpu_initialization=True, ...)
```
大量 `tests/unit_tests/` 下的测试就是这样运行的。

**Q6：`context_parallel_size > 1` 时，注意力怎么计算？KV 不是在不同卡上吗？**

A：CP 使用 Ring Attention：每张卡持有完整的 Q 的一个切片，但 K/V 会通过 Ring AllGather 在 CP 组内循环传递。每张卡依次与所有其他卡的 K/V 做注意力计算，最终每张卡得到对应序列切片的注意力输出。这比朴素 AllGather 全量 KV 更节省内存。

**Q7：`--recompute-granularity selective` 和 `full` 分别在什么场景下用？**

A：`selective`（默认重计算 `core_attn`）：只重计算注意力的核心计算（QK^T → softmax → V），这部分内存占用大但计算相对便宜，是显存/速度 tradeoff 最优的选项，推荐大多数场景。`full`：重计算整个 Transformer 层，显存节省最大（约 60-80%），但训练时间增加约 30-40%，适合显存极度紧张的场景。

**Q8：`TransformerConfig` 里的 `ffn_hidden_size` 设置为多少合适？**

A：传统 GPT（ReLU FFN）：`4 × hidden_size`。使用 SwiGLU（`gated_linear_unit=True`）时：通常是 `8/3 × hidden_size` 向上取整到 64 的倍数，例如 Llama-7B 的 `hidden_size=4096`，`ffn_hidden_size=11008`（约 `8/3 × 4096 ≈ 10922`，取最近的 64 倍数）。

**Q9：Megatron 的 `--transformer-impl` 参数有哪些选项，默认是哪个？**

A：常见选项：
- `local`：纯 PyTorch 实现，不依赖 TE，适合调试
- `transformer_engine`（默认，如果有 TE）：使用 NVIDIA Transformer Engine，支持 FP8、Flash Attention、融合 LayerNorm+Linear
- `inference_optimized`：推理优化的实现，需要 TE

如果环境没有安装 TE，即使指定 `transformer_engine` 也会回退到 `local`（并打印警告）。

**Q10：Megatron 的 checkpoint 是什么格式？可以直接加载到 HuggingFace 吗？**

A：Megatron 使用 `dist_checkpointing`（ShardedTensor 格式），每个 PP/TP rank 保存自己的分片。不能直接加载到 HuggingFace——需要先用 `tools/checkpoint/` 下的转换脚本（如 `convert_checkpoint_from_megatron_to_transformers.py`）将 Megatron checkpoint 合并并转换为 HuggingFace 格式。反方向同理。

---

## 附录 F：术语速记小测验（带答案）

以下 5 题用于检验你对核心概念的掌握。

**题 1**：`world_size=32`, `TP=4`, `PP=2`, `CP=1`，问 `DP=?`

> **答**：`DP = 32 / (4×2×1) = 4`

**题 2**：`global_batch_size=256`, `micro_batch_size=4`, `DP=4`，问 `num_microbatches=?`

> **答**：`num_microbatches = 256 / (4×4) = 16`

**题 3**：PP=4, VPP=2，32 层模型，PP rank=3, vp_stage=1 持有哪些全局层？

> **答**：每个 chunk = 32/(4×2) = 4 层。全局偏移 = vp_stage × PP + pp_rank = 1×4 + 3 = 7，即第 7 个 chunk → 全局层 28-31。

**题 4**：`recompute_granularity='selective'` 默认重计算哪个子模块？

> **答**：`core_attn`（核心注意力计算：QK^T → softmax → V），由 `TransformerConfig.__post_init__` 中 `if self.recompute_modules is None: self.recompute_modules = ["core_attn"]` 设置。

**题 5**：使用 `DistributedOptimizer` 时，为什么 `train_step` 需要同时调用 `zero_grad_buffer()` 和 `optimizer.zero_grad()`？

> **答**：`zero_grad_buffer()` 清零 `param.main_grad`（连续内存 buffer，用于 ReduceScatter 通信）；`optimizer.zero_grad()` 清零 `param.grad`（PyTorch 标准梯度）。两者可能指向不同内存区域，都不清零会导致梯度跨步累积。

---

---

## 附录 G：num_microbatches_calculator —— 动态 batch size 调度

`megatron/core/num_microbatches_calculator.py` 是一个常被忽略但非常重要的文件。它实现了两种 batch size 策略：

### G.1 恒定 batch size（`ConstantNumMicroBatchesCalculator`）

默认模式。计算公式如前所述：

```python
num_microbatches = global_batch_size // (micro_batch_size * data_parallel_size)
```

初始化时一次性确定，之后调用 `get_num_microbatches()` 始终返回同一个数。

### G.2 阶梯式 batch size 调度（`StepBatchsizeNumMicroBatchesCalculator`）

由 `--step-batch-size-schedule` 参数触发。格式为 `"阈值1:bs1 阈值2:bs2 ..."`，阈值支持 K/M/B/T 后缀：

```bash
# 示例：训练过程中逐步增大 batch size
# 前 250B tokens 用 bs=768，之后用 1536，再之后用 3072
--step-batch-size-schedule "0:768 250B:1536 500B:3072 750B:6144"
--seq-length 4096
```

**为什么要动态增大 batch size？**

训练初期，小 batch size 有更强的梯度噪声，帮助逃离局部极小值；训练后期，大 batch size 提高硬件利用率（更高的 MFU）。这是 GPT-3 论文和后续工作验证的有效策略。

每步调用 `update_num_microbatches(consumed_samples)` 时，计算器会检查当前 consumed_samples 是否达到下一个阈值，如果是，则切换到更大的 `global_batch_size`，`num_microbatches` 随之增加。

```python
# 伪代码展示阶梯调度逻辑
def update(self, consumed_samples, consistency_check, verbose=False):
    self.current_global_batch_size = self._get_batch_size_for_samples(consumed_samples)
    self.num_micro_batches = (
        self.current_global_batch_size // self.micro_batch_times_data_parallel_size
    )
```

**注意**：切换阈值时，`current_global_batch_size` 必须能被 `micro_batch_size × DP` 整除，否则 `consistency_check=True` 时会报错。这是生产使用时需要仔细规划各阶段 batch size 的原因。

---

## 附录 H：`megatron/core` 代码风格约定速查

阅读 Megatron Core 代码时，了解以下约定可以减少困惑：

| 约定 | 说明 |
|------|------|
| `pg_collection` 参数 | 新代码通过 `ProcessGroupCollection` 显式传递进程组，而不是从 `parallel_state` 读取全局变量 |
| `tp_group` 参数 | 某些旧接口仍接受 `tp_group: ProcessGroup` 而非 `pg_collection`，两种风格并存 |
| `pre_process` / `post_process` | 控制 PP stage 是否持有 embedding / output_layer，始终显式传递 |
| `vp_stage` 参数 | VPP 场景下标识当前是第几个虚拟 stage（0-indexed） |
| `@dataclass` 配置类 | `TransformerConfig`、`DistributedDataParallelConfig` 等都是 frozen dataclass，创建后不可修改 |
| `build_module(spec, ...)` | 统一的模块装配接口，避免在代码中直接写 `if use_te: ... else: ...` |
| `ShardedTensor` | checkpoint 中每个参数的分布式存储描述符，包含 global_shape / local_shape / global_offset |

---

## 附录 I：阅读清单速查卡

把这张卡打印出来，阅读时随时参考：

```
初次阅读顺序（推荐）:
1. pretrain_gpt.py           → 入口，掌握 __main__ 结构
2. megatron/training/training.py  pretrain()  → 训练总指挥
3. megatron/training/arguments.py validate_args()  → 参数验证，理解所有约束
4. megatron/core/models/gpt/gpt_model.py  → GPTModel 结构
5. megatron/core/transformer/transformer_layer.py  → 单层计算
6. megatron/core/parallel_state.py  initialize_model_parallel()  → 进程组
7. megatron/core/pipeline_parallel/schedules.py  → 调度算法（最后读）

遇到问题时查阅:
- batch size / DP 不整除 → arguments.py validate_args
- 形状报错 → transformer_layer.py + attention.py
- 梯度同步 → megatron/core/distributed/
- checkpoint 格式 → megatron/core/dist_checkpointing/
```

*下一篇：[精读 Megatron 源码（2）：完整拆解一次训练——从 `__main__` 到 `optimizer.step`](./02-pretrain-loop.md)*
