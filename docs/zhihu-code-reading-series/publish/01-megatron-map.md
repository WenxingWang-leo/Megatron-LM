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

### Phase 3：进阶专题（第 8-10 篇）

8. **第 8 篇（优化器与 Checkpoint）**：DistributedOptimizer 内部、ShardedTensor 格式
9. **第 9 篇（MoE）**：Expert 路由、All-to-All 通信、EP 分组
10. **第 10 篇（前沿扩展）**：CP、FP8、speculative decoding、multi-token prediction

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

## 11. 本文小结

现在你手里有了：

1. **目录结构地图**：知道每个文件大概做什么
2. **完整术语表**：再也不会被 TP/PP/DP/CP/EP/SP/VPP/DistOpt/MFU 搞混
3. **一句话故事**：能向别人解释"Megatron 一步训练发生了什么"
4. **推荐阅读路线**：Phase 1 → 2 → 3，有先后依赖
5. **环境搭建方法**：能运行最小示例
6. **避坑清单**：7 个常见错误，提前规避

下一篇，我们将用"探针法"从 `pretrain_gpt.py` 的第一行开始，逐行追踪到 `optimizer.step`，让整个训练循环在脑子里留下清晰的印记。

---

## 课后练习

**练习 1**：打开 `/workspace/pretrain_gpt.py`，找到 `BATCH_KEYS` 列表（第 83-94 行），回答：这个列表的顺序有什么特殊之处？（提示：看注释）

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

*下一篇：[精读 Megatron 源码（2）：完整拆解一次训练——从 `__main__` 到 `optimizer.step`](./02-pretrain-loop.md)*
