# 精读 Megatron 源码（10）：从「读通主路径」到「能改框架」——CP、推理、RL、精度与贡献指南

> **专栏**：Megatron 源码精读 · 第 10 篇（终篇）  
> **上一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)

---

你已经读完了前九篇。这不是一篇"总结大纲"，而是把那些**前九篇只点到为止、但值得深挖**的主题彻底展开：Context Parallel 的实现原理、推理栈的组织方式、RL 训练与预训练的差异、精度优化的代码位置、CUDA Graph 约束、如何写单元测试，以及如何真正参与 Megatron-LM 的贡献。最后用一份完整的 8 周自学日历和 FAQ 收尾。

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

### 2.3 get_batch_on_this_cp_rank 的调用链与 seq=8192，CP=4 的精确推演

```python
# megatron/core/utils.py  get_batch_on_this_cp_rank（精简）
def get_batch_on_this_cp_rank(batch, is_hybrid_cp, cp_group=None, ...):
    if use_per_sequence_balancing or batch.get("cu_seqlens") is None:
        # 标准 zigzag 模式
        batch = _get_batch_on_this_cp_rank_per_sequence_balancing(batch, cp_group)
    elif is_hybrid_cp:
        # Hybrid CP：先按 local_cp_size 切，再按文档 zigzag
        hybrid_cp_group = hybrid_cp_group_func(group_size=batch['local_cp_size'].item())
        batch = _get_batch_on_this_cp_rank_per_sequence_balancing(batch, hybrid_cp_group)
    else:
        # packed sequence（cu_seqlens 存在）：按文档边界做 zigzag
        batch = _get_batch_on_this_cp_rank_per_document_balancing(batch, cp_group)
    return batch
```

**seq=8192, CP=4 的精确 zigzag 分配**：

每个 CP rank 分配 `seq/CP = 8192/4 = 2048` 个 token，但不是连续的 2048 个，而是交错的：

```
rank 的 token 来源公式（每个 chunk_size = seq / (2*CP) = 8192/8 = 1024）：
  rank r 取：
    chunk [r * chunk_size : (r+1) * chunk_size]         # 前半部分
    chunk [(2*CP-1-r) * chunk_size : (2*CP-r) * chunk_size]  # 后半部分

CP=4, chunk_size=1024：
  rank 0 取：[0:1024]    + [7168:8192]  = 位置 0-1023, 7168-8191
  rank 1 取：[1024:2048] + [6144:7168]  = 位置 1024-2047, 6144-7167
  rank 2 取：[2048:3072] + [5120:6144]  = 位置 2048-3071, 5120-6143
  rank 3 取：[3072:4096] + [4096:5120]  = 位置 3072-4095, 4096-5119
```

**负载均衡的数学直觉**（causal 注意力）：

```
token i 在 causal 注意力中需要处理 (i+1) 个 KV（与前面所有 token 的注意力）

rank 0 负担（位置 0-1023 + 7168-8191）：
  前半  sum(1..1024)     = 1024 * 1025 / 2 = 524800
  后半  sum(7169..8192)  ≈ 7680 * 1024     ≈ 7864320
  总计  ≈ 8389120

rank 3 负担（位置 3072-4095 + 4096-5119）：
  前半  sum(3073..4096)  ≈ 3584 * 1024 ≈ 3670016
  后半  sum(4097..5120)  ≈ 4608 * 1024 ≈ 4718592
  总计  ≈ 8388608

各 rank 的计算量几乎完全相同（相差 < 0.01%）
```

### 2.4 CP 注意力的 Ring Attention 逐步图

切分序列后，每个 CP rank 只有 Q/K/V 的一部分。自注意力要求每个 Q 能看到所有 K/V（causal 下看到之前的所有 K/V），这就需要跨 CP rank 的 K/V 通信。

#### 方案 1：AllGather（直觉但贵）

```text
每个 rank AllGather 全序列的 K/V → 本地完整注意力
通信量 = S × H × 2(K+V) × (CP-1)       # 每 rank 接收量随 CP 线性增长
```

#### 方案 2：Ring Attention（Megatron 默认 CP 方案）

4 张卡排成环，K/V 沿环逆时针转，每轮只传 `S/CP` 个 token 的 K/V：

```text
设 CP=4, S=8192, 每 rank 持有 S/4=2048 tokens 的 Q/K/V
  Q 固定不动，只有 K/V 在环上轮转

                ┌── rank0 ──┐
                │  Q0 K0 V0 │
 rank3 ─────────┤           ├───────── rank1
  Q3 K3 V3     │  ← ring   │         Q1 K1 V1
                └─── rank2 ─┘
                   Q2 K2 V2

round 0: 每个 rank 用本地 K/V 算一部分 attn
           rank0: Attn(Q0, K0, V0)
           rank1: Attn(Q1, K1, V1)  …

         同时发 K0/V0→rank1, 收 K3/V3←rank3（send/recv 并行）

round 1: rank0 拿到 K3/V3 → Attn(Q0, K3, V3)
         rank1 拿到 K0/V0 → Attn(Q1, K0, V0)  …
         再发 K3/V3→rank1, 收 K2/V2←rank3

round 2: rank0 拿到 K2/V2 → Attn(Q0, K2, V2)  …
         再发/收最后一轮

round 3 结束后: 每个 Q 都看到了所有 K/V
```

**通信量（seq=8192，CP=4，每轮传 2048 tokens 的 K+V）**：

```text
Ring: CP-1 = 3 轮 × 2048×H×2 / 轮 = 3×2048×H×2 = 12288 × H
AllGather:                                          = (S×2×(CP-1))×H = 8192×2×3×H = 49152 × H
比值: Ring / AllGather = 12288 / 49152 = 25%
```

Ring 的带宽节省来自「每轮只传 S/CP 而不是 S」。代价：需要 CP-1 轮顺序依赖（但每轮内 compute 可以 overlap send/recv）。

**causal mask 下的优化**：rank r 的 Q 只需要看到「位置 ≤ 本 rank 最后一个 token」的 K/V。zigzag 分配后，靠后的 K/V 块可以跳过不算（`skip_mask`），进一步减少无效 GEMM。

#### Hybrid CP：Ring + Ulysses 混合

Megatron 的 `is_hybrid_cp=True` 模式把 CP 组拆成两级：

- **内层（local CP group）**：用 Ulysses（AllToAll 重排 QKV，不是 Ring）；  
- **外层（global CP group）**：用 Ring Attention。  

适合 CP 很大时（如 CP=16）减少 Ring 的轮数：令外层 4 路 Ring × 内层 4 路 Ulysses，Ring 只需 3 轮（而非 15 轮）。

### 2.5 DP-CP 联合梯度同步

CP 组内不同 rank 处理同一序列的不同段，**权重是完全相同的副本**（和 DP 同理），因此权重梯度需要在 CP 维上 AllReduce。

Megatron 维护 `dp_cp_group`（大小 = DP × CP），DDP 的梯度 AllReduce 在这个联合组上执行：

```python
# distributed_data_parallel.py
self.ddp_config.data_parallel_group  # 实际是 dp_cp_group
```

为什么不分开做「先 CP AllReduce 再 DP AllReduce」？合并成一个大组可以让 NCCL 用更大消息做一次 AllReduce / ReduceScatter，而不是两次小消息——带宽利用率更高，且 SHARP 只能绑一个组（§8 parallel_state 建组顺序解释了 dp-cp 第一个创建的原因）。

---

## 三、推理栈：megatron/core/inference/ 完整地图

### 3.1 推理栈层次

```
megatron/core/inference/
├── apis/
│   ├── llm.py               # LLM 类：阻塞式批量生成
│   └── async_llm.py         # AsyncLLM：异步流式生成
├── engines/
│   ├── abstract_engine.py   # 引擎接口定义
│   ├── static_engine.py     # StaticInferenceEngine（已废弃，转用 DynamicEngine）
│   ├── dynamic_engine.py    # DynamicInferenceEngine：主力推理引擎
│   └── mcore_engine.py      # MCoreEngine 别名（= StaticInferenceEngine）
├── contexts/
│   ├── static_context.py    # 固定 batch size KV cache 上下文
│   ├── dynamic_context.py   # 动态 KV block 分配（类 PagedAttention）
│   └── kv_block_allocator.py # KV 缓存块分配器
├── text_generation_controllers/
│   └── text_generation_controller.py  # 控制 prefill/decode 循环
├── config.py                # InferenceConfig（beam_size, top_k, max_new_tokens 等）
└── sampling_params.py       # 单请求的采样参数
```

### 3.2 一次完整的生成路径：从 LLM.generate 到模型 forward

```
LLM.generate(["Hello, how are"])
  ↓ StaticInferenceEngine.generate(requests)
    ↓ DynamicInferenceEngine.generate(requests)
      ↓ scheduler.add_request(request)         # 把请求放入调度队列
      ↓ 内部事件循环（asyncio）
        ↓ scheduler.get_next_batch()            # 决定这次 step 处理哪些请求
        ↓ text_generation_controller.run_one_forward_step(batch)
          ↓ 准备 input_ids, position_ids, attention_mask
          ↓ model.forward(...)                   # 通过 PP schedule 调度各 stage
            ↓ GPTModel.forward(input_ids, ...)
              ↓ Embedding → TransformerBlock.forward
                ↓ TransformerLayer.forward（每层）
                  ↓ SelfAttention.forward（使用 KV cache 中已有的 K/V）
                  ↓ MoELayer.forward（若有 MoE 层）
              ↓ output_layer → logits
          ↓ 对 logits 采样（temperature, top_k, top_p）→ next_token
          ↓ 若非 EOS 且未达到 max_new_tokens，继续下一轮 decode
      ↓ 返回生成的 token ids
  ↓ tokenizer.decode → text
```

**关键文件路径**：

| 步骤 | 文件 | 函数 |
|------|------|------|
| 入口 | `megatron/core/inference/apis/llm.py` | `LLM.generate` |
| 引擎调度 | `megatron/core/inference/engines/dynamic_engine.py` | `DynamicInferenceEngine._generate_loop` |
| 控制器 | `megatron/core/inference/text_generation_controllers/text_generation_controller.py` | `run_one_forward_step` |
| 采样 | `megatron/core/inference/sampling_params.py` | `SamplingParams` |
| KV 分配 | `megatron/core/inference/contexts/kv_block_allocator.py` | `KVBlockAllocator.allocate` |

### 3.3 推理时 KV cache 的工作原理

```
Prefill 阶段（处理 prompt）：
  input_ids = [Hello, ,, how, are]  # 4 tokens
  模型一次性处理所有 token
  每层 Attention 生成 K/V，存入 KV cache context

Decode 阶段（逐 token 生成）：
  Step 1：
    input_ids = [?]  # 只有 1 个 token（上一步预测的）
    Attention 使用 cache 中的 K/V（不需要重算 prompt 的 K/V）
    生成下一个 token
  Step 2, 3, ...：类似，cache 逐步增长

内存使用：
  KV cache 大小 = num_layers × 2 × seq_len × num_heads × head_dim
  静态上下文（StaticInferenceContext）：一次性分配全部 KV cache
  动态上下文（DynamicInferenceContext）：按需分配 block，类似 PagedAttention
```

---

## 四、RL 训练：train_rl.py 与 pretrain_gpt.py 的真正差异

### 4.1 核心差异对比

```
预训练（pretrain_gpt.py forward_step）：
  ↓ get_batch(data_iterator)       # 从 DataLoader 取 tokens/labels/loss_mask
  ↓ model(tokens, position_ids, ...) → logits
  ↓ cross_entropy(logits, labels) * loss_mask
  ↓ 返回 (loss, loss_mask, {'lm loss': loss})

RL 训练（train_rl.py forward_step）：
  ↓ next(data_iterator)            # 取 RL batch（tokens + advantages + old_logprobs + ref_logprobs）
  ↓ get_logprobs(model, tokens, ...) → current_logprobs
  ↓ calculate_grpo_loss(
       current_logprobs, old_logprobs, ref_logprobs,
       advantages, clamp_eps, kl_beta, ...
     ) → loss + diagnostic tensors
  ↓ 返回 (loss, total_tokens, {'lm loss': ..., 'rl/kl_term': ..., 'rl/pi_over_pi_old': ...})
```

### 4.2 train_rl.py forward_step 真实代码路径

```python
# train_rl.py  forward_step（第 194 行起，精简并标注）
def forward_step(data_iterator, model: GPTModel, loss_only: bool = False):
    runtime_state = get_rl_runtime_state()  # RL 特有的 runtime 状态（rollout 数据、采样历史）
    args = get_args()

    # ① 从数据迭代器取 RL batch（而非调用 get_batch）
    batch_data = next(data_iterator)

    if args.rl_use_sequence_packing:
        # ② 解包 packed sequence 格式的 RL batch
        (tokens, advantages, old_logprobs, loss_mask, position_ids,
         ref_logprobs, inference_logprobs, seq_starts, seq_lengths,
         seq_indices, packed_seq_params) = load_packed_data_by_index(
            bin_tensor.item(), runtime_state.packing_context, ...
        )
    else:
        # ③ 解包普通格式 RL batch
        (tokens, advantages, old_logprobs, loss_mask, position_ids,
         ref_logprobs, inference_logprobs) = batch_data

    # ④ 当前策略的 log_probs（需要梯度）
    logprobs_or_hidden_states = get_logprobs(
        model_to_use, tokens, position_ids, no_grad=False,
        packed_seq_params=packed_seq_params
    )

    # ⑤ PP 最后一个 stage 才有 logits，计算 GRPO loss
    if is_pipeline_last_stage():
        current_logprobs = logprobs_or_hidden_states
        loss, kl_term, ratios, entropy_term, ... = calculate_grpo_loss(
            current_logprobs=current_logprobs,
            old_logprobs=old_logprobs,
            ref_logprobs=ref_logprobs,
            advantages=advantages,
            clamp_eps_lower=args.grpo_clamp_eps_lower,
            clamp_eps_upper=args.grpo_clamp_eps_upper,
            kl_beta=args.grpo_kl_beta,
            ...
        )
        output_tensor = loss
    else:
        output_tensor = logprobs_or_hidden_states  # 中间 stage 传激活

    return output_tensor, partial(loss_func, loss_mask, kl_term, ratios, ...)
```

### 4.3 pretrain_gpt.py forward_step 对比

```python
# pretrain_gpt.py  forward_step（第 282 行起，精简并标注）
def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    # ① 标准 get_batch（处理 TP broadcast + CP split）
    (attention_mask, cu_seqlens, ..., tokens) = get_batch(data_iterator, vp_stage)

    # ② PackedSeqParams（若有 cu_seqlens）
    if cu_seqlens is not None:
        packed_seq_params = PackedSeqParams(qkv_format="thd", ...)

    # ③ 直接调用模型（labels 不传入模型，而是传给 loss_func）
    output_tensor = model(tokens, position_ids, attention_mask,
                         packed_seq_params=packed_seq_params)

    # ④ 返回 output_tensor 和 loss_func（cross entropy）
    return output_tensor, partial(loss_func, labels, loss_mask)
```

**关键差异总结**：

| 方面 | pretrain_gpt | train_rl |
|------|-------------|---------|
| 数据来源 | `get_batch()` → DataLoader → IndexedDataset | `next(data_iterator)` → RL rollout buffer |
| loss 函数 | Cross-Entropy（next-token 预测） | GRPO loss（PPO 变体，含 KL 惩罚） |
| 输入组成 | tokens, labels, loss_mask | tokens, advantages, old/ref logprobs |
| 返回的 metric | `{'lm loss': loss}` | `{'lm loss', 'rl/kl_term', 'rl/pi_over_pi_old', 'rl/entropy_term', ...}` |
| 内存压力 | 1 份模型 | 2 份（policy + reference 需要同时在 GPU 上） |
| 并行特殊处理 | CP split 在 get_batch 完成 | sequence packing 用 `PackedSeqParams(qkv_format='thd')` |

---

## 五、精度优化：FP8/FP4 代码入口与配置字段

### 5.1 fp8_utils.py 的角色与主要入口

`megatron/core/fp8_utils.py` 是 FP8 功能的统一出口，封装了对 TransformerEngine 的所有 FP8 操作：

| 函数 / 类 | 作用 |
|-----------|------|
| `get_fp8_recipe(config)` | 根据 config 构建 TE 的 FP8 recipe 对象 |
| `get_fp8_context(config, layer_no)` | 返回 `fp8_autocast` 上下文管理器 |
| `FP8_TENSOR_CLASS` | FP8 tensor 的基类（TE >= 2.0 是 `QuantizedTensor`） |
| `_wrap_te_linear_for_padding` | 推理时自动 padding 序列长度以满足 FP8 对齐要求 |

### 5.2 FP8 recipe 配置字段

```bash
--fp8-format hybrid          # 前向 E4M3，反向 E5M2（最常用）
--fp8-recipe delayed         # 延迟缩放（TE < 2.1 唯一选项）
--fp8-recipe tensorwise      # 张量级缩放（TE >= 2.2，精度最高）
--fp8-recipe blockwise       # 块级缩放（TE >= 2.3）
--fp8-recipe mxfp8           # MXFP8 块缩放（需 Blackwell）
--fp8-interval 1             # 每步更新 amax 历史
--fp8-margin 0               # amax 缩放的安全裕度
--fp8-wgrad                  # weight gradient 也用 FP8（节省内存，可能影响精度）
```

**FP8 recipe 决策树**：

```
你的 TE 版本 < 2.1？
  → 只能用 --fp8-recipe delayed

TE >= 2.2 且有 H100/H200？
  → 推荐 --fp8-recipe tensorwise（精度最接近 BF16）

TE >= 2.3 且模型较大（> 13B）？
  → 尝试 --fp8-recipe blockwise（内存效率高）

有 Blackwell (SM100+)？
  → 使用 --fp8-recipe mxfp8（硬件原生支持，速度最快）
```

### 5.3 `get_fp8_context` 的调用位置

在模型 forward 中，每层 Transformer 被包裹在 `get_fp8_context` 上下文管理器中：

```python
# megatron/core/extensions/transformer_engine.py（精简）
with fp8_utils.get_fp8_context(config, layer_no=global_layer_no):
    # 这个 context 内的所有 TE Linear/Attention 操作都用 FP8
    output = te_layer(input, ...)
```

`layer_no` 参数允许对某些层（如 embedding 层、output head）禁用 FP8，保持 BF16 精度：

```bash
--fp8-first-num-layers-no-fp8 1   # 第一层不用 FP8
--fp8-last-num-layers-no-fp8 1    # 最后一层不用 FP8
```

### 5.4 FP4 训练（NVIDIA Blackwell 专属）

```bash
--fp4   # 启用 FP4（需要 SM100+ 硬件和 TE 支持）
```

FP4 相关文件：
- `megatron/core/fp4_utils.py`：FP4 tensor 量化工具
- `megatron/core/optimizer/distrib_optimizer.py` 中 `quantize_nvfp4_param_shard`：FP4 master 参数量化存储

FP4 相比 FP8 的进一步优势：显存减半、带宽减半，但精度损失更大，通常需要更仔细的量化 calibration。

---

## 六、CUDA Graph：约束与适用场景

### 6.1 CUDA Graph 的基本原理

CUDA Graph 把 GPU kernel 调用序列录制成一个图，之后每次 step 直接重放（replay），消除 CPU-GPU 同步开销：

```
传统执行模式：
  CPU: launch kernel1 → launch kernel2 → launch kernel3 → ...
  GPU:  [kernel1] [kernel2] [kernel3] ...
  开销：每次 launch 约 5-20 μs CPU 时间

CUDA Graph 模式：
  录制阶段：trace 所有 kernel launch，建立图
  重放阶段：一次 cudaGraphLaunch 触发整个序列
  开销：约 1-5 μs，节省大量 CPU overhead
```

### 6.2 Megatron 的 CUDA Graph 配置

```bash
--cuda-graph-modules "attention,mlp"   # 子模块级别的 CUDA graph
--cuda-graph-modules "full"            # 整个迭代（含 optimizer）的 CUDA graph
```

代码位置：
- `megatron/core/transformer/cuda_graph_config.py`：`normalize_cuda_graph_modules`
- `megatron/core/transformer/cuda_graphs.py`：`CudaGraphManager`

`cuda_graph_impl` 的四种模式：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `"none"` | 不使用 CUDA graph | 调试、动态 shape |
| `"local"` | 子模块级别 graph（Attention/MLP 等） | 大多数训练场景 |
| `"transformer_engine"` | TE 内置 graph | TE 特化 kernel |
| `"full_iteration"` | 整个训练 step 的 graph | 极致性能优化 |

### 6.3 CUDA Graph 的硬约束

CUDA Graph **要求所有 kernel 的输入 shape 在录制时确定，重放时不变**。以下情况无法使用（或需要特殊处理）：

| 约束 | 原因 | 解决方案 |
|------|------|---------|
| 动态 sequence length | 不同 batch 的序列长度不同 | 使用 packed sequence + static shape，或 padding 到固定长度 |
| MoE AllToAll 动态路由 | 每个 token 路由不同，通信大小变化 | `--moe-expert-capacity-factor` + 固定 capacity；或 `ncclep` + `moe_ncclep_static_shape=True` |
| Python 控制流（if/else） | 录制时选定一个分支，重放时不能切换 | 消除运行时 Python 条件判断 |
| NCCL 通信 | 大多数通信 op 不能被 graph 捕获 | 只 graph 纯计算部分（Attention/MLP），通信在 graph 外 |
| 推理 `DynamicInferenceContext` | KV block 分配动态变化 | 使用 `InferenceCudaGraphScope.layer` 或 `.block` 级别 |

**实践中的最优配置**（A100/H100 训练）：

```bash
--cuda-graph-modules attention,mlp   # Attention 和 MLP 用 graph
--moe-token-dispatcher-type alltoall  # MoE 用 AllToAll（在 graph 外）
--moe-expert-capacity-factor 1.0      # 固定 shape
```

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
        group = parallel_state.get_tensor_model_parallel_group()
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
        group = self.tp_group
        ...
```

**允许的例外**：`parallel_state.py` 本身、测试代码、`process_groups_config.py` 的初始化代码、以及带有显式注释说明的迁移兼容路径。

### 7.2 写一个最小单元测试：手把手示例

单元测试位于 `tests/unit_tests/`，可以不需要 GPU（使用 `use_cpu_initialization=True` 和 mock 进程组）。

以下是一个完整的最小单元测试示例，测试一个新的 MLP 变体：

```python
# tests/unit_tests/transformer/test_my_mlp.py
import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from my_module import MyMLP  # 你要测试的模块


class TestMyMLP:

    def setup_method(self, method):
        # ① 初始化 1×1 进程组（单 GPU 测试，无需多卡）
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(42)

        # ② 构建最小 TransformerConfig
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=256,
            use_cpu_initialization=True,  # 不需要 GPU
        )
        self.mlp = MyMLP(self.config)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        """验证模块可以正常构建，参数数量符合预期"""
        assert isinstance(self.mlp, MyMLP)
        num_params = sum(p.numel() for p in self.mlp.parameters())
        # 预期参数量 = 2 * hidden_size * ffn_hidden_size = 2 * 64 * 256 = 32768
        assert num_params == 32768

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_shape(self):
        """验证 forward 输出 shape 正确"""
        self.mlp.cuda()
        # [seq_len, batch, hidden]
        x = torch.randn(16, 2, 64, device='cuda')
        output, bias = self.mlp(x)
        assert output.shape == (16, 2, 64)
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward(self):
        """验证梯度可以正常反传"""
        self.mlp.cuda()
        x = torch.randn(16, 2, 64, device='cuda', requires_grad=True)
        output, _ = self.mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

**运行单元测试**：

```bash
# 在 workspace 根目录
uv run pytest tests/unit_tests/transformer/test_my_mlp.py -v

# 只跑 CPU 测试（无 GPU）
uv run pytest tests/unit_tests/transformer/test_my_mlp.py -v -k "not cuda"
```

### 7.3 PR 提交规范

根据 `CLAUDE.md` 和 `docs/developer/contribute.md`：

```bash
# 1. Fork 仓库，在 fork 上工作
git remote add origin https://github.com/<your-username>/Megatron-LM
git checkout -b feature/my-feature

# 2. 修改代码后运行 isort 修复 import 顺序
uv run isort megatron/core/transformer/my_module.py

# 3. 运行 linting（参考 skills/mcore-linting-and-formatting/SKILL.md）
bash tools/autoformat.sh

# 4. 提交时必须同时加 -s（Signed-off-by）和 -S（GPG 签名）
git commit -s -S -m "feat: add ProcessGroupCollection to MyModule"

# 5. 创建草稿 PR（必须是 draft，不要直接创建 ready for review）
gh pr create --draft --title "feat: ..." --body "..."
```

---

## 八、8 周自学日历

这是一个结合系列文章和实验任务的 8 周课表，适合已有 PyTorch 基础、想深入理解大模型框架的工程师。

| 周 | 主题 | 阅读 | 实验 |
|----|------|------|------|
| 第 1 周 | 全局地图与启动链 | 第 1、2 篇 | 启动 `pretrain_gpt.py --mock-data`，用 1 GPU 跑通 100 步，观察日志 |
| 第 2 周 | GPTModel 与 ModuleSpec | 第 3 篇 | 修改 `get_gpt_layer_local_submodules`，替换一个 MLP 为自定义模块（继承 MLP），验证 loss 不变 |
| 第 3 周 | 并行状态与 TP | 第 4、5 篇 | 2 GPU TP=2 跑通，在 rank 0/1 打印 Q/K/V weight 的 shape，验证列切 |
| 第 4 周 | PP 与流水线调度 | 第 6 篇 | 4 GPU PP=4 跑通，统计每个 stage 的 forward/backward 时间，观察 bubble |
| 第 5 周 | 数据并行与 DistOpt | 第 7、8 篇前半 | 4 GPU DP=4，分别用 `use_distributed_optimizer=False/True`，对比显存使用 |
| 第 6 周 | 数据管线与 Checkpoint | 第 8 篇后半 | 用真实数据集训练 100 步保存 checkpoint，换 TP 度加载验证 loss 连续 |
| 第 7 周 | MoE | 第 9 篇 | 8 GPU TP=2+EP=4 的 MoE 模型，跑 200 步，画出 `tokens_per_expert` 曲线 |
| 第 8 周 | CP + 推理 + RL + 贡献 | 第 10 篇 | 选择一个实践项目（见下节），从代码修改到 PR draft 全流程 |

**每周实验时间估计**：约 4-6 小时（含环境准备、调试、记录）

---

## 九、五个深入实践项目

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

## 九点五、Activation Recompute 与 Fused Kernels

### 9.5.1 Activation Recompute（梯度检查点）

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
  - `"shared_experts"`：重算共享专家的激活
  - `"mlp"`：重算整个 MLP 子模块

**各种配置的显存/算力权衡**：

| 配置 | 显存节省 | 额外算力开销 | 推荐场景 |
|------|---------|------------|---------|
| `recompute_granularity=None` | 0% | 0% | GPU 显存充足 |
| `selective core_attn` | ~40% | ~5% | 大多数 A100/H100 训练 |
| `selective mlp` | ~30% | ~8% | 模型很深时 |
| `selective moe_act` | ~20% | ~3% | MoE 专属 |
| `full` | ~65% | ~33% | 显存极度紧张 |

代码位置：
- `megatron/core/transformer/transformer_config.py`：所有 recompute 相关字段
- `megatron/core/transformer/transformer_layer.py`：`te_checkpoint` 调用点

### 9.5.2 Fused Kernels

```bash
--apply-query-key-layer-scaling   # QK scaling 融合进 attention kernel
--bias-dropout-fusion             # bias + dropout 融合
--masked-softmax-fusion           # masked softmax 融合
```

这些 fusions 通过减少 kernel 启动次数和中间 tensor 的内存读写来提速。TransformerEngine 的 `TELinear` 会自动判断是否使用 cuBLAS 或 cuDNN 的 fused op。

**FlashAttention 的隐式 fusion**：Megatron 默认使用 FlashAttention（通过 TransformerEngine），它把 Q×K×softmax×V 的整个注意力计算融合为一个 CUDA kernel，内存访问量从 O(N²) 降到 O(N)（N=序列长度），是最重要的单点优化。

---

## 十、个人知识系统：如何维护 Megatron 理解

读完系列只是起点。大型框架的代码会持续演进，维护理解比第一次学更难。推荐以下方法：

### 10.1 配置三元组笔记

每次遇到一个新的训练配置，在笔记里记录：

```
配置名: TP=4, PP=2, EP=8, CP=1 MoE 训练
---
进程组：
  tp_group: 4个rank
  pp_group: 2个stage
  ep_group: 8个rank（跨pp）
  dp_group: 总进程数 / (TP*PP*EP)

关键 shape（hidden=4096, seq=8192）：
  tokens per rank: [B, 8192]（CP=1 时不切）
  TP0 attention weight: [4096/4, 4096] = [1024, 4096]
  每 EP rank local experts: total/EP

通信模式：
  TP: AllReduce / AllGather
  EP: AllToAll
  DP: ReduceScatter + AllGather (DDP bucket)
```

### 10.2 维护调用图

```
pretrain()
  → train_step()
    → forward_backward_func()
      → forward_step()
        → model.forward()
          → TransformerLayer.forward()
            → SelfAttention.forward()
            → MoELayer.forward()
      → backward_step()
    → finalize_model_grads()
    → optimizer.step()
```

---

## 十一、掌握度评分标准（进阶版）

### 11.1 分级评估标准

**Level 1（60分）：能运行框架**
- [ ] 能在单 GPU 上跑通 `pretrain_gpt.py --mock-data`
- [ ] 理解 `--tensor-model-parallel-size/--pipeline-model-parallel-size` 的含义
- [ ] 能修改超参数（层数、hidden_size）并验证 loss 下降
- [ ] 会查看 training log，理解 loss/tokens/throughput 指标

**Level 2（75分）：能读懂主路径**
- [ ] 能从 `pretrain()` 追溯到 `model.forward()` 的完整调用链
- [ ] 理解 TP 切分的 Column/Row Parallel Linear 各自的通信方式
- [ ] 能解释 PP 的 1F1B schedule 与 bubble rate 公式
- [ ] 能独立排查 shuffle index 缓存问题、checkpoint 加载失败

**Level 3（85分）：能读懂复杂特性**
- [ ] 能解释 DistributedOptimizer 的三阶段内存收益
- [ ] 能分析 MoE AllToAll 的通信量并选择合适的 dispatcher
- [ ] 能理解 ShardedTensor resharding 的工作原理
- [ ] 能阅读并理解一个中等复杂度的 PR（如新增一个 attention variant）

**Level 4（95分）：能改框架**
- [ ] 能实现一个新的 MoE 路由策略并添加单元测试
- [ ] 能按照 ProcessGroupCollection 规范重构一个模块
- [ ] 能为 CP 或 RL 功能添加新的配置选项并维护向后兼容
- [ ] 能追踪并修复一个涉及多维并行交互的 bug

**Level 5（100分）：能架构框架**
- [ ] 能设计新的并行策略（如新的 EP + PP 交互方式）
- [ ] 能评审涉及核心通信逻辑的 PR
- [ ] 能优化关键路径的性能（如识别并消除通信瓶颈）
- [ ] 能从零开始为新硬件适配 Megatron 的 dispatch 后端

---

## 十二、完整系列索引与一行精华

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
| 10 | 进阶与贡献 | CP = zigzag seq 切分 + Ring Attention；fp8_utils = FP8 统一出口；贡献必须 draft PR |

---

## 十三、十五道"what next" FAQ

**Q1：学完这个系列，下一步应该读什么代码？**  
A：推荐按以下顺序：(1) `megatron/core/models/mamba/`——了解 SSM 如何复用同一框架；(2) `megatron/core/models/multimodal/llava_model.py`——了解多模态如何与 CP 协作；(3) 一个真实的大模型训练 PR（在 GitHub 上筛选 `feature` 标签）。

**Q2：Megatron-LM 和 vLLM 的推理栈有什么本质区别？**  
A：Megatron 的推理栈是训练框架内的扩展，复用了相同的模型、并行组和 checkpoint 格式，适合"训完立刻部署"的场景。vLLM 是专门为推理吞吐优化的独立框架，PagedAttention 和 continuous batching 实现更成熟，但需要额外的模型转换步骤。大规模部署通常将 Megatron 训练的模型转换格式后用 vLLM 或 TensorRT-LLM 部署。

**Q3：CP 和 Ulysses/Ring-Attention 的关系？**  
A：Megatron 的 CP 内部使用了 Ring Attention 的思想（KV 在 CP ranks 之间环形传递），与 Ulysses（AllToAll 重新排列 QKV）并列为两种序列并行方案。Megatron 的 Hybrid CP 模式（`is_hybrid_cp=True`）允许两者结合：内部 CP group 用 Ulysses，外部 CP group 用 Ring Attention。

**Q4：FP8 训练会影响模型质量吗？**  
A：对大多数任务影响可忽略（< 0.1% perplexity 差异），但有些任务（如数学、代码）对精度更敏感。推荐策略：先用 `--fp8-recipe tensorwise` 训练，定期与 BF16 基线对比 loss 曲线；若发现发散，降低 `--fp8-interval` 或切换到 `delayed` recipe。

**Q5：DP 越大越好吗？增大 DP 有什么代价？**  
A：不是。DP 增大意味着每步处理的 global batch 增大（`global_batch_size = micro_batch × GAS × DP`），需要相应增大 learning rate（通常遵循线性缩放规则），否则训练不稳定。此外，DP AllReduce 通信量随 DP 增大，大 DP 时 ReduceScatter/AllGather 成为瓶颈。实践中 DP=128~1024 时需要精细的学习率调度。

**Q6：TP+PP+DP+EP+CP 五维并行同时使用时，进程组如何组织？**  
A：同一套 `world_size` 上有**两套编组**，不是五维简单连乘再解释一次。  
- dense：`world = TP × PP × CP × dense_DP`  
- expert：`world = expert_TP × EP × PP × expert_DP`（expert 侧 `cp` 强制为 1）  
两套的 PP ranks 必须相同。EP 不「新变出 GPU」，而是从专家副本维里抽出卡来切不同专家。进 MoE 时序列仍按 CP 分片，不做全序列 AllGather。详见第 4、11 篇。

**Q7：如何估计某个配置下的显存使用？**  
A：粗略公式：`显存 ≈ (模型参数 × 精度字节 × 2 + 激活 × GAS × micro_batch) / TP / PP`。精确计算需考虑：(1) BF16 模型参数 + FP32 master；(2) Adam 的 m+v 在 DistOpt 下按 DP 分片；(3) KV cache（推理）或激活 cache（训练+recompute）；(4) 通信 buffer（AllToAll/AllGather）。

**Q8：Megatron 的 Hybrid Model（Attention + Mamba 混合）如何处理并行？**  
A：Attention 层照常使用 TP+CP；Mamba/SSM 层的 TP 支持有限（SSM 状态张量的切分方式不同），通常在 TP 维度上复制（duplicated 模式），不做切分。`hybrid_layer_specs.py` 中每层独立设置 TP 策略，通过 ModuleSpec 注入。

**Q9：如何在不重新训练的情况下增加模型的专家数（MoE 扩展）？**  
A：理论上可以通过 "weight cloning"（把原有专家 weight 复制到新专家），但这需要修改 checkpoint 的 ShardedTensor 结构，更新 `num_moe_experts` 和 `axis_fragmentations`。Megatron 目前没有内置的专家扩展工具，但可以写脚本手动操作 checkpoint 文件（numpy 操作 `.bin` 文件或修改 torch_dist checkpoint 的 tensor）。

**Q10：`recompute_granularity='selective'` 和 `'full'` 各适合什么情况？**  
A：`'full'`：反向时重算整个 layer，显存最省（约为 `'none'` 的 30%），但计算量增加约 33%。适合显存极度紧张时。`'selective'`：只重算选定的子模块（如 `core_attn`），通常节省 40-60% 的激活显存，计算开销仅增加 5-10%。实践中优先用 `'selective'`，只有在极端情况下才用 `'full'`。

**Q11：如何验证一个新的并行策略实现是否正确？**  
A：最可靠的方法是"等价性验证"：(1) 用相同 seed 和数据，分别在 TP=1（参考）和新配置（被测）下各跑 N 步；(2) 比较每步的 loss 值（应在 FP 误差范围内相同）；(3) 比较参数更新后的 weight 之和（数值守恒）。Megatron 的 `test_thd_correctness.py` 是很好的参考实现。

**Q12：为什么 MoE 的 router weight 通常保持在 FP32？**  
A：路由决策对数值极其敏感——BF16 下两个 token 的相对路由得分可能因精度损失而颠倒，导致不必要的路由抖动。FP32 保留了足够的精度让路由器学到稳定的专家分配。代码中 `Router.reset_parameters()` 把 weight 转为 `params_dtype`，但 `expert_bias` 和 `qb_beta` 始终保持 FP32。

**Q13：如何在 Megatron 框架内添加对新硬件的 AllToAll 后端支持？**  
A：扩展 `MoEFlexTokenDispatcher`，在 `moe_flex_dispatcher_backend` 中添加新的 backend 名称，在 `token_dispatcher.py` 中实现对应的 `dispatch_preprocess/token_dispatch/token_combine` 接口。关键是实现 `split_sizes` 的动态计算和通信原语的调用。参考 `deepep` 后端的实现作为模板。

**Q14：从零开始 profiling 一个 Megatron 训练 run 应该关注哪些指标？**  
A：优先级排序：(1) Compute throughput（MFU，模型 FLOP 利用率）——目标 > 50% A100；(2) 通信/计算比例（用 NSIGHT 观察 kernel 时间线）；(3) AllReduce/AllToAll 的 bus bandwidth 利用率；(4) 内存碎片（通过 `torch.cuda.memory_stats` 监控 `reserved_bytes` vs `allocated_bytes`）；(5) CPU overhead（通过 `torch.profiler` 的 `use_cuda=True` 检查 CPU-GPU 同步点）。

**Q15：想为 Megatron 贡献一个完整特性，大概需要多少工作量？**  
A：这取决于特性的侵入性。小型改动（新增一个配置选项、修复 bug）：2-5 天。中型特性（新的 attention variant、优化某个 dispatcher）：2-4 周（含单元测试 + 功能测试 golden values 更新）。大型特性（新的并行维度、新的推理后端）：1-3 个月（含架构讨论、多轮 PR review、CI 稳定化）。贡献前建议先开 GitHub issue 讨论设计方案，避免大量返工。

---

## 十三点五、黄金测试集与维护建议

维护一组自己写的最小测试，覆盖"最容易出错"的并行组合：

```
tests/unit_tests/transformer/test_my_tp2_mlp.py      # TP=2 的 MLP 等价性
tests/unit_tests/transformer/test_my_moe_ep4.py      # EP=4 MoE 路由
tests/unit_tests/dist_checkpointing/test_reshard.py  # TP 变化时的 reshard
tests/unit_tests/datasets/test_sft_cu_seqlens.py     # SFT packed seq 边界
```

**维护建议**：
- 每次更新 Megatron 版本时，先跑这组测试（5 分钟之内发现 regression）。
- 遇到生产 bug 后，先写一个能复现 bug 的最小测试，修复后测试自动成为 regression 保护。
- 把测试与业务逻辑解耦：测试只验证框架行为，不包含私有数据或模型权重。

---

## 十四、系列闭幕词：让你成为框架贡献者的思维转变

读完这 10 篇，你应该能感受到 Megatron-LM 在架构上的一个核心哲学：

> **"所有复杂性都是可组合的"**

TP × PP × DP × CP × EP——五个维度的并行，每一个都有清晰的边界，通过进程组抽象彼此正交。`ModuleSpec` 体系让 Attention/MLP/MoE 可以任意替换而不影响训练循环。`ShardedTensor` 让 checkpoint 与并行度解耦。

这不是偶然的，而是经过多年工程演进的结果。每一个看似"过度设计"的抽象（如 `ProcessGroupCollection`、`Range` 类、`MoESubmodules`），都是在解决实际训练时遇到的具体问题。

当你面对一个新的硬件、新的模型架构、或新的训练范式时，不要问"Megatron 支持吗"，而应该问：

1. **哪个抽象层最适合插入这个新能力？**（ModuleSpec / ProcessGroup / Strategy / Dataset）
2. **这个修改是否破坏了现有的正交性？**
3. **我能写一个可以在 CI 中自动验证的测试吗？**

从"读懂框架"到"能改框架"，需要的不是更多阅读，而是**动手实验**。从最小的改动开始，每次只改一个变量，验证等价性，逐步建立对框架行为的直觉。

欢迎你加入 Megatron-LM 的贡献者行列。

---

> **系列全部文章**：[精读 Megatron 源码（1-10）系列索引](./README.md)  
> **上一篇**：[精读 Megatron 源码（9）：MoE 完全精读](./09-moe.md)
