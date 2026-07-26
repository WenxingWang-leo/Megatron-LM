# 精读 Megatron 源码（6）：流水线并行——1F1B 不是玄学，是一张调度表

> **专栏**：Megatron 源码精读 · 第 6 篇  
> **核心文件**：`megatron/core/pipeline_parallel/schedules.py`、`p2p_communication.py`

---

流水线并行（PP）一句话：

> **按深度切模型；把 global batch 拆成多个 microbatch，在 stage 之间形成流水，减少设备空等。**

训练侧入口你已经在第 2 篇见过：`train_step` → `forward_backward_func(...)`。  
这篇只精读一件事：**这个 func 怎么被选中，1F1B 里 warmup / steady / cooldown 各自在干什么。**

---

## 调度函数如何选型

`get_forward_backward_func` 的核心分支：

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

| 条件 | 函数 |
|------|------|
| `PP == 1` | `forward_backward_no_pipelining` |
| `PP > 1` 且无 VPP | `..._without_interleaving`（经典 1F1B） |
| `PP > 1` 且有 VPP | `..._with_interleaving`（交错） |

文档字符串还规定了回调契约：`forward_step_func` 必须返回 `(output, loss_func)`——这就是第 2 篇强调的接口。

**精读策略**：先确认自己配置命中哪条，再只精读那一个大函数。三个全读容易晕。

---

## Microbatch：流水线里的「水」

没有足够多的 microbatch，PP 几乎必然严重气泡。

- `micro_batch_size`：单个 microbatch 样本数  
- global batch / micro → `num_microbatches`  

调度内部会算：

- **warmup**：先灌多少个 forward  
- **steady 1F1B**：一前一后交替  
- **cooldown**：收尾的 backward  

不同 PP rank 的 warmup 长度不同——这正是「流水线」的字面意义。

---

## 经典 1F1B（非 interleaving）直觉

`PP=4` 示意：

```text
时间 →
Stage0: F0 F1 F2 F3 B0 F4 B1 ...
Stage1:    F0 F1 F2 B0 F3 B1 ...
Stage2:       F0 F1 B0 F2 B1 ...
Stage3:          F0 B0 F1 B1 ...
```

- 前向：stage \(i\) 把激活 P2P 发给 \(i+1\)  
- 反向：\(i+1\) 把梯度 P2P 回 \(i\)  
- **只有最后一个 stage 在 forward 末尾调用 `loss_func`**

对应实现：`forward_backward_pipelining_without_interleaving`。  
建议边读边在纸上画自己的 `PP` 与 `num_microbatches`。

---

## Virtual PP：用层交错换气泡

当 `virtual_pipeline_model_parallel_size > 1`：

- 每个 PP rank 持有**多个 model chunk**（层交错）  
- 接口上 `model` / `data_iterator` 变成 **list**  
- 调度表更复杂（`get_schedule_table` 一类给出 `(microbatch_id, model_chunk_id)`）  

收益是减小气泡；代价是调试复杂度上升，且对 `num_microbatches` 更敏感。

---

## P2P：激活和梯度怎么搬家

`p2p_communication.py` 里的 `P2PCommunicator` 负责：

- 约定相邻 stage 收发的 tensor shape  
- 处理 sequence parallel 等导致的 shape 变化  
- 与 warmup/steady/cooldown 点位对齐  

在 schedule 里看到 send/recv forward/backward，把它们标在你的时序图上。

---

## 和 `train_step` 的衔接

```text
取 forward_backward_func
  → 传入 forward_step / data_iterator / model / num_microbatches / seq_length
  → 收集 loss 字典打日志
  → 之后才 optimizer.step
```

因此：**PP 的正确性与性能问题，优先查 schedules + P2P，而不是查 Adam。**

---

## 气泡排查清单

1. `num_microbatches` 是否明显大于 `PP`（过小≈全是 warmup）  
2. 是否意外启用了 VPP（model 是否变成 list）  
3. timer 里 PP 通信是否过度暴露  
4. 进阶重叠路径（如 `combined_1f1b.py`）是否符合预期  

可对照官方文档：`docs/user-guide/features/pipeline_parallel_layout.md`。

---

## 常见坑

1. VPP 开启但仍传单个 model  
2. `num_microbatches` 太小，看起来像「PP 没用」  
3. 忽略中间 stage 没有 loss，日志只在部分 rank 有意义  
4. shape 与 SP/可变序列不一致 → P2P hang  
5. `deallocate_output_tensor` 省显存，但调试时更诡异  

---

## 动手验证

1. 确认你的配置会选中哪一个 `forward_backward_*`。  
2. 对 `PP=2, num_microbatches=4` 手画非 interleaving 时序。  
3. 在 schedule 内搜索：最后一个 stage 在哪里调用 `loss_func`。  

下一篇补齐并行三角最后一边：数据并行、梯度桶，以及所有并行模式在 optimizer.step 之前的汇合点——`finalize_model_grads`。
