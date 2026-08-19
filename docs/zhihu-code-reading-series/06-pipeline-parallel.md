# 知乎系列 06｜流水线并行：1F1B 与微批次调度

> 目标：读懂 `get_forward_backward_func` 如何选型，以及经典 1F1B 里 warmup / steady / cooldown 与 P2P 激活传递。

---

## PP 一句话

**流水线并行按深度切模型**：不同 PP stage 持有不同层；一个 global batch 拆成多个 microbatch，在 stage 之间形成流水，以提高设备利用率。

关键文件：

- `megatron/core/pipeline_parallel/schedules.py`  
- `megatron/core/pipeline_parallel/p2p_communication.py`  
- `megatron/core/pipeline_parallel/combined_1f1b.py`（进阶重叠）  
- `megatron/core/pipeline_parallel/utils.py`

训练侧入口回顾：`train_step` → `forward_backward_func(...)`。

---

## 调度函数如何被选中

`schedules.get_forward_backward_func` 的核心分支（概念上）：

| 条件 | 返回 |
|------|------|
| `pipeline_model_parallel_size == 1` | `forward_backward_no_pipelining` |
| `PP > 1` 且无 virtual PP | `forward_backward_pipelining_without_interleaving` |
| `PP > 1` 且有 virtual PP | `forward_backward_pipelining_with_interleaving` |

文档字符串里还详细描述了回调契约：`forward_step_func(data_iterator, model)` 必须返回 `(output, loss_func)`。  

**阅读策略**：先把选型读懂，再只精读你当前配置对应的那一个大函数——三个全读容易晕。

---

## Microbatch：流水线的「水」

没有 microbatch，PP 几乎必然严重气泡（前面 stage 算完就干等）。  

训练配置里常见：

- `micro_batch_size`：单个 microbatch 的样本数  
- global batch / microbatch → `num_microbatches`

调度内部会用类似 `get_pp_rank_microbatches` 的逻辑计算：

- **warmup**：先灌多少个 forward  
- **steady 1F1B**：一前一后交替  
- **cooldown**：收尾的 backward  

不同 PP rank 的 warmup 长度不同——这是「流水线」字面意义所在。

---

## 经典 1F1B（non-interleaving）直觉

以 `PP=4` 为例（示意）：

```text
时间 →

Stage0: F0 F1 F2 F3 B0 F4 B1 F5 B2 ... B*
Stage1:    F0 F1 F2 B0 F3 B1 F4 B2 ...
Stage2:       F0 F1 B0 F2 B1 F3 B2 ...
Stage3:          F0 B0 F1 B1 F2 B2 ...
```

- 前向时：stage \(i\) 把激活 P2P 发给 stage \(i+1\)  
- 反向时：stage \(i+1\) 把梯度 P2P 回 stage \(i\)  
- 只有最后一个 stage 在 forward 末尾调用 `loss_func`

对应实现：`forward_backward_pipelining_without_interleaving`。  
建议边读边在纸上画自己的 `PP` 与 `num_microbatches`。

---

## Virtual PP（Interleaving）：用层交错换气泡

当 `virtual_pipeline_model_parallel_size`（VPP）> 1 时：

- 每个 PP rank 持有**多个 model chunk**（层交错）  
- `model` 与 `data_iterator` 在接口上变成 **list**  
- 调度表更复杂：`get_schedule_table` 一类逻辑给出 `(microbatch_id, model_chunk_id)`  

收益是减小气泡；代价是调度与调试复杂度上升，且对 `num_microbatches` 更敏感。  

实现入口：`forward_backward_pipelining_with_interleaving`。

---

## P2P：激活与梯度怎么搬家

文件：`p2p_communication.py` 中的 `P2PCommunicator`

职责概括：

- 约定相邻 PP stage 之间发送/接收的 tensor shape  
- 处理 sequence parallel 等导致的 shape 变化  
- 与 schedule 中的 warmup/steady/cooldown 点位对齐  

读 schedule 时，看到 `send_forward` / `recv_forward` / `send_backward` / `recv_backward`（或以 communicator 方法出现的等价物），把它标在你的时序图上。

---

## 和 `train_step` 的衔接点

回到 `megatron/training/training.py`：

1. 取 `forward_backward_func`  
2. 传入 `forward_step`、`data_iterator`、`model`、`num_microbatches`、`seq_length`…  
3. 收集 loss 字典用于日志  
4. 之后才是 optimizer step  

因此：**PP 的正确性与性能问题，优先查 schedules + P2P，而不是查 Adam**。

---

## 气泡与性能：读代码时能验证的几件事

1. `num_microbatches` 是否明显大于 `PP`（过小会几乎全是 warmup）  
2. 是否意外启用了 VPP（model 是否变成 list）  
3. 是否开启了更多重叠（`combined_1f1b` 等进阶路径）  
4. timer 日志里 PP 通信是否暴露过多  

官方文档 `docs/user-guide/features/pipeline_parallel_layout.md` 可与自定义 layout 对照阅读。

---

## 建议阅读顺序

```text
get_forward_backward_func（选型）
  → forward_step / backward_step 辅助函数
  → without_interleaving 主循环（画 warmup/1F1B/cooldown）
  → P2PCommunicator
  → （需要时）with_interleaving + get_schedule_table
  → （进阶）combined_1f1b
```

---

## 常见坑

1. **VPP 开启但仍传单个 model** → 接口契约破坏。  
2. **`num_microbatches` 太小** → 利用率差，看起来像「PP 没用」。  
3. **忽略中间 stage 没有 loss** → 日志只在部分 rank 有意义。  
4. **shape 约定与 SP/可变序列不一致** → P2P hang 或 silent corruption。  
5. **`deallocate_output_tensor` 省显存** → 调试时若提前释放，观感更诡异。

---

## 本周作业

1. 在源码中确认：你当前（或假设）配置会选中哪一个 `forward_backward_*`。  
2. 对 `PP=2, num_microbatches=4` 手画非 interleaving 时序（标出 F/B 与 P2P）。  
3. 找到最后一个 stage 调用 `loss_func` 的代码位置（在 schedule 内搜索）。  

下一篇补齐并行三角的最后一边：数据并行、梯度桶与 `finalize_model_grads`。
