# 精读 Megatron 源码（7）：数据并行——为什么 Megatron 要自己写一套 DDP

> **专栏**：Megatron 源码精读 · 第 7 篇  
> **核心文件**：`distributed/distributed_data_parallel.py`、`param_and_grad_buffer.py`、`finalize_model_grads.py`

---

在 TP/PP/CP 决定「模型怎么切开」之后，**数据并行决定：同一份（或分片后的）参数如何吃不同数据，并在正确的 process group 上对齐梯度。**

很多人第一反应是：直接套 `torch.nn.parallel.DistributedDataParallel` 不就行了？  
Megatron 没有这么做——原因比「自己造轮子」更具体。

---

## 为什么不能简单套原生 DDP

Megatron 需要和这些特性深度耦合：

- 流水线 stage 上只有部分参数  
- Tensor / Sequence Parallel 下某些梯度要先在 TP 组处理  
- 共享 embedding 跨 PP  
- MoE 的 expert / router 梯度走不同组  
- 与 Distributed Optimizer 的连续 buffer / reduce-scatter 布局  
- 通信与反传重叠（bucketing、async）  

所以它实现了自己的 `DistributedDataParallel` 包装器。

关键路径：

- `megatron/core/distributed/distributed_data_parallel.py`  
- `param_and_grad_buffer.py`  
- `finalize_model_grads.py`  
- （进阶）`distributed/fsdp/`、`torch_fully_sharded_data_parallel.py`

---

## 梯度桶与连续 buffer

核心思想：

1. 把参数/梯度放入连续存储（`_ParamAndGradBuffer`）  
2. 按 bucket 做 all-reduce 或 reduce-scatter  
3. 尽量与反向传播重叠，隐藏通信延迟  

精读 `DistributedDataParallel` 时抓住三点：

- 构造期如何注册参数、划分 bucket  
- `finish_grad_sync`（或等价 API）何时被调用  
- 哪些 stage 会禁用 bucketing（常见于非首 PP stage 的策略差异）  

配置侧对照 `DistributedDataParallelConfig`，以及 CLI 里的 `--overlap-grad-reduce` 等旗标。

---

## `finalize_model_grads`：梯度世界的收银台

当一层层反向结束、DP 桶通信也差不多时，仍可能有一组「跨切分维度」的收尾，例如：

- 数据并行维最终同步（注意经常是 `dp-cp` 组）  
- Sequence Parallel 下 LayerNorm 等梯度处理  
- 共享 embedding / 输出权重在 PP 首尾之间同步  
- MoE router 相关梯度  
- 按 token 数缩放（变长、打包序列）  

把它当成：

> **所有并行模式在 `optimizer.step` 之前的汇合点。**

排「不 crash 但数值不对」类 bug 时，这个文件优先级很高。建议把文件头/函数文档翻译成自己的检查清单。

---

## DP 与 CP、TP、MoE 的交叉点

| 场景 | 常见通信组 |
|------|------------|
| 普通参数梯度同步 | `dp` 或 `dp-cp` |
| TP 内激活相关 | TP group（更多在 mappings/layers） |
| Expert 参数 | expert DP / EP 相关组 |

看到 `expt_dp`、`dp_cp` 不要混为一谈：MoE 下 dense 与 expert 的数据并行拓扑可以不同。

---

## FSDP 路线：知道入口即可

若启用 Megatron-FSDP 或 Torch FSDP 封装：

- 参数/梯度/优化器状态更激进分片  
- 前向 all-gather、反向 reduce-scatter 的时序与 DDP+DistOpt 不同  

文档：`docs/user-guide/features/megatron_fsdp.md`  

建议先吃透经典 DDP + DistOpt（下一篇），再进 FSDP，否则两套术语容易揉乱。

---

## 和 PP 的分工（再强调）

- PP 决定「梯度在深度维何时产生」  
- DP 决定「副本之间何时对齐」  

一个 microbatch 的反向可能只更新本地 stage 参数；多个 microbatch 累积后，再在 DP 维同步。VPP/overlap 只会让时序更细，不会改变这个分工。

---

## 建议精读顺序

```text
DistributedDataParallelConfig
  → DistributedDataParallel.__init__
  → param_and_grad_buffer 的 bucket 逻辑
  → train_step 里 finish_grad_sync / finalize 的调用点
  → finalize_model_grads（逐段做笔记）
```

---

## 常见坑

1. 在错误的 process group 上 all-reduce → 静默数值错误  
2. 忘记 embedding 跨 PP 同步 → 首尾权重漂移  
3. MoE 与 dense 混用同一 DP 假设  
4. overlap 打开后的竞态 → 确认 `finish_grad_sync` 栅栏  
5. 把 DistOpt 的 reduce-scatter 当成「没做 DP」  

---

## 动手验证

1. 在 `train_step` 中定位梯度同步与 `finalize_model_grads` 的调用顺序。  
2. 阅读 `finalize_model_grads`，列出不少于 4 类它处理的梯度。  
3. 说明：`CP>1` 时，为什么讨论 DP 常提到 `dp-cp`。  

下一篇转向「数据从哪来、状态往哪存」：Dataset、DistributedOptimizer、Dist Checkpoint——把训练闭环的另外三块拼上。
