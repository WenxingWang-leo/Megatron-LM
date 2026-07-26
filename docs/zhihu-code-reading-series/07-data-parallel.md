# 知乎系列 07｜数据并行：DDP、梯度桶与 `finalize_model_grads`

> 目标：理解 Megatron 自定义 DDP 如何做梯度同步，以及为何需要统一的 `finalize_model_grads` 收尾。

---

## DP 在 Megatron 里扮演什么角色

在 TP/PP/CP 决定「模型怎么切开」之后，**数据并行决定「同一份（或分片后的）参数如何吃不同数据并同步梯度」**。

关键文件：

- `megatron/core/distributed/distributed_data_parallel.py`  
- `megatron/core/distributed/param_and_grad_buffer.py`  
- `megatron/core/distributed/finalize_model_grads.py`  
- `megatron/core/distributed/fsdp/`（Megatron-FSDP，进阶）  
- `megatron/core/distributed/torch_fully_sharded_data_parallel.py`

以及优化器侧将在下一篇出现的 `DistributedOptimizer`（ZeRO-1 风格状态分片，常与 DDP buffer 协同）。

---

## 为什么不是直接用原生 `torch.nn.parallel.DistributedDataParallel`

Megatron 需要与下列特性深度耦合：

- 流水线 stage 上只有部分参数  
- Tensor / Sequence Parallel 下某些梯度要先在 TP 组处理  
- 共享 embedding 跨 PP  
- MoE 的 expert 梯度与 router 梯度走不同组  
- 与 Distributed Optimizer 的连续 buffer / reduce-scatter 布局  
- 通信与反传重叠（bucketing、async）

因此它实现了自己的 `DistributedDataParallel` 包装器，而不是简单套一层 PyTorch DDP。

---



## 梯度桶与连续 buffer

核心思想：

1. 把参数/梯度放入连续存储（`_ParamAndGradBuffer` 一类结构）
2. 按 bucket 做 all-reduce 或 reduce-scatter
3. 尽量与反向传播重叠，隐藏通信延迟

阅读 `DistributedDataParallel` 时抓住：

- 构造期如何注册参数、划分 bucket  
- `finish_grad_sync`（或等价 API）何时被调用  
- 哪些 stage 会禁用 bucketing（常见于非首 PP stage 的策略差异）

配置侧常有 `DistributedDataParallelConfig`（overlap、平均方式、fp32 累积等）——对照 args 里的 `--overlap-grad-reduce` 等旗标阅读。

---



## `finalize_model_grads`：梯度世界的「收银台」

文件：`finalize_model_grads.py`

当一层层反向结束、DP 桶通信也差不多时，仍可能有一组「跨切分维度」的收尾工作，例如：

- 数据并行维上的最终同步（注意经常是 `dp-cp` 组）  
- Sequence Parallel 下 LayerNorm 等梯度的处理  
- 共享 embedding / 输出权重在 PP 首尾之间的同步  
- MoE router 相关梯度  
- 按 token 数缩放损失/梯度（变长、打包序列场景）

把它当成：**所有并行模式在「optimizer.step 之前」的汇合点**。  

排「数值不对b又不 crash」类 bug 时，这个文件的优先级很高。

---



## DP 与 CP、TP 的交叉点（再强调）


| 场景        | 常见通信组                           |
| --------- | ------------------------------- |
| 普通参数梯度同步  | `dp` 或 `dp-cp`                  |
| TP 内激活相关  | TP group（更多在 mappings / layers） |
| Expert 参数 | expert DP / EP 相关组              |


读代码时看到 `expt_dp`、`dp_cp` 不要混为一谈：MoE 下 dense 与 expert 的数据并行拓扑可以不同。

---



## FSDP 路线（知道入口即可）

若启用 Megatron-FSDP 或 Torch FSDP 封装：

- 参数/梯度/优化器状态更激进地分片  
- 前向 all-gather、反向 reduce-scatter 的时序与 DDP+DistOpt 不同

文档：`docs/user-guide/features/megatron_fsdp.md`  
代码：`distributed/fsdp/`、`torch_fully_sharded_data_parallel.py`

建议在吃透经典 DDP + DistOpt 后再进 FSDP，否则容易把两套术语揉乱。

---



## 建议阅读顺序

```text
DistributedDataParallelConfig（字段）
  → DistributedDataParallel.__init__
  → param_and_grad_buffer 的 bucket 逻辑
  → train_step 里 finish_grad_sync / finalize 的调用点
  → finalize_model_grads（逐段注释翻译成自己的清单）
  → （可选）FSDP adapter
```

---



## 和上一篇 PP 的衔接

PP 决定「梯度在深度维何时产生」；DP 决定「副本之间何时对齐」。  

一个 microbatch 的反向可能只更新本地 stage 参数；多个 microbatch 累积后，再在 DP 维同步。VPP/overlap 只会让时序更细，不会改变这个分工。

---



## 常见坑

1. **在错误的 process group 上 all-reduce** → 静默数值错误。
2. **忘记 embedding 跨 PP 同步** → 首尾 stage 权重漂移。
3. **MoE 与 dense 混用同一 DP 假设** → expert 梯度少同步/多同步。
4. **overlap 打开后的竞态** → 需确认 `finish_grad_sync` 栅栏位置。
5. **把 DistOpt 的 reduce-scatter 当成「没做 DP」** → 其实同步形态变了。

---



## 本周作业

1. 在 `training.py` 的 `train_step` 中定位梯度同步与 `finalize_model_grads` 的调用顺序。
2. 阅读 `finalize_model_grads` 文件头/函数文档，列出不少于 4 类它处理的梯度。
3. 说明：当 `CP>1` 时，为什么讨论 DP 时常提到 `dp-cp`。

下一篇转向「数据从哪来、状态往哪存」：Dataset、DistributedOptimizer、Dist Checkpoint。