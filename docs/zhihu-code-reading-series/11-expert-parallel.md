# 提纲笔记（11）：专家并行 EP

> 正式精读成稿见 [`publish/11-expert-parallel.md`](./publish/11-expert-parallel.md)。

## 本篇要解决的困惑

- EP 切的是专家列表，不是 batch
- EP ≠ DP；与 EP 配对的「数据并行」叫 expert_DP
- Token 必须 AlltoAll 搬家；权重尽量留在持有该专家的卡上

## 必读符号

| 文件 | 符号 |
|------|------|
| `parallel_state.py` | `expert_decoder_rank_generator`、`get_expert_model_parallel_group` |
| `moe_layer.py` | `BaseMoELayer.local_expert_indices` |
| `token_dispatcher.py` | `MoEAlltoAllTokenDispatcher` |

## 公式

```text
expert_DP = world / (expert_TP × EP × PP)
# expert_TP=TP 且 CP=1 时：dense_DP = EP × expert_DP
```

## 与第 9 篇分工

- **11**：拓扑与概念（本篇）
- **09**：Router / aux loss / Dispatcher 实现细节
