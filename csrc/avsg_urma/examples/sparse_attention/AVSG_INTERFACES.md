# AVSG Match 和 Copy 接口入门

一次 Decode 中，Sparse Attention 只读取 Indexer 选中的历史 KV。**Match 决定缺哪些 KV、该放在哪，并把 TopK 改成 Selection Cache 的索引。Copy 按 Match 给出的清单搬数据。** 当前目录的三层示例把 Full Cache 放在远端 NPU，把 Selection Cache 放在模型所在的 NPU。每条 KV 是一个 656 维 FP8 向量，占 656 字节。 [示例模型](model.py)

## 1. 一轮推理里这两个算子在哪

| 阶段 | 每层完成的工作 |
| --- | --- |
| Prefill | 为 prompt 的每个位置生成 Q 和 KV，用本端临时缓存做 Sparse Attention，并把本层 KV 写入远端 Full Cache。这一步只走 URMA WRITE。 |
| Decode | 为当前 token 生成本层 Q 和 KV，先 WRITE 新 KV，再用 Indexer 的 TopK 做 Match，用 Copy 把缺失的 KV READ 到本端，最后做 Attention。 |

Full Cache 保存这层已处理位置的 KV。Selection Cache 较小，保存此前已搬到本端、仍可复用的 KV。Indexer 的 TopK 是它为某个 query 选出的历史**位置编号**。例如 `[12, 200]` 表示这个 query 要读请求内位置 12 和 200 对应的 KV。

每个请求的逻辑位置都从 0 开始。多个请求共用一块缓存时，即使每个请求的 KV 都按顺序存放，起始槽位也可能不同。同样的逻辑位置 200，A 和 B 要分别查各自页表。若每个请求预留 512 个连续槽位，A 的 200 对应物理槽位 200，B 的 200 对应 712。 [vLLM 的物理槽与暂存区](../../../../vllm_ascend/attention/avsg_transport.py)

普通 Decode 每个请求有一个 query。主模型做 MTP 草稿验证时，一个请求可以在同一次 forward 中有多个 query。这里的三层示例每步只处理一个请求的一个 query，也不启用 MTP。Prefill 一次处理 prompt 的多个位置，因此每层要为每个位置算 Q。即使最终只用最后一个位置出词，前面位置在较早层的 Attention 输出仍会用来生成后续层的 KV。 [示例 Prefill 和 Decode](model.py)

三层示例的每层都有独立的随机 TopK 和 Match 状态。vLLM 的 GLM-5.2 在共享 indexer 的层间保留原始逻辑 TopK，每层用本层远端 KV 和选择缓冲区计算注意力。[GLM-5.2 的 AVSG 调用](../../../../vllm_ascend/attention/sfa_avsg.py)

## 2. `npu_avsg_match` 接收和返回什么

```python
reused_num, mapping, count = torch.ops._C_ascend.npu_avsg_match(
    selection_kv_block_table,
    selection_kv_block_status,
    selection_topk_indices,
    full_kv_block_table,
    full_kv_actual_seq,
    full_q_actual_seq,
    selection_topk_block_size=1,
    kv_block_size=128,
)
```

六个张量输入均为 `int32`。以下用 `B` 表示请求数，`S` 表示 BSND 的 query 轴长度，`T` 表示 TND 中打包的 query 数，`N` 表示每个 query 的 TopK 列表数，`K` 表示每条列表的容量。

| 参数 | Shape | 含义 |
| --- | --- | --- |
| `selection_kv_block_table` | `[B_sel, P_sel]`，`B_sel >= B` | Selection 页表。每个请求一行，逻辑页号用于查该行中的物理页号。 |
| `selection_kv_block_status` | `[B_sel, 2 * state_size]` | 每个请求的缓存匹配状态。两项 `int32` 组成一条记录。冷启动填 `-1`，Match 原地更新，后续 Decode 继续使用。 |
| `selection_topk_indices` | TND `[T, N, K]` 或 BSND `[B, S, N, K]` | 输入是请求内的 Full 逻辑位置。Match 将有效项原地改写为 Selection 逻辑槽位，`-1` 用作无效项。当前 `N=1`。 |
| `full_kv_block_table` | `[B, P_full]` | Full 页表。Match 用它把某个请求的历史位置转换成可供 Copy 使用的源槽位。 |
| `full_kv_actual_seq` | `[B]` | 每个请求的实际 KV 长度。例如 `[400, 80]` 表示两个请求分别有 400 和 80 条 KV。 |
| `full_q_actual_seq` | `[B]` | 本轮各请求 query 数的累加和。两个请求分别有 2 和 1 个 query 时，传 `[2, 3]`。这三行依次属于 A 的两个 query 和 B 的一个 query。 |

`selection_topk_block_size=1` 表示每个 TopK 元素选一个 token。vLLM Ascend 的 Torch 接口要求显式传入两个块大小。示例中两边的 KV 页大小共用 `kv_block_size=128`，须为 2 的幂并能被 `selection_topk_block_size` 整除。每个请求的 `state_size = P_sel * kv_block_size`，须为 2 的幂。当前 `K <= 2048`，当 `K > 32` 时必须传 `selection_topk_block_size=1`。 [Torch schema](../../../torch_binding.cpp) · [Match 参数校验](../../../attention/avsg_match/op_host/avsg_match_tiling.cpp#L67)

三个返回值也都是 `int32`。把 `K4` 定义为 `K` 向上取整到 4 的倍数。

| 返回值 | TND shape | BSND shape | 含义 |
| --- | --- | --- | --- |
| `reused_num`，算子输出名为 `selection_kv_actual_seq` | `[B]` | `[B]` | 每个请求累计的命中条目数。同一次 Match 中，后面的 query 复用前面已经安排搬运的槽位时，也会计为命中。 |
| `mapping`，即 `stage_two_copy_mapping` | `[T, N, K4, 2]` | `[B, S, N, K4, 2]` | 每条有效记录为 `[src, dst]`，值分别是 Full 和 Selection 的物理 token 槽位。 |
| `count`，即 `stage_two_copy_count` | `[T, N]` | `[B, S, N]` | 每个 query 对应的有效搬运条目数。Copy 只读对应 mapping 行的前 `count` 项。 |

读取改写后的 TopK 时，使用传入的 `selection_topk_indices` 张量。示例先复制本层的原始 TopK，再交给 Match，便于在图外按原始位置核对数据。CANN 算子接口还把 Selection 页表列为原地输出，当前 kernel 读取它来寻址。 [Torch 返回值形状](../../../attention/avsg_match/avsg_match_torch_adpt.h) · [Match 算子输出定义](../../../attention/avsg_match/op_host/avsg_match_def.cpp#L50) · [示例调用](model.py#L118)

## 3. 用两个 query 手算一次

设请求 A 当前有 400 条 KV，本次处理两个 query。每页 128 个 token，Selection 有 256 个槽位。为了看清地址转换，这里选用一组手算页表。当前三层示例采用单请求的连续页表，Full 为 `[0, 1, 2, 3]`，Selection 为 `[0, 1]`。

| 页表 | 逻辑页 0 | 逻辑页 1 | 逻辑页 2 | 逻辑页 3 |
| --- | ---: | ---: | ---: | ---: |
| 请求 A 的 Full 页表 | 7 | 9 | 4 | 6 |
| 请求 A 的 Selection 页表 | 3 | 5 |  |  |

假设 Selection 已经缓存 `Full 12 → Selection 5` 和 `Full 130 → Selection 133`。这一轮 `q0` 选 `[12, 200]`，`q1` 选 `[200, 130, 270]`。表格只展示有效项，其他 TopK 槽位填 `-1`。为便于手算，假设 Match 为新来的 200 和 270 分别分配 Selection 逻辑槽位 6 和 134。

以位置 200 为例：

1. Full 的逻辑位置 200 拆成逻辑页 1、页内偏移 72。查 Full 页表得到物理页 9，源槽位是 `9 × 128 + 72 = 1224`。
2. Selection 的逻辑槽位 6 拆成逻辑页 0、页内偏移 6。查 Selection 页表得到物理页 3，目标槽位是 `3 × 128 + 6 = 390`。
3. Match 把 `q0` 的 TopK 里的 200 改写成 `6`，并生成一条 `[1224, 390]` 的记录。之后处理 `q1` 时，200 已经有槽位和搬运计划，`q1` 直接复用槽位 6。

| Query | 原始 TopK | 改写后的 TopK | `mapping` 有效部分 | `count` |
| --- | --- | --- | --- | ---: |
| `q0` | `[12, 200]` | `[5, 6]` | `[[1224, 390]]` | 1 |
| `q1` | `[200, 130, 270]` | `[6, 133, 134]` | `[[526, 646]]` | 1 |

对这组缓存状态，`reused_num = [3]`，两个 query 合计只生成两条搬运记录。Copy 把物理槽位 1224 处的数据写到物理槽位 390。Attention 收到的是逻辑槽位 `6`，再用 Selection 页表定位到物理槽位 `390`，因此两个 query 都能读到同一份 KV。示例的 `selection_slots()` 明确写出了这一步页表转换。 [Match 生成物理地址](../../../attention/avsg_match/op_kernel/avsg_match_split_bs_reuse_vec.h#L847) · [示例将逻辑槽位转为物理槽位](model.py#L24)

当前一个 query 只有一份 TopK，所以 `N=1`。如果多个 Attention Q head 共享这份 TopK 和同一份 KV 缓存，Match 仍只查一次这份列表，Copy 仍只搬一份缺失 KV。每个 Q head 用自己的 Q 计算注意力。如果每个 head 都有独立 TopK，就需要 `N>1`，当前 Match 会在 tiling 检查时拒绝。 [head 维校验](../../../attention/avsg_match/op_host/avsg_match_tiling.cpp#L154)

## 4. 使用 URMA 读回 KV

```python
rope, selection_kv, context, scratch, seen, status = session.prepare_graph_copy(
    layer_id=layer, mapping_capacity=32,
)

torch.ops.avsg_urma.copy_graph_(
    rope, selection_kv, full_q_actual_seq, mapping, count,
    context, scratch, seen, status, write_status,
)
```

先在图外完成两端内存注册、连接和每层 context 准备，再把固定张量传给算子。这里的 D2D Copy 消费 Match 输出的同一份 `mapping/count`。设备上的 AIV kernel 根据 context 提交 URMA READ，并等待完成。

| 参数 | 本目录示例中的用途 |
| --- | --- |
| `rope` | 零元素的 FP8 张量，满足调用签名。远端 layout 的 `rope_dim=0`，每条有效记录只搬一行 KV。 |
| `selection_kv` | 本端目标张量，shape 为 `[2, 128, 656]`，dtype 为 `float8_e4m3fn`。Copy 原地写入。 |
| `full_q_actual_seq` | 同 Match。单请求单 query 时是 `[1]`。 |
| `mapping`、`count` | Match 生成的计划。示例每层分别是 `[1, 1, 32, 2]` 和 `[1, 1]`。某行 count 为 0 时，该行没有数据需要搬。 |
| `context` | 保存远端 Full Cache、本端 Selection Cache 和设备通信通道的信息。远端 Full Cache 由服务端注册，Copy 通过 context 定位它。 |
| `scratch`、`seen` | 预先分配的描述符工作区和用于检查目标槽位的位图。 |
| `status` | Copy 原地写入的完成状态。示例在图外用 `session.check_graph_copy(status)` 校验每个 jetty 的状态，并汇总搬运条数。 |
| `write_status` | 可选的前序 WRITE 状态。示例中 READ 位于 WRITE 之后，两者使用同一流，READ 读取这份状态检查当前 token 的 KV 已成功写到远端。 |

Torch 调用返回 `None`。后续 Attention 直接读取已更新的 `selection_kv`。示例把 WRITE、Match、READ 和 Attention 编进同一个 Decode 图，索引生成、传输状态检查和 `--verify` 字节对比在图外执行。WRITE 的记录顺序是 `[远端目标槽位, 本端源槽位]`。传给 READ 的 Match 清单顺序是 `[远端源槽位, 本端目标槽位]`。 [URMA Torch schema](../../csrc/binding.cpp#L455) · [图资源准备与状态检查](../../avsg_urma/session.py) · [示例 Decode](model.py#L118)

## 5. 为什么 GE 和 Torch 的 Match 输出数不同

GE 的 AvsgMatch 声明 6 个输出，依次是 Selection 页表、缓存状态、`selection_kv_actual_seq`、`mapping`、`count` 和改写后的 TopK。vLLM Ascend 的可变 Torch schema 把页表、缓存状态及 TopK 声明为可写输入，返回三个单独分配的统计与计划张量。示例读取改写后的 TopK，并将 `mapping/count` 传给 URMA。`aclnnAvsgMatchGetWorkspaceSize` 和 `aclnnAvsgMatch` 由插件绑定调用。[Match ACLNN 声明](../../../attention/avsg_match/op_api/aclnn_avsg_match.h) · [vLLM Ascend Torch 绑定](../../../attention/avsg_match/avsg_match_torch_adpt.h)

## 6. 对照三层入图示例

| 想看什么 | 入口 |
| --- | --- |
| Prefill 的 Q/KV 生成和远端写入 | [`AvsgSparseModel.prefill()`](model.py#L97) |
| Decode 的 WRITE → Match → READ → Attention | [`AvsgSparseModel.decode()`](model.py#L118) |
| 图外生成固定 shape 的随机 TopK、捕获与重放 | [`run()`](run.py) |
| 用原始 TopK 逐层核对读回的 FP8 字节 | [`check_decode()`](run.py) |
| 两端 JSON、安装和运行命令 | [运行说明](README.md#运行) |
