# FP8 Lightning Indexer with AVSG KVMatch on Ascend 950

The operator selects 2,048 tokens per query and resolves them to persistent hot KV slots in one mixed Cube/Vector kernel launch. It runs the Ascend 950 CANN FP8 indexer, writes the selected logical IDs into the slots output, waits for all producers, and rewrites that buffer with the AVSG slot indices. One vector core owns all the query rows of a request during matching. The caller applies the returned copy plan before consuming the slots.

## Torch API

```python
slots, hits, mapping, counts = torch.ops._C_ascend.npu_fused_quant_lightning_indexer_avsg_match(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, actual_seq_lengths_key,
    block_table, hot_block_table, status,
)
```

All tensors must be contiguous on the same device. Use distinct hot pages and state for simultaneously active requests. Initialize each new request's status to `-1` and preserve it across decode steps. When recycling or reordering requests, reset or reorder their state and page tables together. Keep the page-to-physical-row mapping and KV contents consistent with the saved state.

| Input | Dtype and shape | Meaning |
|---|---|---|
| query, key | FP8 E4M3FN `[T,32,128]`, `[pages,128,1,128]` | TND queries and paged index keys |
| weights | BF16 `[T,32]` | Per-query head weights |
| query_dequant_scale, key_dequant_scale | FP32 `[T,32]`, `[pages,128,1]` | Per-token, per-head scales |
| actual_seq_lengths_query | int32 `[B]` | Cumulative ends `[Q,2Q,…,BQ]` with `T=B*Q` |
| actual_seq_lengths_key | int32 `[B]` | Valid key length per request, at least `2048+Q-1` under the causal mask |
| block_table | int32 `[B,full_pages_per_request]` | Physical source page IDs in the key cache |
| hot_block_table | int32 `[B,C/128]` | Physical destination page IDs for each request's hot KV pool |
| status | int32 `[B,2*C]`, mutable | Two packed AVSG state words per persistent slot |

For Q=1 use C=4,096 or 8,192 slots per request. For Q=4 use C=8,192 so all four top-k sets fit even when disjoint. Page size is 128. The valid prefix must fit its block table and the 24-bit logical token field. The page IDs for referenced source and destination rows must be valid for the corresponding storage, with physical row addresses representable as int32.

| Output | Shape, all int32 | Meaning |
|---|---|---|
| slots | `[T,1,2048]` | Request-local logical hot slot per selected token |
| hits | `[B]` | Total number of AVSG lookup hits across a request's Q rows |
| mapping | `[T,1,2048,2]` | First `counts[t,0]` pairs are physical full-cache source and hot-cache destination row addresses |
| counts | `[T,1]` | Number of KV rows to copy for each query |

For example, use `hot_block_table[b,slot//128]*128 + slot%128` to translate a returned slot to the hot physical row. On invalid device sequence lengths, `hits` and `counts` are all `-1`, the operator preserves the request state, and the slots and mapping are undefined. Shape and dtype errors are reported before launch.

## Validation

The Ascend 950 package includes the operator through the normal vLLM Ascend custom-op build. The nightly test uses official FP8 LI selections and replays the physical KV transfer plan across successive steps on non-identity page tables. It covers both batch sizes, all supported 4K/8K configurations, cold and reused state, multi-query eviction, and invalid lengths. Run it against a built extension on an idle device.

```sh
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_quant_lightning_indexer_avsg_match.py
```

The indexer sources are pinned in the [source notice](op_kernel/qli/NOTICE). The AVSG implementation is based on [the recipes Match kernel](https://gitcode.com/wangzhenwu/cann-recipes-infer_sjy/blob/d107a3e106866344c05c6ba58948ab1f196c6169/ops/ascendc/src/avsg_match/op_kernel/avsg_match_split_bs_reuse_vec.h). The imported device sources carry the [CANN Open Software License 2.0](LICENSE).
