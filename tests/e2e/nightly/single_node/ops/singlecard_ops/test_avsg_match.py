# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the native AVSG mapping and GLM attention with an on-device copy."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif(not is_950(), reason="AVSG match is built for A5")
def test_avsg_match_paged_attention_cold_and_warm():
    assert enable_custom_op()
    rng = torch.Generator().manual_seed(37)
    block_size, topk, full_blocks = 128, 2048, 24
    selected_blocks = topk // block_size
    full_table = torch.randperm(full_blocks, generator=rng).to(torch.int32).view(1, -1)
    selection_table = torch.arange(selected_blocks - 1, -1, -1, dtype=torch.int32).view(1, -1)
    tables = full_table.npu(), selection_table.npu()
    status = torch.full((1, 2 * topk), -1, device="npu", dtype=torch.int32)
    logical = torch.tensor([2, 258, 1100, 2561])
    original = torch.full((1, 1, topk), -1, dtype=torch.int32)
    original[0, 0, : len(logical)] = logical.to(torch.int32)
    source = full_table[0, logical // block_size].long() * block_size + logical % block_size
    q_len = torch.tensor([1], dtype=torch.int32, device="npu")
    kv_len = torch.tensor([full_blocks * block_size], dtype=torch.int32, device="npu")

    def random_tensor(*shape):
        return torch.randn(shape, generator=rng, dtype=torch.bfloat16).npu()

    q, q_rope = random_tensor(1, 64, 512), random_tensor(1, 64, 64)
    full = (random_tensor(full_blocks, block_size, 1, 512), random_tensor(full_blocks, block_size, 1, 64))
    selected = tuple(torch.zeros_like(cache[:selected_blocks]) for cache in full)

    def attend(cache, indices, table, length):
        result, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q,
            key=cache[0],
            value=cache[0],
            sparse_indices=indices,
            scale_value=1 / 24,
            sparse_block_size=1,
            block_table=table,
            actual_seq_lengths_query=q_len,
            actual_seq_lengths_kv=length,
            query_rope=q_rope,
            key_rope=cache[1],
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        return result.cpu()

    baseline = attend(full, original.npu(), tables[0], kv_len)
    for expected_misses in (len(logical), 0):
        remapped = original.npu()
        _, mapping, counts = torch.ops._C_ascend.npu_avsg_match(
            tables[1], status, remapped, tables[0], kv_len, q_len, 1, block_size
        )
        assert counts.cpu().item() == expected_misses
        rewritten = remapped.cpu().flatten()
        assert (rewritten[len(logical) :] == -1).all()
        slots = rewritten[: len(logical)].long()
        assert ((slots >= 0) & (slots < topk)).all()
        destination = selection_table[0, slots // block_size].long() * block_size + slots % block_size
        if expected_misses:
            pairs = mapping[0, 0, :expected_misses].cpu().long()
            assert sorted(map(tuple, pairs.tolist())) == sorted(zip(source.tolist(), destination.tolist(), strict=True))
            # Exercise the addresses returned by Match. Remote URMA transfer has separate two-endpoint coverage.
            src, dst = pairs.npu().unbind(-1)
            for cache, hot_cache in zip(full, selected, strict=True):
                hot_cache.flatten(0, 1).index_copy_(0, dst, cache.flatten(0, 1).index_select(0, src))
        for cache, hot_cache in zip(full, selected, strict=True):
            actual = hot_cache.flatten(0, 1).index_select(0, destination.npu())
            expected = cache.flatten(0, 1).index_select(0, source.npu())
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        output = attend(selected, remapped, tables[1], kv_len.clamp_max(topk))
        torch.testing.assert_close(output, baseline, atol=0.03, rtol=0.01)
