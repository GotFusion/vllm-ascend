# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_avsg import AscendAvsgSFAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def test_decode_publishes_and_reads_the_correct_physical_rows():
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 3
    table = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([3, 2], dtype=torch.int32)
    slots = torch.tensor([0, 3, -1])
    metadata = SimpleNamespace(
        offload_dummy=False,
        attn_state=AscendAttentionState.DecodeOnly,
        num_actual_tokens=2,
        block_size=2,
        block_table=table,
        seq_lens_cpu=seq_lens,
        slot_mapping=slots,
    )
    full_kv = torch.arange(6, dtype=torch.float32).view(3, 2, 1, 1)
    full_rope = full_kv + 100
    topk = torch.tensor([[[0, 2]], [[1, 8]], [[-1, -1]]], dtype=torch.int32)
    original_topk = topk.clone()
    query = torch.ones(3, 1, 1)
    transport = impl.avsg_transport = Mock()
    transport.selection_block_table = torch.tensor([[0]], dtype=torch.int32)

    def match(sel_table, status, indices, full_table, kv_len, q_len, sparse_size, block_size):
        assert (status == -1).all() and q_len.tolist() == [1] and sparse_size == 1
        assert full_table.shape[1] == (int(kv_len[0]) + block_size - 1) // block_size
        assert torch.equal(sel_table, transport.selection_block_table)
        status.fill_(12)
        logical = indices[indices >= 0].long()
        physical = full_table[0, logical // block_size] * block_size + logical % block_size
        selection_slots = torch.arange(len(logical), dtype=torch.int32)
        indices[indices >= 0] = selection_slots
        return None, torch.stack((physical, selection_slots), dim=-1), len(logical)

    def gather(layer, q_len, mapping, count):
        assert layer == 3 and count == len(mapping)
        return tuple(component.flatten()[mapping[:, 0].long()] for component in (full_kv, full_rope))

    def sfa(q, q_rope, selected, indices, meta, q_len, kv_len, selection_table):
        assert torch.equal(selection_table, transport.selection_block_table)
        assert kv_len.tolist() == [2]
        kv, rope = selected
        valid = indices[indices >= 0].long()
        return (kv[valid] + rope[valid]).sum().reshape(q.shape)

    transport.gather.side_effect = gather
    impl._on_kv_cache_written((full_kv, full_rope), metadata)
    (layer, kv, rope, published_slots), _ = transport.publish.call_args
    assert layer == 3 and kv is full_kv and rope is full_rope
    assert published_slots.tolist() == [0, 3]
    with (
        patch.object(torch.ops._C_ascend, "npu_avsg_match", side_effect=match, create=True),
        patch.object(AscendSFAImpl, "_execute_sparse_flash_attention_process", side_effect=sfa),
    ):
        output = impl._execute_sparse_flash_attention_process(
            query,
            query,
            (full_kv, full_rope),
            topk,
            metadata,
            torch.tensor([1, 2]),
            seq_lens,
        )
    assert output.flatten().tolist() == [208.0, 106.0, 0.0]
    assert torch.equal(topk, original_topk)


@pytest.mark.parametrize("dummy", [False, True])
def test_prefill_uses_native_attention_and_only_publishes_real_rows(dummy):
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 2
    impl.avsg_transport = Mock()
    metadata = SimpleNamespace(
        offload_dummy=dummy,
        attn_state=AscendAttentionState.ChunkedPrefill,
        num_actual_tokens=3,
        slot_mapping=torch.tensor([7, 3, 4, -1]),
    )
    kv_cache = (Mock(), Mock())
    impl._on_kv_cache_written(kv_cache, metadata)
    assert impl.avsg_transport.publish.call_count == int(not dummy)
    with patch.object(AscendSFAImpl, "_execute_sparse_flash_attention_process") as native:
        result = impl._execute_sparse_flash_attention_process(None, None, kv_cache, None, metadata, None, None)
    assert result is native.return_value
    impl.avsg_transport.gather.assert_not_called()
