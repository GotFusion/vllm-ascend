# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention import sfa_avsg, sfa_v1
from vllm_ascend.attention.sfa_avsg import AscendAvsgSFAImpl
from vllm_ascend.device.device_op import DeviceOperator


@pytest.mark.parametrize("is_chunked", [False, True])
def test_read_selected_physical_rows_per_causal_query(is_chunked):
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 3
    table = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.int32)
    # Request 0 starts with a two-token cached prefix. In the mixed case it
    # appends two tokens, then request 1 emits one decode token.
    if is_chunked:
        query_starts, lengths, slots = [0, 2, 3], [4, 2], [0, 1, 3, -1]
        logical_topk = [[3, 2], [1, 3], [1, 8], [-1, -1]]
        expected = [100, 212, 106, 0]
    else:
        query_starts, lengths, slots = [0, 1, 2], [3, 2], [0, 3, -1]
        logical_topk = [[0, 2], [1, 8], [-1, -1]]
        expected = [208, 106, 0]
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    slots = torch.tensor(slots)
    metadata = SimpleNamespace(
        offload_dummy=False,
        query_start_loc_cpu=torch.tensor(query_starts),
        num_actual_tokens=query_starts[-1],
        block_size=2,
        block_table=table,
        seq_lens_cpu=seq_lens,
        slot_mapping=slots,
    )
    full_kv = torch.arange(6, dtype=torch.float32).view(3, 2, 1, 1)
    full_rope = full_kv + 100
    topk = torch.tensor(logical_topk, dtype=torch.int32).unsqueeze(1)
    original_topk = topk.clone()
    query = torch.ones(len(logical_topk), 1, 1)
    transport = impl.avsg_transport = Mock()
    transport.selection_block_table = torch.tensor([[0]], dtype=torch.int32)

    def match(sel_table, status, indices, full_table, kv_len, q_len, sparse_size, block_size):
        assert (status == -1).all() and q_len.tolist() == [1] and sparse_size == 1
        assert (int(kv_len[0]) + block_size - 1) // block_size <= full_table.shape[1] <= 2
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

    def sfa(owner, q, q_rope, selected, indices, meta, q_len, kv_len, *, block_table, sparse_mode, return_lse):
        assert owner is impl and sparse_mode == 0 and return_lse
        assert torch.equal(block_table, transport.selection_block_table)
        assert kv_len.tolist() == [2]
        kv, rope = selected
        valid = indices[indices >= 0].long()
        assert torch.equal(indices.view(-1)[: valid.numel()], valid.int())
        return (kv[valid] + rope[valid]).sum().reshape(q.shape), None, None

    transport.gather.side_effect = gather
    impl._on_kv_cache_written((), metadata)
    (layer, published_slots), _ = transport.publish_staged.call_args
    assert layer == 3 and published_slots.tolist() == slots.tolist()[:-1]
    with (
        patch.object(torch.ops._C_ascend, "npu_avsg_match", side_effect=match, create=True),
        patch.object(DeviceOperator, "execute_sparse_flash_attention_process", side_effect=sfa),
    ):
        output = impl._execute_sparse_flash_attention_process(
            query,
            query,
            (),
            topk,
            metadata,
            metadata.query_start_loc_cpu[1:],
            seq_lens,
        )
    assert output.flatten().tolist() == expected
    assert transport.gather.call_count == metadata.num_actual_tokens
    assert torch.equal(topk, original_topk)


def test_dummy_attention_does_not_use_remote_or_address_unused_logical_pages():
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 2
    impl.avsg_transport = Mock()
    metadata = SimpleNamespace(
        offload_dummy=True,
        num_actual_tokens=3,
        slot_mapping=torch.tensor([7, 3, 4, -1]),
    )
    query = torch.ones(3, 1, 2)
    assert (impl._get_sfa_kv_slot_mapping(metadata) == -1).all()
    impl.avsg_transport.staging_slots.assert_not_called()
    impl._on_kv_cache_written((), metadata)
    with patch.object(DeviceOperator, "execute_sparse_flash_attention_process") as native:
        result = impl._execute_sparse_flash_attention_process(query, query, (), None, metadata, None, None)
    assert torch.equal(result, torch.zeros_like(query))
    native.assert_not_called()
    impl.avsg_transport.publish_staged.assert_not_called()
    impl.avsg_transport.gather.assert_not_called()


def test_profiling_forward_without_transport_or_kv_cache():
    impl = object.__new__(AscendAvsgSFAImpl)
    impl._parallel_o_proj = None
    impl.enable_sparse_sfa_turboquant = False
    hidden = torch.ones(2, 4)
    output = torch.ones_like(hidden)
    with patch.object(torch, "npu", create=True) as npu:
        npu.is_current_stream_capturing.return_value = False
        assert impl.forward("model.layers.3", hidden, None, None, output) is output
    assert torch.equal(output, torch.zeros_like(output))


def test_avsg_refuses_to_capture_dummy_attention():
    impl = object.__new__(AscendAvsgSFAImpl)
    with patch.object(torch, "npu", create=True) as npu, pytest.raises(RuntimeError, match="mla_forward"):
        npu.is_current_stream_capturing.return_value = True
        impl.forward("model.layers.3", torch.ones(2, 4), (), None)


def test_decode_graph_records_attention_even_for_capture_dummy():
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 3
    transport = impl.avsg_transport = Mock()
    remote_slot = torch.tensor([-1], dtype=torch.int32)
    transport.decode_inputs_for_layer.side_effect = {
        3: (remote_slot, torch.tensor([0]), torch.tensor([[0]]))
    }.__getitem__
    transport.staging_slots.side_effect = lambda slot: torch.where(slot >= 0, 0, -1)
    metadata = SimpleNamespace(offload_dummy=True, num_actual_tokens=1, num_input_tokens=1, seq_lens=torch.tensor([0]))
    with (
        patch.object(impl, "_in_full_decode", return_value=True),
        patch.object(sfa_v1.AscendSFAImpl, "forward") as base,
    ):
        hidden = torch.ones(1, 4)
        impl.forward("model.layers.3", hidden, (), metadata)
        base.assert_called_once()
        for live_slot in (-1, 42):
            remote_slot.fill_(live_slot)
            local = impl._get_sfa_kv_slot_mapping(metadata)
            assert local.tolist() == ([-1] if live_slot < 0 else [0])
            impl._on_kv_cache_written((), metadata)
            transport.publish_decode_graph.assert_called_with(3)
        metadata.num_actual_tokens = 0
        with pytest.raises(RuntimeError, match="one request and one token"):
            impl.forward("model.layers.3", hidden, (), metadata)


def test_full_decode_capture_scope_accepts_pinned_vllm_none_mode():
    impl = object.__new__(AscendAvsgSFAImpl)
    transport = impl.avsg_transport = SimpleNamespace(decode_graph_capture=False)
    context = SimpleNamespace(cudagraph_runtime_mode=sfa_avsg.CUDAGraphMode.NONE)
    with (
        patch.object(sfa_avsg, "is_forward_context_available", return_value=True),
        patch.object(sfa_avsg, "get_forward_context", return_value=context),
    ):
        assert not impl._in_full_decode()
        transport.decode_graph_capture = True
        assert impl._in_full_decode()
        context.cudagraph_runtime_mode = sfa_avsg.CUDAGraphMode.PIECEWISE
        assert not impl._in_full_decode()
        context.cudagraph_runtime_mode = sfa_avsg.CUDAGraphMode.FULL
        assert impl._in_full_decode()


def test_fixed_decode_graph_inputs_drive_packed_indices_and_empty_results():
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 1
    transport = impl.avsg_transport = Mock()
    remote_slot = torch.tensor([-1], dtype=torch.int32)
    remote_length = torch.tensor([0], dtype=torch.int32)
    remote_table = torch.zeros((1, 25), dtype=torch.int32)
    transport.decode_inputs_for_layer.side_effect = {1: (remote_slot, remote_length, remote_table)}.__getitem__
    transport.selection_block_table = torch.arange(64, dtype=torch.int32).reshape(1, -1)
    meta = SimpleNamespace(offload_dummy=True, block_size=128)
    topk = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    q = torch.ones((1, 2, 3))
    selected = torch.empty(8192)
    expected = []

    def match(local, state, packed, table, seq_len, q_len, _, block_size):
        assert (state == -1).all() and table is remote_table
        assert seq_len.tolist() == [max(1, int(remote_length[0]))] and q_len.tolist() == [1]
        assert packed.flatten()[: len(expected)].tolist() == expected
        assert (packed.flatten()[len(expected) :] == -1).all()
        logical = packed.flatten()[: len(expected)].long()
        physical = table[0, logical // block_size] * block_size + logical % block_size
        packed.flatten()[: len(expected)] = torch.arange(len(expected), dtype=torch.int32)
        pairs = torch.zeros((1, 1, 2048, 2), dtype=torch.int32)
        pairs[0, 0, : len(expected), 0] = physical
        # If Match reports a bogus nonzero count for the dummy shape, no READ may run.
        return None, pairs, torch.tensor([[len(expected) or 1]], dtype=torch.int32)

    def gather(layer, pairs, count):
        assert layer == 1 and count.tolist() == [[len(expected)]]
        selected.fill_(float("nan"))
        selected[: len(expected)].copy_(pairs[0, 0, : len(expected), 0])
        return selected, selected

    def sfa(owner, q, qr, caches, indices, metadata, q_len, kv_len, *, block_table, sparse_mode, return_lse):
        assert owner is impl and sparse_mode == 0 and return_lse
        assert block_table is transport.selection_block_table
        valid = indices.flatten()[indices.flatten() >= 0].long()
        assert valid.tolist() == list(range(max(1, len(expected))))
        return caches[0][valid].sum().expand_as(q), None, None

    transport.gather_decode_graph.side_effect = gather
    with (
        patch.object(impl, "_in_full_decode", return_value=True),
        patch.object(torch.ops._C_ascend, "npu_avsg_match", side_effect=match, create=True),
        patch.object(DeviceOperator, "execute_sparse_flash_attention_process", side_effect=sfa),
    ):
        for slot, length, offset, incoming, expected in (
            (3, 3072, 1, [-1, 177, 3072, 3, -2, 788, 0], [177, 3, 788, 0]),
            (6, 3073, 2, [3072, 3073, -1, 129, 1], [3072, 129, 1]),
            (6, 3073, 2, [3073, -1], []),
            (-1, 0, 0, [0, 1], []),
            (-1, 3073, 1, [0, 1], []),
        ):
            remote_slot.fill_(slot)
            remote_length.fill_(length)
            remote_table[0] = (torch.arange(25) + offset) % 25
            topk.fill_(-1)
            topk.flatten()[: len(incoming)] = torch.tensor(incoming)
            original = topk.clone()
            result = impl._execute_sparse_flash_attention_process(q, q, (), topk, meta, None, None)
            total = sum(int(remote_table[0, pos // 128]) * 128 + pos % 128 for pos in expected)
            torch.testing.assert_close(result, torch.full_like(q, total))
            assert torch.equal(topk, original)


@pytest.mark.parametrize("preprocess_type", ["NATIVE", "PROLOG_V3", "MLAPO"])
def test_forward_writes_staging_before_publishing_physical_slots(preprocess_type):
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 3
    impl.layer_name = "model.layers.3"
    impl.preprocess_type = getattr(sfa_v1.PreprocessType, preprocess_type)
    impl._parallel_o_proj = impl.layerwise_kv_cache_hook = impl.g_proj = None
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.pcp_enabled = impl.has_indexer = False
    impl.skip_topk = True
    impl.q_lora_rank = impl.kv_lora_rank = impl.qk_rope_head_dim = 1
    hidden = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    slots = torch.tensor([19, 7], dtype=torch.int32)
    metadata = SimpleNamespace(
        cos=None,
        sin=None,
        slot_mapping=slots,
        num_input_tokens=2,
        num_actual_tokens=2,
        seq_lens=torch.tensor([13]),
        cum_query_lens=torch.tensor([2]),
        offload_dummy=False,
    )
    transport = impl.avsg_transport = Mock()
    staging = transport.staging_cache = tuple(torch.zeros(1, 2, 1, 1) for _ in range(2))
    transport.staging_slots.return_value = torch.arange(2, dtype=torch.int32)

    def write_staging(cache, local_slots):
        assert cache is staging and local_slots.tolist() == [0, 1]
        for tensor, values in zip(cache, hidden.T[1:]):
            tensor.flatten()[local_slots.long()] = values

    def fused(*, kv_cache, slot_mapping, **kwargs):
        write_staging(kv_cache, slot_mapping)
        return hidden, hidden, hidden, hidden

    def native(kv, cos, sin, cache, local_slots, meta):
        write_staging(cache, local_slots)
        return hidden, hidden

    def publish(layer_id, remote_slots):
        assert layer_id == 3 and remote_slots.tolist() == [19, 7]
        for tensor, values in zip(staging, hidden.T[1:]):
            torch.testing.assert_close(tensor.flatten(), values)

    transport.publish_staged.side_effect = publish
    impl._get_indexer_attn_metadata = Mock(return_value=None)
    impl._get_indexcache_topk_indices = Mock(return_value=None)
    impl.fused_qkv_a_proj = lambda x: (x,)
    impl.q_a_layernorm = torch.nn.Identity()
    impl.exec_kv = native
    impl._q_proj_and_k_up_proj = lambda q: (hidden, hidden)
    impl.rope_single = lambda x, *args: x
    impl._sfa_preprocess_prolog_v3 = impl._sfa_preprocess_mlapo = fused
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = torch.nn.Identity()
    impl._finalize_o_proj = lambda x, output: output.copy_(x)
    output = torch.empty_like(hidden)
    with (
        patch.object(torch, "npu", create=True) as npu,
        patch.object(sfa_v1, "wait_for_kv_layer_from_connector"),
        patch.object(sfa_v1, "notify_kv_cache_written"),
        patch.object(sfa_v1, "attention_transfer_window"),
        patch.object(sfa_v1, "maybe_save_kv_layer_to_connector"),
    ):
        npu.is_current_stream_capturing.return_value = False
        assert impl.forward(impl.layer_name, hidden, (), metadata, output) is output
    assert torch.equal(hidden, output)
    transport.publish_staged.assert_called_once()


@pytest.mark.parametrize("selected_count", [0, 4])
def test_long_history_with_padded_topk_matches_selected_row_reference(selected_count):
    generator = torch.Generator().manual_seed(353)
    kv_len, block_size, topk = 3072, 128, 2048
    layout = torch.randperm(kv_len // block_size, generator=generator, dtype=torch.int32).reshape(1, -1)
    kv = torch.randn(kv_len, 8, generator=generator)
    rope = torch.randn(kv_len, 4, generator=generator)
    q = torch.randn(1, 2, 8, generator=generator)
    q_rope = torch.randn(1, 2, 4, generator=generator)
    logical = torch.tensor([3001, 64, 2050, 511], dtype=torch.int32)[:selected_count]
    physical = layout[0, logical.long() // block_size] * block_size + logical % block_size
    indices = torch.full((1, 1, topk), -1, dtype=torch.int32)
    indices.view(-1)[1 : 1 + 2 * selected_count : 2] = logical
    metadata = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1]),
        seq_lens_cpu=torch.tensor([kv_len]),
        block_size=block_size,
        block_table=layout,
        num_actual_tokens=1,
        offload_dummy=False,
    )
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.avsg_layer_id = 0
    impl.avsg_transport = Mock()
    impl.avsg_transport.selection_block_table = torch.arange(64, dtype=torch.int32).reshape(1, -1)

    def match(sel_table, status, packed, full_table, kv_lens, q_lens, sparse_size, page_size):
        assert kv_lens.tolist() == [kv_len] and q_lens.tolist() == [1]
        assert torch.equal(packed.flatten()[:selected_count], logical)
        assert (packed.flatten()[selected_count:] == -1).all()
        packed.flatten()[:selected_count] = torch.arange(selected_count, dtype=torch.int32)
        return None, physical, selected_count

    def reference_sfa(
        owner, query, query_rope, selected, packed, meta, q_lens, kv_lens, *, block_table, sparse_mode, return_lse
    ):
        assert sparse_mode == 0 and return_lse
        assert kv_lens.tolist() == [3072] and packed.shape == (1, 1, 2048)
        k, r = selected
        valid = packed.flatten()[:selected_count].long()
        scores = query @ k[valid].T + query_rope @ r[valid].T
        return scores.softmax(-1) @ k[valid], None, None

    impl.avsg_transport.gather.side_effect = lambda _, q_len, src, count: (kv[src.long()], rope[src.long()])
    with (
        patch.object(torch.ops._C_ascend, "npu_avsg_match", side_effect=match, create=True),
        patch.object(DeviceOperator, "execute_sparse_flash_attention_process", side_effect=reference_sfa),
    ):
        actual = impl._execute_sparse_flash_attention_process(
            q, q_rope, (), indices, metadata, torch.tensor([1]), torch.tensor([kv_len])
        )
    keys = kv[physical.long()]
    scores = q @ keys.T + q_rope @ rope[physical.long()].T
    torch.testing.assert_close(actual, scores.softmax(-1) @ keys)
    assert impl.avsg_transport.gather.call_count == int(selected_count > 0)


@pytest.mark.parametrize("has_indexer", [False, True])
def test_main_kv_preprocessing_uses_registered_staging_and_preserves_indexer_cache(has_indexer):
    impl = object.__new__(AscendAvsgSFAImpl)
    impl.qk_rope_head_dim = 64
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.has_indexer = has_indexer
    indexer_cache = (torch.ones(4, 2, 1, 2), torch.ones(4, 2, 1, 1))
    impl.indexer = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=indexer_cache), num_cache_tensors=2)
    transport = impl.avsg_transport = Mock()
    transport.staging_cache = (torch.empty(1, 2, 1, 2), torch.empty(1, 2, 1, 1))
    remote_slots = torch.tensor([30, 7, -1])
    transport.staging_slots.return_value = torch.tensor([0, 1, -1])

    composed = impl._compose_sfa_kv_cache(())
    assert composed[:2] == transport.staging_cache
    assert composed[2:] == (indexer_cache if has_indexer else ())
    slots = impl._get_sfa_kv_slot_mapping(SimpleNamespace(slot_mapping=remote_slots, offload_dummy=False))
    transport.staging_slots.assert_called_once_with(remote_slots)
    assert slots.tolist() == [0, 1, -1] and remote_slots.tolist() == [30, 7, -1]
