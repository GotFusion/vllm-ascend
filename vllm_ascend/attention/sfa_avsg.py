# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM sparse attention over paged KV held in peer HBM."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata
from vllm_ascend.device.device_op import DeviceOperator

if TYPE_CHECKING:
    from vllm_ascend.attention.avsg_transport import AvsgTransport


class AscendAvsgSFAImpl(AscendSFAImpl):
    store_kv_cache_on_peer = True
    # The worker owns one connection and one registered selection buffer per rank.
    avsg_transport: AvsgTransport

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avsg_layer_id = extract_layer_index(self.layer_name)

    def _in_full_decode(self) -> bool:
        if not is_forward_context_available():
            return False
        mode = get_forward_context().cudagraph_runtime_mode
        # Upstream passes NONE to the model during both FULL warmup and
        # capture. The V2 manager arms only the configured decode graph.
        transport = getattr(self, "avsg_transport", None)
        return mode == CUDAGraphMode.FULL or (
            mode == CUDAGraphMode.NONE and transport is not None and transport.decode_graph_capture
        )

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        if self._in_full_decode():
            # The base forward needs a real-shaped query even on inactive capture
            # warmup so the indexer, projections and peer operations all get recorded.
            if (
                attn_metadata is None
                or hidden_states.shape[0] != 1
                or attn_metadata.num_input_tokens != 1
                or attn_metadata.num_actual_tokens != 1
                or attn_metadata.seq_lens.numel() != 1
                or not self.avsg_transport.decode_block_table.shape[1]
            ):
                raise RuntimeError("AVSG FULL decode requires one request and one token with graph inputs")
        elif torch.npu.is_current_stream_capturing():
            raise RuntimeError("AVSG attention cannot run inside ACLGraph capture; keep vllm::mla_forward eager")
        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def _compose_sfa_kv_cache(self, kv_cache):
        # All main-KV preprocessing paths write the registered staging pair.
        # The base class appends the layer's separately allocated indexer cache.
        return super()._compose_sfa_kv_cache(self.avsg_transport.staging_cache)

    def _get_sfa_kv_slot_mapping(self, attn_metadata: AscendSFAMetadata) -> torch.Tensor:
        if self._in_full_decode():
            return self.avsg_transport.staging_slots(self.avsg_transport.decode_slot)
        if attn_metadata.offload_dummy:
            # A padded graph warmup has no physical rows to publish.
            return torch.full_like(attn_metadata.slot_mapping, -1)
        return self.avsg_transport.staging_slots(attn_metadata.slot_mapping)

    def _on_kv_cache_written(self, kv_cache, attn_metadata: AscendSFAMetadata) -> None:
        if self._in_full_decode():
            self.avsg_transport.publish_decode_graph(self.avsg_layer_id)
            return
        if attn_metadata.offload_dummy or not attn_metadata.num_actual_tokens:
            return
        slots = attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens]
        self.avsg_transport.publish_staged(self.avsg_layer_id, slots)

    def _execute_decode_graph(self, ql_nope, q_pe, topk_indices, attn_metadata):
        transport = self.avsg_transport
        selection_table = transport.selection_block_table
        selection_slots = selection_table.shape[1] * attn_metadata.block_size
        indices = topk_indices.reshape(1, 1, -1).to(torch.int32)
        valid = (transport.decode_slot >= 0) & (indices >= 0) & (indices < transport.decode_length)
        # Stably move valid indices to the prefix with a permutation of fixed
        # size. Match mutates its input; other layers reuse the indexer's tensor.
        prefix = valid.cumsum(-1, dtype=torch.int32)
        count = prefix[..., -1:]
        ordinal = torch.arange(indices.shape[-1], dtype=torch.int32, device=indices.device)
        destination = torch.where(valid, prefix - 1, count + ordinal - prefix)
        remapped = torch.empty_like(indices).scatter_(-1, destination.long(), torch.where(valid, indices, -1))
        # Match skips initialization when the logical length is zero. Length
        # one plus an all-padding top-k represents a completed, empty transfer.
        seq_len = transport.decode_length.clamp(1, transport.decode_block_table.shape[1] * attn_metadata.block_size)
        query_len = selection_table.new_ones(1)
        _, mapping, counts = torch.ops._C_ascend.npu_avsg_match(
            selection_table,
            selection_table.new_full((1, 2 * selection_slots), -1),
            remapped,
            transport.decode_block_table,
            seq_len,
            query_len,
            1,
            attn_metadata.block_size,
        )
        has_rows = count > 0
        selected = transport.gather_decode_graph(
            self.avsg_layer_id, mapping, torch.where(has_rows.view(1, 1), counts, 0)
        )
        # Give SFA one in-bounds row even for a no-op and discard that output.
        # This also masks NaNs from a never-written or previously used selection.
        sparse_indices = torch.where((~has_rows) & (ordinal == 0), 0, remapped)
        row_output, _, _ = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            selected,
            sparse_indices,
            attn_metadata,
            query_len,
            seq_len.clamp_max(selection_slots),
            block_table=selection_table,
            sparse_mode=0,
            return_lse=True,
        )
        return torch.where(has_rows, row_output, 0)

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        if self._in_full_decode():
            return self._execute_decode_graph(ql_nope, q_pe, topk_indices, attn_metadata)
        output = torch.zeros_like(ql_nope)
        if attn_metadata.offload_dummy or not attn_metadata.num_actual_tokens:
            return output

        # Submit queries serially so all requests and layers can share one registered selection
        # buffer. The adapter's checked copy and attention run on the current stream.
        full_table = attn_metadata.block_table if block_table is None else block_table
        selection_table = self.avsg_transport.selection_block_table
        block_size = attn_metadata.block_size
        selection_slots = selection_table.shape[1] * block_size
        status = selection_table.new_empty((1, 2 * selection_slots))
        query_len = selection_table.new_ones(1)
        # Both host tensors are already populated by the model runner. The device
        # KV lengths remain authoritative for masking and for AVSG Match.
        assert attn_metadata.query_start_loc_cpu is not None and attn_metadata.seq_lens_cpu is not None
        query_starts = attn_metadata.query_start_loc_cpu.tolist()
        for request, (start, stop) in enumerate(pairwise(query_starts)):
            stop = min(stop, attn_metadata.num_actual_tokens)
            if start >= stop:
                continue
            # Trim the full page table to live blocks before copying it to Match's UB.
            blocks = cdiv(int(attn_metadata.seq_lens_cpu[request]), block_size)
            row_table = full_table[request : request + 1, :blocks].contiguous()
            seq_len = actual_seq_lengths_key[request : request + 1].to(torch.int32)
            for row in range(start, stop):
                # All KV for this scheduled chunk has been published. Each query
                # can see its cached prefix and only its own prefix of this chunk.
                visible_len = seq_len - (stop - row - 1)
                indices = topk_indices[row : row + 1].reshape(1, 1, -1)
                # A5 SFA finds the valid prefix before the first -1. Pack it in
                # order on a private tensor because Match rewrites the entries
                # and the next shared-indexer layer needs the original indices.
                # Eager execution can synchronize to size this dynamic selection.
                valid = indices.masked_select((indices >= 0) & (indices < visible_len))
                if not valid.numel():
                    continue
                remapped = torch.full(indices.shape, -1, device=indices.device, dtype=torch.int32)
                remapped.view(-1)[: valid.numel()].copy_(valid)
                status.fill_(-1)
                _, mapping, counts = torch.ops._C_ascend.npu_avsg_match(
                    selection_table,
                    status,
                    remapped,
                    row_table,
                    visible_len,
                    query_len,
                    1,
                    block_size,
                )
                selected = self.avsg_transport.gather(self.avsg_layer_id, query_len, mapping, counts)
                # Causality was enforced in logical-token space above. After
                # remapping, mode 0 with LSE makes A5 bound softmax by the actual
                # valid prefix even when the KV length is greater than the top-k.
                row_output, _, _ = DeviceOperator.execute_sparse_flash_attention_process(
                    self,
                    ql_nope[row : row + 1],
                    q_pe[row : row + 1],
                    selected,
                    remapped,
                    attn_metadata,
                    query_len,
                    visible_len.clamp_max(selection_slots),
                    block_table=selection_table,
                    sparse_mode=0,
                    return_lse=True,
                )
                output[row : row + 1] = row_output
        return output
