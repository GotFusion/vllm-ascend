# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager GLM sparse attention over paged KV held in peer HBM."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
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

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        # In PIECEWISE, vllm::mla_forward must run eagerly even during warmup.
        # A captured dummy would replay zero attention and skip the peer transfers.
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("AVSG attention cannot run inside ACLGraph capture; keep vllm::mla_forward eager")
        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def _compose_sfa_kv_cache(self, kv_cache):
        # All main-KV preprocessing paths write the registered staging pair.
        # The base class appends the layer's separately allocated indexer cache.
        return super()._compose_sfa_kv_cache(self.avsg_transport.staging_cache)

    def _get_sfa_kv_slot_mapping(self, attn_metadata: AscendSFAMetadata) -> torch.Tensor:
        if attn_metadata.offload_dummy:
            # A padded graph warmup has no physical rows to publish.
            return torch.full_like(attn_metadata.slot_mapping, -1)
        return self.avsg_transport.staging_slots(attn_metadata.slot_mapping)

    def _on_kv_cache_written(self, kv_cache, attn_metadata: AscendSFAMetadata) -> None:
        if attn_metadata.offload_dummy or not attn_metadata.num_actual_tokens:
            return
        slots = attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens]
        self.avsg_transport.publish_staged(self.avsg_layer_id, slots)

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
