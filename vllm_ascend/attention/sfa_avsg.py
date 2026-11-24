# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager GLM sparse attention using a remote HBM mirror and AVSG selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata

if TYPE_CHECKING:
    from vllm_ascend.attention.avsg_transport import AvsgTransport


class AscendAvsgSFAImpl(AscendSFAImpl):
    # The worker owns one connection and one registered selection buffer per rank.
    avsg_transport: AvsgTransport

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avsg_layer_id = extract_layer_index(self.layer_name)

    def _on_kv_cache_written(self, kv_cache, attn_metadata: AscendSFAMetadata) -> None:
        if attn_metadata.offload_dummy or not attn_metadata.num_actual_tokens:
            return
        slots = attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens]
        self.avsg_transport.publish(self.avsg_layer_id, kv_cache[0], kv_cache[1], slots)

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
        if attn_metadata.offload_dummy or attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            return super()._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
            )

        # Each real request schedules one token in eager, non-speculative decode.
        # Submit requests serially so all layers can share one registered selection
        # buffer. The adapter's checked copy and attention run on the current stream.
        full_table = attn_metadata.block_table if block_table is None else block_table
        selection_table = self.avsg_transport.selection_block_table
        block_size = attn_metadata.block_size
        selection_slots = selection_table.shape[1] * block_size
        status = selection_table.new_empty((1, 2 * selection_slots))
        query_len = selection_table.new_ones(1)
        output = torch.zeros_like(ql_nope)
        # Unquantized GLM SFA already has a host mirror of the sequence lengths.
        # Trimming to live blocks avoids copying a max-model-length table to Match's UB.
        assert attn_metadata.seq_lens_cpu is not None
        seq_lens = attn_metadata.seq_lens_cpu[: attn_metadata.num_actual_tokens].tolist()
        for row, seq_len in enumerate(seq_lens):
            kv_len = actual_seq_lengths_key[row : row + 1].to(torch.int32)
            row_table = full_table[row : row + 1, : cdiv(seq_len, block_size)].contiguous()
            indices = topk_indices[row : row + 1].reshape(1, 1, -1)
            # Match rewrites logical indices to selection slots. Shared-indexer
            # layers still need the original logical top-k, including its padding.
            remapped = torch.where((indices >= 0) & (indices < kv_len), indices, -1).to(torch.int32).contiguous()
            # The scratch row changes owner between requests and between layers.
            status.fill_(-1)
            _, mapping, counts = torch.ops._C_ascend.npu_avsg_match(
                selection_table,
                status,
                remapped,
                row_table,
                kv_len,
                query_len,
                1,
                block_size,
            )
            selected = self.avsg_transport.gather(self.avsg_layer_id, query_len, mapping, counts)
            # For a single decode token, every selected slot is causally visible.
            # SFA's effective KV length must fit its supplied selection page table.
            selected_len = kv_len.clamp_max(selection_slots)
            output[row : row + 1] = super()._execute_sparse_flash_attention_process(
                ql_nope[row : row + 1],
                q_pe[row : row + 1],
                selected,
                remapped,
                attn_metadata,
                query_len,
                selected_len,
                selection_table,
            )
        return output
