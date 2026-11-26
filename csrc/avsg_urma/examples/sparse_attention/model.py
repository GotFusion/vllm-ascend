# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Three sparse-attention layers with an FP8 KV-only remote cache."""

import torch
from torch import nn


NUM_LAYERS = 3
TOKEN_DIM = 656
FP8_DTYPE = torch.float8_e4m3fn
BLOCK_SIZE = 128
FULL_BLOCKS = 4
SELECTION_BLOCKS = 2
TOPK = 32


def token_rows(cache, slots):
    """Gather raw FP8 bytes, then reinterpret them for attention."""
    payload = cache.view(torch.uint8).reshape(-1, TOKEN_DIM)
    return payload.index_select(0, slots.reshape(-1).long()).view(*slots.shape, TOKEN_DIM)


def selection_slots(indices, table):
    logical = indices.clamp_min(0).long()
    pages = table.reshape(-1).index_select(0, (logical // BLOCK_SIZE).reshape(-1))
    return pages.reshape_as(logical) * BLOCK_SIZE + logical % BLOCK_SIZE


class SparseAttention(nn.Module):
    """Shared K/V projection, single head, residual, and no positional encoding."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False)
        self.kv = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False)
        self.output = nn.Linear(TOKEN_DIM, TOKEN_DIM, bias=False)

    def project(self, hidden):
        return self.query(hidden), self.kv(hidden).to(FP8_DTYPE)

    def attend(self, hidden, q, kv_bytes, valid):
        kv = kv_bytes.view(FP8_DTYPE).to(q.dtype)
        scores = torch.bmm(q.unsqueeze(1), kv.transpose(1, 2)).squeeze(1)
        scores = (scores.float() * (TOKEN_DIM ** -0.5)).masked_fill(~valid, float("-inf"))
        weights = scores.softmax(dim=-1).to(q.dtype)
        attended = torch.bmm(weights.unsqueeze(1), kv).squeeze(1)
        return hidden + self.output(attended)


def write_tokens(args, q, mapping, count):
    rope, source, context, scratch, seen, status, remote_slots = args
    torch.ops.avsg_urma.write_graph_(
        rope, source, q, mapping, count, context, scratch, seen, status, remote_slots,
    )


class AvsgSparseModel(nn.Module):
    def __init__(self, device, session, remote_layers):
        super().__init__()
        self.layers = nn.ModuleList(SparseAttention() for _ in range(NUM_LAYERS))
        self.layers.to(device=device, dtype=torch.bfloat16)

        self.full_table = torch.arange(FULL_BLOCKS, dtype=torch.int32, device=device).view(1, -1)
        self.selection_table = torch.arange(SELECTION_BLOCKS, dtype=torch.int32, device=device).view(1, -1)
        self.match_status = tuple(torch.full(
            (1, 2 * SELECTION_BLOCKS * BLOCK_SIZE), -1, dtype=torch.int32, device=device,
        ) for _ in self.layers)
        self.one_query = torch.ones(1, dtype=torch.int32, device=device)
        self.write_one = self.one_query.view(1, 1)
        self.staging_slot = torch.zeros_like(self.one_query)

        empty_rope = torch.empty(0, dtype=FP8_DTYPE, device=device)
        prefill_source = torch.empty((FULL_BLOCKS, BLOCK_SIZE, TOKEN_DIM), dtype=FP8_DTYPE, device=device)
        session.bind_source("prefill", prefill_source, empty_rope)
        for layer in range(NUM_LAYERS):
            selection = torch.empty((SELECTION_BLOCKS, BLOCK_SIZE, TOKEN_DIM), dtype=FP8_DTYPE, device=device)
            decode_source = torch.empty((1, BLOCK_SIZE, TOKEN_DIM), dtype=FP8_DTYPE, device=device)
            session.bind_selection(layer, selection)
            session.bind_source(f"decode-{layer}", decode_source, empty_rope)

        self.remote_layers = remote_layers

    def prepare_transfers(self, session):
        """Prepare fixed graph arguments after Connect."""
        self.prefill_writes, self.decode_writes, self.reads = [], [], []
        for layer, remote in enumerate(self.remote_layers):
            session.prepare_layer(layer, remote)
            self.prefill_writes.append(session.prepare_graph_write(
                layer_id=layer, source="prefill", mapping_capacity=FULL_BLOCKS * BLOCK_SIZE,
            ))
            self.decode_writes.append(session.prepare_graph_write(
                layer_id=layer, source=f"decode-{layer}", mapping_capacity=1,
            ))
            self.reads.append(session.prepare_graph_copy(layer_id=layer, mapping_capacity=TOPK))

    def prefill(self, tokens, topk, reference=None):
        """Write prompt KV to remote HBM while attending from the local staging cache."""
        length = tokens.size(0)
        slots = torch.arange(length, dtype=torch.int32, device=tokens.device)
        mapping = torch.stack((slots, slots), dim=-1).view(1, 1, length, 2)
        count = torch.full_like(self.write_one, length)
        hidden = tokens.to(torch.bfloat16)
        for layer, block in enumerate(self.layers):
            q, kv = block.project(hidden)
            args = self.prefill_writes[layer]
            stage = args[1].view(torch.uint8).reshape(-1, TOKEN_DIM)
            stage[:length].copy_(kv.view(torch.uint8))
            # WRITE pairs are [remote destination, local source].
            write_tokens(args, self.one_query, mapping, count)
            indices = topk[layer]
            selected = token_rows(args[1], indices.clamp_min(0))
            hidden = block.attend(hidden, q, selected, indices >= 0)
            if reference is not None:
                reference[layer][:length].copy_(stage[:length])
        return hidden[-1:].to(FP8_DTYPE)

    def decode(self, token, position, topk):
        """One full graph with per-layer WRITE -> Match -> READ -> Attention."""
        hidden = token.to(torch.bfloat16)
        kv_length = position + 1
        write_mapping = torch.stack((position, self.staging_slot), dim=-1).view(1, 1, 1, 2)
        remapped, misses = [], []
        for layer, block in enumerate(self.layers):
            q, kv = block.project(hidden)
            write = self.decode_writes[layer]
            write[1].view(torch.uint8).reshape(-1, TOKEN_DIM)[:1].copy_(kv.view(torch.uint8))
            write_tokens(write, self.one_query, write_mapping, self.write_one)

            indices = topk[layer].view(1, 1, TOPK).clone()
            _, mapping, count = torch.ops._C_ascend.npu_avsg_match(
                self.selection_table, self.match_status[layer], indices,
                self.full_table, kv_length, self.one_query,
                selection_topk_block_size=1, kv_block_size=BLOCK_SIZE,
            )
            rope, selection, context, scratch, seen, status = self.reads[layer]
            # READ pairs are [remote source, local destination]. READ waits for WRITE status.
            torch.ops.avsg_urma.copy_graph_(
                rope, selection, self.one_query, mapping, count, context, scratch, seen, status, write[-2],
            )
            indices = indices.reshape(1, TOPK)
            selected = token_rows(selection, selection_slots(indices, self.selection_table))
            hidden = block.attend(hidden, q, selected, indices >= 0)
            remapped.append(indices)
            misses.append(count)
        return hidden.to(FP8_DTYPE), tuple(remapped), tuple(misses)
