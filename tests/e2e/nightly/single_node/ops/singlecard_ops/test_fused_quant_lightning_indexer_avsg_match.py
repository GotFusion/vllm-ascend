# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fused FP8 LI + AVSG against official LI and a CPU replay of its physical copy plan.

Running this module directly on an idle 950 records eager NPU event timings for all six hot-cache cases.
"""

import json
import statistics
from importlib import import_module

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import bootstrap_custom_op_env

PAGE = 128
TOPK = 2048
HEADS = 32
DIM = 128
TOKEN_MASK = 0xFFFFFF
CASES = [(1, 1, 4096), (8, 1, 4096), (1, 1, 8192), (8, 1, 8192), (1, 4, 8192), (8, 4, 8192)]

pytestmark = [
    pytest.mark.skipif(not is_950(), reason="FP8 LI + AVSG KVMatch requires Ascend 950"),
    pytest.mark.e2e_coverage(
        arch="", feature="", parallel="", deploy="", hardware="A5", quantization="fp8", graph_mode="eager"
    ),
]


def load_fused():
    bootstrap_custom_op_env()
    import_module("vllm_ascend.vllm_ascend_C")
    return torch.ops._C_ascend.npu_fused_quant_lightning_indexer_avsg_match


def physical_rows(pages, logical):
    return pages[logical // PAGE].long() * PAGE + logical % PAGE


class Case:
    def __init__(self, batch, lanes, capacity, length, steps, *, key_lengths=None):
        self.batch = batch
        self.lanes = lanes
        self.capacity = capacity
        self.length = length
        self.fused = load_fused()
        generator = torch.Generator().manual_seed(1024)
        pages = batch * ((length + PAGE - 1) // PAGE)
        self.full_pages = torch.randperm(pages, dtype=torch.int32, generator=generator).view(batch, -1)
        self.hot_pages = torch.randperm(batch * capacity // PAGE, dtype=torch.int32, generator=generator).view(
            batch, -1
        )
        self.full_table = self.full_pages.npu()
        self.hot_table = self.hot_pages.npu()
        self.q_ends = torch.arange(lanes, batch * lanes + 1, lanes, dtype=torch.int32).npu()
        self.key_lengths = [length] * batch if key_lengths is None else key_lengths
        self.k_lens = torch.tensor(self.key_lengths, dtype=torch.int32, device="npu")
        self.status = torch.full((batch, 2 * capacity), -1, dtype=torch.int32, device="npu")

        key = torch.randn((pages, PAGE, 1, DIM), generator=generator, dtype=torch.bfloat16)
        self.key = key.to(torch.float8_e4m3fn).npu()
        key_scale = 0.9 + 0.2 * torch.rand((pages, PAGE, 1), generator=generator)
        # Ensure the MTP tail is selected so clearing uncommitted rows is observable.
        for b, k_len in enumerate(self.key_lengths):
            tail = torch.arange(k_len - lanes, k_len)
            key_scale[self.full_pages[b, tail // PAGE].long(), tail % PAGE, :] = 32.0
        self.key_scale = key_scale.npu()
        shape = (batch * lanes, HEADS)
        self.query_scale = (0.9 + 0.2 * torch.rand(shape, generator=generator)).npu()
        self.weights = (0.5 + torch.rand(shape, generator=generator)).bfloat16().npu()
        self.queries = [
            torch.randn((*shape, DIM), generator=generator).to(torch.float8_e4m3fn).npu() for _ in range(steps)
        ]

    def run(self, query):
        return self.fused(
            query,
            self.key,
            self.weights,
            self.query_scale,
            self.key_scale,
            self.q_ends,
            self.k_lens,
            self.full_table,
            self.hot_table,
            self.status,
        )

    def reference(self, query):
        indices = torch_npu.npu_quant_lightning_indexer(
            query=query,
            key=self.key,
            weights=self.weights,
            query_dequant_scale=self.query_scale,
            key_dequant_scale=self.key_scale,
            actual_seq_lengths_query=self.q_ends,
            actual_seq_lengths_key=self.k_lens,
            block_table=self.full_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=TOPK,
            sparse_mode=3,
        )
        if isinstance(indices, (tuple, list)):
            indices = indices[0]
        return indices.cpu().long().reshape(self.batch, self.lanes, TOPK)


def check_step(case, query, memory):
    expected = case.reference(query)
    slots, hits, plan, counts = (output.cpu() for output in case.run(query))
    rows = case.batch * case.lanes
    assert slots.shape == (rows, 1, TOPK) and hits.shape == (case.batch,)
    assert plan.shape == (rows, 1, TOPK, 2) and counts.shape == (rows, 1)
    assert all(output.dtype == torch.int32 for output in (slots, hits, plan, counts))
    slots = slots.long().reshape_as(expected)
    plan = plan.long().reshape(case.batch, case.lanes, TOPK, 2)
    counts = counts.reshape(case.batch, case.lanes)
    assert ((slots >= 0) & (slots < case.capacity)).all()
    assert ((counts >= 0) & (counts <= TOPK)).all()
    assert torch.equal(hits, case.lanes * TOPK - counts.sum(dim=-1))
    status = case.status.cpu().view(case.batch, case.capacity, 2)
    for b in range(case.batch):
        hot = physical_rows(case.hot_pages[b], torch.arange(case.capacity))
        expected_source = physical_rows(case.full_pages[b], expected[b])
        for lane in range(case.lanes):
            count = int(counts[b, lane])
            src, dst = plan[b, lane, :count].unbind(-1)
            assert len(src.unique()) == len(dst.unique()) == count
            assert torch.isin(src, expected_source[lane]).all()
            assert torch.isin(dst, hot).all()
            assert (plan[b, lane, count:] == -1).all()
            memory[dst] = src

        # Check all lanes *after* the last write: a later lane must not evict an earlier lane's top-k.
        actual_source = memory[hot[slots[b]]]
        torch.testing.assert_close(actual_source, expected_source)
        tokens = status[b, :, 0] & TOKEN_MASK
        valid = tokens != TOKEN_MASK
        last_committed = case.key_lengths[b] - case.lanes
        assert (tokens[valid] <= last_committed).all()
        if case.lanes > 1:
            assert (expected[b] > last_committed).any()
        torch.testing.assert_close(memory[hot[valid]], physical_rows(case.full_pages[b], tokens[valid]))
        memory[hot[~valid]] = -1
    return hits


@pytest.mark.parametrize(("batch", "lanes", "capacity"), CASES)
def test_persistent_match(batch, lanes, capacity):
    case = Case(batch, lanes, capacity, length=16 * 1024, steps=2)
    memory = torch.full((batch * capacity,), -1, dtype=torch.long)
    check_step(case, case.queries[0], memory)
    hits = check_step(case, case.queries[0], memory)
    assert (hits > 0).all()
    check_step(case, case.queries[1], memory)


@pytest.mark.parametrize("lanes", [1, 4])
def test_ragged_key_tiles_keep_order_and_state(lanes):
    # Uneven requests exercise the first/last tile of each score producer.
    # The first request's earliest query sees exactly one full top-k.
    lengths = [TOPK + lanes - 1, 8191, 8193, 32767, 65533]
    case = Case(len(lengths), lanes, 8192, length=max(lengths), steps=2, key_lengths=lengths)
    memory = torch.full((case.batch * case.capacity,), -1, dtype=torch.long)
    for query in (case.queries[0], case.queries[1], case.queries[0]):
        check_step(case, query, memory)


@pytest.mark.parametrize("invalid", ["q_end", "key_too_long", "key_too_short"])
def test_bad_lengths_preserve_status(invalid):
    case = Case(batch=8, lanes=1, capacity=4096, length=16 * 1024, steps=1)
    case.run(case.queries[0])
    before = case.status.clone()
    assert (before != -1).any()
    if invalid == "q_end":
        case.q_ends[1] = case.q_ends[0]
    elif invalid == "key_too_long":
        case.k_lens[-1] = case.length + 1
    else:
        case.k_lens[0] = TOPK - 1
    _, hits, _, counts = case.run(case.queries[0])
    assert (hits.cpu() == -1).all() and (counts.cpu() == -1).all()
    assert torch.equal(case.status.cpu(), before.cpu())


def benchmark():
    if not is_950():
        raise RuntimeError("A 950 device is required for NPU event timing")
    for batch, lanes, capacity in CASES:
        case = Case(batch, lanes, capacity, length=64 * 1024, steps=12)
        for query in case.queries:
            case.run(query)
        torch.npu.synchronize()
        events = []
        for i in range(48):
            query = case.queries[i % len(case.queries)]
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            _, hits, _, counts = case.run(query)
            end.record()
            events.append((start, end))
        torch.npu.synchronize()
        assert (hits.cpu() >= 0).all() and (counts.cpu() >= 0).all()
        samples = [start.elapsed_time(end) * 1000 for start, end in events]
        print(
            json.dumps(
                {
                    "batch": batch,
                    "queries": lanes,
                    "capacity": capacity,
                    "steady_p50_us": statistics.median(samples),
                    "samples_us": samples,
                }
            )
        )


if __name__ == "__main__":
    benchmark()
