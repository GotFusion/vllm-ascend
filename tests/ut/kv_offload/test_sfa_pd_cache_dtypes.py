# SPDX-License-Identifier: Apache-2.0
"""Preserve component dtypes from producer allocation through the wire metadata."""

from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch

pytest.importorskip("vllm")

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import worker as worker_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import MF_META  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.send_thread import (  # noqa: E402
    MembPullSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import (  # noqa: E402
    SFAPDRD2HProducerWorker,
)


@pytest.mark.parametrize("indexer_format", ["shared", "bf16", "c8"])
def test_producer_registration_serializes_every_component_dtype(monkeypatch, indexer_format):
    main_name = "model.layers.0.self_attn.attn"
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"
    caches = {
        main_name: (
            torch.empty(4, 2, 1, 8, dtype=torch.bfloat16),
            torch.empty(4, 2, 1, 2, dtype=torch.bfloat16),
        )
    }
    groups = [SimpleNamespace(layer_names=[main_name])]
    if indexer_format != "shared":
        indexer = torch.empty(4, 2, 1, 4, dtype=torch.bfloat16 if indexer_format == "bf16" else torch.int8)
        caches[indexer_name] = (indexer,)
        if indexer_format == "c8":
            caches[indexer_name] += (torch.empty(4, 2, 1, 1, dtype=torch.float32),)
        groups.append(SimpleNamespace(layer_names=[indexer_name]))

    class SenderWithoutIO:
        def __init__(self, *, ready_event, state):
            self._state = state
            self.ready_event = ready_event
            self.startup_error = None

        def start(self):
            self.ready_event.set()

    monkeypatch.setattr(worker_module, "MembPullSendingThread", SenderWithoutIO)
    monkeypatch.setattr(
        worker_module, "global_memfabric_te", SimpleNamespace(unique_id="prefill", register_buffer=Mock())
    )
    monkeypatch.setattr(
        worker_module,
        "collect_storage_merged_register_regions",
        lambda _: SimpleNamespace(ptrs=[], lengths=[]),
    )
    monkeypatch.setattr(worker_module, "validate_register_region_count", lambda _: None)
    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker._backend = worker_module.BACKEND_MEMFABRIC
    worker.kv_cache_config = SimpleNamespace(num_blocks=2, kv_cache_groups=groups)
    worker.dspark_draft_layer_names = ()
    worker.dspark_draft_kv_metadata = {}
    worker.layer_metadata = {}
    worker.main_group_idx = 0
    worker.indexer_group_idx = len(groups) - 1
    worker.block_size = [4] * len(groups)
    worker.pp_rank = 0
    worker.pp_size = 1
    worker.register_kv_caches(caches)

    sender = worker.kv_send_layer_thread
    sender.timeout = 0.01
    sender._mf_meta_sent_paths = set()
    dealer = Mock()
    dealer.poll.return_value = True
    dealer.recv_multipart.return_value = [b"ACK"]
    MembPullSendingThread._send_mf_meta(sender, "tcp://decode:20050", dealer, msgspec.msgpack.Encoder())
    envelope = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    assert envelope[0] == MF_META
    layer = msgspec.msgpack.decode(envelope[2])[main_name]
    expected = ["torch.bfloat16", "torch.bfloat16"]
    if indexer_format == "bf16":
        expected.append("torch.bfloat16")
    elif indexer_format == "c8":
        expected.extend(["torch.int8", "torch.float32"])
    assert layer["cache_dtypes"] == expected
    assert len(layer["cache_dtypes"]) == len(layer["base_addrs"])
    assert layer["block_size_scale"] == [2] * len(expected)
