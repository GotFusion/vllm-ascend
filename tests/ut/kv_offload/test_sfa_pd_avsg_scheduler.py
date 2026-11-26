# SPDX-License-Identifier: Apache-2.0
"""PD admission must preserve already-published AVSG prefix pages."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("vllm")

from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs  # noqa: E402
from vllm.v1.kv_cache_layout import KVCacheLayout  # noqa: E402

from vllm_ascend.core.kv_cache_interface import (  # noqa: E402
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import scheduler as scheduler_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.scheduler import (  # noqa: E402
    SFAPDRD2HProducerScheduler,
    SFAPDRD2HScheduler,
)


@pytest.mark.parametrize("mixed_group", [False, True])
def test_peer_cache_detection_uses_planned_layer_specs(monkeypatch, mixed_group):
    main = AscendMLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_peer=True
    )
    indexer = AscendSFAIndexerCacheSpec(block_size=256, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    if mixed_group:
        specs = {"model.layers.0.attn": main, "model.layers.0.indexer": indexer}
        groups = [KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=specs))]
    else:
        groups = [
            KVCacheGroupSpec(["model.layers.0.attn"], main),
            KVCacheGroupSpec(["model.layers.0.indexer"], indexer),
        ]
    runtime = SimpleNamespace(
        additional_config={},
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(engine_id="decode", kv_port=1234),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC if mixed_group else KVCacheLayout.BLHNC,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        parallel_config=SimpleNamespace(data_parallel_rank=0, data_parallel_size=1, tensor_parallel_size=1),
    )
    plan = get_kv_cache_config_from_groups(runtime, groups, 4 * (main.page_size_bytes + indexer.page_size_bytes))
    monkeypatch.setattr(scheduler_module, "get_ip", lambda: "decode-host")
    monkeypatch.setattr(SFAPDRD2HScheduler, "_resolve_copy_sfa_hot_tokens", staticmethod(lambda: 0))
    monkeypatch.setattr(SFAPDRD2HScheduler, "_try_create_copy_sfa_slot_allocator", staticmethod(lambda _: None))
    scheduler = SFAPDRD2HScheduler(runtime, True, plan)
    try:
        assert scheduler._main_store_on_peer
    finally:
        scheduler.shutdown()


@pytest.fixture
def consumer():
    scheduler = SFAPDRD2HScheduler.__new__(SFAPDRD2HScheduler)
    scheduler.main_group_idx = 0
    scheduler.indexer_group_idx = 1
    scheduler.block_size = [128, 256]
    scheduler._main_store_on_peer = True
    scheduler.engine_id = "decode-engine"
    scheduler.side_channel_host = "decode-host"
    scheduler.side_channel_port = 1234
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        )
    )
    scheduler._request_trackers = {}
    scheduler._reqs_need_recv = set()
    scheduler._dspark_context_groups = ()
    scheduler._dspark_pending_recv = set()
    scheduler._avsg_pending_recv = set()
    scheduler._deferred_finished_req_ids = set()
    scheduler._metaserver_lock = threading.Lock()
    scheduler._cancelled_metaserver_requests = set()
    scheduler._metaserver_futures = {}
    scheduler._metaserver_retry_timers = {}
    scheduler._submit_metaserver_request = MagicMock()
    return scheduler


def make_request(prompt_tokens=517):
    return SimpleNamespace(
        request_id="prefix-request",
        prompt_token_ids=list(range(prompt_tokens)),
        all_token_ids=list(range(prompt_tokens)),
        # vLLM has not assigned the lookup hit at update_state_after_alloc.
        num_computed_tokens=0,
        kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://metaserver"},
    )


@pytest.mark.parametrize("prefix_tokens", [0, 256, 512])
def test_avsg_admission_advertises_accepted_prefix(consumer, prefix_tokens):
    request = make_request()
    external_tokens, asynchronous = consumer.get_num_new_matched_tokens(request, prefix_tokens)
    assert asynchronous
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13, 14], [20, 21, 22])

    consumer.update_state_after_alloc(request, blocks, external_tokens)

    message = consumer._submit_metaserver_request.call_args.kwargs["message"]
    assert message["remote_cached_tokens"] == prefix_tokens
    assert request.num_computed_tokens == 0
    assert request.kv_transfer_params["do_remote_prefill"] is False
    metadata = consumer.build_connector_meta(MagicMock())
    # Keep the whole logical table so P chunk offsets index the right suffix.
    assert metadata.requests[0].main_block_ids == [10, 11, 12, 13, 14]
    assert metadata.requests[0].indexer_block_ids == [20, 21, 22]


def test_avsg_local_only_admission_does_not_start_a_receive(consumer):
    request = make_request()
    blocks = MagicMock()

    consumer.update_state_after_alloc(request, blocks, num_external_tokens=0)

    consumer._submit_metaserver_request.assert_not_called()
    blocks.get_block_ids.assert_not_called()
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert consumer._reqs_need_recv == set()


@pytest.mark.parametrize("external_tokens", [-1, 518])
def test_avsg_rejects_invalid_admission_before_publishing_destinations(consumer, external_tokens):
    request = make_request()
    with pytest.raises(ValueError, match="external tokens"):
        consumer.update_state_after_alloc(request, MagicMock(), external_tokens)
    consumer._submit_metaserver_request.assert_not_called()
    assert consumer._request_trackers == {}


def test_host_copy_sfa_keeps_whole_prompt_notification(consumer):
    consumer._main_store_on_peer = False
    request = make_request()
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13, 14], [20, 21, 22])

    consumer.update_state_after_alloc(request, blocks, num_external_tokens=261)

    message = consumer._submit_metaserver_request.call_args.kwargs["message"]
    # Host Copy-SFA also initializes a fresh dense top-k row from P's KV.
    assert message["remote_cached_tokens"] == 0


def test_cancelled_avsg_admission_retains_rendezvous_and_destinations_until_receive(consumer):
    request = make_request()
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13, 14], [20, 21, 22])
    consumer.update_state_after_alloc(request, blocks, num_external_tokens=517)
    rendezvous = Future()
    consumer._metaserver_futures[request.request_id] = rendezvous

    assert consumer.request_finished_all_groups(request, blocks.get_block_ids()) == (True, None)
    assert not rendezvous.cancelled()
    assert request.request_id not in consumer._cancelled_metaserver_requests
    assert request.request_id in consumer._request_trackers
    assert request.request_id in consumer._avsg_pending_recv
    # Cancellation can precede the worker's first metadata delivery.
    assert consumer.build_connector_meta(MagicMock()).requests[0].req_id == request.request_id

    consumer.update_connector_output(SimpleNamespace(finished_recving=None))
    assert request.request_id in consumer._request_trackers
    assert not rendezvous.cancelled()

    consumer.update_connector_output(SimpleNamespace(finished_recving={request.request_id}))
    assert consumer._request_trackers == {}
    assert consumer._avsg_pending_recv == set()
    assert consumer._deferred_finished_req_ids == set()
    assert rendezvous.cancelled()


def test_completed_avsg_receive_can_release_normally(consumer):
    request = make_request()
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13, 14], [20, 21, 22])
    consumer.update_state_after_alloc(request, blocks, num_external_tokens=517)
    consumer.update_connector_output(SimpleNamespace(finished_recving={request.request_id}))

    # Receiving finishes before the user request, so keep the live lookup.
    assert request.request_id in consumer._request_trackers
    assert consumer._avsg_pending_recv == set()
    assert consumer.request_finished_all_groups(request, blocks.get_block_ids()) == (False, None)
    assert consumer._request_trackers == {}


def test_producer_keeps_decode_prefix_across_prefill_chunks(consumer):
    request = make_request()
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([10, 11, 12, 13, 14], [20, 21, 22])
    consumer.update_state_after_alloc(request, blocks, num_external_tokens=261)
    request.kv_transfer_params = consumer._submit_metaserver_request.call_args.kwargs["message"]

    cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=size)) for size in (128, 256)]
    )
    producer = SFAPDRD2HProducerScheduler(SimpleNamespace(), cache_config, "prefill-engine")
    producer.update_state_after_alloc(request, blocks, num_external_tokens=0)
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, num_computed_tokens=0)],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={request.request_id: 128},
    )
    first = producer.build_connector_meta(output).requests[request.request_id]
    assert first.remote_cache_tokens == 256
    assert first.local_computed_tokens == 128
    assert not first.chunk_finish

    output.scheduled_new_reqs = []
    output.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[request.request_id], new_block_ids=[None], num_computed_tokens=[128]
    )
    output.num_scheduled_tokens = {request.request_id: 389}
    final = producer.build_connector_meta(output).requests[request.request_id]
    assert final.remote_cache_tokens == 256
    assert final.local_transed_tokens == 128
    assert final.local_computed_tokens == 517
    assert final.chunk_finish
