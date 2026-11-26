# SPDX-License-Identifier: Apache-2.0
"""CPU transport doubles exercise the real PD ingress, reader and worker hooks."""

import ctypes
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
import zmq
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.avsg_ingress import AvsgPDIngress
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    MF_META,
    MF_META_ACK,
    READ_DONE,
    READ_FAILED,
    READ_READY_BATCH,
    SFAPD_PROTOCOL_VERSION,
    SfaPDConsumerMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import MembPullReadThread
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HConsumerWorker

MAIN_BLOCK_TOKENS = 4
NUM_BLOCKS = 8
LAYER_NAMES = ("model.layers.0.self_attn", "model.layers.2.self_attn")


class _Peer:
    block_size = 2
    remote_block_capacity = NUM_BLOCKS * MAIN_BLOCK_TOKENS // block_size

    def __init__(self):
        self.staging_cache = (
            torch.empty(1, 2, 1, 3, dtype=torch.bfloat16),
            torch.empty(1, 2, 1, 2, dtype=torch.bfloat16),
        )
        self.kv = {layer: torch.zeros(NUM_BLOCKS * MAIN_BLOCK_TOKENS, 3, dtype=torch.bfloat16) for layer in (0, 2)}
        self.rope = {layer: torch.zeros(NUM_BLOCKS * MAIN_BLOCK_TOKENS, 2, dtype=torch.bfloat16) for layer in (0, 2)}
        self.published = []

    def publish_received(self, layer_id, kv_rows, rope_rows, slots):
        self.kv[layer_id][slots] = kv_rows
        self.rope[layer_id][slots] = rope_rows
        self.published.append((layer_id, slots.tolist(), threading.get_ident()))


class _MemoryCopyEngine:
    def __init__(self):
        self.calls = []
        self.fail = False

    def batch_transfer_sync_read(self, session, local, peer, lengths):
        self.calls.append((session, list(local), list(peer), list(lengths)))
        if self.fail:
            return -1
        for destination, source, length in zip(local, peer, lengths):
            ctypes.memmove(destination, source, length)
        return 0


def _wait_pending(ingress):
    deadline = time.monotonic() + 5
    while ingress._queue.empty():
        if time.monotonic() > deadline:
            raise AssertionError("read thread did not enqueue main ingress")
        time.sleep(0.001)


@pytest.fixture
def ingress():
    engine, peer = _MemoryCopyEngine(), _Peer()
    ingress = AvsgPDIngress(engine, {name: (layer, peer) for name, layer in zip(LAYER_NAMES, (0, 2))}, 4, NUM_BLOCKS)
    yield ingress, peer, engine
    ingress.close()


def _source(layer=0):
    kv = torch.arange(NUM_BLOCKS * 4 * 3, dtype=torch.float32).view(NUM_BLOCKS, 4, 3) + 1000 * layer
    rope = torch.arange(NUM_BLOCKS * 4 * 2, dtype=torch.float32).view(NUM_BLOCKS, 4, 2) + 2000 * layer
    return kv.bfloat16(), rope.bfloat16()


def test_main_pages_are_bounded_and_published_on_the_model_thread(ingress):
    receiver, peer, engine = ingress
    with ThreadPoolExecutor(1) as pool:
        for layer, name in zip((0, 2), LAYER_NAMES):
            kv, rope = _source(layer)
            future = pool.submit(receiver.read_main, name, "prefill", (kv.data_ptr(), rope.data_ptr()), [6, 2], [1, 5])
            _wait_pending(receiver)
            assert not future.done()
            assert len(engine.calls) == 2 * (layer // 2)
            receiver.drain()
            future.result(timeout=5)
            torch.testing.assert_close(peer.kv[layer][4:8], kv[6])
            torch.testing.assert_close(peer.kv[layer][20:24], kv[2])
            torch.testing.assert_close(peer.rope[layer][20:24], rope[2])
    assert receiver.kv_rows.shape == (4, 3)
    assert receiver.rope_rows.shape == (4, 2)
    assert {entry[2] for entry in peer.published} == {threading.get_ident()}
    assert len({tuple(call[1]) for call in engine.calls}) == 1


@pytest.mark.parametrize("failure", ["memfabric", "avsg"])
def test_failed_main_transfer_releases_waiter_without_acknowledging_success(ingress, failure):
    receiver, peer, engine = ingress
    kv, rope = _source()
    if failure == "memfabric":
        engine.fail = True
    else:
        peer.publish_received = Mock(side_effect=RuntimeError("peer write failed"))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(receiver.read_main, LAYER_NAMES[0], "prefill", (kv.data_ptr(), rope.data_ptr()), [0], [1])
        _wait_pending(receiver)
        with pytest.raises(RuntimeError):
            receiver.drain()
        with pytest.raises(RuntimeError, match="AVSG PD main transfer failed"):
            future.result(timeout=5)
    assert not peer.published


def test_shutdown_cancels_queued_main_read(ingress):
    receiver, _, engine = ingress
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(receiver.read_main, LAYER_NAMES[0], "prefill", (100, 200), [0], [1])
        _wait_pending(receiver)
        receiver.close()
        with pytest.raises(RuntimeError, match="AVSG PD main transfer failed"):
            future.result(timeout=5)
        receiver.drain()
    assert not engine.calls


def test_shutdown_cannot_release_a_source_during_an_active_dma(ingress):
    receiver, _, engine = ingress
    entered, release = threading.Event(), threading.Event()

    def blocking_read(*args):
        entered.set()
        assert release.wait(5)
        return -1

    engine.batch_transfer_sync_read = blocking_read
    with ThreadPoolExecutor(2) as pool:
        reading = pool.submit(receiver.read_main, LAYER_NAMES[0], "prefill", (100, 200), [0], [1])
        _wait_pending(receiver)
        draining = pool.submit(receiver.drain)
        assert entered.wait(5)
        receiver.close()
        assert not reading.done()
        release.set()
        for future in (draining, reading):
            with pytest.raises(RuntimeError):
                future.result(timeout=5)


@pytest.mark.parametrize("source,destination", [([-1], [0]), ([0], [-1]), ([0], [NUM_BLOCKS]), ([0], [])])
def test_invalid_main_page_table_is_rejected_before_submission(ingress, source, destination):
    receiver, _, engine = ingress
    with pytest.raises(ValueError):
        receiver.read_main(LAYER_NAMES[0], "prefill", (100, 200), source, destination)
    assert receiver._queue.empty()
    assert not engine.calls


@pytest.fixture
def consumer(monkeypatch):
    peer, engine = _Peer(), _MemoryCopyEngine()
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_rank=0),
        kv_transfer_config=SimpleNamespace(kv_port=24000, kv_connector_extra_config={}),
        compilation_config=SimpleNamespace(
            static_forward_context={
                name: SimpleNamespace(impl=SimpleNamespace(avsg_layer_id=layer, avsg_transport=peer))
                for name, layer in zip(LAYER_NAMES, (0, 2))
            }
        ),
    )
    groups = [
        SimpleNamespace(layer_names=list(LAYER_NAMES), kv_cache_spec=SimpleNamespace(block_size=4)),
        SimpleNamespace(
            layer_names=[f"{name}.indexer" for name in LAYER_NAMES], kv_cache_spec=SimpleNamespace(block_size=2)
        ),
    ]
    module = "vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker"
    monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(f"{module}.get_ip", lambda: "127.0.0.1")
    worker = SFAPDRD2HConsumerWorker(config, True, SimpleNamespace(num_blocks=NUM_BLOCKS, kv_cache_groups=groups))

    def ensure_engine():
        worker.engine = engine
        return engine

    def start_reader():
        worker._mf_read_thread = MembPullReadThread(0, 24000, engine, worker._build_consumer_read_state())

    monkeypatch.setattr(worker, "_ensure_engine", ensure_engine)
    monkeypatch.setattr(worker, "_start_memfabric_read_thread", start_reader)
    caches = {name: (torch.empty(0), torch.empty(0)) for name in LAYER_NAMES}
    caches.update({f"{name}.indexer": (torch.zeros(NUM_BLOCKS, 2, 3),) for name in LAYER_NAMES})
    worker._register_avsg_memfabric_pull(caches)
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("request", [5, 1, 6], [2, 4, 7])
    worker.start_load_kv(metadata)
    yield worker, peer, engine, caches
    worker.shutdown()


def _p_metadata(kv, rope, indexer):
    return {
        "base_addrs": [kv.data_ptr(), rope.data_ptr(), indexer.data_ptr()],
        "block_len": [tensor[0].numel() * tensor.element_size() for tensor in (kv, rope, indexer)],
        "block_size_scale": [1, 1, 1],
        "main_tensor_count": 2,
        "has_indexer": True,
        "cache_dtypes": [str(t.dtype) for t in (kv, rope, indexer)],
    }


def test_real_reader_routes_main_to_peer_and_indexer_to_local_after_prefix(consumer):
    worker, peer, engine, caches = consumer
    kv, rope = _source()
    indexer = torch.arange(NUM_BLOCKS * 2 * 3, dtype=torch.float32).view(NUM_BLOCKS, 2, 3)
    name = LAYER_NAMES[0]
    read = worker._mf_read_thread
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            read._do_read_batch,
            name,
            [("request", [3, 0], [1, 6], 1, 1)],
            p_session="prefill",
            p_layer_meta={name: _p_metadata(kv, rope, indexer)},
        )
        _wait_pending(worker._avsg_ingress)
        assert worker.get_finished() == (set(), set())
        future.result(timeout=5)
    torch.testing.assert_close(peer.kv[0][4:8], kv[3])
    torch.testing.assert_close(peer.kv[0][24:28], kv[0])
    assert peer.kv[0][20:24].count_nonzero() == 0  # cached prefix main page is untouched
    local_indexer = caches[f"{name}.indexer"][0]
    torch.testing.assert_close(local_indexer[4], indexer[1])
    torch.testing.assert_close(local_indexer[7], indexer[6])
    assert local_indexer[2].count_nonzero() == 0
    assert all(t.numel() == 0 for t in caches[name])
    assert worker.offload_manager is None
    assert len(engine.calls) == 3  # two bounded main pages, then one indexer batch
    read._record_chunk_done(["request"], 0, 1)
    assert worker.get_finished() == (set(), {"request"})


def test_cancellation_retains_mapping_until_late_transfer_completion(consumer):
    worker, _, _, _ = consumer
    assert worker.get_finished({"request"}) == (set(), set())
    assert worker.request_map == {"request": "request"}
    assert worker._dest_blocks_by_req
    worker._mf_read_thread._record_chunk_done(["request"], 0, 1)
    assert worker.get_finished() == (set(), {"request"})
    assert not worker.request_map
    assert not worker._dest_blocks_by_req
    assert not worker._deferred_finished_req_ids


@pytest.mark.parametrize("component", ["main", "indexer"])
def test_reader_rejects_out_of_bounds_destination_before_dma(consumer, component):
    worker, _, engine, _ = consumer
    name = LAYER_NAMES[0]
    kv, rope = _source()
    indexer = torch.zeros(NUM_BLOCKS, 2, 3)
    worker._dest_blocks_by_req["request"] = ([NUM_BLOCKS], [0]) if component == "main" else ([0], [NUM_BLOCKS])
    with pytest.raises(ValueError, match="physical page table"):
        worker._mf_read_thread._do_read_batch(
            name,
            [("request", [0], [0], 0, 0)],
            p_session="prefill",
            p_layer_meta={name: _p_metadata(kv, rope, indexer)},
        )
    assert not engine.calls


def test_reader_rejects_unequal_tp_contributors(consumer):
    worker, _, engine, _ = consumer
    name = LAYER_NAMES[0]
    kv, rope = _source()
    indexer = torch.zeros(NUM_BLOCKS, 2, 3)
    with pytest.raises(ValueError, match="equal TP=1"):
        worker._mf_read_thread._do_read_batch(
            name,
            [("request", [0], [0], 0, 0)],
            p_session="prefill",
            p_layer_meta={name: _p_metadata(kv, rope, indexer)},
            ratio=2,
        )
    assert not engine.calls


def test_avsg_transfer_failure_never_falls_back_to_local_prefill(consumer):
    worker, _, _, _ = consumer
    worker._mf_read_thread._failed_requests.add("request")
    with pytest.raises(RuntimeError, match="AVSG remote prompt transfer failed"):
        worker.get_finished()


@pytest.mark.parametrize("has_sync_loads", [False, True])
def test_no_forward_progresses_ingress_while_requests_wait_for_remote_prefill(consumer, monkeypatch, has_sync_loads):
    worker, peer, _, _ = consumer
    name = LAYER_NAMES[0]
    kv, rope = _source()
    indexer = torch.zeros(NUM_BLOCKS, 2, 3)
    worker.request_map.clear()
    worker._dest_blocks_by_req.clear()
    metadata = SfaPDConsumerMetadata()
    metadata.add_request("request", [5, 1, 6], [2, 4, 7])
    connector = Mock()
    connector.bind_connector_metadata.side_effect = lambda value: setattr(connector, "metadata", value)
    connector.start_load_kv.side_effect = lambda *a, **kw: worker.start_load_kv(connector.metadata)

    def results(finished):
        sending, receiving = worker.get_finished(finished)
        return SimpleNamespace(finished_sending=sending, finished_recving=receiving, failed_recving=set())

    connector.get_transfer_results.side_effect = results
    connector.get_block_ids_with_load_errors.side_effect = worker.get_block_ids_with_load_errors
    connector.get_kv_connector_stats.return_value = None
    connector.get_kv_connector_kv_cache_events.return_value = None
    connector.build_connector_worker_meta.return_value = None
    active = object.__new__(ActiveKVConnector)
    active.kv_connector = connector
    active._disabled = False
    active._pending_load_kwargs = None
    active.vllm_config = worker.vllm_config
    module = "vllm.v1.worker.gpu.kv_connector"
    monkeypatch.setattr(f"{module}.is_forward_context_available", lambda: True)
    monkeypatch.setattr(f"{module}.get_forward_context", lambda: SimpleNamespace())
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=metadata, has_sync_kv_loads=has_sync_loads, finished_req_ids=set()
    )

    def remote_transfer():
        worker._mf_read_thread._do_read_batch(
            name,
            [("request", [3], [1], 1, 1)],
            p_session="prefill",
            p_layer_meta={name: _p_metadata(kv, rope, indexer)},
        )
        worker._mf_read_thread._record_chunk_done(["request"], 0, 1)

    with ThreadPoolExecutor(1) as pool:
        transfer = pool.submit(remote_transfer)
        deadline = time.monotonic() + 5
        while True:
            output = active.no_forward(scheduler_output)
            scheduler_output.kv_connector_metadata = SfaPDConsumerMetadata()
            result = output.kv_connector_output
            if result is not None and result.finished_recving == {"request"}:
                break
            if time.monotonic() > deadline:
                raise AssertionError("no_forward did not complete the waiting PD request")
            time.sleep(0.001)
        transfer.result(timeout=5)
    torch.testing.assert_close(peer.kv[0][4:8], kv[3])
    assert connector.finish_forward.called
    assert not connector.wait_for_save.called


@pytest.mark.parametrize("invalid", ["missing", "count", "main_fp16", "indexer", "scale"])
def test_reader_rejects_mismatched_cache_dtypes_before_dma(consumer, invalid):
    worker, _, engine, _ = consumer
    name = LAYER_NAMES[0]
    kv, rope = _source()
    indexer = torch.zeros(NUM_BLOCKS, 2, 3)
    meta = _p_metadata(kv, rope, indexer)
    if invalid == "missing":
        meta.pop("cache_dtypes")
    elif invalid == "count":
        meta["cache_dtypes"].pop()
    elif invalid == "main_fp16":
        meta["cache_dtypes"][0] = "torch.float16"
    elif invalid == "indexer":
        meta["cache_dtypes"][2] = "torch.int32"  # same bytes per element as float32
    else:
        scale = torch.zeros(NUM_BLOCKS, 2, 1)
        worker._mf_read_thread._state.indexer_scale_tensors[0] = scale
        meta["base_addrs"].append(scale.data_ptr())
        meta["block_len"].append(scale[0].numel() * scale.element_size())
        meta["block_size_scale"].append(1)
        meta["cache_dtypes"].append("torch.int32")
    with pytest.raises(RuntimeError, match="cache dtype mismatch"):
        worker._mf_read_thread._do_read_batch(
            name, [("request", [0], [0], 0, 0)], p_session="prefill", p_layer_meta={name: meta}
        )
    assert not engine.calls


@pytest.fixture
def wire(consumer):
    worker, peer, engine, caches = consumer
    with socket.socket() as temporary:
        temporary.bind(("127.0.0.1", 0))
        port = temporary.getsockname()[1]
    read = worker._mf_read_thread
    read.side_channel_port = port
    read._host = "127.0.0.1"
    read.start()
    assert read.ready_event.wait(5)
    assert read.startup_error is None
    context = zmq.Context()
    dealer = context.socket(zmq.DEALER)
    dealer.setsockopt(zmq.RCVTIMEO, 5000)
    dealer.connect(f"tcp://127.0.0.1:{port}")
    try:
        yield worker, peer, engine, caches, dealer
    finally:
        worker.shutdown()
        dealer.close(linger=0)
        context.term()


def _receive_message(dealer):
    return msgspec.msgpack.Decoder(type=tuple).decode(dealer.recv_multipart()[-1])


@pytest.mark.parametrize("failure", [False, True])
def test_wire_acknowledgement_waits_for_peer_publication(wire, failure):
    worker, peer, engine, caches, dealer = wire
    kv, rope = _source()
    indexer = torch.arange(NUM_BLOCKS * 2 * 3, dtype=torch.float32).view(NUM_BLOCKS, 2, 3)
    p_meta = {name: _p_metadata(kv, rope, indexer) for name in LAYER_NAMES}
    dealer.send(
        msgspec.msgpack.encode((MF_META, "prefill", msgspec.msgpack.encode(p_meta), SFAPD_PROTOCOL_VERSION, 0, 1))
    )
    assert _receive_message(dealer) == (MF_META_ACK, SFAPD_PROTOCOL_VERSION)
    dealer.send(
        msgspec.msgpack.encode((READ_READY_BATCH, 0, LAYER_NAMES[0], [("request", [3], [1], 1, 1)], ["request"], 0, 1))
    )
    _wait_pending(worker._avsg_ingress)
    assert not dealer.poll(timeout=0)
    if failure:
        engine.fail = True
        with pytest.raises(RuntimeError, match="MemFabric main read failed"):
            worker.get_finished()
        reply = _receive_message(dealer)
        assert reply[:2] == (READ_FAILED, 0)
        assert not peer.published
    else:
        worker.get_finished()
        assert _receive_message(dealer) == (READ_DONE, 0)
        torch.testing.assert_close(peer.kv[0][4:8], kv[3])
        torch.testing.assert_close(caches[f"{LAYER_NAMES[0]}.indexer"][0][4], indexer[1])


@pytest.mark.parametrize("bad_pp,missing_layer", [(True, False), (False, True)])
def test_wire_rejects_unsupported_prefill_topology(wire, bad_pp, missing_layer):
    _, _, _, _, dealer = wire
    kv, rope = _source()
    indexer = torch.zeros(NUM_BLOCKS, 2, 3)
    p_meta = {name: _p_metadata(kv, rope, indexer) for name in (LAYER_NAMES[:1] if missing_layer else LAYER_NAMES)}
    dealer.send(
        msgspec.msgpack.encode(
            (MF_META, "prefill", msgspec.msgpack.encode(p_meta), SFAPD_PROTOCOL_VERSION, 0, 2 if bad_pp else 1)
        )
    )
    message = _receive_message(dealer)
    assert message[:2] == (READ_FAILED, -1)
    assert "AVSG PD" in message[2]
