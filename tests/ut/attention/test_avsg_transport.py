# SPDX-License-Identifier: Apache-2.0
"""Exercise the serving adapter with CPU tensors and an in-memory URMA peer."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.avsg_transport import AvsgTransport


class MemoryPeer:
    def __init__(self):
        self.session = Mock()
        self.session.bind_staging.side_effect = self.bind_staging
        self.session.prepare_layer.side_effect = self.prepare_layer
        self.session.prepare_graph_write.side_effect = self.prepare_write
        self.session.prepare_graph_copy.side_effect = self.prepare_read
        self.session.check_graph_copy.side_effect = self.check
        self.rows = {}
        self.failure = None

    def bind_staging(self, layer_id, source, kv, rope, selection_blocks):
        self.selection = kv[:selection_blocks], rope[:selection_blocks]
        self.source = kv[selection_blocks:], rope[selection_blocks:]
        return *self.selection, *self.source

    def prepare_layer(self, layer_id, remote):
        self.base = remote.kv_base
        self.slots = remote.blocks * remote.block_size
        self.rows[self.base] = (
            torch.zeros(self.slots, remote.kv_dim),
            torch.zeros(self.slots, remote.rope_dim),
        )

    def prepare_write(self, *, layer_id, source, mapping_capacity, workspace=None):
        kv, rope = self.source
        return rope, kv, self.base, None, None, torch.zeros(1), self.slots

    def prepare_read(self, *, layer_id, mapping_capacity, workspace=None):
        kv, rope = self.selection
        return rope, kv, self.base, None, None, torch.zeros(1)

    def transfer(self, direction, rope, kv, q, mapping, count, base, scratch, seen, status, *args):
        assert mapping.dtype == count.dtype == q.dtype == torch.int32
        active_groups = int(q[-1]) * count.shape[-1]
        assert active_groups <= count.numel()
        pairs = mapping.reshape(-1, mapping.shape[-2], 2)
        remote_kv, remote_rope = self.rows[base]
        local_kv, local_rope = kv.flatten(0, 1), rope.flatten(0, 1)
        for group, length in enumerate(count.flatten()[:active_groups]):
            for remote, local in pairs[group, :length].tolist():
                assert remote >= 0 and local >= 0
                if direction == "write":
                    remote_kv[remote].copy_(local_kv[local])
                    remote_rope[remote].copy_(local_rope[local])
                else:
                    local_kv[local].copy_(remote_kv[remote])
                    local_rope[local].copy_(remote_rope[remote])
        status.fill_(int(self.failure == direction))

    @staticmethod
    def check(status):
        if status.item():
            raise RuntimeError("peer reported a failed transfer")


@pytest.fixture
def transport(tmp_path, monkeypatch, request):
    peer = MemoryPeer()
    use_kvgather = getattr(request, "param", False)
    kv_dim, rope_dim, dtype = (512, 64, torch.bfloat16) if use_kvgather else (2, 1, torch.float16)
    # Load the vendored wire-format validator without importing its NPU extension.
    package = ModuleType("avsg_urma")
    monkeypatch.setitem(sys.modules, "avsg_urma", package)
    spec = importlib.util.spec_from_file_location(
        "avsg_urma.protocol", Path(__file__).resolve().parents[3] / "csrc/avsg_urma/avsg_urma/protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, protocol)
    spec.loader.exec_module(protocol)
    package.RemoteCache = protocol.RemoteCache
    package.AvsgUrmaSession = Mock(return_value=peer.session)
    ops = torch.ops.avsg_urma
    monkeypatch.setattr(ops, "write_graph_", lambda *args: peer.transfer("write", *args), raising=False)
    monkeypatch.setattr(ops, "copy_graph_", lambda *args: peer.transfer("read", *args), raising=False)
    if use_kvgather:
        # This tests adapter ownership/routing only. Native HCOMM and Match
        # prefix validation have separate package and two-card tests.
        gather_package = ModuleType("ascend_kv_gather")
        peer.gather_client = Mock()

        def gather(layer, q_lens, mapping, counts):
            kv, rope = peer.selection
            status = torch.zeros(1)
            peer.transfer("read", rope, kv, q_lens, mapping, counts, (layer + 1) * 8192, None, None, status)
            peer.check(status)
            return kv, rope

        peer.gather_client.gather.side_effect = gather
        peer.gather_factory = gather_package.KVGatherClient = Mock(return_value=peer.gather_client)
        gather_package.parse_endpoint = lambda options, base_dir: options
        monkeypatch.setitem(sys.modules, "ascend_kv_gather", gather_package)
    manifest = {}
    for layer in (0, 3):
        base = (layer + 1) * 8192
        manifest[str(layer)] = protocol.RemoteCache(
            kv_base=base,
            kv_bytes=256 * kv_dim * 2,
            blocks=2,
            block_size=128,
            kv_dim=kv_dim,
            kv_stride0=128 * kv_dim * 2,
            kv_stride1=kv_dim * 2,
            dtype=str(dtype).removeprefix("torch."),
            rope_base=base + 2048,
            rope_bytes=256 * rope_dim * 2,
            rope_dim=rope_dim,
            rope_stride0=128 * rope_dim * 2,
            rope_stride1=rope_dim * 2,
        ).to_dict()
    (tmp_path / "layers.json").write_text(json.dumps(manifest))
    (tmp_path / "rank2.json").write_text(
        json.dumps(
            {
                "local_engine": "model",
                "peer_engine": "cache",
                "local_comm_res": {},
                "remote_manifest": "layers.json",
                **({"kvgather": {"host": "localhost", "port": 19780}} if use_kvgather else {}),
            }
        )
    )

    def create_adapter(**overrides):
        kwargs = dict(
            endpoint_file=str(tmp_path / "rank{rank}.json"),
            rank=2,
            device=torch.device("cpu", 0),
            layer_ids=[0, 3],
            block_size=128,
            kv_dim=kv_dim,
            rope_dim=rope_dim,
            max_tokens=4,
            dtype=dtype,
        )
        kwargs.update(overrides)
        return AvsgTransport(**kwargs)

    peer.create_adapter = create_adapter
    adapter = create_adapter()
    yield adapter, peer
    adapter.close()


def paged_cache():
    # Both real cache components are strided views into a paged allocation.
    packed = torch.arange(256 * 3, dtype=torch.float16).reshape(2, 128, 1, 3)
    return packed[..., :2], packed[..., 2:]


def test_publish_and_gather_permuted_rows(transport):
    adapter, peer = transport
    kv, rope = paged_cache()
    slots = torch.tensor([129, -1, 2, 7])
    adapter.publish(0, kv, rope, slots)
    adapter.publish(3, -kv, -rope, slots)
    q = torch.tensor([1, 2], dtype=torch.int32)
    pairs = torch.tensor([[[[7, 6], [129, 131], [-1, -1]]], [[[2, 1], [-1, -1], [-1, -1]]]], dtype=torch.int32)
    counts = torch.tensor([[2], [1]], dtype=torch.int32)
    for layer, sign in ((0, 1), (3, -1)):
        selected_kv, selected_rope = adapter.gather(layer, q, pairs, counts)
        assert selected_kv.ndim == selected_rope.ndim == 4
        for selected, source in ((selected_kv, kv), (selected_rope, rope)):
            torch.testing.assert_close(selected.flatten(0, 1)[[6, 131, 1]], sign * source.flatten(0, 1)[[7, 129, 2]])
    assert peer.session.check_graph_copy.call_count == 4
    for remote_kv, remote_rope in peer.rows.values():
        assert remote_kv.count_nonzero() == 6
        assert remote_rope.count_nonzero() == 3


def test_padding_only_batch_preserves_remote_cache(transport):
    adapter, peer = transport
    kv, rope = paged_cache()
    adapter.publish(0, kv, rope, torch.tensor([-1, -1]))
    peer.session.check_graph_copy.assert_not_called()
    for remote_kv, remote_rope in peer.rows.values():
        assert remote_kv.count_nonzero() == remote_rope.count_nonzero() == 0


def test_publish_step_without_local_history(transport):
    adapter, peer = transport
    assert adapter.remote_block_capacity == 2
    kv, rope = adapter.staging_cache
    assert kv.shape == (1, 128, 1, 2)
    assert rope.shape == (1, 128, 1, 1)
    remote_slots = torch.tensor([255, -1, 128, 3])
    assert adapter.staging_slots(remote_slots).tolist() == [0, -1, 2, 3]
    for layer in (0, 3):
        kv.flatten(0, 1)[:4].copy_(torch.arange(8).reshape(4, 1, 2) + layer)
        rope.flatten(0, 1)[:4].copy_(torch.arange(4).reshape(4, 1, 1) + layer)
        adapter.publish_staged(layer, remote_slots)
    for layer in (0, 3):
        remote_kv, remote_rope = peer.rows[(layer + 1) * 8192]
        assert remote_kv[[255, 128, 3]].tolist() == [[layer, layer + 1], [layer + 4, layer + 5], [layer + 6, layer + 7]]
        assert remote_rope[[255, 128, 3], 0].tolist() == [layer, layer + 2, layer + 3]

    # Reuse step row 0 for a new token. A request can still gather the old
    # prefix in slot 255 after the layer and request occupying staging changed.
    kv.flatten(0, 1)[0].fill_(40)
    rope.flatten(0, 1)[0].fill_(50)
    adapter.publish_staged(0, torch.tensor([4]))
    selected_kv, selected_rope = adapter.gather(
        0,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[[[255, 0], [4, 2]]]], dtype=torch.int32),
        torch.tensor([[2]], dtype=torch.int32),
    )
    assert selected_kv.flatten(0, 1)[[0, 2], 0].tolist() == [[0, 1], [40, 40]]
    assert selected_rope.flatten(0, 1)[[0, 2], 0, 0].tolist() == [0, 50]
    assert kv.flatten(0, 1)[0, 0].tolist() == [40, 40]
    with pytest.raises(ValueError, match="capacity"):
        adapter.publish_staged(0, torch.ones(5, dtype=torch.int32))


@pytest.mark.parametrize("direction", ["write", "read"])
def test_failed_transfer_is_reported(transport, direction):
    adapter, peer = transport
    kv, rope = paged_cache()
    peer.failure = direction
    with pytest.raises(RuntimeError, match="failed transfer"):
        if direction == "write":
            adapter.publish(0, kv, rope, torch.tensor([2]))
        else:
            adapter.gather(
                0,
                torch.ones(1, dtype=torch.int32),
                torch.tensor([[[[2, 0]]]], dtype=torch.int32),
                torch.ones((1, 1), dtype=torch.int32),
            )
    peer.session.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.publish(0, kv, rope, torch.tensor([2]))


def test_pd_page_larger_than_decode_batch_preserves_prefix_and_layers(transport):
    adapter, peer = transport
    source_kv, source_rope = paged_cache()
    # A previously cached prefix belongs to page 0; import a new suffix into
    # page 1 using many small WRITE batches (the serving capacity is four).
    adapter.publish(0, source_kv, source_rope, torch.tensor([1, 2]))
    kv, rope = source_kv.flatten(0, 2)[128:], source_rope.flatten(0, 2)[128:]
    slots = torch.arange(255, 127, -1)
    for layer, sign in ((0, 1), (3, -1)):
        adapter.publish_received(layer, sign * kv, sign * rope, slots)
        received_kv, received_rope = peer.rows[(layer + 1) * 8192]
        torch.testing.assert_close(received_kv[slots], (sign * kv).float())
        torch.testing.assert_close(received_rope[slots], (sign * rope).float())
    torch.testing.assert_close(peer.rows[8192][0][[1, 2]], source_kv.flatten(0, 2)[[1, 2]].float())
    assert peer.session.check_graph_copy.call_count == 65


@pytest.mark.parametrize("bad_slot", [-1, 256, 2**32])
def test_pd_validates_entire_page_before_first_write(transport, bad_slot):
    adapter, peer = transport
    kv, rope = paged_cache()
    with pytest.raises(ValueError, match="destination slots"):
        adapter.publish_received(0, kv.flatten(0, 2)[:5], rope.flatten(0, 2)[:5], torch.tensor([1, 2, 3, 4, bad_slot]))
    peer.session.check_graph_copy.assert_not_called()


def test_pd_write_failure_stops_page_and_poison_transport(transport):
    adapter, peer = transport
    kv, rope = paged_cache()
    peer.failure = "write"
    with pytest.raises(RuntimeError, match="failed transfer"):
        adapter.publish_received(0, kv.flatten(0, 2)[:8], rope.flatten(0, 2)[:8], torch.arange(8))
    peer.session.check_graph_copy.assert_called_once()
    peer.session.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.publish_received(0, kv.flatten(0, 2)[:8], rope.flatten(0, 2)[:8], torch.arange(8))


def test_pd_rejects_layout_mismatch(transport):
    adapter, peer = transport
    kv, rope = paged_cache()
    with pytest.raises(ValueError, match="matching KV/RoPE"):
        adapter.publish_received(0, kv.flatten(0, 2)[:8], rope.flatten(0, 2)[:7], torch.arange(8))
    with pytest.raises(ValueError, match="no layer"):
        adapter.publish_received(1, kv.flatten(0, 2)[:8], rope.flatten(0, 2)[:8], torch.arange(8))
    peer.session.check_graph_copy.assert_not_called()


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_kvgather_keeps_hixl_publish_and_bounded_selection(transport, monkeypatch):
    adapter, peer = transport
    original_read = Mock(side_effect=AssertionError("KV gather must not use the HIXL READ kernel"))
    monkeypatch.setattr(torch.ops.avsg_urma, "copy_graph_", original_read)
    source_kv, source_rope = adapter.staging_cache
    source_kv.flatten(0, 2)[:3].fill_(11)
    source_rope.flatten(0, 2)[:3].fill_(19)
    adapter.publish_staged(3, torch.tensor([255, -1, 129]))
    pairs = torch.tensor([[[[255, 7], [129, 131], [-1, -1]]]], dtype=torch.int32)
    selected_kv, selected_rope = adapter.gather(
        3, torch.tensor([1], dtype=torch.int32), pairs, torch.tensor([[2]], dtype=torch.int32)
    )
    assert selected_kv.shape == (64, 128, 1, 512)
    assert selected_rope.shape == (64, 128, 1, 64)
    assert source_kv.shape == (1, 128, 1, 512)
    assert source_rope.shape == (1, 128, 1, 64)
    assert torch.all(selected_kv.flatten(0, 2)[[7, 131]] == 11)
    assert torch.all(selected_rope.flatten(0, 2)[[7, 131]] == 19)
    peer.session.check_graph_copy.assert_called_once()  # HIXL WRITE only.
    peer.session.prepare_graph_copy.assert_not_called()
    original_read.assert_not_called()
    options = peer.gather_factory.call_args.kwargs
    assert options["host"] == "localhost" and options["port"] == 19780
    assert options["layer_ids"] == [0, 3]
    assert options["layer_layouts"]["3"]["kv_base"] == 32768
    assert peer.gather_factory.call_args.args[0] is peer.selection[0]


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_kvgather_pd_ingress_keeps_hixl_write_order(transport):
    adapter, peer = transport
    kv = torch.full((9, 512), 5, dtype=torch.bfloat16)
    rope = torch.full((9, 64), 7, dtype=torch.bfloat16)
    slots = torch.arange(200, 209)
    adapter.publish_received(0, kv, rope, slots)
    assert peer.session.check_graph_copy.call_count == 3
    peer.gather_client.gather.assert_not_called()
    for received, expected in zip(peer.rows[8192], (kv, rope)):
        torch.testing.assert_close(received[slots], expected.float())


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_kvgather_failure_poison_transport_without_fallback(transport, monkeypatch):
    adapter, peer = transport
    fallback = Mock()
    monkeypatch.setattr(torch.ops.avsg_urma, "copy_graph_", fallback)
    peer.gather_client.gather.side_effect = RuntimeError("HCOMM completion failed")
    with pytest.raises(RuntimeError, match="HCOMM completion"):
        adapter.gather(
            0,
            torch.ones(1, dtype=torch.int32),
            torch.zeros((1, 1, 1, 2), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.int32),
        )
    peer.gather_client.close.assert_called_once()
    peer.session.close.assert_called_once()
    fallback.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.publish_staged(0, torch.zeros(1, dtype=torch.int32))


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_kvgather_initialization_failure_closes_hixl(transport):
    adapter, peer = transport
    adapter.close()
    peer.session.close.reset_mock()
    peer.gather_factory.side_effect = RuntimeError("HCOMM bootstrap failed")
    with pytest.raises(RuntimeError, match="HCOMM bootstrap"):
        peer.create_adapter()
    peer.session.close.assert_called_once()


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_kvgather_close_error_still_closes_hixl_and_can_retry_cleanup(transport):
    adapter, peer = transport
    peer.gather_client.close.side_effect = RuntimeError("HCOMM destroy failed")
    with pytest.raises(RuntimeError, match="HCOMM destroy"):
        adapter.close()
    peer.session.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.publish_staged(0, torch.zeros(1, dtype=torch.int32))
    peer.gather_client.close.side_effect = None
    adapter.close()
    adapter.close()
    assert peer.gather_client.close.call_count == 2
    peer.session.close.assert_called_once()
