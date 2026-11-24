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
        self.session.prepare_layer.side_effect = self.prepare_layer
        self.session.prepare_graph_write.side_effect = self.prepare_write
        self.session.prepare_graph_copy.side_effect = self.prepare_read
        self.session.check_graph_copy.side_effect = self.check
        self.rows = {}
        self.failure = None

    def prepare_layer(self, layer_id, remote):
        self.base = remote.kv_base
        self.slots = remote.blocks * remote.block_size
        self.rows[self.base] = (
            torch.zeros(self.slots, remote.kv_dim),
            torch.zeros(self.slots, remote.rope_dim),
        )

    def prepare_write(self, *, layer_id, source, mapping_capacity, workspace=None):
        _, kv, rope = self.session.bind_source.call_args.args
        return rope, kv, self.base, None, None, torch.zeros(1), self.slots

    def prepare_read(self, *, layer_id, mapping_capacity, workspace=None):
        _, kv, rope = self.session.bind_selection.call_args.args
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
def transport(tmp_path, monkeypatch):
    peer = MemoryPeer()
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
    manifest = {}
    for layer in (0, 3):
        base = (layer + 1) * 8192
        manifest[str(layer)] = protocol.RemoteCache(
            kv_base=base,
            kv_bytes=1024,
            blocks=2,
            block_size=128,
            kv_dim=2,
            kv_stride0=512,
            kv_stride1=4,
            dtype="float16",
            rope_base=base + 2048,
            rope_bytes=512,
            rope_dim=1,
            rope_stride0=256,
            rope_stride1=2,
        ).to_dict()
    (tmp_path / "layers.json").write_text(json.dumps(manifest))
    (tmp_path / "rank2.json").write_text(
        json.dumps(
            {
                "local_engine": "model",
                "peer_engine": "cache",
                "local_comm_res": {},
                "remote_manifest": "layers.json",
            }
        )
    )
    adapter = AvsgTransport(
        endpoint_file=str(tmp_path / "rank{rank}.json"),
        rank=2,
        device=torch.device("cpu", 0),
        layer_ids=[0, 3],
        block_size=128,
        kv_dim=2,
        rope_dim=1,
        max_tokens=4,
        dtype=torch.float16,
    )
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
