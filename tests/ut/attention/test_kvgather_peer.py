# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle checks for the peer launcher; native HCOMM is replaced by a stub."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch


@pytest.mark.parametrize("failure", [None, "start", "health"])
def test_peer_publishes_shared_storage_and_releases_hcomm_first(tmp_path, monkeypatch, capsys, failure):
    events = []
    manifest = tmp_path / "manifest.json"
    config_path = tmp_path / "peer.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest": manifest.name,
                "device": 0,
                "local_engine": "peer",
                "local_comm_res": {},
                "cache": {
                    "layers": 2,
                    "blocks": 1,
                    "block_size": 128,
                    "kv_dim": 512,
                    "rope_dim": 64,
                    "dtype": "bfloat16",
                },
                "kvgather": {"host": "127.0.0.1", "port": 19780},
            }
        )
    )
    hixl = MagicMock()
    hixl.__enter__.return_value = hixl
    hixl.__exit__.side_effect = lambda *_: events.append("hixl_close")
    hixl.publish_layout.return_value = {"registered": True}
    urma_package = ModuleType("avsg_urma")
    urma_package.AvsgUrmaCacheServer = Mock(return_value=hixl)
    monkeypatch.setitem(sys.modules, "avsg_urma", urma_package)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    npu = SimpleNamespace(set_device=Mock())
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda shape, *, dtype, device: real_zeros(shape, dtype=dtype))

    gather = Mock()
    gather.stats = {
        "transfer_requests": 7,
        "transferred_rows": 35,
        "native_transfers": 14,
        "native_library_path": "/test/libascend_kv_gather.so",
    }

    def start():
        assert not manifest.exists()
        events.append("gather_start")
        if failure == "start":
            raise RuntimeError("start failed")

    def check_health():
        if manifest.exists():
            assert json.loads(manifest.read_text()) == {"registered": True}
        events.append("gather_health")
        if failure == "health":
            raise RuntimeError("health failed")

    gather.start.side_effect = start
    gather.check_health.side_effect = check_health
    gather.close.side_effect = lambda: events.append("gather_close")
    package = ModuleType("ascend_kv_gather")
    package.KVGatherPeer = Mock(return_value=gather)
    package.parse_endpoint = lambda options, base_dir: options
    monkeypatch.setitem(sys.modules, "ascend_kv_gather", package)
    spec = importlib.util.spec_from_file_location(
        "_test_kvgather_peer_launcher",
        Path(__file__).resolve().parents[3] / "csrc/avsg_urma/examples/d2d_cache_server.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    done = Mock()
    done.wait.side_effect = [False, True]
    monkeypatch.setattr(launcher.threading, "Event", Mock(return_value=done))
    monkeypatch.setattr(launcher.signal, "signal", Mock())
    monkeypatch.setattr(sys, "argv", ["d2d_cache_server.py", "--config", str(config_path)])
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            launcher.main()
    else:
        launcher.main()

    assert not manifest.exists()
    assert events[-2:] == ["gather_close", "hixl_close"]
    _, kv, rope = hixl.register_layers.call_args.args
    assert package.KVGatherPeer.call_args.args[0] is kv
    assert package.KVGatherPeer.call_args.args[1] is rope
    assert kv.shape == (2, 1, 128, 512)
    assert rope.shape == (2, 1, 128, 64)
    assert kv.dtype == rope.dtype == torch.bfloat16
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(records) == 1
    assert records[0]["event"] == "kvgather_peer_summary"
    assert records[0]["observed_before_close"] == gather.stats
    assert records[0]["status"] == ("failed" if failure else "closed")
    assert records[0]["healthy_before_close"] == (failure != "health")
    if failure:
        assert f"{failure} failed" in records[0]["session_error"]


def test_peer_summary_reports_cleanup_failure_without_hiding_counts(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    package = ModuleType("avsg_urma")
    package.AvsgUrmaCacheServer = Mock()
    monkeypatch.setitem(sys.modules, "avsg_urma", package)
    spec = importlib.util.spec_from_file_location(
        "_test_kvgather_peer_cleanup",
        Path(__file__).resolve().parents[3] / "csrc/avsg_urma/examples/d2d_cache_server.py",
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    peer = Mock()
    peer.stats = {"transfer_requests": 3, "transferred_rows": 11, "native_transfers": 6}
    peer.close.side_effect = RuntimeError("native destroy failed")
    with pytest.raises(RuntimeError, match="native destroy"):
        launcher.close_gather_peer(peer)
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "failed"
    assert result["healthy_before_close"] is True
    assert result["observed_before_close"] == peer.stats
    assert "native destroy failed" in result["error"]
