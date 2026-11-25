# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.worker.v2 import aclgraph_utils as graphs


@pytest.fixture
def manager():
    manager = graphs.ModelAclGraphManager.__new__(graphs.ModelAclGraphManager)
    manager.avsg_layer_name = "main.0"
    manager.avsg_transport = Mock()
    manager.update_stream = object()
    manager.vllm_config = object()
    return manager


def test_avsg_binding_checked_before_capture(manager):
    runner = SimpleNamespace(cudagraph_manager=manager, avsg_decode_layer_names=("main.0",))
    graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, Mock())
    runner.avsg_decode_layer_names = ("other",)
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)
    runner.cudagraph_manager = None
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)


def test_avsg_capture_clears_stale_inputs_and_restores_scope_on_failure(manager, monkeypatch):
    transport = manager.avsg_transport
    transport.decode_graph_capture = False

    def capture(*args, **kwargs):
        transport.prepare_decode_replay.assert_called_once_with(None, None, None, active=False)
        assert transport.decode_graph_capture
        raise RuntimeError("capture failed")

    monkeypatch.setattr(graphs, "ModelWithContext", lambda model, **kwargs: model)
    monkeypatch.setattr(graphs, "communicator_switch", nullcontext)
    monkeypatch.setattr(graphs, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(graphs.ModelCudaGraphManager, "capture", capture)
    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(object(), SimpleNamespace(), None, None, None, [], None)
    assert not transport.decode_graph_capture


@pytest.mark.parametrize("dummy", [True, False])
def test_avsg_replay_checks_status_before_return(manager, monkeypatch, dummy):
    layer = SimpleNamespace(
        slot_mapping=object(), seq_lens=object(), block_table=object(), offload_dummy=dummy, num_actual_tokens=1
    )
    manager.model_runner = SimpleNamespace(attn_groups=[], model_state=SimpleNamespace(attn_metadata={"main.0": layer}))
    desc = SimpleNamespace(num_tokens=1, num_reqs=1)
    calls = Mock()
    calls.attach_mock(manager.avsg_transport.prepare_decode_replay, "prepare")
    calls.attach_mock(manager.avsg_transport.check_decode_graph_status, "check")
    manager._graph_relay = calls.replay
    monkeypatch.setattr(graphs, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(graphs, "_get_graph_update_backend", lambda *_: object())
    monkeypatch.setattr(graphs, "use_updatable_graph", lambda *_: False)
    assert manager.run_fullgraph(desc) is calls.replay.return_value
    assert [call[0] for call in calls.mock_calls] == ["prepare", "replay", "check"]
    calls.prepare.assert_called_once_with(layer.slot_mapping, layer.seq_lens, layer.block_table, active=not dummy)
    calls.check.side_effect = RuntimeError("failed peer READ")
    with pytest.raises(RuntimeError, match="failed peer READ"):
        manager.run_fullgraph(desc)
