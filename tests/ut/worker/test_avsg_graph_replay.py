# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm_ascend.worker.v2 import aclgraph_utils as graphs


@pytest.fixture
def manager():
    manager = graphs.ModelAclGraphManager.__new__(graphs.ModelAclGraphManager)
    manager.avsg_transport = Mock()
    manager.update_stream = object()
    manager.vllm_config = object()
    manager._bind_avsg_groups(
        {"main.0": 0, "main.1": 1, "main.2": 2, "shared.3": 3},
        [
            [SimpleNamespace(layer_names=["indexer.0", "indexer.1", "indexer.2"])],
            [SimpleNamespace(layer_names=["main.2", "main.0", "main.1"])],
            [SimpleNamespace(layer_names=["shared.3"])],
        ],
    )
    return manager


def test_shared_index_layers_use_their_own_main_kv_group(manager):
    manager.avsg_transport.bind_decode_groups.assert_called_once_with({0: 0, 1: 0, 2: 0, 3: 3})
    assert manager.avsg_group_sources == {0: "main.0", 3: "shared.3"}
    manager._bind_avsg_groups({"main.0": 0, "main.1": 1}, [[SimpleNamespace(layer_names=["main.0", "main.1"])]])
    assert manager.avsg_group_sources == {0: "main.0"}


@pytest.mark.parametrize("groups,error", [([["main.0"]], "missing"), ([["main.0"], ["main.1", "main.0"]], "ambiguous")])
def test_graph_rejects_missing_or_ambiguous_metadata_source(manager, groups, error):
    with pytest.raises(ValueError, match=error):
        manager._bind_avsg_groups(
            {"main.0": 0, "main.1": 1}, [[SimpleNamespace(layer_names=names)] for names in groups]
        )


def test_avsg_binding_checked_before_capture(manager):
    runner = SimpleNamespace(cudagraph_manager=manager, avsg_decode_layer_ids=dict(manager.avsg_layer_ids))
    graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, Mock())
    del runner.avsg_decode_layer_ids["shared.3"]
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)
    runner.cudagraph_manager = None
    with pytest.raises(RuntimeError, match="own the active transport"):
        graphs.ModelAclGraphManager.validate_avsg_binding(runner, manager.avsg_transport)


def test_avsg_capture_clears_stale_inputs_and_restores_scope_on_failure(manager, monkeypatch):
    transport = manager.avsg_transport
    transport.decode_graph_capture = False

    def capture(*args, **kwargs):
        assert transport.prepare_decode_replay.call_args_list == [
            call(None, None, None, active=False, group_id=0),
            call(None, None, None, active=False, group_id=3),
        ]
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
    def layer_metadata(page):
        return SimpleNamespace(
            slot_mapping=torch.tensor([page * 128 + 1]),
            seq_lens=torch.tensor([130]),
            block_table=torch.tensor([[0, page]]),
            offload_dummy=dummy,
            num_actual_tokens=1,
        )

    regular, shared = layer_metadata(4), layer_metadata(9)
    metadata = {"main.0": regular, "main.1": regular, "main.2": regular, "shared.3": shared}
    manager.model_runner = SimpleNamespace(attn_groups=[], model_state=SimpleNamespace(attn_metadata=metadata))
    desc = SimpleNamespace(num_tokens=1, num_reqs=1)
    calls = Mock()
    calls.attach_mock(manager.avsg_transport.prepare_decode_replay, "prepare")
    calls.attach_mock(manager.avsg_transport.check_decode_graph_status, "check")
    manager._graph_relay = calls.replay
    monkeypatch.setattr(graphs, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(graphs, "_get_graph_update_backend", lambda *_: object())
    monkeypatch.setattr(graphs, "use_updatable_graph", lambda *_: False)
    assert manager.run_fullgraph(desc) is calls.replay.return_value
    assert [invocation[0] for invocation in calls.mock_calls] == ["prepare", "prepare", "replay", "check"]
    assert calls.prepare.call_args_list == [
        call(regular.slot_mapping, regular.seq_lens, regular.block_table, active=not dummy, group_id=0),
        call(shared.slot_mapping, shared.seq_lens, shared.block_table, active=not dummy, group_id=3),
    ]
    calls.check.side_effect = RuntimeError("failed peer READ")
    with pytest.raises(RuntimeError, match="failed peer READ"):
        manager.run_fullgraph(desc)
