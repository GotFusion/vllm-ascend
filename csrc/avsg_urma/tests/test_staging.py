# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Exercise the HIXL registration and per-layer lookup contract with CPU tensor storage."""
import importlib.util
import sys
import threading
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import avsg_urma

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch is required for tensor storage checks")
class StagingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(avsg_urma.__file__).with_name("session.py")
        spec = importlib.util.spec_from_file_location("avsg_urma._test_session", path)
        cls.session = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch_npu": Mock(), spec.name: cls.session}), \
                patch.object(avsg_urma, "_native", Mock(), create=True):
            spec.loader.exec_module(cls.session)

    def setUp(self):
        def init(endpoint, **kwargs):
            endpoint.device = torch.device("cpu")
            endpoint._lock = threading.RLock()
            endpoint._owner = Mock()
            endpoint._closed = endpoint._poisoned = endpoint._connected = False

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(self.session._Endpoint, "__init__", init))
        stack.enter_context(patch.object(torch, "npu", SimpleNamespace(device=lambda _: nullcontext(),
                                                                       synchronize=lambda: None), create=True))

    def test_fixed_region_count_and_layer_addresses(self):
        for n in (1, 2, 78):
            with self.subTest(layers=n):
                peer = self.session.AvsgUrmaCacheServer()
                client = self.session.AvsgUrmaSession()
                banks = tuple(torch.empty(n, 2, 4, width, dtype=torch.bfloat16) for width in (8, 2))
                peer.register_layers(0, *banks)
                manifest = peer.publish_layout()
                staging = tuple(torch.zeros(5, 4, width, dtype=torch.bfloat16) for width in (8, 2))
                selected_kv, selected_rope, source_kv, source_rope = client.bind_staging(0, "step", *staging, 3)
                for selected, source, backing in zip((selected_kv, selected_rope), (source_kv, source_rope), staging):
                    self.assertEqual(selected.data_ptr(), backing.data_ptr())
                    self.assertEqual(source.data_ptr(), selected.data_ptr() + selected.nbytes)
                    self.assertEqual(selected.nbytes + source.nbytes, backing.nbytes)
                    source.fill_(2)
                    self.assertEqual(selected.count_nonzero().item(), 0)

                self.assertEqual(peer._owner.register_tensor.call_count, 2)
                self.assertEqual(client._owner.register_tensor.call_count, 2)
                for endpoint, tensors in ((peer, banks), (client, staging)):
                    for call, tensor in zip(endpoint._owner.register_tensor.call_args_list, tensors):
                        self.assertIs(call.args[0], tensor)

                client._owner.connect.return_value = [1]
                client.connect("peer")

                def get_addr(base, size):
                    self.assertTrue(any(bank.data_ptr() <= base and base + size <= bank.data_ptr() + bank.nbytes
                                        for bank in banks))
                    return base

                client._owner.remote_address.side_effect = get_addr
                for layer in range(n):
                    client.prepare_layer(0, manifest[str(layer)])
                    for prefix, bank in zip(("kv", "rope"), banks):
                        view = bank[layer]
                        self.assertEqual(manifest[str(layer)][f"{prefix}_base"], view.data_ptr())
                        self.assertEqual(manifest[str(layer)][f"{prefix}_bytes"], view.nbytes)
                        client._owner.remote_address.assert_any_call(view.data_ptr(), view.nbytes)

    def test_bad_layout_leaves_no_registrations(self):
        peer = self.session.AvsgUrmaCacheServer()
        kv = torch.empty((2, 1, 4, 8), dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            peer.register_layers(0, kv, kv[:1])
        peer._owner.register_tensor.assert_not_called()
        client = self.session.AvsgUrmaSession()
        with self.assertRaises(ValueError):
            client.bind_staging(0, "step", kv[0], None, 1)
        client._owner.register_tensor.assert_not_called()

    def test_auxiliary_registration_does_not_publish_an_extra_layer(self):
        peer = self.session.AvsgUrmaCacheServer()
        data = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
        peer.register_layer(3, data)
        auxiliary = torch.empty_like(data)
        peer.register_auxiliary_cache(auxiliary)
        self.assertIs(peer._owner.register_tensor.call_args.args[0], auxiliary)
        self.assertEqual(set(peer.publish_layout()), {"3"})
        with self.assertRaisesRegex(ValueError, "before publishing"):
            peer.register_auxiliary_cache(auxiliary)
        self.assertEqual(peer._owner.register_tensor.call_count, 2)

    def test_check_graph_copy_reports_per_channel_context_detail(self):
        client = self.session.AvsgUrmaSession(channels=2)
        client._channels = (1, 2)
        status = torch.zeros((2, self.session.STATUS_BYTES // 8), dtype=torch.int64)
        status[:, 1] = 4
        status[:, 2] = torch.tensor([2, 3])
        self.assertEqual(client.check_graph_copy(status), 5)
        status[1, :5] = torch.tensor([100, 1, 0, 0, 3])
        with self.assertRaisesRegex(RuntimeError, r"jetty=1, code=100, phase=1.*Context check 3: local KV address"):
            client.check_graph_copy(status)
        with self.assertRaisesRegex(RuntimeError, "closed or failed"):
            client.check_graph_copy(status)
