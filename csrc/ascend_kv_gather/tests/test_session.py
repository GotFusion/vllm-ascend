# SPDX-License-Identifier: Apache-2.0
"""CPU contract and socket tests; these do not validate HCOMM/NPU execution."""

import base64
import ctypes
import socket
import struct
import unittest
from contextlib import ExitStack, suppress
from pathlib import Path
from unittest.mock import patch

import torch
from ascend_kv_gather import session


class EndpointTest(unittest.TestCase):
    def test_defaults_and_library_resolution(self):
        self.assertEqual(
            session.parse_endpoint({"host": "127.0.0.1", "port": 1234}),
            {"host": "127.0.0.1", "port": 1234, "cores": 1, "timeout": 300.0},
        )
        self.assertEqual(
            session.parse_endpoint({"host": "localhost", "port": 1234, "library": "lib.so"}, "/tmp")["library"],
            str(Path("/tmp/lib.so").resolve()),
        )

    def test_invalid_endpoint_values(self):
        base = {"host": "127.0.0.1", "port": 1234}
        for bad in (
            {"host": ""},
            {"host": " host"},
            {"port": 0},
            {"port": True},
            {"port": 65536},
            {"cores": 0},
            {"cores": 9},
            {"cores": True},
            {"timeout": float("nan")},
            {"timeout": float("inf")},
            {"timeout": 0},
            {"timeout": True},
            {"library": ""},
            {"unrecognized": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                session.parse_endpoint(base | bad)

    def test_mapping_consumes_only_active_prefixes(self):
        q = torch.tensor([1, 2], dtype=torch.int32)
        counts = torch.tensor([[2, 0], [1, 1], [-999, -999]], dtype=torch.int32)
        mapping = torch.full((3, 2, 3, 2), -999, dtype=torch.int32)
        mapping[0, 0, :2] = torch.tensor([[20001, 8191], [1, 3]])
        mapping[1, 0, 0] = torch.tensor([2, 5])
        mapping[1, 1, 0] = torch.tensor([20001, 8])
        self.assertEqual(session._mapping_rows(q, mapping, counts, 30000), ([20001, 1, 2, 20001], [8191, 3, 5, 8]))

    def test_mapping_zero_queries_and_zero_misses(self):
        mapping = torch.full((1, 1, 3, 2), -999, dtype=torch.int32)
        self.assertEqual(
            session._mapping_rows(
                torch.tensor([0], dtype=torch.int32), mapping, torch.tensor([[-999]], dtype=torch.int32), 2
            ),
            ([], []),
        )
        self.assertEqual(
            session._mapping_rows(
                torch.tensor([1], dtype=torch.int32), mapping, torch.tensor([[0]], dtype=torch.int32), 2
            ),
            ([], []),
        )

    def test_bad_mapping_rejected_before_transfer(self):
        q = torch.tensor([1], dtype=torch.int32)
        counts = torch.tensor([[2]], dtype=torch.int32)
        for pairs in ([[0, 1], [1, 1]], [[-1, 0], [1, 1]], [[5, 0], [1, 1]], [[1, -1], [2, 1]], [[1, 8192], [2, 1]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                session._mapping_rows(q, torch.tensor([[pairs]], dtype=torch.int32), counts, 5)
        mapping = torch.zeros((1, 1, 2, 2), dtype=torch.int32)
        for query, count in (([-1], [[0]]), ([2, 1], [[0]]), ([2], [[0]]), ([1], [[3]]), ([1], [[-1]])):
            with self.subTest(query=query, count=count), self.assertRaises(ValueError):
                session._mapping_rows(
                    torch.tensor(query, dtype=torch.int32), mapping, torch.tensor(count, dtype=torch.int32), 5
                )

    def test_frames_have_bounded_length_and_object_schema(self):
        for frame in (
            struct.pack("!I", 0),
            struct.pack("!I", session.MAX_FRAME_BYTES + 1),
            struct.pack("!I", 2) + b"[]",
        ):
            left, right = socket.socketpair()
            with left, right:
                left.sendall(frame)
                with self.assertRaises(ValueError):
                    session._receive(right)

    def test_partial_frame_eof(self):
        left, right = socket.socketpair()
        with right:
            left.sendall(struct.pack("!I", 6) + b"{}")
            left.close()
            with self.assertRaises(ConnectionError):
                session._receive(right)

    def test_idle_wait_does_not_expire_session_but_partial_frame_does(self):
        left, right = socket.socketpair()
        with left, right:
            right.settimeout(0.01)
            self.assertIsNone(session._receive(right, allow_idle=True))
            left.sendall(b"\x00")
            with self.assertRaises(TimeoutError):
                session._receive(right, allow_idle=True)


class FakeNative:
    """Exercise the Python wire/scatter contract with CPU bytes, never an NPU proof."""

    def __init__(self):
        self.root_bytes = 16
        self.library_path = "cpu-test-native"
        self.transfers = []
        self.created = []
        self.destroyed = []
        self.fail_on_transfer = None

    def root_info(self):
        return b"x" * self.root_bytes

    def create(self, rank, root_info, banks, cores):
        if root_info != self.root_info():
            raise ValueError("bad fake root")
        self.created.append((rank, tuple(session._span(bank) for bank in banks), cores))
        return rank + 1

    def transfer(self, handle, source, destination, row_bytes, indices):
        if self.fail_on_transfer == len(self.transfers):
            raise RuntimeError("injected native transfer failure")
        self.transfers.append((handle, source, destination, row_bytes, tuple(indices)))
        for offset, row in enumerate(indices):
            ctypes.memmove(destination + offset * row_bytes, source + row * row_bytes, row_bytes)

    def destroy(self, handle):
        self.destroyed.append(handle)


class SocketSessionTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.native = FakeNative()
        self.stack.enter_context(patch.object(session, "_Native", lambda library=None: self.native))
        self.stack.enter_context(patch.object(session, "_device", lambda device: torch.device("cpu")))
        self.stack.enter_context(patch.object(session, "_set_device", lambda device: None))
        self.stack.enter_context(patch.object(session, "_synchronize", lambda device: None))
        self.stack.enter_context(patch.object(session, "_eager_only", lambda: None))
        self.kv = torch.arange(2 * 3 * 4 * session.KV_DIM, dtype=torch.float32).to(torch.bfloat16).view(2, 3, 4, -1)
        self.rope = (
            -torch.arange(2 * 3 * 4 * session.ROPE_DIM, dtype=torch.float32).to(torch.bfloat16).view(2, 3, 4, -1)
        )
        self.selection = (
            torch.full((64, 128, session.KV_DIM), 17, dtype=torch.bfloat16),
            torch.full((64, 128, session.ROPE_DIM), 23, dtype=torch.bfloat16),
        )
        self.peer = session.KVGatherPeer(self.kv, self.rope, host="127.0.0.1", port=0, device="cpu", timeout=2).start()
        self.addCleanup(self._cleanup_peer)
        self.client = None
        self.addCleanup(self._cleanup_client)

    def _cleanup_client(self):
        if self.client is not None:
            self.client.close()

    def _cleanup_peer(self):
        # Tests deliberately poison the peer; assertions below inspect errors.
        with suppress(RuntimeError):
            self.peer.close()

    def connect(self, **kwargs):
        self.client = session.KVGatherClient(
            *self.selection, host="127.0.0.1", port=self.peer.port, device="cpu", timeout=2, **kwargs
        )
        return self.client

    def gather(self, layer, pairs):
        q = torch.tensor([1], dtype=torch.int32)
        counts = torch.tensor([[len(pairs)]], dtype=torch.int32)
        mapping = torch.full((1, 1, max(1, len(pairs) + 1), 2), -999, dtype=torch.int32)
        if pairs:
            mapping[0, 0, : len(pairs)] = torch.tensor(pairs, dtype=torch.int32)
        return self.client.gather(layer, q, mapping, counts)

    def test_real_socket_layers_holes_reuse_and_zero_miss(self):
        client = self.connect(layer_ids=[0, 1])
        actual = self.gather(1, [[11, 8191], [3, 4], [1, 9]])
        self.assertIs(actual[0], self.selection[0])
        for selected, source, sentinel in zip(actual, (self.kv, self.rope), (17, 23)):
            rows = selected.view(session.SELECTION_ROWS, -1)
            self.assertTrue(torch.equal(rows[[8191, 4, 9]], source[1].flatten(0, 1)[[11, 3, 1]]))
            self.assertTrue(torch.all(rows[7] == sentinel))
        before = [selected.clone() for selected in actual]
        self.gather(1, [])
        self.assertEqual(len(self.native.transfers), 2)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, actual)))
        self.kv[0, 0, 0].fill_(41)
        self.rope[0, 0, 0].fill_(-37)
        self.gather(0, [[0, 9]])
        self.assertTrue(torch.all(actual[0].view(session.SELECTION_ROWS, -1)[9] == 41))
        self.assertTrue(torch.all(actual[1].view(session.SELECTION_ROWS, -1)[9] == -37))
        self.assertEqual(client.stats["transfer_requests"], 2)
        self.assertEqual(client.stats["transferred_rows"], 4)
        client.close()
        self.peer.close()
        self.assertCountEqual(self.native.destroyed, [1, 2])
        self.assertEqual(self.peer.stats["native_transfers"], 4)

    def test_manifest_banks_must_be_the_registered_hixl_storage(self):
        def manifest():
            result = {}
            for layer in range(2):
                fields = {
                    "blocks": 3,
                    "block_size": 4,
                    "kv_dim": session.KV_DIM,
                    "rope_dim": session.ROPE_DIM,
                    "dtype": "bfloat16",
                }
                for name, bank in zip(("kv", "rope"), (self.kv, self.rope)):
                    fields.update(
                        {
                            f"{name}_base": bank[layer].data_ptr(),
                            f"{name}_bytes": bank[layer].numel() * 2,
                            f"{name}_stride0": bank[layer].stride(0) * 2,
                            f"{name}_stride1": bank[layer].stride(1) * 2,
                        }
                    )
                result[str(layer)] = fields
            return result

        wrong = manifest()
        wrong["1"]["kv_base"] += 1024
        with self.assertRaisesRegex(ValueError, "HIXL remote manifest"):
            self.connect(layer_ids=[1], layer_layouts=wrong)
        self.assertEqual(self.native.created, [])

    def test_error_after_first_bank_never_scatters_partial_result(self):
        self.connect()
        self.native.fail_on_transfer = 1
        before = [tensor.clone() for tensor in self.selection]
        with self.assertRaisesRegex(RuntimeError, "injected native transfer failure"):
            self.gather(0, [[0, 3]])
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, self.selection)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.gather(0, [[0, 3]])
        self.assertEqual(self.client.stats["transfer_requests"], 0)

    def test_invalid_mapping_is_rejected_without_native_transfer(self):
        self.connect()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.gather(0, [[0, 3], [1, 3]])
        self.assertEqual(self.native.transfers, [])

    def test_peer_revalidates_wire_row_ranges(self):
        self.connect()
        session._send(self.client._socket, {"op": "gather", "sequence": 1, "layer": 0, "indices": [12]})
        with self.assertRaisesRegex(RuntimeError, "source row"):
            session._receive(self.client._socket)
        self.assertEqual(self.native.transfers, [])
        self.client._abort()

    def test_peer_rejects_replayed_or_skipped_request_sequence(self):
        self.connect()
        session._send(self.client._socket, {"op": "gather", "sequence": 2, "layer": 0, "indices": [0]})
        with self.assertRaisesRegex(RuntimeError, "sequence mismatch"):
            session._receive(self.client._socket)
        self.assertEqual(self.native.transfers, [])
        self.client._abort()

    def test_peer_checks_receive_range_before_native_create(self):
        with socket.create_connection(("127.0.0.1", self.peer.port), timeout=2) as connection:
            session._send(
                connection,
                {
                    "op": "open",
                    "version": session.PROTOCOL_VERSION,
                    "cores": 1,
                    "root_info": base64.b64encode(self.native.root_info()).decode(),
                    "receive": [{"base": 1, "bytes": 8}, {"base": 32, "bytes": 8}],
                },
            )
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                session._receive(connection)
        self.assertEqual(self.native.created, [])

    def test_destroy_failure_retains_handle_and_allows_cleanup_only(self):
        client = self.connect()
        destroy = self.native.destroy
        failures = []

        def fail_once(handle):
            if handle == 1 and not failures:
                failures.append(handle)
                raise RuntimeError("injected incomplete drain")
            destroy(handle)

        self.native.destroy = fail_once
        with self.assertRaisesRegex(RuntimeError, "incomplete drain"):
            client.close()
        self.assertEqual(client._handle, 1)
        self.assertEqual(len(client.receive), 2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.gather(0, [[0, 1]])
        client.close()
        self.assertIsNone(client._handle)
        self.assertIn(1, self.native.destroyed)

    def test_partial_create_handle_is_cleaned_up(self):
        create = self.native.create

        def fail_client(rank, *args):
            handle = create(rank, *args)
            if rank == 0:
                raise session._CreateFailure("injected create cleanup failure", handle)
            return handle

        self.native.create = fail_client
        with self.assertRaisesRegex(RuntimeError, "injected create cleanup failure"):
            self.connect()
        self.assertIn(1, self.native.destroyed)


if __name__ == "__main__":
    unittest.main()
