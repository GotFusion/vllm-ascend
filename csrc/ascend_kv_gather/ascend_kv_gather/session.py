# SPDX-License-Identifier: Apache-2.0
"""Bounded eager gather sessions; HIXL remains the owner of KV publication.

The TCP connection carries bootstrap metadata, source row IDs and completion
messages only. KV/RoPE bytes travel through the native HCOMM multi-SGE kernel.
The CPU mapping read, socket exchange and Torch scatter are intentionally part
of this first integration's end-to-end latency. This is not a graph interface.
"""

from __future__ import annotations

import base64
import ctypes
import json
import math
import socket
import struct
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

SELECTION_ROWS = 8192
KV_DIM = 512
ROPE_DIM = 64
ELEMENT_BYTES = 2
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 1024 * 1024
MAX_ROOT_INFO_BYTES = 64 * 1024
ERROR_BYTES = 4096
UINT64_MAX = (1 << 64) - 1
MAX_CLOSE_WAIT_SECONDS = 30


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def parse_endpoint(options: Mapping[str, Any], base_dir: str | Path = ".") -> dict[str, Any]:
    """Validate the explicit endpoint configuration; no runtime env switches."""
    if not isinstance(options, Mapping):
        raise ValueError("kvgather must be an object")
    allowed = {"host", "port", "library", "cores", "timeout"}
    if set(options) - allowed or not {"host", "port"} <= set(options):
        raise ValueError("kvgather requires host and port and accepts only library, cores and timeout")
    host = options["host"]
    if not isinstance(host, str) or not host.strip() or host != host.strip():
        raise ValueError("kvgather.host must be a nonempty host name or IP address")
    port = _integer(options["port"], "kvgather.port", 1, 65535)
    cores = _integer(options.get("cores", 1), "kvgather.cores", 1, 8)
    timeout = options.get("timeout", 300)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("kvgather.timeout must be finite and positive")
    result = {"host": host, "port": port, "cores": cores, "timeout": float(timeout)}
    if "library" in options:
        library = options["library"]
        if not isinstance(library, str) or not library.strip():
            raise ValueError("kvgather.library must be a nonempty path")
        result["library"] = str((Path(base_dir) / library).resolve())
    return result


def _device(device: int | str | torch.device) -> torch.device:
    # Importing torch_npu is the embedding worker's responsibility. Never make
    # a CPU fallback for a missing NPU extension or unavailable device.
    selected = torch.device(f"npu:{device}" if isinstance(device, int) else device)
    if selected.type not in ("npu", "privateuseone") or selected.index is None:
        raise ValueError("KV gather requires an explicit NPU device index")
    return selected


def _set_device(device: torch.device) -> None:
    torch.npu.set_device(device)


def _synchronize(device: torch.device) -> None:
    torch.npu.current_stream(device).synchronize()


def _eager_only() -> None:
    # Do not call host socket I/O while a graph capture is active.
    if torch.compiler.is_compiling():
        raise RuntimeError("KV gather supports eager execution only")
    capturing = getattr(torch.npu, "is_current_stream_capturing", None)
    if capturing is not None and capturing():
        raise RuntimeError("KV gather supports eager execution only")


def _span(tensor: torch.Tensor) -> dict[str, int]:
    return {"base": tensor.data_ptr(), "bytes": tensor.numel() * tensor.element_size()}


def _check_tensor(tensor: torch.Tensor, device: torch.device, dim: int, *, peer: bool) -> None:
    if (
        tensor.device != device
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or tensor.ndim < 2
        or tensor.shape[-1] != dim
        or tensor.numel() == 0
        or (peer and tensor.ndim != 4)
        or (not peer and tensor.numel() != SELECTION_ROWS * dim)
    ):
        kind = "[layer, block, token, dim] peer bank" if peer else "8192-row selection bank"
        raise ValueError(f"KV gather requires a contiguous BF16 {kind} on {device}, dim={dim}")


def _check_span(value: Any, name: str, expected_bytes: int | None = None) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"base", "bytes"}:
        raise ValueError(f"invalid {name} memory descriptor")
    base = _integer(value["base"], f"{name}.base", 1, UINT64_MAX)
    size = _integer(value["bytes"], f"{name}.bytes", 1, UINT64_MAX)
    if base > UINT64_MAX - size or (expected_bytes is not None and size != expected_bytes):
        raise ValueError(f"invalid {name} address range or capacity")
    return {"base": base, "bytes": size}


def _check_disjoint(banks: Sequence[dict[str, int]]) -> None:
    first, second = banks
    if max(first["base"], second["base"]) < min(first["base"] + first["bytes"], second["base"] + second["bytes"]):
        raise ValueError("KV and RoPE memory descriptors overlap")


def _send(sock: socket.socket, value: dict[str, Any]) -> None:
    # Keep JSON as the control-plane format; no pickle or executable payloads.
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("KV gather control frame is too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _receive(sock: socket.socket, *, allow_idle: bool = False) -> dict[str, Any] | None:
    def exact(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            data = sock.recv(size - len(chunks))
            if not data:
                raise ConnectionError("KV gather peer closed the control connection")
            chunks.extend(data)
        return bytes(chunks)

    # Model loading or a long Prefill may legitimately leave the established
    # peer idle longer than a transfer timeout. Once any frame byte arrives,
    # partial headers and payloads still have a bounded socket read deadline.
    try:
        first = sock.recv(1)
    except TimeoutError:
        if allow_idle:
            return None
        raise
    if not first:
        raise ConnectionError("KV gather peer closed the control connection")
    size = struct.unpack("!I", first + exact(3))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError("invalid KV gather control frame length")
    value = json.loads(exact(size))
    if not isinstance(value, dict):
        raise ValueError("KV gather control frame must be an object")
    if value.get("op") == "error":
        raise RuntimeError(f"KV gather peer failed: {value.get('error', 'unknown error')}")
    return value


def _shutdown(sock: socket.socket | None) -> None:
    if sock is not None:
        with suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()


class _CreateFailure(RuntimeError):
    """Initialization failed while native teardown still owns registered banks."""

    def __init__(self, message: str, handle: ctypes.c_void_p):
        super().__init__(message)
        self.handle = handle


class _Native:
    """ctypes releases the GIL while HCOMM initializes, launches or drains."""

    def __init__(self, library: str | Path | None = None):
        path = Path(library) if library is not None else Path(__file__).with_name("libascend_kv_gather.so")
        self.library_path = str(path.resolve())
        self.library = ctypes.CDLL(self.library_path)
        pointer, word = ctypes.c_void_p, ctypes.c_uint64
        u32, size = ctypes.c_uint32, ctypes.c_size_t
        error_args = [pointer, size]
        functions = {
            "kvg_root_info": [pointer, size, *error_args],
            "kvg_create": [
                u32,
                pointer,
                ctypes.POINTER(word),
                ctypes.POINTER(word),
                u32,
                u32,
                ctypes.POINTER(pointer),
                *error_args,
            ],
            "kvg_transfer": [pointer, word, word, u32, ctypes.POINTER(word), u32, *error_args],
            "kvg_destroy": [pointer, *error_args],
        }
        for name, args in functions.items():
            function = getattr(self.library, name)
            function.argtypes = args
            function.restype = ctypes.c_int
        self.library.kvg_root_info_size.argtypes = []
        self.library.kvg_root_info_size.restype = size
        self.root_bytes = _integer(int(self.library.kvg_root_info_size()), "root info bytes", 1, MAX_ROOT_INFO_BYTES)

    def _call(self, name: str, *args: Any) -> None:
        error = ctypes.create_string_buffer(ERROR_BYTES)
        status = getattr(self.library, name)(*args, error, len(error))
        if status != 0:
            message = error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"{name} failed ({status}): {message}")

    def root_info(self) -> bytes:
        root = ctypes.create_string_buffer(self.root_bytes)
        self._call("kvg_root_info", root, len(root))
        return root.raw

    def create(self, rank: int, root_info: bytes, banks: Sequence[torch.Tensor], cores: int) -> ctypes.c_void_p:
        if len(root_info) != self.root_bytes:
            raise ValueError("HCCL root-info size differs from the loaded native library")
        root = ctypes.create_string_buffer(root_info, len(root_info))
        bases = (ctypes.c_uint64 * len(banks))(*(bank.data_ptr() for bank in banks))
        sizes = (ctypes.c_uint64 * len(banks))(*(bank.numel() * bank.element_size() for bank in banks))
        handle = ctypes.c_void_p()
        try:
            self._call("kvg_create", rank, root, bases, sizes, len(banks), cores, ctypes.byref(handle))
        except BaseException as exc:
            if handle.value:
                raise _CreateFailure(str(exc), handle) from exc
            raise
        if not handle.value:
            raise RuntimeError("kvg_create returned a null session")
        return handle

    def transfer(
        self, handle: ctypes.c_void_p, source: int, destination: int, row_bytes: int, indices: Sequence[int]
    ) -> None:
        rows = (ctypes.c_uint64 * len(indices))(*indices)
        self._call("kvg_transfer", handle, source, destination, row_bytes, rows, len(indices))

    def destroy(self, handle: ctypes.c_void_p) -> None:
        self._call("kvg_destroy", handle)


def _mapping_rows(
    q_lens: torch.Tensor, mapping: torch.Tensor, counts: torch.Tensor, source_rows: int
) -> tuple[list[int], list[int]]:
    """Read only the active query/head groups and each group's valid prefix."""
    if (
        any(t.dtype != torch.int32 for t in (q_lens, mapping, counts))
        or q_lens.ndim != 1
        or q_lens.numel() == 0
        or counts.ndim < 1
        or counts.numel() == 0
        or mapping.ndim != counts.ndim + 2
        or tuple(mapping.shape[:-2]) != tuple(counts.shape)
        or mapping.shape[-1] != 2
        or mapping.shape[-2] == 0
        or mapping.device != counts.device
        or mapping.device != q_lens.device
    ):
        raise ValueError("expected INT32 cumulative query lengths and matching counts/mapping[..., entries, 2]")
    # This synchronization is explicit and belongs to the eager implementation.
    queries = q_lens.cpu().tolist()
    previous = 0
    for length in queries:
        if length < previous:
            raise ValueError("query lengths must be nonnegative cumulative lengths")
        previous = length
    groups = previous * counts.shape[-1]
    if groups > counts.numel():
        raise ValueError("active query/head groups exceed mapping capacity")
    active_counts = counts.reshape(-1)[:groups].cpu().tolist()
    entries = mapping.shape[-2]
    if any(count < 0 or count > entries for count in active_counts) or sum(active_counts) > SELECTION_ROWS:
        raise ValueError("invalid Match count or gather exceeds the receive capacity")
    # Padding after a valid prefix is undefined; never inspect it for validity.
    active = mapping.reshape(-1, entries, 2)[:groups].cpu()
    sources, destinations = [], []
    seen = set()
    for group, count in enumerate(active_counts):
        for source, destination in active[group, :count].tolist():
            if not 0 <= source < source_rows or not 0 <= destination < SELECTION_ROWS or destination in seen:
                raise ValueError("invalid source row, destination row or duplicate Match destination")
            sources.append(source)
            destinations.append(destination)
            seen.add(destination)
    return sources, destinations


class KVGatherPeer:
    """Serve one model session from the same contiguous banks registered by HIXL."""

    def __init__(
        self,
        kv: torch.Tensor,
        rope: torch.Tensor,
        *,
        host: str,
        port: int,
        device: int | str | torch.device,
        library: str | Path | None = None,
        cores: int = 1,
        timeout: float = 300,
    ):
        dynamic_port = type(port) is int and port == 0
        options = parse_endpoint(
            {"host": host, "port": 1 if dynamic_port else port, "cores": cores, "timeout": timeout}
        )
        self.device = _device(device)
        _check_tensor(kv, self.device, KV_DIM, peer=True)
        _check_tensor(rope, self.device, ROPE_DIM, peer=True)
        if kv.shape[:3] != rope.shape[:3]:
            raise ValueError("peer KV and RoPE layers/pages/tokens must match")
        self.banks = (kv, rope)
        _check_disjoint([_span(bank) for bank in self.banks])
        self.host, self.port = options["host"], 0 if dynamic_port else options["port"]
        self.cores, self.timeout = options["cores"], options["timeout"]
        self._native = _Native(library)
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._connection: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._handle: ctypes.c_void_p | None = None
        self._transfer_requests = 0
        self._transferred_rows = 0
        self._native_transfers = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Completed operation counts, suitable for one end-of-run evidence log."""
        return {
            "transfer_requests": self._transfer_requests,
            "transferred_rows": self._transferred_rows,
            "native_transfers": self._native_transfers,
            "native_library_path": self._native.library_path,
        }

    def start(self) -> KVGatherPeer:
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("KV gather peer cannot be started twice")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            self.port = listener.getsockname()[1]
            listener.listen(1)
            listener.settimeout(0.5)
            self._listener = listener
            self._thread = threading.Thread(target=self._serve, name="kv-gather-peer", daemon=True)
            self._thread.start()
        except BaseException:
            listener.close()
            self._listener = None
            raise
        return self

    def _open(self, connection: socket.socket) -> list[dict[str, int]]:
        hello = _receive(connection)
        if set(hello) != {"op", "version", "cores", "root_info", "receive"} or (
            hello["op"] != "open"
            or type(hello["version"]) is not int
            or hello["version"] != PROTOCOL_VERSION
            or type(hello["cores"]) is not int
            or hello["cores"] != self.cores
        ):
            raise ValueError("KV gather bootstrap version, cores or schema mismatch")
        root_info = base64.b64decode(hello["root_info"], validate=True)
        if len(root_info) != self._native.root_bytes:
            raise ValueError("KV gather bootstrap root-info size mismatch")
        if not isinstance(hello["receive"], list) or len(hello["receive"]) != 2:
            raise ValueError("KV gather requires two receive banks")
        receive = [
            _check_span(bank, name, SELECTION_ROWS * dim * ELEMENT_BYTES)
            for bank, name, dim in zip(hello["receive"], ("KV receive", "RoPE receive"), (KV_DIM, ROPE_DIM))
        ]
        _check_disjoint(receive)
        shape = self.banks[0].shape
        _send(
            connection,
            {
                "op": "layout",
                "version": PROTOCOL_VERSION,
                "layers": shape[0],
                "blocks": shape[1],
                "block_size": shape[2],
                "kv_dim": KV_DIM,
                "rope_dim": ROPE_DIM,
                "dtype": "bfloat16",
                "banks": [_span(bank) for bank in self.banks],
            },
        )
        # Do not enter collective initialization until the client has checked
        # that these are exactly the HIXL banks described by its manifest.
        if _receive(connection) != {"op": "accept"}:
            raise ValueError("KV gather client rejected the peer layout")
        try:
            self._handle = self._native.create(1, root_info, self.banks, self.cores)
        except _CreateFailure as exc:
            self._handle = exc.handle
            raise
        _send(connection, {"op": "ready"})
        return receive

    def _serve(self) -> None:
        connection = None
        try:
            _set_device(self.device)
            _synchronize(self.device)
            while not self._stop.is_set():
                try:
                    connection, _ = self._listener.accept()
                    break
                except TimeoutError:
                    continue
            if connection is None:
                return
            self._connection = connection
            connection.settimeout(self.timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            receive = self._open(connection)
            sequence = 1
            layers, blocks, block_size = self.banks[0].shape[:3]
            rows_per_layer = blocks * block_size
            while not self._stop.is_set():
                request = _receive(connection, allow_idle=True)
                if request is None:
                    continue
                if type(request.get("sequence")) is not int or request["sequence"] != sequence:
                    raise ValueError("KV gather request sequence mismatch")
                if request == {"op": "close", "sequence": sequence}:
                    _send(connection, {"op": "closed", "sequence": sequence})
                    return
                if set(request) != {"op", "sequence", "layer", "indices"} or request["op"] != "gather":
                    raise ValueError("invalid KV gather request schema")
                layer = _integer(request["layer"], "layer", 0, layers - 1)
                indices = request["indices"]
                if not isinstance(indices, list) or not 0 < len(indices) <= SELECTION_ROWS:
                    raise ValueError("KV gather request exceeds receive capacity")
                for index in indices:
                    _integer(index, "source row", 0, rows_per_layer - 1)
                for bank, destination, dim in zip(self.banks, receive, (KV_DIM, ROPE_DIM)):
                    source = bank.data_ptr() + layer * rows_per_layer * dim * ELEMENT_BYTES
                    self._native.transfer(self._handle, source, destination["base"], dim * ELEMENT_BYTES, indices)
                    self._native_transfers += 1
                # Both native launches drain their completions and synchronize
                # their private stream before this message becomes observable.
                _send(connection, {"op": "done", "sequence": sequence, "rows": len(indices)})
                self._transfer_requests += 1
                self._transferred_rows += len(indices)
                sequence += 1
        except BaseException as exc:
            if not self._stop.is_set():
                self._error = exc
                if connection is not None:
                    with suppress(OSError, ValueError):
                        _send(connection, {"op": "error", "error": str(exc)[:ERROR_BYTES]})
        finally:
            _shutdown(connection)
            self._connection = None
            # This peer owns exactly one model session. A later connection must
            # fail promptly instead of entering a backlog with no accept loop.
            _shutdown(self._listener)
            if self._handle is not None:
                try:
                    self._native.destroy(self._handle)
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
                else:
                    self._handle = None

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"KV gather peer failed: {self._error}") from self._error

    def close(self) -> None:
        self._stop.set()
        _shutdown(self._connection)
        _shutdown(self._listener)
        if self._thread is not None:
            self._thread.join(timeout=min(self.timeout, MAX_CLOSE_WAIT_SECONDS))
            if self._thread.is_alive():
                # The thread retains self/banks until the native call finishes;
                # never free registered allocations under an active DMA.
                raise RuntimeError("KV gather peer is still inside a native operation; bank ownership retained")
        if self._handle is not None:
            _set_device(self.device)
            self._native.destroy(self._handle)
            self._handle = None
        self.check_health()

    def __enter__(self) -> KVGatherPeer:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()


class KVGatherClient:
    """Registered bounded receive banks plus a caller-owned Match selection pool."""

    def __init__(
        self,
        selection_kv: torch.Tensor,
        selection_rope: torch.Tensor,
        *,
        host: str,
        port: int,
        device: int | str | torch.device,
        library: str | Path | None = None,
        cores: int = 1,
        timeout: float = 300,
        layer_ids: Sequence[int] | None = None,
        layer_layouts: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        options = parse_endpoint({"host": host, "port": port, "cores": cores, "timeout": timeout})
        self.device = _device(device)
        _set_device(self.device)
        _eager_only()
        _check_tensor(selection_kv, self.device, KV_DIM, peer=False)
        _check_tensor(selection_rope, self.device, ROPE_DIM, peer=False)
        self.selection = (selection_kv, selection_rope)
        _check_disjoint([_span(bank) for bank in self.selection])
        self.receive = tuple(
            torch.empty((SELECTION_ROWS, dim), dtype=torch.bfloat16, device=self.device) for dim in (KV_DIM, ROPE_DIM)
        )
        self._lock = threading.Lock()
        self._native = _Native(library)
        self._handle: ctypes.c_void_p | None = None
        self._socket: socket.socket | None = None
        self._closed = False
        self._sequence = 1
        self._layers: set[int] = set()
        self._rows_per_layer = 0
        self._transfer_requests = 0
        self._transferred_rows = 0
        try:
            _synchronize(self.device)
            root_info = self._native.root_info()
            self._socket = socket.create_connection((host, options["port"]), timeout=options["timeout"])
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _send(
                self._socket,
                {
                    "op": "open",
                    "version": PROTOCOL_VERSION,
                    "cores": options["cores"],
                    "root_info": base64.b64encode(root_info).decode("ascii"),
                    "receive": [_span(bank) for bank in self.receive],
                },
            )
            self._check_layout(_receive(self._socket), layer_ids, layer_layouts)
            _send(self._socket, {"op": "accept"})
            self._handle = self._native.create(0, root_info, self.receive, options["cores"])
            if _receive(self._socket) != {"op": "ready"}:
                raise ValueError("KV gather peer did not complete initialization")
        except BaseException as exc:
            if isinstance(exc, _CreateFailure):
                self._handle = exc.handle
            self._abort_after_error(exc)
            raise

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "transfer_requests": self._transfer_requests,
            "transferred_rows": self._transferred_rows,
            "native_library_path": self._native.library_path,
        }

    def _check_layout(
        self,
        value: dict[str, Any],
        layer_ids: Sequence[int] | None,
        layer_layouts: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        fields = {"op", "version", "layers", "blocks", "block_size", "kv_dim", "rope_dim", "dtype", "banks"}
        if (
            set(value) != fields
            or type(value["version"]) is not int
            or (value["op"], value["version"], value["kv_dim"], value["rope_dim"], value["dtype"])
            != ("layout", PROTOCOL_VERSION, KV_DIM, ROPE_DIM, "bfloat16")
        ):
            raise ValueError("KV gather peer layout/schema mismatch")
        layers = _integer(value["layers"], "layers", 1, (1 << 31) - 1)
        blocks = _integer(value["blocks"], "blocks", 1, (1 << 31) - 1)
        block_size = _integer(value["block_size"], "block size", 1, (1 << 31) - 1)
        rows = blocks * block_size
        if rows > (1 << 31) - 1 or not isinstance(value["banks"], list) or len(value["banks"]) != 2:
            raise ValueError("invalid KV gather peer row capacity")
        banks = [
            _check_span(bank, name, layers * rows * dim * ELEMENT_BYTES)
            for bank, name, dim in zip(value["banks"], ("peer KV", "peer RoPE"), (KV_DIM, ROPE_DIM))
        ]
        _check_disjoint(banks)
        selected = list(range(layers)) if layer_ids is None else list(layer_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("KV gather requires distinct nonempty layer IDs")
        for layer in selected:
            _integer(layer, "layer ID", 0, layers - 1)
            if layer_layouts is not None:
                layout = layer_layouts.get(str(layer))
                expected = {
                    "blocks": blocks,
                    "block_size": block_size,
                    "kv_dim": KV_DIM,
                    "rope_dim": ROPE_DIM,
                    "dtype": "bfloat16",
                }
                for prefix, bank, dim in zip(("kv", "rope"), banks, (KV_DIM, ROPE_DIM)):
                    row_bytes = dim * ELEMENT_BYTES
                    expected.update(
                        {
                            f"{prefix}_base": bank["base"] + layer * rows * row_bytes,
                            f"{prefix}_bytes": rows * row_bytes,
                            f"{prefix}_stride0": block_size * row_bytes,
                            f"{prefix}_stride1": row_bytes,
                        }
                    )
                if layout is None or any(layout.get(key) != field for key, field in expected.items()):
                    raise ValueError(f"KV gather layer {layer} does not match its HIXL remote manifest")
        self._layers = set(selected)
        self._rows_per_layer = rows

    def gather(
        self, layer_id: int, q_lens: torch.Tensor, mapping: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            if self._closed:
                raise RuntimeError("KV gather client is closed")
            try:
                _set_device(self.device)
                _eager_only()
                if type(layer_id) is not int or layer_id not in self._layers:
                    raise ValueError(f"KV gather peer has no requested layer {layer_id}")
                if any(t.device != self.device for t in (q_lens, mapping, counts)):
                    raise ValueError("Match tensors must be on the KV gather worker device")
                # The previous selection consumer and any previous scatter must
                # finish before the peer reuses the registered receive banks.
                _synchronize(self.device)
                sources, destinations = _mapping_rows(q_lens, mapping, counts, self._rows_per_layer)
                if not sources:
                    return self.selection
                _send(self._socket, {"op": "gather", "sequence": self._sequence, "layer": layer_id, "indices": sources})
                expected = {"op": "done", "sequence": self._sequence, "rows": len(sources)}
                if _receive(self._socket) != expected:
                    raise ValueError("KV gather completion sequence or row count mismatch")
                destination = torch.tensor(destinations, dtype=torch.int64, device=self.device)
                for selection, receive in zip(self.selection, self.receive):
                    selection.view(-1, selection.shape[-1]).index_copy_(0, destination, receive[: len(sources)])
                self._sequence += 1
                self._transfer_requests += 1
                self._transferred_rows += len(sources)
                return self.selection
            except BaseException as exc:
                self._abort_after_error(exc)
                raise

    def _abort_after_error(self, error: BaseException) -> None:
        try:
            self._abort()
        except BaseException as cleanup_error:
            # Preserve the original transfer/init failure as well as the reason
            # cleanup could not release registrations. The worker must stop;
            # only close(), never another gather(), may be attempted afterwards.
            failure = RuntimeError(f"{error}; KV gather cleanup incomplete: {cleanup_error}; worker must exit")
            failure.session = self
            raise failure from error

    def _abort(self) -> None:
        self._closed = True
        _shutdown(self._socket)
        self._socket = None
        if self._handle is not None:
            self._native.destroy(self._handle)
            self._handle = None

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                _set_device(self.device)
            if self._closed:
                # Native destroy failure retains its handle and both tensor
                # banks. Permit a later cleanup attempt, never a data retry.
                if self._handle is not None:
                    self._abort()
                return
            try:
                _set_device(self.device)
                _synchronize(self.device)
                _send(self._socket, {"op": "close", "sequence": self._sequence})
                if _receive(self._socket) != {"op": "closed", "sequence": self._sequence}:
                    raise ValueError("KV gather peer did not acknowledge close")
            finally:
                self._abort()

    def __enter__(self) -> KVGatherClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
