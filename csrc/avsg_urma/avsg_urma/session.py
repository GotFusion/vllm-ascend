# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Eager-only resource owners. Each endpoint privately owns its HIXL engine."""
from dataclasses import dataclass
import json
import threading
import time

import torch
import torch_npu  # noqa: F401 - registers the PrivateUse1 device backend

from . import _native
from .protocol import (DTYPES, HEADER_WORDS, MAX_CHANNELS, graph_bitmap_words,
                       STATUS_BYTES, SYNC_BYTES, TIMING_ROWS, TIMING_WORDS, RemoteCache,
                       pack_context, descriptor_bytes, transfer_plan, device_timing_stats, host_timing_stats)


def _options(channels, local_comm_res):
    if not isinstance(channels, int) or not 0 < channels <= min(32, MAX_CHANNELS):
        raise ValueError("channels must be an integer between 1 and 32 for the HIXL launcher")
    if not isinstance(json.loads(local_comm_res), dict):
        raise ValueError("local_comm_res must be an explicit communication resource JSON object")
    return {"AivJettyNum": str(channels), "LocalCommRes": local_comm_res}


def _check_cache(tensor, device, *, dim_limit):
    if tensor.requires_grad or tensor.device != device or tensor.ndim != 3 or tensor.stride(-1) != 1:
        raise ValueError("cache must be rank-3 with contiguous channels on the endpoint Device")
    name = str(tensor.dtype).removeprefix("torch.")
    if name not in DTYPES or not 0 < tensor.shape[-1] <= dim_limit or tensor.numel() == 0:
        raise ValueError("unsupported cache dtype, dimensions or empty storage")
    if any(stride <= 0 for stride in tensor.stride()):
        raise ValueError("cache strides must be positive")
    return name


def _eager_only():
    if torch.compiler.is_compiling():
        raise RuntimeError("AVSG URMA Session does not support torch.compile or graph capture")
    capturing = getattr(torch.npu, "is_current_stream_capturing", None)
    if capturing is not None and capturing():
        raise RuntimeError("AVSG URMA Session cannot run during graph capture")


class _Endpoint:
    def __init__(self, *, local_engine, device, local_comm_res, channels=1):
        if not isinstance(device, int) or device < 0:
            raise ValueError("an explicit nonnegative NPU device index is required")
        self.device = torch.device("npu", device)
        self._lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._connected = False
        self._owner = _native.Endpoint(local_engine, device, _options(channels, local_comm_res))

    def _check_open(self):
        _eager_only()
        if self._closed or self._poisoned:
            raise RuntimeError("Session is closed or failed; reset AVSG match state and create a new Session")

    def _register_storage(self, *tensors):
        try:
            for tensor in tensors:
                if tensor is not None:
                    self._owner.register_tensor(tensor)
        except Exception:
            # A previous tensor may already be registered. Do not retry partial setup.
            self._poisoned = True
            raise

    def close(self):
        """Caller must ensure that the peer will no longer read registered source storage."""
        with self._lock:
            if not self._closed:
                self._owner.close()
                self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


class AvsgUrmaCacheServer(_Endpoint):
    """Register remote full caches before the client connects; exchange returned layouts out of band."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._layouts = {}
        self._sealed = False

    def register_layer(self, layer_id, full_kv, full_rope=None):
        with self._lock:
            self._check_open()
            if self._sealed or layer_id in self._layouts:
                raise ValueError("registration is sealed or layer is already registered")
            layout = self._layer_layout(full_kv, full_rope)
            self._register_storage(full_kv, full_rope)
            self._layouts[layer_id] = layout
            return layout

    def register_layers(self, first_layer_id, full_kv, full_rope=None):
        """Register contiguous [layer, block, token, dim] caches once per tensor."""
        with self._lock:
            self._check_open()
            if full_kv.ndim != 4 or not full_kv.is_contiguous() or full_kv.size(0) == 0:
                raise ValueError("expected a nonempty contiguous rank-4 KV cache")
            if full_rope is not None and (full_rope.ndim != 4 or not full_rope.is_contiguous() or
                                          full_rope.shape[:3] != full_kv.shape[:3]):
                raise ValueError("expected a contiguous rank-4 RoPE cache matching the KV layers and pages")
            layer_ids = range(first_layer_id, first_layer_id + full_kv.size(0))
            if self._sealed or any(layer in self._layouts for layer in layer_ids):
                raise ValueError("registration is sealed or layer is already registered")
            layouts = {
                layer: self._layer_layout(full_kv[i], full_rope[i] if full_rope is not None else None)
                for i, layer in enumerate(layer_ids)
            }
            self._register_storage(full_kv, full_rope)
            self._layouts.update(layouts)

    def _layer_layout(self, full_kv, full_rope):
        dtype = _check_cache(full_kv, self.device, dim_limit=1024)
        if full_rope is not None:
            rope_dtype = _check_cache(full_rope, self.device, dim_limit=64)
            if full_rope.shape[:2] != full_kv.shape[:2] or rope_dtype != dtype:
                raise ValueError("full KV and RoPE layouts/dtypes differ")
        width = full_kv.element_size()

        def span(t):
            return (sum((size - 1) * stride for size, stride in zip(t.shape, t.stride())) + 1) * width

        return RemoteCache(
            kv_base=full_kv.data_ptr(), kv_bytes=span(full_kv), blocks=full_kv.shape[0],
            block_size=full_kv.shape[1], kv_dim=full_kv.shape[2],
            kv_stride0=full_kv.stride(0) * width, kv_stride1=full_kv.stride(1) * width, dtype=dtype,
            rope_base=full_rope.data_ptr() if full_rope is not None else 0,
            rope_bytes=span(full_rope) if full_rope is not None else 0,
            rope_dim=full_rope.shape[2] if full_rope is not None else 0,
            rope_stride0=full_rope.stride(0) * width if full_rope is not None else 0,
            rope_stride1=full_rope.stride(1) * width if full_rope is not None else 0,
        )

    def publish_layout(self):
        """Seal registrations and return JSON-serializable metadata after producer completion.

        The caller sends this manifest to the consumer through its control plane;
        this method does not itself send a network message or track later updates.
        """
        with self._lock:
            self._check_open()
            if not self._layouts:
                raise RuntimeError("register full caches before publishing")
            with torch.npu.device(self.device):
                torch.npu.synchronize()
            self._sealed = True
            return {str(key): value.to_dict() for key, value in self._layouts.items()}


@dataclass
class _Binding:
    kv: torch.Tensor
    rope: torch.Tensor
    context: torch.Tensor | None = None
    remote: RemoteCache | None = None


class AvsgUrmaSession(_Endpoint):
    """One peer, privately owned channels and fixed registered selection allocations.

    copy_into synchronizes and validates metadata/status on the host. This initial
    implementation is for correctness and intentionally does not promise overlap.
    """
    def __init__(self, *, batch_descriptors=32, transfer_cores=None, jetty_strategy="auto",
                 measure_devbw=False, measure_hostbw=False, **kwargs):
        if not isinstance(batch_descriptors, int) or batch_descriptors not in (8, 32, 64):
            raise ValueError("batch_descriptors selects HIXL SIMT lanes: use 8, 32 or 64")
        transfer_plan(kwargs.get("channels", 1), transfer_cores, jetty_strategy)
        if type(measure_devbw) is not bool or type(measure_hostbw) is not bool:
            raise ValueError("measure_devbw and measure_hostbw must be bool")
        super().__init__(**kwargs)
        self._batch = batch_descriptors
        self._transfer_cores = transfer_cores
        self._strategy = jetty_strategy
        self._measure_devbw = measure_devbw
        self._measure_hostbw = measure_hostbw
        self._plan = None
        self._sync = None
        self._timing = None
        self._last_transfer_stats = None
        self._layers = {}
        self._sources = {}
        self._channels = ()
        self._scratch = None
        self._status = None

    def bind_selection(self, layer_id, kv, rope=None):
        with self._lock:
            self._check_open()
            if self._connected or layer_id in self._layers:
                raise ValueError("bind each layer once before Connect")
            dtype = _check_cache(kv, self.device, dim_limit=1024)
            if not kv.is_contiguous():
                raise ValueError("selection KV must be contiguous")
            if rope is not None:
                if (_check_cache(rope, self.device, dim_limit=64) != dtype or
                        rope.shape[:2] != kv.shape[:2] or not rope.is_contiguous()):
                    raise ValueError("selection RoPE must match KV and be contiguous")
            self._register_storage(kv, rope)
            if rope is None:
                rope = torch.empty(0, dtype=kv.dtype, device=self.device)
            self._layers[layer_id] = _Binding(kv, rope)

    def bind_staging(self, layer_id, source, full_kv, full_rope, selection_blocks):
        """Register KV/RoPE once and split each into disjoint READ and WRITE buffers."""
        with self._lock:
            self._check_open()
            if self._connected or layer_id in self._layers or source in self._sources:
                raise ValueError("bind each selection and WRITE source once before Connect")
            dtype = _check_cache(full_kv, self.device, dim_limit=1024)
            if not full_kv.is_contiguous() or not 0 < selection_blocks < full_kv.size(0):
                raise ValueError("staging KV needs contiguous, nonempty selection and WRITE blocks")
            selection_kv, source_kv = full_kv[:selection_blocks], full_kv[selection_blocks:]
            if full_rope is None:
                selection_rope = source_rope = torch.empty(0, dtype=full_kv.dtype, device=self.device)
            else:
                if (_check_cache(full_rope, self.device, dim_limit=64) != dtype or
                        not full_rope.is_contiguous() or full_rope.shape[:2] != full_kv.shape[:2]):
                    raise ValueError("staging RoPE must match KV and be contiguous")
                selection_rope, source_rope = full_rope[:selection_blocks], full_rope[selection_blocks:]
            self._register_storage(full_kv, full_rope)
            self._layers[layer_id] = _Binding(selection_kv, selection_rope)
            self._sources[source] = _Binding(source_kv, source_rope)
            return selection_kv, selection_rope, source_kv, source_rope

    def bind_source(self, name, kv, rope):
        with self._lock:
            self._check_open()
            if self._connected or name in self._sources:
                raise ValueError("bind each WRITE source once before Connect")
            dtype = _check_cache(kv, self.device, dim_limit=1024)
            if (not kv.is_contiguous() or not rope.is_contiguous() or
                    (rope.numel() and (_check_cache(rope, self.device, dim_limit=64) != dtype or
                     rope.shape[:2] != kv.shape[:2]))):
                raise ValueError("WRITE source KV/RoPE must have matching contiguous layouts")
            self._register_storage(kv, rope if rope.numel() else None)
            self._sources[name] = _Binding(kv, rope)

    def connect(self, peer_engine, *, timeout_ms=30000):
        """Both endpoints must have registered all caches before this call."""
        with self._lock:
            self._check_open()
            if self._connected or not self._layers:
                raise RuntimeError("bind selection before the single allowed Connect")
            try:
                self._channels = tuple(self._owner.connect(peer_engine, timeout_ms))
                self._connected = True
                self._plan = transfer_plan(len(self._channels), self._transfer_cores, self._strategy)
                self._sync = torch.empty(SYNC_BYTES if self._plan["strategy"] == "shared" else 0,
                                         dtype=torch.uint8, device=self.device)
                self._timing = torch.empty((TIMING_ROWS, TIMING_WORDS) if self._measure_devbw else (0,),
                                           dtype=torch.int64, device=self.device)
                self._scratch = torch.empty(descriptor_bytes(0, False),
                                            dtype=torch.uint8, device=self.device)
                self._status = torch.empty((len(self._channels), STATUS_BYTES // 8),
                                           dtype=torch.int64, device=self.device)
            except Exception:
                self._poisoned = True
                raise

    def prepare_layer(self, layer_id, remote):
        with self._lock:
            self._check_open()
            if not self._connected:
                raise RuntimeError("Connect before preparing context")
            if isinstance(remote, dict):
                remote = RemoteCache(**remote)
            binding = self._layers[layer_id]
            dtype = str(binding.kv.dtype).removeprefix("torch.")
            if (remote.dtype != dtype or binding.kv.shape[1:] != (remote.block_size, remote.kv_dim) or
                    binding.rope.numel() != binding.kv.shape[0] * remote.block_size * remote.rope_dim):
                raise ValueError("remote layout differs from local selection")
            # A layer can start inside a larger MR. HIXL validates its whole subrange
            # and returns the supplied URMA RVA; a zero-address query would lose its offset.
            if self._owner.remote_address(remote.kv_base, remote.kv_bytes) != remote.kv_base:
                raise RuntimeError("URMA GetAddr did not return the advertised RVA")
            if remote.rope_dim:
                if self._owner.remote_address(remote.rope_base, remote.rope_bytes) != remote.rope_base:
                    raise RuntimeError("URMA RoPE RVA differs from advertised layout")
            binding.context = self._make_context(remote, binding)
            binding.remote = remote

    def _make_context(self, remote, binding):
        blob = pack_context(remote, self._channels, generation=1, batch=self._batch,
                            local_kv=binding.kv.data_ptr(), local_kv_bytes=binding.kv.nbytes,
                            local_rope=binding.rope.data_ptr() if remote.rope_dim else 0,
                            local_rope_bytes=binding.rope.nbytes if remote.rope_dim else 0)
        host = torch.frombuffer(bytearray(blob), dtype=torch.int64).clone()
        assert host.numel() == HEADER_WORDS + len(self._channels)
        return host.to(self.device)

    def copy_into(self, *, layer_id, full_q_actual_seq, mapping, count):
        begin_ns = time.perf_counter_ns() if self._measure_hostbw else 0
        with self._lock:
            self._last_transfer_stats = None
            self._check_open()
            binding = self._layers[layer_id]
            if binding.context is None:
                raise RuntimeError("prepare_layer before copy_into")
            # Full compact list, not the former per-core ring staging area.
            # Capacity covers all candidate entries; native code checks actual
            # descriptorCount after validating counts/mapping on the Host.
            required = descriptor_bytes(mapping.numel() // 2, bool(binding.rope.numel()))
            if self._scratch is None or self._scratch.numel() < required:
                self._scratch = torch.empty(required, dtype=torch.uint8, device=self.device)
            with torch.npu.device(self.device):
                try:
                    self._owner.validate_context(binding.context)
                    # Explicitly order remote producers via the application's ready protocol first.
                    descriptor_ns, transfer_ns = torch.ops.avsg_urma.copy_(
                        binding.rope, binding.kv, full_q_actual_seq, mapping, count,
                        binding.context, self._scratch, self._status,
                        self._sync, self._timing, self._plan["transfer_cores"], self._measure_hostbw,
                    )
                    if self._measure_devbw or self._measure_hostbw:
                        tokens = int(self._status.cpu()[:, 2].sum().item())
                        rope_dim = binding.rope.shape[-1] if binding.rope.numel() else 0
                        payload = tokens * (binding.kv.shape[-1] + rope_dim) * binding.kv.element_size()
                        stats = dict(self._plan, lanes=self._batch, payload_bytes=payload,
                                     descriptors=tokens * (2 if rope_dim else 1), valid=bool(tokens))
                        if self._measure_devbw:
                            stats.update(device_timing_stats(
                                self._timing.cpu().tolist(), self._plan, tokens=tokens,
                                kv_dim=binding.kv.shape[-1], rope_dim=rope_dim,
                                element_bytes=binding.kv.element_size(), lanes=self._batch,
                            ))
                        if self._measure_hostbw:
                            # End after optional Device timing readback, just before
                            # publishing stats/returning. Output validation is external.
                            stats.update(host_timing_stats(time.perf_counter_ns() - begin_ns, payload,
                                                          descriptor_ns, transfer_ns))
                        self._last_transfer_stats = stats
                except Exception:
                    self._poisoned = True
                    raise
            return binding.kv, binding.rope

    def prepare_graph_copy(self, *, layer_id, mapping_capacity, workspace=None):
        """Return fixed tensors for copy_graph_. Retain this Session through graph teardown."""
        with self._lock:
            self._check_open()
            binding = self._layers[layer_id]
            if binding.context is None:
                raise RuntimeError("prepare_layer before prepare_graph_copy")
            per_token = 2 if binding.rope.numel() else 1
            if not isinstance(mapping_capacity, int) or not 0 < mapping_capacity <= (2**32 - 1) // per_token:
                raise ValueError("mapping_capacity must cover the static mapping entries")
            self._owner.validate_context(binding.context)
            slots = binding.kv.shape[0] * binding.kv.shape[1]
            scratch, seen, status = self._graph_buffers(mapping_capacity, per_token == 2, slots, workspace)
            return binding.rope, binding.kv, binding.context, scratch, seen, status

    def prepare_graph_write(self, *, layer_id, source, mapping_capacity, workspace=None):
        """Prepare the same remote layer with a registered local WRITE source."""
        with self._lock:
            self._check_open()
            target = self._layers[layer_id]
            if target.remote is None:
                raise RuntimeError("prepare_layer before prepare_graph_write")
            remote = target.remote
            binding = self._sources[source]
            if (binding.kv.shape[1:] != (remote.block_size, remote.kv_dim) or
                    str(binding.kv.dtype).removeprefix("torch.") != remote.dtype or
                    binding.rope.numel() != binding.kv.shape[0] * remote.block_size * remote.rope_dim):
                raise ValueError("WRITE source differs from remote full cache")
            context = self._make_context(remote, binding)
            self._owner.validate_context(context)
            slots = remote.blocks * remote.block_size
            scratch, seen, status = self._graph_buffers(mapping_capacity, bool(remote.rope_dim), slots, workspace)
            return binding.rope, binding.kv, context, scratch, seen, status, slots

    def _graph_buffers(self, capacity, has_rope, slots, workspace):
        if self._plan["strategy"] != "exclusive" or self._batch != 32:
            raise ValueError("single-kernel transfer requires one AIV per jetty and 32 lanes")
        size = descriptor_bytes(capacity, has_rope)
        channels = len(self._channels)
        words = graph_bitmap_words(slots, channels)
        if workspace is None:
            scratch = torch.empty(size, dtype=torch.uint8, device=self.device)
            seen = torch.empty(words, dtype=torch.int32, device=self.device)
        else:
            scratch, seen = workspace
            if scratch.numel() < size or seen.numel() < words:
                raise ValueError("shared graph workspace is too small")
        status = torch.full((channels * (STATUS_BYTES // 8),), -1, dtype=torch.int64, device=self.device)
        return scratch, seen, status

    def check_graph_copy(self, status):
        """Check completed graph work at its output boundary before using the result."""
        with self._lock:
            self._check_open()
            rows = status.reshape(len(self._channels), STATUS_BYTES // 8)[:, :3].cpu().tolist()
            for jetty, (code, phase, _) in enumerate(rows):
                if code != 0 or phase != 4:  # kComplete in avsg_urma_abi.h
                    self._poisoned = True
                    raise RuntimeError(
                        f"AVSG URMA copy failed: jetty={jetty}, code={code}, phase={phase}. "
                        "Reset match state and Session."
                    )
            return sum(tokens for _, _, tokens in rows)

    @property
    def last_transfer_stats(self):
        """Last successful call's enabled Host/Device stats; None when disabled/failed.

        Zero work returns valid=False and zero bandwidth, never the previous
        sample. Device profiling adds a timing-buffer readback after completion.
        Host time includes that readback, lock wait, allocation and validation.
        """
        with self._lock:
            return dict(self._last_transfer_stats) if self._last_transfer_stats is not None else None
