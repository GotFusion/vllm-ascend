# SPDX-License-Identifier: Apache-2.0
"""Bounded PD ingress, drained by the model thread that owns AVSG's session."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

INGRESS_POLL_SECONDS = 0.1


@dataclass
class _MainRead:
    layer_name: str
    session: str
    source_bases: tuple[int, int]
    source_blocks: tuple[int, ...]
    destination_blocks: tuple[int, ...]
    done: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    error: BaseException | None = None


class AvsgPDIngress:
    """Keep one external main-KV page in HBM, shared by every request/layer.

    The ZMQ thread waits for each submission. The model thread performs both
    the synchronous MemFabric read and the checked AVSG publication before
    releasing that wait. Consequently READ_DONE never races the peer write,
    and attention and PD never concurrently use AVSG's registered step cache.
    """

    def __init__(
        self,
        engine: Any,
        layers: dict[str, tuple[int, Any]],
        block_tokens: int,
        num_blocks: int,
    ) -> None:
        if not layers or block_tokens <= 0 or num_blocks <= 0:
            raise ValueError("AVSG PD requires layers and positive external page geometry")
        self.engine = engine
        self.layers = layers
        self.block_tokens = block_tokens
        self.num_blocks = num_blocks
        _, transport = next(iter(layers.values()))
        kv, rope = transport.staging_cache
        self.kv_rows = torch.empty((block_tokens, kv.shape[-1]), dtype=kv.dtype, device=kv.device)
        self.rope_rows = torch.empty((block_tokens, rope.shape[-1]), dtype=rope.dtype, device=rope.device)
        self.block_bytes = tuple(t.numel() * t.element_size() for t in (self.kv_rows, self.rope_rows))
        self._offsets = torch.arange(block_tokens, device=kv.device, dtype=torch.int64)
        for _, layer_transport in layers.values():
            if layer_transport is not transport:
                raise ValueError("AVSG PD layers must share the worker-owned transport")
            if block_tokens % layer_transport.block_size:
                raise ValueError("AVSG PD external page must contain whole AVSG kernel pages")
            if num_blocks * block_tokens > layer_transport.remote_block_capacity * layer_transport.block_size:
                raise ValueError("AVSG PD allocated page table exceeds the remote manifest")
        self._queue: queue.Queue[_MainRead] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._pending: _MainRead | None = None
        self._closed = False

    def read_main(
        self,
        layer_name: str,
        session: str,
        source_bases: tuple[int, int],
        source_blocks: list[int],
        destination_blocks: list[int],
    ) -> None:
        """Called by the read thread; return only after peer publication."""
        if layer_name not in self.layers or len(source_blocks) != len(destination_blocks):
            raise ValueError("AVSG PD main source/destination layout is inconsistent")
        if any(block < 0 for block in source_blocks) or any(
            not 0 <= block < self.num_blocks for block in destination_blocks
        ):
            raise ValueError("AVSG PD main block is outside its physical page table")
        if not source_blocks:
            return
        task = _MainRead(layer_name, session, source_bases, tuple(source_blocks), tuple(destination_blocks))
        with self._lock:
            if self._closed:
                raise RuntimeError("AVSG PD ingress is closed")
            if self._pending is not None:
                raise RuntimeError("AVSG PD ingress requires a single serialized reader")
            self._pending = task
            self._queue.put_nowait(task)
        # No timeout may release P's source pages while a DMA is still active.
        # Shutdown wakes queued tasks; a running task retains ownership until
        # its synchronous read/write returns and drain() sets done.
        while not task.done.wait(INGRESS_POLL_SECONDS):
            pass
        if task.error is not None:
            raise RuntimeError("AVSG PD main transfer failed") from task.error

    def drain(self) -> None:
        """Run at the worker's ordered completion hook, never on the ZMQ thread."""
        try:
            task = self._queue.get_nowait()
        except queue.Empty:
            return
        with self._lock:
            if task.done.is_set():
                return
            task.started = True
        try:
            layer_id, transport = self.layers[task.layer_name]
            for source_block, destination_block in zip(task.source_blocks, task.destination_blocks):
                with self._lock:
                    if self._closed:
                        raise RuntimeError("AVSG PD ingress closed during transfer")
                source_ptrs = [base + source_block * size for base, size in zip(task.source_bases, self.block_bytes)]
                local_ptrs = [self.kv_rows.data_ptr(), self.rope_rows.data_ptr()]
                result = self.engine.batch_transfer_sync_read(
                    task.session, local_ptrs, source_ptrs, list(self.block_bytes)
                )
                if result != 0:
                    raise RuntimeError(f"AVSG PD MemFabric main read failed: ret={result}")
                slots = destination_block * self.block_tokens + self._offsets
                transport.publish_received(layer_id, self.kv_rows, self.rope_rows, slots)
            logger.debug(
                "AVSG PD published main KV: layer=%s pages=%d rows=%d destination_pages=[%d..%d]",
                task.layer_name,
                len(task.destination_blocks),
                len(task.destination_blocks) * self.block_tokens,
                min(task.destination_blocks),
                max(task.destination_blocks),
            )
        except BaseException as error:
            task.error = error
            raise
        finally:
            with self._lock:
                self._pending = None
                task.done.set()

    def close(self) -> None:
        """Cancel queued reads; never announce completion of an active DMA."""
        with self._lock:
            self._closed = True
            task = self._pending
            if task is not None and not task.started:
                task.error = RuntimeError("AVSG PD ingress closed before transfer")
                self._pending = None
                task.done.set()
