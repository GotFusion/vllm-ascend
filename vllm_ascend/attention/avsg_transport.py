# SPDX-License-Identifier: Apache-2.0
"""One eager worker's connection to its GLM sparse KV cache in peer HBM."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch

# Match uses the same power-of-two selection pool as the GLM recipe.
AVSG_SELECTION_TOKENS = 8192


class AvsgTransport:
    """A worker serializes layers and requests on its current compute stream."""

    def __init__(
        self,
        endpoint_file: str,
        rank: int,
        device: torch.device,
        layer_ids: Sequence[int],
        block_size: int,
        kv_dim: int,
        rope_dim: int,
        max_tokens: int,
        dtype: torch.dtype,
    ) -> None:
        # The native extension depends on the worker's NPU runtime and patched HIXL.
        from avsg_urma import AvsgUrmaSession, RemoteCache
        from avsg_urma.protocol import descriptor_bytes, graph_bitmap_words

        path = Path(endpoint_file.replace("{rank}", str(rank)))
        endpoint = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((path.parent / endpoint["remote_manifest"]).read_text(encoding="utf-8"))
        remote = {layer: RemoteCache(**manifest[str(layer)]) for layer in layer_ids}
        if not remote or any(
            layout.blocks * layout.block_size > torch.iinfo(torch.int32).max
            or (layout.block_size, layout.kv_dim, layout.rope_dim, layout.dtype)
            != (block_size, kv_dim, rope_dim, str(dtype).removeprefix("torch."))
            for layout in remote.values()
        ):
            raise ValueError("AVSG remote manifest must match the GLM layers, cache dtype and page layout")
        kvgather_options = None
        if "kvgather" in endpoint:
            # Import only inside an NPU worker that explicitly selected this
            # backend. Missing binaries or a failed connection must not silently
            # fall back to the original HIXL READ implementation.
            from ascend_kv_gather import KVGatherClient, parse_endpoint

            kvgather_options = parse_endpoint(endpoint["kvgather"], path.parent)
            if (kv_dim, rope_dim, dtype) != (512, 64, torch.bfloat16):
                raise ValueError("KV gather requires BF16 GLM KV512 and RoPE64 caches")
        self._remote_blocks = {layer: layout.blocks for layer, layout in remote.items()}
        self._block_size = block_size
        self._max_tokens = max_tokens
        selection_blocks = AVSG_SELECTION_TOKENS // block_size
        source_blocks = (max_tokens + block_size - 1) // block_size
        # HIXL pairs the two registered regions with the peer's KV and RoPE banks.
        # Disjoint views keep read results and unpublished step rows independent.
        buffer_blocks = selection_blocks + source_blocks
        kv_buffer = torch.empty((buffer_blocks, block_size, kv_dim), device=device, dtype=dtype)
        rope_buffer = torch.empty((buffer_blocks, block_size, rope_dim), device=device, dtype=dtype)
        self._step_rows = torch.arange(max_tokens, device=device, dtype=torch.int32)
        self.selection_block_table = torch.arange(selection_blocks, device=device, dtype=torch.int32).unsqueeze(0)
        self._one_query = torch.ones(1, device=device, dtype=torch.int32)
        channels = endpoint.get("channels", 1)
        capacity = max(AVSG_SELECTION_TOKENS, max_tokens)
        slots = max(AVSG_SELECTION_TOKENS, max(self._remote_blocks.values()) * block_size)
        workspace = (
            torch.empty(descriptor_bytes(capacity, True), device=device, dtype=torch.uint8),
            torch.empty(graph_bitmap_words(slots, channels), device=device, dtype=torch.int32),
        )
        self._reads = {}
        self._writes = {}
        self._gather_client = None
        self._closed = False
        self._gather_closed = False
        self._hixl_closed = False
        self._session = AvsgUrmaSession(
            local_engine=endpoint["local_engine"],
            device=device.index,
            local_comm_res=json.dumps(endpoint["local_comm_res"]),
            channels=channels,
        )
        try:
            # Register each local allocation once. Each context then pairs that
            # allocation with a different layer's remote base addresses.
            self._selection_kv, self._selection_rope, self._source_kv, self._source_rope = self._session.bind_staging(
                0, "step", kv_buffer, rope_buffer, selection_blocks
            )
            self._session.connect(endpoint["peer_engine"], timeout_ms=endpoint.get("timeout_ms", 30000))
            for layer, layout in remote.items():
                self._session.prepare_layer(0, layout)
                if kvgather_options is None:
                    self._reads[layer] = self._session.prepare_graph_copy(
                        layer_id=0, mapping_capacity=AVSG_SELECTION_TOKENS, workspace=workspace
                    )
                self._writes[layer] = self._session.prepare_graph_write(
                    layer_id=0, source="step", mapping_capacity=max_tokens, workspace=workspace
                )
            if kvgather_options is not None:
                self._gather_client = KVGatherClient(
                    self._selection_kv,
                    self._selection_rope,
                    device=device.index,
                    layer_ids=list(remote),
                    layer_layouts={str(layer): manifest[str(layer)] for layer in remote},
                    **kvgather_options,
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._closed = True
        try:
            if self._gather_client is not None and not self._gather_closed:
                self._gather_client.close()
                self._gather_closed = True
        finally:
            if not self._hixl_closed:
                self._session.close()
                self._hixl_closed = True

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def remote_block_capacity(self) -> int:
        """Number of physical kernel pages available to every main-KV layer on this worker."""
        return min(self._remote_blocks.values())

    @property
    def staging_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Paged views of the WRITE buffers shared across layers for this scheduler step."""
        return self._source_kv.unsqueeze(2), self._source_rope.unsqueeze(2)

    def staging_slots(self, slots: torch.Tensor) -> torch.Tensor:
        """Map remote slot IDs to step-local row numbers, retaining padding holes."""
        if slots.numel() > self._max_tokens:
            raise ValueError("AVSG batch exceeds registered WRITE capacity")
        rows = self._step_rows[: slots.numel()].view_as(slots).to(slots.dtype)
        return torch.where(slots >= 0, rows, -1)

    def publish_staged(self, layer_id: int, slots: torch.Tensor) -> None:
        """Write fresh rows from the registered step cache into their physical slots on the peer."""
        if self._closed:
            raise RuntimeError("AVSG transport is closed")
        slots = slots.reshape(-1)
        if slots.numel() > self._max_tokens:
            raise ValueError("AVSG batch exceeds registered WRITE capacity")
        local = (slots >= 0).nonzero().flatten()
        count = local.numel()
        if not count:
            return
        mapping = torch.stack((slots[local].int(), local.int()), dim=-1).view(1, 1, count, 2)
        counts = torch.full((1, 1), count, device=slots.device, dtype=torch.int32)
        src_rope, src_kv, context, scratch, seen, status, remote_slots = self._writes[layer_id]
        try:
            torch.ops.avsg_urma.write_graph_(
                src_rope, src_kv, self._one_query, mapping, counts, context, scratch, seen, status, remote_slots
            )
            self._session.check_graph_copy(status)
        except Exception:
            self.close()
            raise

    def publish(self, layer_id: int, kv: torch.Tensor, rope: torch.Tensor, slots: torch.Tensor) -> None:
        """Publish from a caller-owned paged reference cache, e.g. for two-endpoint parity checks."""
        if self._closed:
            raise RuntimeError("AVSG transport is closed")
        if kv.shape[:3] != (kv.shape[0], self._block_size, 1) or rope.shape[:3] != kv.shape[:3]:
            raise ValueError("AVSG requires separate paged KV and RoPE tensors with one cache head")
        if kv.shape[0] > self._remote_blocks[layer_id]:
            raise ValueError(f"AVSG peer layer {layer_id} needs at least {kv.shape[0]} physical blocks")
        slots = slots.reshape(-1)
        if slots.numel() > self._max_tokens:
            raise ValueError("AVSG batch exceeds registered WRITE capacity")
        local = (slots >= 0).nonzero().flatten()
        valid = slots[local].long()
        blocks, offsets = valid // self._block_size, valid % self._block_size
        for staging, cache in zip(self.staging_cache, (kv, rope)):
            staging.flatten(0, 1).index_copy_(0, local, cache[blocks, offsets])
        self.publish_staged(layer_id, slots)

    def publish_received(self, layer_id: int, kv: torch.Tensor, rope: torch.Tensor, slots: torch.Tensor) -> None:
        """Import a PD ingress page on the model thread before acknowledging its producer.

        The connector owns the bounded ingress tensors. The source page can be
        larger than a Decode step (e.g. 128 rows versus 60 scheduled tokens), so
        reuse the registered WRITE source in checked, synchronous chunks.
        """
        if self._closed:
            raise RuntimeError("AVSG transport is closed")
        if layer_id not in self._writes:
            raise ValueError(f"AVSG peer has no layer {layer_id}")
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or kv.shape != (slots.numel(), self._source_kv.shape[-1])
            or rope.shape != (slots.numel(), self._source_rope.shape[-1])
            or any(t.dtype != self._source_kv.dtype or t.device != self._source_kv.device for t in (kv, rope))
            or slots.device != self._source_kv.device
        ):
            raise ValueError("AVSG PD ingress requires matching KV/RoPE rows and integer slots on the worker device")
        # PD is an admission path, outside attention's hot loop. Validate before
        # publishing any rows, rather than allowing a partial invalid transfer.
        if torch.any((slots < 0) | (slots >= self._remote_blocks[layer_id] * self._block_size)):
            raise ValueError("AVSG PD destination slots exceed the peer cache")
        for start in range(0, slots.numel(), self._max_tokens):
            end = min(start + self._max_tokens, slots.numel())
            self._source_kv.flatten(0, 1)[: end - start].copy_(kv[start:end])
            self._source_rope.flatten(0, 1)[: end - start].copy_(rope[start:end])
            self.publish_staged(layer_id, slots[start:end])

    def gather(
        self, layer_id: int, q_lens: torch.Tensor, mapping: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("AVSG transport is closed")
        try:
            if self._gather_client is not None:
                kv, rope = self._gather_client.gather(layer_id, q_lens, mapping, counts)
            else:
                rope, kv, context, scratch, seen, status = self._reads[layer_id]
                torch.ops.avsg_urma.copy_graph_(rope, kv, q_lens, mapping, counts, context, scratch, seen, status, None)
                self._session.check_graph_copy(status)
        except Exception:
            self.close()
            raise
        return kv.unsqueeze(2), rope.unsqueeze(2)
