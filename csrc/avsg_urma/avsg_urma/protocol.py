# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""CPU-only context and layout protocol; no torch import or address dereference."""
from dataclasses import asdict, dataclass
import struct

MAGIC = 0x4156534755524D41
VERSION = 2
HEADER_WORDS = 24
MAX_CHANNELS = 64
MAX_BATCH = 64
STATUS_BYTES = 128
INFO_BYTES = 128
UINT64_MAX = (1 << 64) - 1
SYNC_BYTES = 4096
TIMING_ROWS = 32
TIMING_WORDS = 16
DTYPES = {"float16": (0, 2), "bfloat16": (1, 2), "int8": (2, 1),
          "float8_e4m3fn": (3, 1), "float8_e5m2": (4, 1)}


def descriptor_bytes(entries, has_rope):
    """Capacity of the flat list, including alignment padding and the zero case."""
    if not isinstance(entries, int) or entries < 0:
        raise ValueError("descriptor capacity requires nonnegative entry count")
    size = entries * (2 if has_rope else 1) * 24
    return max(128, (size + 127) // 128 * 128)


def graph_bitmap_words(slots, channels):
    """Number of int32 words for one private duplicate-check bitmap per AIV."""
    words = (slots + 31) // 32
    if channels > 1:
        line_words = STATUS_BYTES // 4
        words = (words + line_words - 1) // line_words * line_words
    return channels * words


def transfer_plan(jettys, cores=None, strategy="auto"):
    """Match HIXL avsg's ceil(cores/jettys) grouping, including unused jetties."""
    if type(jettys) is not int or not 1 <= jettys <= 32:
        raise ValueError("actual jetty count must be between 1 and 32")
    if cores is None:
        cores = jettys
    if type(cores) is not int or not jettys <= cores <= 64:
        raise ValueError("transfer_cores must be between actual jetty count and 64")
    if strategy not in ("auto", "exclusive", "shared"):
        raise ValueError("jetty_strategy must be auto, exclusive or shared")
    shared = cores > jettys
    if (strategy == "exclusive" and shared) or (strategy == "shared" and not shared):
        raise ValueError("exclusive requires cores == jettys; shared requires cores > jettys")
    step = (cores + jettys - 1) // jettys
    return dict(jettys=jettys, transfer_cores=cores, strategy="shared" if shared else "exclusive",
                active_jettys=(cores + step - 1) // step)


def device_timing_stats(rows, plan, *, tokens, kv_dim, rope_dim, element_bytes, lanes):
    """One launch: HIXL max per-jetty completion duration, ticks treated as ns.

    This matches the benchmark's simt_urma conversion; it is not Host latency or
    a cross-core absolute wall-clock interval. All rows are reset on every call.
    """
    count = tokens * (2 if rope_dim else 1)
    payload = tokens * (kv_dim + rope_dim) * element_bytes
    slots = rows[:plan["active_jettys"]]
    if len(slots) != plan["active_jettys"] or any(len(row) != TIMING_WORDS for row in slots):
        raise ValueError("invalid HIXL timing layout")
    if any(row[2] != 0 for row in slots):
        raise RuntimeError("HIXL timing reports a device error")
    work = sum(row[3] for row in slots)
    if work != count:
        raise RuntimeError(f"HIXL timing descriptor coverage mismatch: {work} != {count}")
    completion_ns = max((row[1] for row in slots), default=0)
    issue_ns = max((row[0] for row in slots), default=0)
    if count and completion_ns <= 0:
        raise RuntimeError("nonempty transfer has no completion timing")
    return dict(plan, lanes=lanes, payload_bytes=payload, descriptors=count,
                completion_us=completion_ns / 1000, issue_us=issue_ns / 1000,
                publish_count=sum(row[4] for row in slots),
                devbw_gib_s=payload * 1e9 / completion_ns / (1024 ** 3) if completion_ns else 0.0,
                valid=bool(count))


def host_timing_stats(elapsed_ns, payload_bytes, descriptor_ns=0, transfer_ns=0):
    """Host copy_into wall time, including enabled profiling overhead."""
    if elapsed_ns <= 0 or payload_bytes < 0:
        raise ValueError("Host elapsed time must be positive and payload nonnegative")
    if descriptor_ns < 0 or transfer_ns < 0 or descriptor_ns + transfer_ns > elapsed_ns:
        raise ValueError("Host stage durations must be nonnegative and within copy_into wall time")
    return dict(host_copy_us=elapsed_ns / 1000,
                host_descriptor_us=descriptor_ns / 1000,
                host_transfer_us=transfer_ns / 1000,
                host_other_us=(elapsed_ns - descriptor_ns - transfer_ns) / 1000,
                host_bandwidth_gib_s=payload_bytes * 1e9 / elapsed_ns / (1024 ** 3))


@dataclass(frozen=True)
class RemoteCache:
    """Exact registered storage layout, exchanged out of band after registration."""
    kv_base: int
    kv_bytes: int
    blocks: int
    block_size: int
    kv_dim: int
    kv_stride0: int
    kv_stride1: int
    dtype: str
    rope_base: int = 0
    rope_bytes: int = 0
    rope_dim: int = 0
    rope_stride0: int = 0
    rope_stride1: int = 0

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError("unsupported cache dtype; use FP16, BF16, INT8 or supported FP8")
        if self.blocks <= 0 or self.block_size <= 0 or self.block_size & (self.block_size - 1):
            raise ValueError("blocks must be positive; block_size must be a power of two")
        if not 0 < self.kv_dim <= 1024 or not 0 <= self.rope_dim <= 64:
            raise ValueError("KV/RoPE dimension exceeds AVSG limits")
        self._validate_region(self.kv_base, self.kv_bytes, self.kv_dim, self.kv_stride0, self.kv_stride1)
        if self.rope_dim:
            self._validate_region(self.rope_base, self.rope_bytes, self.rope_dim,
                                  self.rope_stride0, self.rope_stride1)
        elif any((self.rope_base, self.rope_bytes, self.rope_stride0, self.rope_stride1)):
            raise ValueError("absent RoPE must have zero address, size and strides")

    def _validate_region(self, base, size, dim, stride0, stride1):
        row = dim * DTYPES[self.dtype][1]
        span = (self.blocks - 1) * stride0 + (self.block_size - 1) * stride1 + row
        if base <= 0 or size <= 0 or base + size > UINT64_MAX:
            raise ValueError("invalid or overflowing registered address range")
        if stride1 < row or stride0 < self.block_size * stride1 or span > size:
            raise ValueError("cache strides/span exceed the registered region")

    def to_dict(self):
        return asdict(self)

    def addresses(self, src, dst, local_kv, local_rope):
        """Golden descriptor generation used by protocol tests and deployment tools."""
        if not 0 <= src < self.blocks * self.block_size or dst < 0:
            raise ValueError("invalid physical token index")
        block, offset = divmod(src, self.block_size)
        width = DTYPES[self.dtype][1]
        result = [(self.kv_base + block * self.kv_stride0 + offset * self.kv_stride1,
                   local_kv + dst * self.kv_dim * width, self.kv_dim * width)]
        if self.rope_dim:
            result.append((self.rope_base + block * self.rope_stride0 + offset * self.rope_stride1,
                           local_rope + dst * self.rope_dim * width, self.rope_dim * width))
        return result


def pack_context(remote, channels, *, generation, batch, local_kv, local_kv_bytes,
                 local_rope=0, local_rope_bytes=0):
    channels = tuple(channels)
    if not 0 < len(channels) <= MAX_CHANNELS or len(set(channels)) != len(channels):
        raise ValueError("channels must be nonempty, unique and within the ABI capacity")
    if any(not 0 < ch <= UINT64_MAX for ch in channels):
        raise ValueError("invalid device channel pointer")
    if batch not in (8, 32, 64) or generation <= 0:
        raise ValueError("invalid batch capacity or generation")
    if local_kv <= 0 or local_kv_bytes <= 0 or local_kv + local_kv_bytes > UINT64_MAX:
        raise ValueError("invalid local KV registration")
    if remote.rope_dim and (local_rope <= 0 or local_rope_bytes <= 0 or
                            local_rope + local_rope_bytes > UINT64_MAX):
        raise ValueError("invalid local RoPE registration")
    dtype, width = DTYPES[remote.dtype]
    words = [MAGIC, VERSION, (HEADER_WORDS + len(channels)) * 8, generation,
             len(channels), batch, width, remote.block_size,
             remote.kv_dim, remote.rope_dim, int(bool(remote.rope_dim)), dtype,
             remote.kv_base, remote.kv_bytes, remote.kv_stride0, remote.kv_stride1,
             remote.rope_base, remote.rope_bytes, remote.rope_stride0, remote.rope_stride1,
             local_kv, local_kv_bytes, local_rope, local_rope_bytes, *channels]
    return struct.pack(f"<{len(words)}Q", *words)
