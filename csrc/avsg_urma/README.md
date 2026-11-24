# AVSG copy: HIXL AIV SIMT URMA D2D

[English](README.md) | [简体中文](README_zh.md)

Independent, eager-only extension for pulling selected KV tokens from **remote Device HBM**
into registered local selection HBM. This implements the explicit-context design used by
`ops-transformer/mc2/engram_fetch`: a device Tensor carries remote addresses, layouts and
channel pointers. The original `src/avsg_copy` and its model integration remain available.

## Scope and implementation

```text
Remote process                            Model process
full KV / K-RoPE in HBM                    selection KV / K-RoPE in HBM
  RegisterMem(MEM_DEVICE)                    RegisterMem(MEM_DEVICE)
            |                                         |
            +------------ HIXL Connect ---------------+
                                                      |
                                context Tensor = layout + channel pointers
                                                      |
                           AVSG match -> [src, dst] mapping/count
                                                      |
                           descriptor-generation kernel -> flat GM list
                                                      |
                           HixlLaunchAivSimt -> HixlAivSimtDevGetList
                                  internal READ batches -> Drain
                                                      |
                           target cache maintenance (small kernel)
                                                      |
                           checked completion -> local attention
```

- The selected implementation route is the design's **raw ASC fast-launch option**, not an
  additional ACLNN operator stack. It registers `torch.ops.avsg_urma.copy_` and uses
  `OpCommand::RunOpApiV2` to launch the ASC kernel on the current NPU stream.
- Host HIXL calls live in `csrc/hixl_bridge.cpp`, a translation unit that never includes Torch.
  Its PIMPL header uses standard C++ types only, so Torch/torch_npu GE headers and HIXL's
  GE headers are not combined in `binding.cpp`. CMake also separates their include paths:
  `avsg_hixl_host` uses HIXL/CANN headers, `avsg_torch_host` uses the framework's bundled
  headers, and the ASC source uses HIXL/CANN device headers. Do not add a global
  `include_directories` containing both SDK and framework GE trees.
- Include `hixl/aiv/hixl_dev_simt.h`; `hixl_dev_api.h` is the SIMD API.
- One peer, fixed registrations; transfer defaults to one core per channel and also supports
  HIXL-managed multi-core shared jetty groups. Descriptor generation uses 32 SIMT lanes
  per channel/block; HIXL's separate transfer kernel owns the multi-lane WQE
  pipeline. The Session privately owns its HIXL engine; it does not expose
  TransferAsync, reconnect or channel-sharing methods.
- Native launch validates the INT32 cumulative Q lengths, counts, mapping shape, registered
  ranges and unique destination slots **on the CPU before any READ is issued**. This incurs
  synchronization and metadata copies. It is a correctness-first implementation, not a
  claim of asynchronous performance or a drop-in throughput improvement.
- Stage 1 computes AVSG's physical token/block addresses with positive source byte strides,
  compacts valid mapping entries into a full GM descriptor list through UB and bulk DMA.
  KV and RoPE are separate descriptors. Whole-tile core partitions avoid shared GM cache lines.
- Stage 2 calls `HixlLaunchAivSimt` from the referenced HIXL source tree. With more than one
  descriptor it launches `HixlAivSimtDevGetList`; one descriptor uses `HixlAivSimtDevGet`.
  Zero work skips the transfer launch. HIXL handles partitioning, SQ admission, WQE assembly,
  publication and Drain. WQE control data can pass through UB; KV payload does not.
- FP16/BF16/INT8/FP8 (float8_e4m3fn, float8_e5m2) raw transport; KV dimension <= 1024, RoPE dimension <= 64, matching cache
  dtype and power-of-two block size. No quantization or scale conversion is performed.
- `status` contains descriptor-generation coverage and a separate HIXL byte error flag.
  Generation is synchronized and checked before any READ, so partial lists cannot be issued.
  After HIXL, a maintenance kernel invalidates destination cache lines, then the wrapper
  synchronizes and checks the transport flag. Coverage is not an independent CQ completion
  count. A failed call poisons the Session. Do not reuse match metadata or selection
  after failure: reset the AVSG cache state before creating a replacement Session.
- No GE converter, functional clone, `torch.compile`, graph capture, or concurrent channel
  sharing is supported. A raw Torch call is an internal primitive; use the resource-owning
  Python API to keep all registrations and channel entities alive.

### Two-stage baseline and future AICube work

The Python calls (`bind_selection`, `connect`, `prepare_layer`, `copy_into`) and the registered
Torch `copy_` schema are retained. CMake compiles HIXL's own
`cmake/aiv_launch/hixl_aiv_simt_launch.asc` into the extension; `csrc/hixl_transfer.cpp` adapts
plain scalar arguments to its `HixlAivOpParam`, without exposing SDK types to Torch headers.
Use a HIXL source checkout matching the registered-memory/channel ABI of the loaded library.
No HIXL source files are modified by this recipes refactor.

The adapter targets HIXL `avsg`'s **112-byte `HixlAivOpParam` ABI**: the error address
is split into `reserved[0..1]`, jetty count is `reserved[2]`, and lanes is `reserved[5]`.
Other development branches have a different 64-byte named-field ABI; those headers are not
interchangeable. The adapter checks the parameter size and reserved-array offset at compile
time. An error about missing `err_addr`, `jetty_num` or `batch` in the original two-stage
adapter was a branch-ABI mismatch; use the updated adapter rather than modifying SDK headers.

Context ABI is now version 2: rebuild the extension and restart the process. `batch_descriptors`
keeps its Python name but now selects HIXL SIMT lanes (8, 32 or 64, default 32), rather than
the former serial GetList chunk size. Channels are limited to 32. Scratch grows to cover all
candidate mapping entries, at 24 bytes per descriptor, and is reused until a larger call.

Descriptor generation uses fixed 32-token tiles, independently of `batch_descriptors` and
the transfer-core count. Each lane computes one token's addresses and writes its own UB
slice: one KV descriptor, plus one RoPE descriptor when enabled. A full tile is 768 bytes
without RoPE or 1536 bytes with RoPE. The AIV driver waits on V_MTE3, issues one exact-length
`DataCopyPad` from UB to GM, and waits on MTE3_S before reusing the single buffer. It reserves
1536 bytes of UB per generation core; KV/RoPE payload still bypasses UB.

After its last descriptor DMA, each core reuses this buffer to assemble its full 128-byte
status row. Scalar UB writes are ordered with S_MTE3, followed by one UB-to-GM DMA and an
MTE3_S completion wait. Empty cores and context-validation failures also publish a complete
row. There is no provisional status write or separate status.zero_(): every invocation
overwrites code/phase/token count and clears the transport-error/reserved words. The Host
still synchronizes and checks every row before launching HIXL; this is not a claim that
128-byte DMA is atomic against concurrent readers.

The ceil(total_tokens/32) tiles are divided contiguously across channel-count generation
cores, with the remainder assigned to the first cores. Every tile starts at a 128-byte
aligned offset within the scratch tensor. The final tile copies only valid descriptors,
including 24/48-byte one-token tails; there is no padding inside the flat GM list. For the
default 2048-token smoke, generation emits 64 DMA copies totalling 96 KiB with RoPE.
Context ABI v2, Python interfaces, Host barriers and the HIXL transfer stage are unchanged.
Rebuild the native extension to use this kernel; measure HostDescriptor and HostCopy on
the target before judging performance, since cache maintenance and synchronization remain.

This is a sequential correctness baseline: descriptor kernel + Host status barrier + transfer
kernel + cache-maintenance kernel + stream synchronization. The extra launch and status
round trip must be included in end-to-end timings; no speedup or communication/compute overlap
is claimed. Multi-core shared jetty is optional as described below; AICube tile pipelining
is not implemented. The existing `d2d_smoke.py` invocation retains exclusive defaults.
Use the same device pair, resource files, channels, mapping and payload when comparing against
commit `315fc2a`. Profile descriptor generation, HIXL transfer, maintenance and Host wall time
separately; a transfer-kernel-only timing does not represent `copy_into` latency.

### Jetty strategy and Device bandwidth

```python
session = AvsgUrmaSession(
    ..., channels=4, transfer_cores=8, jetty_strategy="shared",
    batch_descriptors=32, measure_devbw=True,
)
# After each successful copy_into (return values are unchanged):
stats = session.last_transfer_stats
print(stats["completion_us"], stats["devbw_gib_s"])
```

`channels` requests jetties on both endpoints; `transfer_cores` configures only the Client's
transfer grid. It defaults to the actual exported channel count. `jetty_strategy="auto"`
selects exclusive for cores==jettys and shared for cores>jettys; explicit `exclusive`/`shared`
reject inconsistent counts. The software bound is actual jettys <= cores <= 64 (jettys<=32).
The requested grid must also be supported by the actual target hardware; a software bound is
not hardware discovery. HIXL checks groupSize*lanes against SQ capacity during kernel setup.

Shared mode passes reserved[6]=1 and a 4096-byte GM barrier via reserved[7..8]. The native
wrapper resets it on every invocation. HIXL partitions cores using step=ceil(cores/jettys),
so active groups=ceil(cores/step); e.g. 5 cores/4 jetties uses only 3 groups. Group leaders
publish and drain while group members fill separate SQ slices. Independent external users
still must not drive these channels concurrently.

With `measure_devbw=True`, reserved[3..4] points to HIXL's 32x16 int64 timing slots (4096B),
reset for every call. Completion is slot[1], issue slot[0], status slot[2], completed descriptors
slot[3], publish count slot[4]. Shared mode uses leader rows; exclusive uses per-core rows.
The reader validates status and descriptor coverage, then uses the benchmark's simt_urma
conversion of 1 ns/tick:

```text
payload_bytes = tokens * (kv_dim + rope_dim) * element_bytes
completion_us = max(active timing rows' completion ticks) / 1000
DevBW/GiB/s = payload_bytes / (completion_us * 1e-6) / 1024^3
```

This is one launch's HIXL transfer window (Start through post-Drain Finish), not Host wall
time, absolute cross-core earliest-start/latest-end, or pure network time. Shared Start
timestamps are after some setup, unlike the earlier exclusive Start timestamp. Descriptor
generation, Host checks, cache-maintenance kernel and timing readback are excluded from the
Device denominator. Profiling adds readback overhead; it defaults off and exposes None.
Zero work returns valid=False with zero bytes/bandwidth, never the preceding sample.

Host profiling is independently enabled with `measure_hostbw=True` (default False for
Session, always enabled in smoke). `host_copy_us` uses perf_counter_ns from entry to
copy_into, before lock acquisition, to just before publishing its stats and returning.
It includes lock wait, first-call workspace growth, context/mapping validation, both stages,
cache maintenance, synchronization, status readback and enabled Device profiling readback.
It excludes input creation and output correctness checks in the caller. The bandwidth is
actual copied payload bytes / Host wall time / 1024^3, exposed as `host_bandwidth_gib_s`.
Turn off Device profiling when measuring Host cost without timing-buffer readback overhead.

Host profiling also returns these nested wall-time stages from the native wrapper:

| Stats field / smoke column | Start | End |
|---|---|---|
| `host_descriptor_us` / HostDescriptor/us | Before descriptor RunOpApiV2 submission | After stream sync, status D2H and all generation coverage checks |
| `host_transfer_us` / HostTransfer/us | Before transfer RunOpApiV2 submission | After HIXL READ/Drain, maintenance kernel and stream sync return |
| `host_other_us` / HostOther/us | Residual | HostCopy minus the two non-overlapping stages |

Both stages use C++ steady_clock, not device events. They include dispatch/queue wait;
RunOpApiV2 is explicitly called with sync=true: its default false may return after enqueue,
even when the callback itself synchronizes. In commit 9284379 this could make HostTransfer
measure only enqueue latency, with execution wait deferred to the following status.cpu()
and misclassified as HostOther. Rebuild the corrected native extension before comparing
HostTransfer with DeviceTime. This fix can add wrapper synchronization overhead.
descriptor synchronization may also wait for earlier stream work such as buffer clears.
The transfer stage excludes the subsequent error-flag readback/check, which is in HostOther
along with allocation, validation, profiling and Python overhead outside the stages.
Zero work still measures generation/confirmation but returns zero transfer time. The smoke
summary prints arithmetic means of stage durations, while bandwidth still uses total bytes
divided by total Host time. Native copy_ now returns `(descriptor_ns, transfer_ns)` (zeroes
when measure_host=False); Session still returns `(kv, rope)`. Rebuild the extension and
restart Python when updating; the internal schema adds measure_host=False.

Client smoke flags: `--transfer-cores 8 --channels 4 --jetty-strategy shared --lanes 32 --devbw
--warmup 2` (the two lines here form one argument list). Keep the Server at `--channels 4`;
only the Client consumes the grid/profiling flags. DEVRESULT prints each of the 8 measured
steps; DEVSUMMARY uses sum(payload_bytes)/sum(completion_time), excluding warmup and zero
calls. It is not the arithmetic mean of bandwidths. Raw Torch copy_ now additionally requires
sync, timing and transfer_cores arguments; use Session to allocate them consistently.

## HIXL dependency patch

This extension needs a public **borrowed channel pointer export**. The referenced HIXL
checkout does not expose it in its original public API.

`patches/hixl_channel_export.patch` is based on HIXL commit
`4b1b53ac3d94175df4760bc4006198a30982e4f5`. It also applies cleanly to master
`259a46668392893944ca865f318650f9f495281b`; the corresponding HIXL `avsg` branch
contains the adapter as commit `a11ce32`. It:

1. Preserves all actual channel pointers collected by `AivEngine::Connect` in the existing
   channel manager, instead of exporting only the first pointer.
2. Adds `Hixl::GetAivChannelPointers(peer, channels)` through the public/impl/engine layers.
3. Rejects duplicate/null/mismatched channel lists and unhealthy/removed peers.
4. Adds HIXL channel-manager unit cases for multiple channels, health and invalidation.

The exporter returns a borrowed snapshot, not deep copies or an automatic transport-wide lease.
Exclusive use and lifetime are enforced by this extension's private engine owner and serialized,
synchronous Python Session. Do not externally drive, disconnect or reset its channels. Device
reset or process failure invalidates the entire Session. Both endpoints must stay alive until
READ completion; the source process must not close while incoming readers still use its memory.

On the Ascend Linux build host, apply the patch **once**, then rebuild HIXL using its own
documented entry point:

```bash
# Set these to your actual checkouts; the patch contains no machine-specific source paths.
git -C "$HIXL_SOURCE_DIR" apply --check "$RECIPES_ROOT/ops/ascendc/avsg_urma/patches/hixl_channel_export.patch"
git -C "$HIXL_SOURCE_DIR" apply "$RECIPES_ROOT/ops/ascendc/avsg_urma/patches/hixl_channel_export.patch"
cd "$HIXL_SOURCE_DIR"
bash build.sh --aiv
```

If the patch is already applied, do not apply it again. Review conflicts against your HIXL
version. A header-only update without rebuilding the library will fail to link the new API.
The dependency's local source changes must be kept with the recipes branch's patch; pushing
only an unrelated HIXL branch is not required to reproduce the extension.

## Build the extension

The referenced SIMT primitives target `dav-3510` (Ascend 950). Do not infer support from the
old AVSG operator's A2/A3 registrations. Confirm the actual target and compatible CANN/ASC,
HIXL and torch_npu versions on your test host. No device IDs or networking configuration are
assumed by the code.

```bash
# Source the CANN environment for your installation first.
# Required environment variables:
# ASCEND_HOME_PATH: active CANN toolkit
# HIXL_SOURCE_DIR: patched HIXL source checkout, including primitive headers
# HIXL_LIBRARY_DIR: directory containing the rebuilt libcann_hixl.so (or libhixl.so)

cd "$RECIPES_ROOT/ops/ascendc/avsg_urma"
python -m pip install --no-build-isolation .
```

Build dependencies: CMake >= 3.20 with ASC language package, a supported Linux host compiler,
Python development headers, setuptools, PyTorch, torch_npu, CANN and rebuilt HIXL. Ensure
HIXL and PyTorch use compatible C++ ABIs. Runtime shared-library dependencies must be visible
through your installation/RPATH/LD_LIBRARY_PATH configuration on both endpoints.
Import the new package with `from avsg_urma import AvsgUrmaSession, AvsgUrmaCacheServer`.
It does not require rebuilding the old `custom_ops` package unless AVSG match is also needed.

If compilation reports redefinitions such as `ge::GRAPH_FAILED` from `graph/error_codes.h`,
include the `hixl_bridge.cpp` isolation fix when updating the extension. Different GE header
families may define the same constants with different include guards. The reported CANN 9.2
case combines its `include/graph/error_codes.h` with the Python environment's
`torch_npu/include/graph/ge_error_codes.h`. Splitting only the source files is insufficient:
when the shared target includes torch_npu first, even `hixl_bridge.cpp` can pick up a
framework `graph/*` header indirectly from an SDK include. The fix uses separate object
targets with independent include lists. Include reordering in C++ or disabling `-Werror`
does not fix that dependency boundary. Do not edit installed CANN headers or
predefine their guards to hide the conflict. Reinstall the updated source with
`python -m pip install --no-build-isolation --no-cache-dir -v .`.
The generated build directory contains `compile_commands.json`. The compile command for
`hixl_bridge.cpp` should target `avsg_hixl_host` and must not contain a torch_npu include
directory; check for user-supplied global `CPATH`/`CPLUS_INCLUDE_PATH`/`CXXFLAGS` as well if
it still selects framework GE headers.
If the error persists, collect the failing translation unit and the compiler's previous-definition
path; a conflict wholly within the framework or SDK requires its own include-stack diagnosis.

## Control plane and remote cache readiness

Use actual deployment settings for `local_engine`, Device ID, communication-resource JSON,
peer engine and channel count. Both endpoints must configure matching jetty counts. The
constructor forces AIV direct on and UBMEM off; remote addresses are URMA RVAs, not local
load/store mappings.

1. Construct an endpoint on each process and register all relevant tensors before Connect.
2. Server calls `publish_layout()`: it waits for current producer writes, seals registrations
   and returns a JSON-compatible layer manifest. Send this through the application's control
   plane after the consumer confirms its registrations are complete.
3. Consumer calls `connect(peer)` and `prepare_layer(layer_id, manifest[layer])`.
   `GetAddr` is called with the exact nonzero RVA, not the ambiguous first-region query.
4. For every later cache update, synchronize the source writer and exchange a data-ready
   version/barrier **before** calling copy. The manifest is a storage layout, not a promise
   that all future writes are ready.
5. After all READs complete, consumer closes its Session and notifies the server before the
   latter closes/releases full caches. The smoke example implements this close handshake.

`publish_layout()` does not start a TCP service or send metadata itself. Transport for the
application's registration/readiness handshake is independent of the URMA data plane.

## Python model interface

```python
from avsg_urma import AvsgUrmaCacheServer, AvsgUrmaSession, RemoteCache

# Remote process, before receiving any Connect:
server = AvsgUrmaCacheServer(
    local_engine=server_engine, device=server_device,
    local_comm_res=server_resource_json, channels=channel_count,
)
server.register_layer(layer_id, full_kv_hbm, full_rope_hbm)
manifest = server.publish_layout()
# Send manifest after the consumer registration barrier; keep server/tensors alive.

# Model process, initialization:
session = AvsgUrmaSession(
    local_engine=client_engine, device=client_device,
    local_comm_res=client_resource_json, channels=channel_count,
)
session.bind_selection(layer_id, selection_kv, selection_rope)
# Application registration barrier and manifest exchange here.
session.connect(server_engine)
session.prepare_layer(layer_id, RemoteCache(**manifest[str(layer_id)]))

# Model forward, after a remote-data-ready barrier and the original AVSG match:
kv, rope = session.copy_into(
    layer_id=layer_id, full_q_actual_seq=cu_q_lens,
    mapping=stage_two_copy_mapping, count=stage_two_copy_count,
)
# attention(..., key=kv, key_rope=rope)

# Shutdown: stop new forwards, close consumer, notify peer, then close server.
session.close()
server.close()  # on the remote process, after receiving the consumer's close notification
```

The full caches must be real remote HBM allocations, not swapped Host allocations. The source
manifest is not enough to migrate a model: its remote full block table must correspond to
the physical `src` indices emitted by AVSG match. The model must explicitly arrange remote
KV production/update and readiness. No model YAML is silently changed to pretend that this
remote-cache placement already exists. Multiple layers may share a Session sequentially;
register/bind every layer before Connect and prepare each layer's remote manifest separately.

When RoPE is absent pass `None` to register/bind. Session creates the required empty device
Tensor. Selection must be contiguous and remain at its registered address; source caches
are rank-3 `[blocks, block_size, dim]` with positive byte strides. Normalize a model's singleton
head axis with `squeeze(-2)` before registering a rank-4 cache.

## Two-process device smoke test

The supplied test allocates full KV/RoPE in the server NPU's HBM and selection in the client
NPU's HBM. Defaults are 8192 full tokens, 4096 selection slots, 2048 copied tokens per step,
block size 128, KV dimension 512, RoPE dimension 64, and one-byte float8_e4m3fn. It verifies
eight steps, untouched slots, zero-copy calls and a source-lifetime handshake with no CPU
transport fallback. `--dtype float8_e5m2` selects the other supported FP8 encoding.

Full cache is 4.5 MiB; selection is 2.25 MiB; each nonzero call copies 1,179,648 bytes
(1.125 MiB) using 4096 descriptors (96 KiB metadata). Eight measured steps copy 9 MiB total.
The source pattern is raw byte `(token*dim+feature)%123`, reinterpreted as FP8 without
quantization; selection starts at sentinel byte 0x80. Validation compares uint8 views exactly,
not floating-point values. No FP8 arithmetic/scale conversion is performed.

Both processes accept `--full-tokens` (default 8192), `--selection-tokens` (4096),
`--copy-tokens` (2048), and `--dtype` (float8_e4m3fn). Full/selection counts must be multiples
of 128 and copy count must fit both. Keep size/dtype settings consistent across endpoints.
Choose e.g. `--copy-tokens 2049` on both to exercise an uneven tail; the new default no longer
guarantees a short tail for every grid/lanes setting.

Every measured call prints HOSTRESULT (HostCopy/us, Bandwidth/GiB/s), even without --devbw.
HOSTSUMMARY divides sum(payload_bytes) by sum(host_copy_us); with --devbw the existing
DEVRESULT/DEVSUMMARY are also printed. Warmup and zero-work calls are excluded from both
summaries. Input generation and output checking are outside HostCopy/us.

```bash
# Start server first. Substitute explicit deployment values and resource JSON paths.
python examples/d2d_smoke.py server \
  --device "$SERVER_DEVICE" --local-engine "$SERVER_ENGINE" \
  --local-comm-res-file "$SERVER_COMM_RES" --channels "$CHANNELS" \
  --control-host "$CONTROL_HOST" --control-port "$CONTROL_PORT"

python examples/d2d_smoke.py client \
  --device "$CLIENT_DEVICE" --local-engine "$CLIENT_ENGINE" --peer-engine "$SERVER_ENGINE" \
  --local-comm-res-file "$CLIENT_COMM_RES" --channels "$CHANNELS" \
  --control-host "$CONTROL_HOST" --control-port "$CONTROL_PORT"
```

The test uses fixed synthetic cache sizes, not model paths. The control host must be reachable
and the server must be able to bind it. Start with one channel, then repeat with the intended
multi-channel count. A real model correctness test must also validate top-k remapping and
cache reuse across steps; this smoke test isolates the new data plane.

## Validation

If linking reports that `avsg_urma.asc.o`, `hixl_bridge.cpp.o` or `binding.cpp.o`
is incompatible with `aicorelinux`, check the final link command, not the GE
include paths. `_native` is a Host shared library containing the packaged kernel
and must be linked by the ASC driver with `-shared`. The build explicitly uses
`SHARED` and `target_link_options(... -shared)` instead of relying on ASC's default
module flags. After updating, rerun the installation with verbose output:

```bash
python -m pip install --no-build-isolation --no-cache-dir -v .
# If needed, locate and inspect the generated final link command:
find build -path '*/CMakeFiles/_native.dir/link.txt' -print -exec cat {} \;
```

The final command must use the ASC compiler driver with `-shared`; do not replace
it with a direct `ld.lld -m aicorelinux` invocation. This does not change the
Device kernel's `dav-3510` compilation target.

CPU-only tests do not require torch, CANN or HIXL:

```bash
cd "$RECIPES_ROOT/ops/ascendc/avsg_urma"
PYTHONPATH=. python -m unittest discover -s tests -v

# Optional but recommended: HIXL_SOURCE_DIR makes the pipeline suite compile the
# real Host adapter with checkout launcher/parameter headers (no CANN required).
HIXL_SOURCE_DIR="$HIXL_SOURCE_DIR" PYTHONPATH=. python -m unittest discover -s tests -p test_pipeline.py -v
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude -Itests/stubs tests/kernel_cpu_test.cpp -o /tmp/avsg_urma_cpu
/tmp/avsg_urma_cpu
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude tests/abi_check.cpp -o /tmp/avsg_urma_abi
/tmp/avsg_urma_abi
g++ -std=c++17 -Wall -Wextra -Werror -Iinclude -Itests/stubs/cann -Itests/stubs \
  csrc/hixl_bridge.cpp tests/bridge_cpu_test.cpp -o /tmp/avsg_urma_bridge
/tmp/avsg_urma_bridge
```

The pipeline CPU tests compile the **actual descriptor kernel body** with test-only
execution-space stubs. They check multi-lane UB assembly, DMA byte counts and event-call
order, whole-tile core partitions, 1/31/32/33-token boundaries, a 145-token multi-tile case,
compact ordering, strided addresses, KV/RoPE, zero work, empty cores and descriptor bounds,
then consume the list with
a CPU copy stand-in. A separate test compiles the real Host adapter against a fake launcher
and checks the channel-table address, READ direction, descriptor count and launch config.
These tests do not execute HIXL's kernel and do not validate its admission/Drain/error paths,
hardware fences, cache maintenance, NIC ordering, CANN syntax or real completion behavior.
The header-isolation regression compiles the real HIXL bridge against fake SDK/framework
trees: a leaked framework include root must reproduce `GRAPH_FAILED` redefinition, while
the SDK-only bridge must compile, link and run. These are compiler regressions, not a test
against the installed CANN 9.2 or torch_npu headers.

For HIXL's affected unit suite, use its own `bash tests/run_test.sh -s hixl` entry point.
Run repository `pre-commit run --all-files` before committing. Device compilation, native
Torch binding checks, HIXL unit execution and real two-endpoint NPU correctness/performance
are still required on an appropriately configured Ascend Linux host.
