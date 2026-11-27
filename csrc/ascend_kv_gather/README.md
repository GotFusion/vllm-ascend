# Ascend KV gather for GLM eager serving

This independently built extension supplies the gather part of the existing
AVSG transport. HIXL still publishes newly computed KV and imports PD pages into
peer HBM. Both libraries register the **same peer KV and RoPE tensors**. The
client checks their addresses, lengths, strides and layer layout against the
HIXL manifest before accepting a connection.

The imported SDK vector multi-SGE builder issues UB_CTP WRITEs from selected
peer rows into two bounded receive buffers on Decode. Decode scatters those
rows into its existing 8192-row selection cache, then runs the existing SFA.
Historical main KV remains offloaded. Each BF16 KV row is 1024 bytes and each
BF16 RoPE row is 128 bytes; the source's synthetic packed 656-byte fixture is
not used.

The first integration uses a finite AIV launch per bank and request. It does
not reserve an AIV core while idle, allowing the peer to share a card with
Prefill. Match mappings are read on the CPU, TCP carries control messages, and
Torch performs the local scatter. These costs are part of end-to-end latency.
This is an eager implementation, not the source's resident AICPU/AIV service
or a device-only pipeline. It never falls back to HIXL READ after an error.

## Build

Use Ascend 950, CANN 9.3 with ASC/HCOMM, and the same Torch/torch_npu runtime as
the serving worker. Source the intended SDK's `set_env.sh` first. This extension
does not install an AICPU package or modify the CANN installation.

```bash
python3 csrc/ascend_kv_gather/sync_vendor.py --check
python3 -m pip install --no-build-isolation ./csrc/ascend_kv_gather
```

The native library has a C ABI and is loaded with `ctypes`. It uses the caller's
current device and does not initialize, reset or finalize Torch's ACL runtime.
KV and RoPE use separate channels, each containing one registered memory
region, so each WRITE uses its bank's matching remote protection token.

## Enable on the existing PD deployment

Add the same `kvgather` section to the peer's `server-0.json` and Decode's
`endpoint-0.json`. Keep all existing HIXL and PD fields. The port is a separate
bootstrap/control listener; it must be unused and reachable by Decode.

```json
{
  "kvgather": {
    "host": "127.0.0.1",
    "port": 20170,
    "cores": 1,
    "timeout": 300
  }
}
```

`library` can optionally name an explicit `libascend_kv_gather.so` path;
relative paths resolve against the JSON file. Otherwise the installed package
library is used. Start the existing `csrc/avsg_urma/examples/d2d_cache_server.py`
entry point with the updated server configuration, then start P/D as before.
The peer publishes its HIXL manifest before waiting for HCOMM initialization.
Do not reuse a manifest from an earlier process.

Add `--shutdown-timeout 60` to both `vllm serve` commands. To stop the services,
signal each top-level API process and wait for its EngineCore to finish before
stopping the peer. Do not signal all child processes at the same time. vLLM
0.30 defaults to a zero-second shutdown wait and can kill EngineCore before
`Worker.shutdown()` closes the gather session. An unexpected control-socket
EOF remains an error; the peer must not treat a crashed client as a successful
protocol close.

The backend is selected only when `kvgather` is present in the endpoint. Without
that section, the original AVSG HIXL gather is unchanged. GLM-5.3 validation
keeps real first-five-layer weights, MTP0, eager, PD TP1, BF16 KV/indexer and
the existing 64K/128K/256K AISBench scenarios: eight requests, concurrency eight,
64 generated tokens per request, seed 1024, target 90% prefix reuse.

## Validation

Host tests exercise validation and control protocol with CPU/mock transfers;
they cannot establish native correctness or performance. On two available
950 cards, run the real native operator test before starting the model:

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 python3 csrc/ascend_kv_gather/examples/d2d_smoke.py \
  --rounds 32 --cores 1 --output /tmp/kvgather-smoke.json
```

This test includes repeated channel use, two layers, source rows above 20,000,
different request sizes up to 8192, all-hit/no-transfer requests, untouched
selection rows, and updated source values observed by a real Torch consumer.
It records native library SHA-256 and completed transfer counts. Deterministic
operator tensors are separate from real-checkpoint model validation. Keep the
model smoke and AISBench logs in opsLearning; do not reuse prior AVSG results
as KV gather measurements.

## Updating the source snapshot

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) locks the imported commit and
per-file SHA-256. The vendor files are unchanged, including licenses. Only the
vector builder and shared protocol helpers are compiled; the imported host
and transfer files are the reference for the separately maintained adapter.

```bash
python3 csrc/ascend_kv_gather/sync_vendor.py \
  --source /path/to/ascend-kv-gather --revision COMMIT_SHA
python3 csrc/ascend_kv_gather/sync_vendor.py --check
```

Review changes against `native/`, rebuild and rerun host, two-device and model
checks after each source update. An updated checksum manifest is not runtime
validation. Native code derived from ascend-kv-gather and its vendor sources
retain the [CANN Open Software License 2.0](vendor/ascend-kv-gather/LICENSE).
