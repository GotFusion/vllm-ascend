# AVSG remote KV for GLM-5.2

On Ascend 950, each model worker can mirror its paged GLM-5.2 KV and RoPE in a dedicated peer NPU's HBM. Prefill and mixed batches publish newly written slots and run ordinary sparse attention. During single-token decode, AVSG Match maps the indexer's selected tokens to physical pages. The worker reads those rows over HIXL URMA, checks transfer completion, and runs sparse attention over the local selection buffer.

This first implementation keeps the full local paged cache. Budget for it on the model NPU as well as for the full mirror on the peer. Execution is eager, with BF16 main KV, one context partition per rank and locally produced KV. Run without speculative decoding, KV connectors, HiSparse, host KV offload or sleep mode. The indexer's KV dtype remains independently configurable. Full-model token parity and device transfers still require validation on a compatible NPU pair.

## Build on Ascend Linux

Use the GLM serving environment for your checkpoint with matching CANN, ASC, PyTorch, torch_npu and vLLM. On the model hosts, build this vLLM Ascend checkout with its custom kernels. For a 950DT build, run from the repository root after sourcing your CANN environment.

```bash
export SOC_VERSION=ascend950dt_9582
COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . --no-build-isolation
```

Both endpoints need the bundled `avsg-urma` extension and the matching HIXL library. Use HIXL's AVSG SIMT source tree with the 112-byte `HixlAivOpParam` ABI and the `GetAivChannelPointers` export. Set `HIXL_SOURCE_DIR` to that source checkout. If the export is missing, apply the supplied patch to the compatible base before building HIXL.

```bash
# From the vLLM Ascend checkout, with HIXL_SOURCE_DIR exported.
git -C "$HIXL_SOURCE_DIR" apply --check "$PWD/csrc/avsg_urma/patches/hixl_channel_export.patch"
git -C "$HIXL_SOURCE_DIR" apply "$PWD/csrc/avsg_urma/patches/hixl_channel_export.patch"
(cd "$HIXL_SOURCE_DIR" && bash build.sh --aiv)
```

For a checkout that already includes the export, run the HIXL build directly. Set `HIXL_LIBRARY_DIR` to the directory containing the rebuilt `libcann_hixl.so` or `libhixl.so`. Build and load the extension against that same library on both hosts.

```bash
export LD_LIBRARY_PATH="$HIXL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
python -m pip install --no-build-isolation ./csrc/avsg_urma
python -c 'import torch, torch_npu; from avsg_urma import AvsgUrmaSession, AvsgUrmaCacheServer'
```

## Configure each worker and its peer

For TP8 with DP1, prepare eight model endpoints and eight corresponding cache servers. Each pair needs a distinct HIXL engine address on each side and a site-provided communication-resource JSON object for each NPU. Share a fresh run directory between the model worker and its peer for their JSON files.

The following example generates the two files for rank 0. Substitute the actual engine addresses and resource file paths. Repeat for each rank with its own filenames, engine addresses and peer device index. Read `layers`, `kv_dim` and `rope_dim` from the checkpoint's `num_hidden_layers`, `kv_lora_rank` and `qk_rope_head_dim`. The example uses 128 physical blocks per layer, each holding 128 tokens.

```bash
# Run in the shared run directory. jq inserts each endpoint's resource object.
jq -n --slurpfile res /path/to/peer-rank-0-comm-res.json \
  --arg engine "$PEER_ENGINE" \
  '{device: 0, local_engine: $engine, local_comm_res: $res[0], channels: 1,
    manifest: "manifest-0.json", cache: {layers: 78, blocks: 128, block_size: 128,
    kv_dim: 512, rope_dim: 64, dtype: "bfloat16"}}' > server-0.json

jq -n --slurpfile res /path/to/model-rank-0-comm-res.json \
  --arg model "$MODEL_ENGINE" --arg peer "$PEER_ENGINE" \
  '{local_engine: $model, peer_engine: $peer, local_comm_res: $res[0],
    channels: 1, timeout_ms: 30000, remote_manifest: "manifest-0.json"}' > endpoint-0.json
```

On the peer host start the cache server from the vLLM Ascend checkout. Wait for each server to print its registered manifest path. Each manifest describes the addresses held by that live process.

```bash
python csrc/avsg_urma/examples/d2d_cache_server.py --config /shared/avsg/server-0.json
```

## Run and compare generated tokens

Start with a GLM-5.2 checkpoint that fits your model workers. This TP8 example assumes Ascend-quantized weights and a main cache that uses BF16 with `--kv-cache-dtype auto`. It caps the scheduler at the peer allocation. If you change the block count, keep the deployment flag and all peer layouts consistent. Preserve the model-specific weight and indexer settings from your existing GLM deployment. [GLM-5.2 deployment](../../tutorials/models/GLM5.2.md)

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 8 --enable-expert-parallel --quantization ascend \
  --served-model-name glm --dtype bfloat16 --kv-cache-dtype auto \
  --enforce-eager --block-size 128 --num-gpu-blocks-override 128 \
  --max-model-len 12288 --max-num-batched-tokens 1024 --max-num-seqs 2 \
  --additional-config '{"avsg_config":{"endpoint_file":"/shared/avsg/endpoint-{rank}.json"}}'
```

Run the same checkpoint and command twice, first with `--additional-config '{}'` and then with AVSG enabled. For each run, capture the returned token IDs for the same prompt and generation settings. Set `RESULT_FILE` to `baseline.json` for the first run and `avsg.json` for the second.

```bash
curl --fail-with-body -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm","prompt":"The first three prime numbers are","temperature":0,"seed":7,"max_tokens":16,"return_token_ids":true}' \
  | jq -e '.choices[0].token_ids | select(length > 1)' > "$RESULT_FILE"

# After both runs:
diff -u baseline.json avsg.json
```

Also compare a prompt longer than 8192 tokens and two concurrently scheduled requests. These exercise selection-slot remapping and changing physical block tables. Stop the model workers first, then stop the cache servers. Restart both ends with freshly generated manifests after a transfer failure.
