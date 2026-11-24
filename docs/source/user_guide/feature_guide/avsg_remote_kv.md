# AVSG remote KV for GLM-5.2

On Ascend 950, each GLM-5.2 model worker can keep its complete per-layer main KV and RoPE cache on a dedicated peer NPU. The model worker retains the Lightning Indexer's cache and shares two buffers across attention layers. The write buffer holds one scheduler step of new rows. The read buffer holds 8192 selection slots.

The worker publishes new rows to the peer before attention. For each query in prefill, chunked prefill, decode or a mixed batch, it maps the indexer's causally visible positions through that request's physical page table and reads the selected rows over HIXL URMA. Prefix cache hits use the same remote page IDs. The available peer pages and the HBM available to the local indexer limit the scheduler's block count.

Use eager execution, BF16 main KV, one context partition per rank and locally produced KV. The indexer's KV dtype remains independently configurable. Speculative decoding, KV connectors (including disaggregated prefill/decode), HiSparse, KV layer parallelism, host KV offload and sleep mode are incompatible with this configuration.

## Build on Ascend Linux

Use the GLM serving environment for your checkpoint with matching CANN, ASC, PyTorch, torch_npu and vLLM. On the model hosts, build this vLLM Ascend checkout with its custom kernels. For a 950DT build, run from the repository root after sourcing your CANN environment.

```bash
export SOC_VERSION=ascend950dt_9582
COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . --no-build-isolation
```

Both endpoints need the bundled `avsg-urma` extension and the matching HIXL library. The following HIXL revision includes the 112-byte `HixlAivOpParam` ABI and the `GetAivChannelPointers` export. Build it on Ascend Linux. [HIXL source revision](https://gitcode.com/wangzhenwu/hixl_avsg/tree/5ac7f35b4aa916db77aac58ea6d3d5c16e20f5fe)

```bash
git clone --branch avsg --single-branch https://gitcode.com/wangzhenwu/hixl_avsg.git /path/to/hixl-avsg
export HIXL_SOURCE_DIR=/path/to/hixl-avsg
git -C "$HIXL_SOURCE_DIR" checkout --detach 5ac7f35b4aa916db77aac58ea6d3d5c16e20f5fe
(cd "$HIXL_SOURCE_DIR" && bash build.sh --aiv)
```

Set `HIXL_LIBRARY_DIR` to the directory containing the rebuilt `libcann_hixl.so` or `libhixl.so`. From the vLLM Ascend checkout, build and load the extension against that same library on both hosts.

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

On the peer host start the cache server from this vLLM Ascend checkout. For BF16 KV and RoPE it registers two regions for all layers combined. The worker also registers two regions. HIXL requires matching registration counts on both ends, so rebuild the extension and restart both processes when updating from the version that registered each layer separately. Wait for each server to print the fresh manifest containing its live per-layer addresses.

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

Compare a prompt longer than 8192 tokens, two requests with a shared prefix longer than one block, and mixed prompt/decode steps. Use a small `--max-num-batched-tokens` for chunked prefill and enable prefix caching on both runs. Check generated tokens in each case. For capacity validation, inspect the scheduler block count and the model worker's main cache allocations: main KV/RoPE layer handles should have zero storage bytes, while indexer pages and the shared step/selection buffers own device storage. Stop the model workers first, then stop the cache servers. Restart both ends with freshly generated manifests after a transfer failure.
