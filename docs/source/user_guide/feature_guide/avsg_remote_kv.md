# GLM-5.3 的 AVSG eager 验证

此分支基于 `releases/v0.30.0@6cbe76554e6f9dd4c487a3c46a15613684f7147e`，将 GLM 的历史主 KV 和 RoPE 放在独立 peer NPU 的 HBM。模型卡保留 Lightning Indexer 缓存，以及所有注意力层共用的当步写入暂存和 8192 行选择缓冲。新 KV 写入 peer 后，AVSG 按每个查询的因果 TopK 读取所需行，再计算 SFA。前缀命中沿用调度器分配的远端物理页。

配套上游 vLLM 为 `v0.30.0@ced6857a`。AVSG eager 与原生算子从已有适配提交 `a5db3fde8` 迁入，其 recipes 源版本为 `d107a3e106866344c05c6ba58948ab1f196c6169`，HIXL 源版本为 `5ac7f35b4aa916db77aac58ea6d3d5c16e20f5fe`。本轮源码与主机验证记录对应当前分支，设备结果需在该组合上重新取得。

本分支提供 GLM-5.3 单服务和日志中的 Prefill/Decode 分离接入。PD 路径为 P 的 `MultiConnector` producer → D 的 `SfaRemoteD2HConnector` consumer → AVSG peer HBM；D 保留本地 indexer，使用 AVSG 读取历史主 KV。P/D 均启用 `VLLM_USE_V2_MODEL_RUNNER=1`、`--enforce-eager` 和 BF16 主 KV，本轮 PD 限定两端 TP1、PP1。下面示例使用 DP1 和一个 D peer。旧 host KV offload、推测解码、上下文并行和图执行不属于此接入范围，设备验收尚待执行。

发布分支的 `SparseFlashAttentionVllm` 保留 `torch.ops._C_ascend.npu_sparse_flash_attention` 入口，本适配使用该分支已有的 SFA。普通 FlashComm1 在注意力前会还原完整 token，可沿用 `enable_flashcomm1`，设备效果待复测。`enable_dsa_cp` 需要关闭。

验证参数来自用户的 `opsLearning/operator-libraries/vllm-ascend/log`。其中模型目录为 `/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916/`，服务 seed 为 1024，初步流程使用 5 层 dummy。日志中的服务长度为 8192/13500，性能场景却请求 131072 个输入 token。这里分别提供 4096-token 冒烟与 131072-token 长输入配置，服务容量还须包含 32 个输出 token。模型使用 MXFP4 权重格式，AVSG 当前使用 BF16 主 KV/RoPE；indexer 格式单独设置。

## 编译与通信配置

在 Ascend Linux 上使用与 vLLM 0.30.0 匹配的 CANN、ASC、PyTorch、torch_npu 环境，并从本分支构建插件。设备编译、模型生成和性能测量在服务器阶段执行。

```bash
export SOC_VERSION=ascend950dt_9582
COMPILE_CUSTOM_KERNELS=1 python3 -m pip install -e . --no-build-isolation
```

两端的 URMA 扩展使用 HIXL `avsg@5ac7f35b4aa916db77aac58ea6d3d5c16e20f5fe` 的 launcher ABI。构建该 HIXL，再将库目录传给扩展。[HIXL 源码版本](https://gitcode.com/wangzhenwu/hixl_avsg/tree/5ac7f35b4aa916db77aac58ea6d3d5c16e20f5fe)

```bash
export HIXL_SOURCE_DIR=/path/to/hixl-avsg
export HIXL_LIBRARY_DIR=/path/to/rebuilt/hixl/lib
export LD_LIBRARY_PATH="$HIXL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
python3 -m pip install --no-build-isolation ./csrc/avsg_urma
python3 -c 'import torch, torch_npu; from avsg_urma import AvsgUrmaSession'
```

以下以 TP1 为例。一张模型卡和一张 peer 卡各自只暴露本端设备，进程内设备编号均为 0。为两端设置唯一 HIXL engine 地址，并使用现场生成的通信资源 JSON。共享目录须为本次启动新建，peer 的 manifest 包含本次分配的地址。读取 checkpoint 的 `num_hidden_layers`、`kv_lora_rank`、`qk_rope_head_dim` 核对层数与维度。下面 5 层与后面的 dummy 服务一致，完整权重验证时改成实际层数。

```bash
RUN_DIR=/shared/glm53-avsg/run-001
mkdir -p "$RUN_DIR"
PEER_ENGINE=peer-address-from-site
MODEL_ENGINE=model-address-from-site
BLOCKS=1024

jq -n --slurpfile res /path/to/peer-comm-res.json \
  --arg engine "$PEER_ENGINE" --argjson blocks "$BLOCKS" \
  '{device: 0, local_engine: $engine, local_comm_res: $res[0], channels: 1,
    manifest: "manifest-0.json", cache: {layers: 5, blocks: $blocks,
    block_size: 128, kv_dim: 512, rope_dim: 64, dtype: "bfloat16"}}' \
  > "$RUN_DIR/server-0.json"
jq -n --slurpfile res /path/to/model-comm-res.json \
  --arg model "$MODEL_ENGINE" --arg peer "$PEER_ENGINE" \
  '{local_engine: $model, peer_engine: $peer, local_comm_res: $res[0],
    channels: 1, timeout_ms: 30000, remote_manifest: "manifest-0.json"}' \
  > "$RUN_DIR/endpoint-0.json"

ASCEND_RT_VISIBLE_DEVICES=7 python3 csrc/avsg_urma/examples/d2d_cache_server.py \
  --config "$RUN_DIR/server-0.json"
```

等 peer 打印注册成功并生成 manifest 后启动模型。多 rank 时，每个 rank 需要独立的 endpoint、peer 和通信资源，`endpoint_file` 中使用 `{rank}`。

## 两组 eager 服务

以下命令从仓库根目录执行。先用本地 KV 启动，再在相同模型卡上重启 AVSG，保存两组各自的命令与日志。将 `ADDITIONAL` 改成 AVSG 配置即可使用同一套参数。

```bash
MODEL_PATH=/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916
RUN_DIR=/shared/glm53-avsg/run-001
ADDITIONAL='{}'
# AVSG 组使用下面这行
# ADDITIONAL="{\"avsg_config\":{\"endpoint_file\":\"$RUN_DIR/endpoint-0.json\"}}"

ASCEND_RT_VISIBLE_DEVICES=6 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 20003 --served-model-name glm53 \
  --tensor-parallel-size 1 --data-parallel-size 1 --seed 1024 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --quantization ascend \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --enforce-eager --enable-prefix-caching --enable-chunked-prefill \
  --max-model-len 8192 --max-num-batched-tokens 512 --max-num-seqs 16 \
  --block-size 128 --num-gpu-blocks-override 1024 --gpu-memory-utilization 0.8 \
  --trust-remote-code --load-format dummy \
  --hf-overrides '{"num_hidden_layers":5,"use_index_cache":true}' \
  --additional-config "$ADDITIONAL"
```

小 prefill 调度窗口用于覆盖分块写出和远端历史读回。5 层 dummy 用于检查服务、量化层初始化及通信路径。精度验收需另用同一份真实裁层或完整 checkpoint，移除 `--load-format dummy`，将 peer 的层数与真实模型配置匹配后比较生成 ID。

```bash
curl --fail http://127.0.0.1:20003/v1/models
curl --fail http://127.0.0.1:20003/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53","prompt":"The first three prime numbers are", "seed":1024,"temperature":0,"max_tokens":32,"ignore_eos":true,"return_token_ids":true}'
```

用相同请求检查两组输出，并保留每层远端 WRITE/READ 完成状态、prefix hit 计数、peer 容量和模型卡主 KV/RoPE backing storage。后者应为 0 字节，实际设备存储由 indexer 与共享暂存、选择缓冲持有。

## 日志中的 PD 服务接入

这个例子使用 P 卡 6、D 卡 7、peer 卡 5。先停止前面的单服务，按通信配置小节重新启动 peer，将 peer 命令的 `ASCEND_RT_VISIBLE_DEVICES` 改为 `5`；endpoint 中的 model engine 和通信资源现在属于 D 卡。P 不启用 AVSG，D 关闭 `sparse_kv_offload_config`。P/D 的主 KV/RoPE 均为 BF16，indexer 均显式设置为 `bf16`，即先关闭 LI C8。仅设置旧日志的 `enable_sparse_li_c8=false` 不能替代本分支的 `attention_config.indexer_kv_dtype`。

P/D 使用本分支相同提交，层数与缓存 dtype 必须匹配。AVSG 会拒绝缺少 dtype 元数据的旧 producer，避免将同字节数的 FP16 误读为 BF16。D 用一页固定大小的主 KV 暂存接收 P 数据，再由模型线程分批写入 peer；该页已计入内存预算。远端写入和本地 indexer 接收完成后才回复 `READ_DONE`，允许 P 复用源页；请求取消也会等待在途传输结束再释放目的页。

P→D 需要现场版本的 `memfabric_hybrid` 及其 `device_urma` 支持，D→peer 需要前面编译的 HIXL/URMA 扩展。两条传输路径的依赖均需安装。保留日志中与现场库匹配的 `MEMFABRIC_HYBRID_EXTEND_LIB_PATH`、`LD_LIBRARY_PATH`；本 connector 直接创建 MemFabric TransferEngine，不从 `MMC_LOCAL_CONFIG_PATH` 读取 KV pool 配置。跨机时将下面 HTTP、proxy 和通信地址替换为双方可访问的地址，proxy 的 `--host` 还用于 D 回调 metaserver，不能填不可路由的地址。

每个服务在独立终端从仓库根目录启动。以下 P/D 配置文件可先在同一终端生成：

```bash
MODEL_PATH=/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916
RUN_DIR=/shared/glm53-avsg/run-001
cat > "$RUN_DIR/p-kv.json" <<'JSON'
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "connectors": [{
      "kv_connector": "SfaRemoteD2HConnector",
      "kv_role": "kv_producer",
      "kv_buffer_device": "npu",
      "kv_parallel_size": 1,
      "kv_port": 20050,
      "kv_connector_extra_config": {
        "transfer_backend": "memfabric",
        "memfabric_transfer_protocol": "device_urma",
        "use_layerwise": true
      }
    }]
  }
}
JSON
cat > "$RUN_DIR/d-kv.json" <<'JSON'
{
  "kv_connector": "SfaRemoteD2HConnector",
  "kv_role": "kv_consumer",
  "kv_port": 20050,
  "kv_connector_extra_config": {
    "transfer_backend": "memfabric",
    "memfabric_transfer_protocol": "device_urma",
    "use_layerwise": true
  }
}
JSON
jq -n --arg endpoint "$RUN_DIR/endpoint-0.json" \
  '{avsg_config: {endpoint_file: $endpoint}, sparse_kv_offload_config: {enabled: false},
    enable_dsa_cp: false, enable_flashcomm1: true}' > "$RUN_DIR/d-additional.json"
```

P 使用普通本地 KV；启用 prefix caching，后面的 ais-bench 统计 P 的命中率：

```bash
MODEL_PATH=/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916
RUN_DIR=/shared/glm53-avsg/run-001
ASCEND_RT_VISIBLE_DEVICES=6 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 20002 --served-model-name glm53 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --seed 1024 --dtype bfloat16 --kv-cache-dtype bfloat16 --quantization ascend \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --enforce-eager --enable-prefix-caching --enable-chunked-prefill \
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 16 \
  --block-size 128 --num-gpu-blocks-override 1024 --gpu-memory-utilization 0.8 \
  --trust-remote-code --load-format dummy \
  --hf-overrides '{"num_hidden_layers":5,"use_index_cache":true}' \
  --kv-transfer-config "$(cat "$RUN_DIR/p-kv.json")" \
  --additional-config '{"sparse_kv_offload_config":{"enabled":false},"enable_dsa_cp":false,"enable_flashcomm1":true}'
```

D 从 P 接收 prompt 的 main KV 和 indexer，main KV 导入 peer，后续 decode 写出新 KV。此处关闭 D 的 prefix caching，让场景 warmup 和正式请求都访问 P，避免 D 缓存命中减少 P 流量后混淆 90% 目标命中率：

```bash
MODEL_PATH=/mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916
RUN_DIR=/shared/glm53-avsg/run-001
ASCEND_RT_VISIBLE_DEVICES=7 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 20003 --served-model-name glm53 \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --seed 1024 --dtype bfloat16 --kv-cache-dtype bfloat16 --quantization ascend \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --enforce-eager --no-enable-prefix-caching --enable-chunked-prefill \
  --max-model-len 8192 --max-num-batched-tokens 60 --max-num-seqs 16 \
  --block-size 128 --num-gpu-blocks-override 1024 --gpu-memory-utilization 0.8 \
  --trust-remote-code --load-format dummy \
  --hf-overrides '{"num_hidden_layers":5,"use_index_cache":true}' \
  --kv-transfer-config "$(cat "$RUN_DIR/d-kv.json")" \
  --additional-config "$(cat "$RUN_DIR/d-additional.json")"
```

P/D 就绪后启动日志同款 layerwise proxy：

```bash
python3 examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 127.0.0.1 --port 20004 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 20002 \
  --decoder-hosts 127.0.0.1 --decoder-ports 20003
```

该 proxy 支持 `/v1/completions`、`/v1/chat/completions`、`/healthcheck` 和 `POST /reset_prefix_cache`；reset 转发所有 P/D，任一失败返回 500。它没有 `/metrics`。以下完成请求必须同时观察 P 的逐层发送、D 的接收和 peer 写入完成，不能只用 HTTP 200 验收：

```bash
curl --fail http://127.0.0.1:20004/healthcheck
curl --fail -X POST http://127.0.0.1:20004/reset_prefix_cache
curl --fail http://127.0.0.1:20002/metrics
curl --fail http://127.0.0.1:20004/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53","prompt":"The first three prime numbers are","seed":1024,"temperature":0,"max_tokens":32,"ignore_eos":true,"return_token_ids":true}'
```

需要验证 D 的 prefix 复用时，单独改成 `--enable-prefix-caching`，记录两端各自的 prefix hit/query 计数和 D 远端请求数量；这与下面专门统计 P 命中的性能口径分开。5 层 dummy 只检查功能，不作模型精度或完整模型性能结论。

## ais-bench 性能配置

配置生成器保留原日志的 5 条请求、32-token 输出、batch size 2、流式输出和 90% 目标前缀命中率。支持 `smoke`、`64k`、`128k`、`256k` 四种输入长度。此次 PD eager 测试使用 **MTP0**：P/D 均不传 `--speculative-config`，关闭推测解码；AISBench 的 `--num-warmups 0` 是性能工具的预热次数，与 MTP 无关。前缀生成器需要日志所用的 `ais_bench_prefix_cache` 扩展。此扩展不随 vLLM Ascend 或 AISBench 安装，需在服务器安装已有验证环境使用的版本。安装后检查两个 CLI 与配置导入。

```bash
ais_bench --help
ais-bench-prefix-cache --help
python3 -c 'from ais_bench_prefix_cache.config import build_dataset_config, build_model_config'
python3 examples/avsg/glm53_scenario.py \
  --model /mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916 \
  --corpus /code/benchmark/ais_bench/datasets/gsm8k/test.jsonl \
  --service-url http://127.0.0.1:20003 --profile smoke \
  --output-dir /shared/glm53-avsg/local-smoke --scenario /shared/glm53-avsg/local-smoke.json
ais-bench-prefix-cache inspect --scenario /shared/glm53-avsg/local-smoke.json
ais-bench-prefix-cache prepare --mode text --scenario /shared/glm53-avsg/local-smoke.json
ais-bench-prefix-cache run --scenario /shared/glm53-avsg/local-smoke.json \
  --config "$PWD/examples/avsg/prefix_cache_perf.py"
```

AVSG 组使用相同命令，改用独立的 `avsg-smoke` 输出路径，并核对生成数据的 SHA256、分词长度、请求顺序与本地组一致。每组从新启动且无历史请求的服务开始，先完成场景自身的 warmup，再统计正式请求。配置写入 `--mode perf`，汇总器使用 `DefaultPerfSummarizer`，采集实际输入/输出 token、前缀命中率、TTFT、TPOT、吞吐与失败率。报告中同时记录模型为 dummy 或真实权重、TP、HBM 用量和数据规模。[AISBench 性能模式](https://github.com/AISBench/benchmark/blob/master/docs/source_en/base_tutorials/scenes_intro/performance_benchmark.md)

PD 使用同一份负载，分别设置推理、P 指标和全体缓存 reset 的 origin。origin 参数只接受协议、地址和端口，不附带 `/metrics` 等路径。未指定时 metrics/reset 沿用 `--service-url`，单服务命令不变。

```bash
python3 examples/avsg/glm53_scenario.py \
  --model /mnt/share/w00936111/GLM-5.3-W4A4C8-MXFP4-0916 \
  --corpus /code/benchmark/ais_bench/datasets/gsm8k/test.jsonl \
  --service-url http://127.0.0.1:20004 \
  --metrics-origin http://127.0.0.1:20002 --reset-origin http://127.0.0.1:20004 \
  --profile smoke --output-dir /shared/glm53-avsg/pd-smoke \
  --scenario /shared/glm53-avsg/pd-smoke.json
ais-bench-prefix-cache inspect --scenario /shared/glm53-avsg/pd-smoke.json
ais-bench-prefix-cache prepare --mode text --scenario /shared/glm53-avsg/pd-smoke.json
ais-bench-prefix-cache run --scenario /shared/glm53-avsg/pd-smoke.json \
  --config "$PWD/examples/avsg/prefix_cache_perf.py"
```

场景 warmup 和正式推理均通过 20004 proxy；缓存 reset 经 proxy 清除 P/D，命中率从 20002 的 P 指标采集。按上面的 D 无 prefix 配置，正式请求会回到 P 复用 warmup 的前缀。核对 P `vllm:prefix_cache_queries` 与 `vllm:prefix_cache_hits` 的正式请求增量后再报告实际命中率，不能用 D 指标或目标值 90% 代替实测值。`ais_bench_prefix_cache` 的现场版本仍需确认其 warmup/reset 顺序及 Prometheus 指标兼容性；本地只验证生成的配置和 URL 路由。

64K/128K/256K 场景的长度均为 token 数，默认输出 32 token。单服务和 PD 的 **P、D 两端**都须配置足够的 `--max-model-len`，不能沿用原日志的 8192/13500：

| profile | 输入 token | 输出 token | 最小 max-model-len | 按 128 对齐的长度 |
| --- | ---: | ---: | ---: | ---: |
| smoke | 4096 | 32 | 4128 | 4224 |
| 64k | 65536 | 32 | 65568 | 65664 |
| 128k | 131072 | 32 | 131104 | 131200 |
| 256k | 262144 | 32 | 262176 | 262272 |

本次 950DT 的量化源权重位于 `/mnt/weight/GLM-5.3-w4a4c8-mxfp4`，模型类型为 `glm_moe_dsa`。`/mnt/share/weight` 下的 GLM-5.3-Flash 使用 `glm5_next` 架构。三档场景使用从量化源权重提取的真实前五层，保留 embedding、生成头、量化元数据与 tokenizer。原始权重中第 0–2 层使用 dense FFN，第 3–4 层使用 MoE。裁层索引包含 3,193 个张量，已逐张量核对原始字节 SHA256。

两端都用同一份裁层启动 `vllm serve "$MODEL_PATH" --load-format auto --quantization ascend`，并保留上述 PD 各自的 connector、eager、BF16 KV 和 indexer 配置。裁层本身设置 `num_hidden_layers=5`、`num_nextn_predict_layers=0`，启动时省略 speculative decoding 配置。以下生成三种长度的独立场景。

```bash
MODEL_PATH=/home/w00959236/avsg-validation-results/glm53-v030-runtime/models/glm53-w4a4c8-5l
test -f "$MODEL_PATH/config.json"
CORPUS_PATH=/code/benchmark/ais_bench/datasets/gsm8k/test.jsonl
RESULTS_ROOT=/shared/glm53-avsg/mtp0-lengths-run-001
for profile in 64k 128k 256k; do
  python3 examples/avsg/glm53_scenario.py \
    --model "$MODEL_PATH" --corpus "$CORPUS_PATH" \
    --service-url http://127.0.0.1:20004 \
    --metrics-origin http://127.0.0.1:20002 --reset-origin http://127.0.0.1:20004 \
    --profile "$profile" --output-dir "$RESULTS_ROOT/$profile" \
    --scenario "$RESULTS_ROOT/$profile.json"
done
```

每档在独立目录运行上述 inspect、prepare 和 run 命令，保存真实裁层模型的 P、D 加载日志、设备显存快照、AISBench 三张原始表及 P/D metrics。两张空闲卡中，P 与 peer 共用一张，D 单独使用另一张。需要核算 P 权重及本地主 KV、peer HBM、D 权重与 indexer、单页 ingress 占用。

每档 warmup 后发送 5 条正式请求，batch 为 2，流式输出 32 个 token。90% 是构造输入时的前缀命中目标。实测值取正式请求期间 P 的 `vllm:prefix_cache_hits_total` 增量除以 `vllm:prefix_cache_queries_total` 增量。报告中保留 warmup、reset、指标采样边界、实际成功和重试数。

启动前同时核对 checkpoint 支持的上下文长度、服务的 `--num-gpu-blocks-override` 和 peer `blocks`；256K 输入还要容纳 32 个输出 token。按实际并发和缓存组分配页数，计入调度器保留的空页，不沿用 smoke 的 1024 页。本次 PD 性能测试按参考日志保留 `--max-num-batched-tokens` **P=8192、D=60**，三种输入长度均使用这个组合，并在结果表中记录两端实际值；比较路径时保持相同负载与容量约束。

上述页数按调度页与算子页均为 128 token 计算。如果运行时调度器合并为更大的页，peer 所需页数应乘以 `scheduler_block_size / 128`。worker 会按该比例检查配置，同时计入本地 indexer 的容量。

新一轮使用 8 条正式请求、并发 8、每条输出 64 token 时，在独立目录生成配置，保留上面的 5/2/32 默认值和既有结果。沿用前面设置的 `MODEL_PATH`、`CORPUS_PATH`：

```bash
RESULTS_ROOT=/shared/glm53-avsg/mtp0-lengths-8x64-run-001
for profile in 64k 128k 256k; do
  python3 examples/avsg/glm53_scenario.py \
    --model "$MODEL_PATH" --corpus "$CORPUS_PATH" \
    --service-url http://127.0.0.1:20004 \
    --metrics-origin http://127.0.0.1:20002 --reset-origin http://127.0.0.1:20004 \
    --request-count 8 --concurrency 8 --output-tokens 64 \
    --profile "$profile" --output-dir "$RESULTS_ROOT/$profile" \
    --scenario "$RESULTS_ROOT/$profile.json"
done
```

每档正式请求的输入总量为 `8 × 输入长度`，输出总量为 512 token，不包含场景 warmup。P、D 的 `--max-num-seqs` 均须至少为 8，`--max-model-len` 至少为 `输入长度 + 64`。同时按 8 路并发、实际缓存组和调度页大小重新核算 P 本地 KV、D indexer 与 peer 页容量，计入调度器保留空页；不能沿用并发 2 的页数。上述 inspect、prepare 和 run 命令使用新场景路径。

在 950DT 的真实五层 smoke 中，vLLM 0.30.0 的 P、D 均完成了 20 个权重分片的加载。提交 `5bb0952e9` 使用 `device_urma` 将 4096-token prompt 的主 KV 逐层经单页 ingress 写到 peer；五层均在写完 32 页后返回 READ_DONE，实际生成 32 个 token。该结果验证了端到端功能。三档长输入的性能取自各自的正式 AISBench 输出。

在 opsLearning 的本轮结果目录中，以下命令会通过 `npu-smi` 选择空闲卡；第二条命令沿用第一条实测得到的缓存组页数和布局。命令与原始日志见 [GLM-5.3 950DT 复测记录](https://github.com/GotFusion/opsLearning/tree/main/operator-libraries/vllm-ascend/glm-5-2-kv-offload/results/20261126-glm53-v030-pd-mtp0)。

```bash
bash service/launch.sh smoke repeat-smoke --peer-kernel-pages 148
bash service/run_lengths.sh repeat-smoke repeat-perf 64k 128k 256k
```
