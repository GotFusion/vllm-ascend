# 三层 Sparse Attention 的 AVSG D2D 入图示例

本示例从 recipes 的三层模型移植，Match 使用 vLLM Ascend 的 `_C_ascend::npu_avsg_match`。FP8 示例可以独立于模型权重运行。GLM-5.2 服务使用自己的配置、分页缓存和 BF16 KV/RoPE。[接口和手算例子](AVSG_INTERFACES.md) · [GLM 服务部署](../../../../docs/source/user_guide/feature_guide/avsg_remote_kv.md) · [上游版本 d107a3e](https://gitcode.com/wangzhenwu/cann-recipes-infer_sjy/tree/d107a3e106866344c05c6ba58948ab1f196c6169/models/avsg_sparse_attention)

示例使用一个请求、三个独立的 Sparse Attention 层。输入和逐层 KV 都是 656 维 `float8_e4m3fn` 向量，每条 KV 占 656 字节。线性权重随机初始化，计算用 BF16，softmax 用 FP32。每层使用单个 Q head 和共享 K/V 投影。模型没有位置编码、RoPE、词表和 MTP。每轮把最后一层的输出转成 FP8，作为下一轮输入。

| 阶段 | 数据流 |
| --- | --- |
| Prefill（eager） | 每层生成所有 prompt 位置的 Q 和 KV。KV 写入本端 staging，并通过 AIV URMA WRITE 放入该层的远端 Full Cache。本轮 Attention 按随机 TopK 从本端 staging 读 KV。 |
| Decode（ACLGraph 或 eager） | 每层生成当前 token 的 Q 和 KV，先用 `write_graph_` 把新 KV 写到远端。接着 `npu_avsg_match` 更新 TopK 和命中状态，输出 `mapping/count`。D2D Copy（`torch.ops.avsg_urma.copy_graph_`）用 AIV SIMT URMA READ 把缺失行搬入本端 Selection Cache。Attention 用改写后的 TopK 从 Selection Cache 取数据。 |

Decode 的 WRITE 和 READ 在同一流，READ 接收本层 WRITE 的完成状态。Full Cache 每层有 512 个槽位，Selection Cache 每层有 256 个槽位，每页 128 个 token。每层有一份匹配状态和一份随机 TopK。随机 Indexer 在图外生成固定形状的索引，随机抽取当前 query 可见的历史位置，并加入当前位置，以便每步都验证新写入的 KV 能立即读回。两阶段都按因果范围选取。默认用 384 个 prompt token，让 Decode 从一开始就走 Full Cache 长度大于 Selection 容量的分支。

## 安装

在两端准备可互通的 Ascend 950、匹配的 CANN、PyTorch、torch_npu 和 HIXL AIV 资源，先安装本仓携带的 URMA 扩展。在模型端安装从当前 checkout 编译的 vLLM Ascend 和 AvsgMatch OPP。下列命令从 vLLM Ascend 仓库根目录执行。HIXL 源码、动态库及 950 型号设置见部署说明。[URMA 构建说明](../../README_zh.md) · [插件构建说明](../../../../docs/source/user_guide/feature_guide/avsg_remote_kv.md)

```bash
# 两端，先配置 CANN、HIXL_SOURCE_DIR 和 HIXL_LIBRARY_DIR
python -m pip install --no-build-isolation ./csrc/avsg_urma

# 模型端，以 950DT 为例
SOC_VERSION=ascend950dt_9582 COMPILE_CUSTOM_KERNELS=1 python -m pip install --no-build-isolation -e .
```

默认由插件注册 Match。若使用验证用的独立共享库，在模型端命令后追加 `--ops-library /absolute/path/to/libmatch.so`，并先加载配套 OPP 的运行环境。该库必须从同一 vLLM Ascend 提交编译，注册 `_C_ascend::npu_avsg_match` 的 NPU 实现。

## 运行

在存储端把 [server.example.json](server.example.json) 复制为 `server.json`，在模型端把 [client.example.json](client.example.json) 复制为 `client.json`。根据两端实际环境填写以下字段。

| 字段 | 怎么填 |
| --- | --- |
| `device` | 填本进程看到的 NPU 编号。模板以 `0` 为例。 |
| `local_engine` | 填本端 HIXL 建链地址，例如实际的 `IP:端口`。 |
| `peer_engine` | 只在模型端填写，取服务端的 `local_engine`。 |
| `local_comm_res` | 填本端 `hixl_tool local_comm_res` 生成的 JSON 文件路径，例如 `"/etc/hixlep/local_comm_res_0_0.json"`。相对路径以当前配置文件的目录为准。也支持直接填完整 JSON 对象。两端分别使用各自设备的资源。 |
| `channels` | 两端填相同的独占 jetty 数，模板为 `1`。 |
| `timeout_ms` | 模型端建链超时，单位毫秒。模板为 `30000`。 |
| `manifest` / `remote_manifest` | 分别指定服务端输出路径和模型端读取路径。相对路径各自以对应配置文件的目录为准。 |

资源生成步骤见 [HIXL LocalCommRes 说明](../../README_zh.md#6-localcommres-生成)。

### `avsg_sparse_manifest.json` 由服务端生成

先启动服务端。它注册三层 Full Cache 后，会按 `manifest` 指定的路径写出 JSON。顶层键是层号 `"0"`、`"1"`、`"2"`。每层包括运行时注册的 `kv_base` 地址，以及容量、dtype、字节步长等布局信息。本示例每层是 `[4, 128, 656]` 的 `float8_e4m3fn` 张量，`kv_bytes=335872`，`rope_dim=0`。运行时地址由服务端填写。

看到服务端打印 `Remote KV ready` 后，把生成的文件原样复制到模型端 `remote_manifest` 指向的位置，也可以通过共享目录读取。模型端退出后再停止服务端。每次重启服务端都使用本次生成的 manifest。 [服务端与模型端实现](run.py)

HIXL AIV 建链要求两端注册相同数量的业务内存。模型端注册 3 块 Selection Cache、3 块 Decode staging 和 1 块 Prefill staging。存储端注册 3 块 Full Cache 和 4 块辅助内存，manifest 只发布 3 层 Full Cache。如果从较早的版本升级，请更新存储端安装的 `avsg_urma` Python 包，再重启服务端并使用新生成的 manifest。 [存储端注册逻辑](run.py)

在两端各自的 vLLM Ascend 仓库根目录运行下列命令。模型端的两个模式分开运行，每次使用新启动的服务端和对应的新 manifest。

```bash
# 服务端
python -m csrc.avsg_urma.examples.sparse_attention.run serve --config /path/to/server.json

# 模型端，直接捕获并重放 Decode
python -m csrc.avsg_urma.examples.sparse_attention.run run --config /path/to/client.json --mode aclgraph --steps 2 --verify

# 模型端，逐算子执行同一段 Decode
python -m csrc.avsg_urma.examples.sparse_attention.run run --config /path/to/client.json --mode eager --steps 2 --verify
```

`--verify` 在图外保留一份本端参考数据，逐层、逐步比较 Selection KV 和原始 TopK 对应的 FP8 字节，并检查每个有效位置都得到可用的 Selection 索引。每步还检查各层 WRITE/READ 的完成状态和实际搬运计数。首次 Decode 每层必须读回全部 32 个不同的位置；以后每步都包含刚写入的位置，每层必须读回 1 至 32 个位置。默认 `--seed 42`、`--prompt-length 384`、`--steps 8`，总长度最多 512。

`--mode aclgraph` 是示例默认模式。程序先用第一步输入做 eager warmup 并检查 WRITE 和 READ，再用 `torch.npu.NPUGraph` 捕获同一段 Decode。首次重放前重置 Match 状态，将 Selection Cache 填为 FP8 NaN，逐字节校验就能检出重放遗漏的 READ。各步复用注册过的缓冲区并更新原地址上的输入。连接、工作区分配、随机 TopK 生成及状态检查在图外执行。传输状态报错时会带出 Prefill 或 Decode、层号以及 WRITE 或 READ。[Decode 计算](model.py) · [捕获与重放](run.py)
