# AVSG Copy：HIXL AIV 直驱 SIMT URMA D2D

[English](README.md) | [简体中文](README_zh.md)

本扩展根据 AVSG 的 `[src, dst]` 映射，将远端 NPU HBM 中的 full KV/K-RoPE 拉取到本端 NPU HBM 的 selection KV/K-RoPE。提供同步 eager 对照路径和异步单 kernel 入口。

原有 `ops/ascendc/src/avsg_copy` 和模型调用路径保留；安装本扩展不会自动替换模型算子、迁移 full cache 或修改模型 YAML。

### 单 kernel 路径

`torch.ops.avsg_urma.copy_graph_` 在一次 AIV launch 内完成动态输入校验、描述符生成、HIXL SIMT READ 和 CQ Drain、目标缓存维护及状态发布。支持 1–32 个独占 jetty，每个 AIV block 对应一个 jetty，使用 32 lanes。连接并调用 `prepare_layer()` 后，在图外用 `prepare_graph_copy()` 按固定 mapping 容量分配工作区。

每个 block 使用独立 bitmap 校验同一份 mapping，保留全表的重复目的 slot 检查。校验通过后，block 只生成和提交本 jetty 的描述符分片。每个 jetty 都写一行 128B 状态，包括本轮无任务的 jetty。READ 依赖 WRITE 时，只有全部 WRITE 状态成功才发 READ。完整校验按 jetty 重复执行，多 jetty 的净收益需上板测量。更换 jetty 数需要重新建链、准备缓冲区并捕获图。

```python
rope, kv, context, scratch, seen, status = session.prepare_graph_copy(
    layer_id=layer_id, mapping_capacity=mapping.numel() // 2,
)
torch.ops.avsg_urma.copy_graph_(rope, kv, q, mapping, count, context, scratch, seen, status)
# 在执行或图重放结束后检查状态。
copied_tokens = session.check_graph_copy(status)
```

在同一张图内重放时，更新 q、mapping 和 count 的内容并保持 shape 和绑定地址。应用需确保对端本轮 full KV/RoPE 已写入可读，再启动本轮 READ。保留 Session 和注册内存直到图停止使用，串行使用同一 Session 的 jetty。出错时丢弃本轮推理结果和 Match/Selection 状态，关闭 Session。

`examples/d2d_smoke.py` 的 client 支持 `--copy-mode single-kernel`、`--copy-mode aclgraph` 和 `--copy-mode npugraph-ex`，覆盖相同输入地址下非零和零 count 交替执行。加 `--check-write` 验证 WRITE→READ、错误传播和 KV/RoPE 字节对比。图重放需要目标环境验证 Selection 原地地址、读回结果和 launch 次数。

`write_graph_` 的 mapping 每项为 `[对端目标 slot, 本地源 slot]`。在同一流调用 READ，或让 READ 流等待 WRITE Drain 后的事件，再将 WRITE 的状态 Tensor 作为 READ 的 `write_status` 输入。

### GLM-5.2 D2D 接入

在存储侧 NPU 上启动 `python examples/d2d_cache_server.py --config <服务端 JSON>`，给对应模型 rank 预留 full cache。服务端 JSON 包含 `device`、`local_engine`、`local_comm_res`（JSON 对象）、`manifest` 和 `cache`。其中 `cache` 包含 `layers`、`blocks`、`block_size`、`kv_dim`、`rope_dim` 和 `dtype`，需要与模型 rank 的 full KV/RoPE 配置一致。每端注册的内存及服务进程应存活至模型端结束图重放并断链。将发布的 manifest 提供给对应模型 rank。

在模型侧为每个 rank 准备 endpoint JSON，其中包含 `local_engine`、`peer_engine`、`local_comm_res` 和 `remote_manifest`。manifest 文件可使用相对 endpoint JSON 的路径。模型在 Prefill 前注册本地 HBM、连接到 peer 并准备每层地址。配置示例如下。

```yaml
exe_mode: "npugraph_ex"
model_config:
  custom_params:
    enable_offload: true
    offload_strategy: "avsg"
    avsg_urma:
      endpoint_file: "/path/to/endpoints/rank-{rank}.json"
        channels: 4
      overlap: true
      prefetch_shared: true
parallel_config:
  cp_size: 1
```

D2D 路径用于主模型、未量化 KV 和 `cp_size: 1`，支持 eager 与 `npugraph_ex` Decode 入口。Prefill 在辅流把临时 HBM 写入 peer，层末等待 WRITE Drain 后复用临时缓存。Decode 先生成本层新增 KV/RoPE，然后按 slot 写入 peer。`overlap` 打开时，通信流的增量 WRITE 可与计算流的 Match/Indexer 重叠。打开 `prefetch_shared` 后，组内下一层的历史条目可提前 READ，本轮新增或覆盖的 slot 在增量 WRITE 完成后补读。相同 jetty 的所有 URMA 请求在通信流串行。SFA 在本层 READ 完成后使用本地 Selection，runner 在推理边界汇总校验传输状态。

存储端的 `blocks` 按单个模型 rank 计算，为 `(pa_max_length / pa_block_size) × batch_size_per_rank`。`layers`、`kv_dim`、`rope_dim`、`dtype` 分别取主模型的 `num_hidden_layers`、`kv_lora_rank`、`qk_rope_head_dim` 和实际 KV dtype。`block_size` 与模型 YAML 的 `pa_block_size` 一致。先启动存储服务并发布 manifest，再启动模型 rank。

存储端 JSON 的顶层 `channels` 与模型 YAML 的 `avsg_urma.channels` 保持一致。两端省略时均使用 1 个 jetty。

模型侧还需按 [GLM-5.2 README](../../../models/glm_5_2/README.md) 安装原 AVSG 的 OPP 和 `custom_ops` wheel，供 `AvsgMatch` 和 runner 的 AVSG 注册使用。在各模型节点按实际拓扑填写 `executor/scripts/set_env.sh`，把 `models/glm_5_2/infer.sh` 的 `YAML_FILE_NAME` 指向加了上述配置的 YAML，再从模型目录执行 `bash infer.sh`。

## 1. 当前实现与调用链

```text
Server 进程                                      Client / 模型进程
full KV/RoPE：NPU HBM                             selection KV/RoPE：NPU HBM
  register_layer()                                  bind_selection()
          │                                                │
          └─────────── HIXL 注册内存、Connect ──────────────┘
                                                           │
                              prepare_layer()：生成 Device context
                                                           │
                              AVSG match → mapping / count
                                                           │
                              session.copy_into(...)
                                                           │
                              torch.ops.avsg_urma.copy_(...)
                                                           │
                              ① 描述符生成 kernel → 完整 GM 列表
                                 同步并检查生成状态
                                                           │
                              ② HixlLaunchAivSimt(...)
                                 → HixlAivSimtDevGetList
                                 → URMA READ → 内部 Drain
                                                           │
                              目标缓存维护 kernel → 同步、检查错误
                                                           │
                              selection 就绪，返回给模型
```

两阶段指“生成描述符”和“执行传输”两个主要阶段。当前另有一个小型传输后缓存维护 kernel，并且在阶段之间保留 Host 状态检查；不是只有两个 launch，也没有通信与计算重叠。

- 使用 raw ASC launcher，通过 `OpCommand::RunOpApiV2` 在当前 NPU stream 下发；未新增 ACLNN 栈。
- `HixlAivSimtDevGetList` 是完整 kernel 入口，由 Host launcher 启动，不是 AVSG kernel 内可直接调用的普通函数。
- CMake 直接编译 `HIXL_SOURCE_DIR/cmake/aiv_launch/hixl_aiv_simt_launch.asc`；本扩展通过 `csrc/hixl_transfer.cpp` 组装 `HixlAivOpParam`。
- 多条 descriptor 使用 `HixlAivSimtDevGetList`；一条使用 `HixlAivSimtDevGet`；零工作量跳过传输。
- 默认每个传输 core 独占一个 channel，也可配置 HIXL 管理的多 core 共享 jetty。描述符生成核数仍等于 channel 数，传输核数独立配置。
- 描述符生成改为每核 32 lanes 并行组装 UB，再用 `DataCopyPad` 批量写 GM；具体分工和尾批规则见第 5 节。
- WQE 控制数据可以经过 UB；KV/RoPE 有效载荷走远端 HBM → 本地 HBM，不经过算子的 UB 中转。

## 2. 环境与 HIXL 依赖

当前编译目标限定为 `dav-3510`（Ascend 950）。不能根据原 AVSG 的 A2/A3 注册推断新实现支持相同平台。请在目标机器确认实际硬件，以及 CANN/ASC、PyTorch、torch_npu、HIXL 和 C++ ABI 的匹配关系。

构建依赖包括 CMake ≥ 3.20、ASC language package、兼容的 Linux C++ 编译器、Python 开发头、setuptools，以及上述运行库。

HIXL 必须提供并实现：

```cpp
Hixl::GetAivChannelPointers(peer, channels);
```

随附 [channel 导出补丁](patches/hixl_channel_export.patch) 基于 `4b1b53a`，也曾验证可应用于 `259a466`；适配提交为 HIXL `avsg` 分支的 `a11ce32`。其他版本需要检查差异，不能保证直接应用。

若源码尚未包含接口，在 Ascend Linux 上检查并应用一次，再重建 HIXL：

```bash
git -C "$HIXL_SOURCE_DIR" apply --check \
  "$RECIPES_ROOT/ops/ascendc/avsg_urma/patches/hixl_channel_export.patch"
git -C "$HIXL_SOURCE_DIR" apply \
  "$RECIPES_ROOT/ops/ascendc/avsg_urma/patches/hixl_channel_export.patch"

cd "$HIXL_SOURCE_DIR"
bash build.sh --aiv
```

已包含接口时不要重复应用补丁。只修改头文件、不重建库，会造成符号缺失。编译 launcher 的 HIXL 源码必须与运行时库的 channel/注册内存 ABI 匹配。

导出的是借用的 Device channel 指针，不转移所有权。Session 私有持有 engine，串行执行调用；不得在外部同时驱动、断开或复用这些 channel。Disconnect、Finalize、设备 reset 后指针失效。

本适配层使用 HIXL `avsg` 分支的 **112 字节 `HixlAivOpParam` ABI**：错误地址拆分到 `reserved[0..1]`，jetty 数放在 `[2]`，SIMT lanes 放在 `[5]`。其他开发分支中的 64 字节具名字段版本不能混用。编译时检查结构体大小和 reserved 偏移；原两阶段适配层的 `err_addr/jetty_num/batch` 不存在报错是分支 ABI 适配遗漏，应更新适配层，不要修改 SDK 系统头。

## 3. 编译、安装与导入

先加载本机实际安装的 CANN 环境，再设置：

| 环境变量 | 含义 |
|---|---|
| `RECIPES_ROOT` | cann-recipes-infer 仓库绝对路径 |
| `ASCEND_HOME_PATH` | 当前 CANN 安装目录，由对应环境脚本配置 |
| `HIXL_SOURCE_DIR` | 包含导出补丁和 raw SIMT launcher 的 HIXL 源码目录 |
| `HIXL_LIBRARY_DIR` | 包含重建后的 `libcann_hixl.so` 或 `libhixl.so` 的目录 |

```bash
cd "$RECIPES_ROOT/ops/ascendc/avsg_urma"
python -m pip install --no-build-isolation --no-cache-dir -v .
```

`setup.py` 使用当前 Python 的 `sys.executable`，先导入 torch/torch_npu，再把框架路径传给 CMake。构建与运行应使用同一套匹配环境；两台机器分别编译，不直接复用另一 Python 版本的 `.so`。

当前 context ABI 已升级为 **v2**，更新后必须重编译扩展、重启进程并重新创建 Session/context。`--no-cache-dir` 不会清理本地 `build/` 的 CMake 缓存；若仍命中旧框架路径，检查 `CMakeCache.txt`。

导入检查建议从源码包目录之外执行，以避免源码目录遮蔽 site-packages：

```bash
cd /tmp
python - <<'PY'
import sys
import avsg_urma
print("Python:", sys.executable)
print("Package:", avsg_urma.__file__)
from avsg_urma import AvsgUrmaSession, AvsgUrmaCacheServer
import torch
print(torch.ops.avsg_urma.copy_)
print("导入和算子注册成功")
PY
```

只写 `import avsg_urma` 不会主动加载 Session/native 扩展。显式导入上述类才会触发 torch_npu 和 `_native.so` 加载；成功不代表建链或 READ 已验证。

需要直接从源码包运行时，可在含 `setup.py` 的目录执行 `python setup.py build_ext --inplace`。两端运行时均需通过安装配置、RPATH 或 `LD_LIBRARY_PATH` 找到 HIXL/CANN/框架依赖库。

## 4. Python 接口与上下文

以下是两个进程各自执行的接口顺序示意；注册屏障、manifest 交换和每轮生产者 ready 协议由外部应用提供。

```python
from avsg_urma import AvsgUrmaCacheServer, AvsgUrmaSession, RemoteCache

# Server：full_kv/full_rope 必须是本端 NPU HBM Tensor。
server = AvsgUrmaCacheServer(
    local_engine=server_engine, device=server_device,
    local_comm_res=server_resource_json, channels=channel_count,
)
server.register_layer(layer_id, full_kv, full_rope)
manifest = server.publish_layout()
# 等待 Client 注册完成，再通过外部控制面发送 manifest。

# Client：在 Connect 前绑定全部层的 selection。
session = AvsgUrmaSession(
    local_engine=client_engine, device=client_device,
    local_comm_res=client_resource_json, channels=channel_count,
    batch_descriptors=32,
)
session.bind_selection(layer_id, selection_kv, selection_rope)
# 两端完成注册屏障，Client 收到 manifest 后：
session.connect(server_engine)
session.prepare_layer(layer_id, RemoteCache(**manifest[str(layer_id)]))

# 每轮 forward：先确认远端生产完成，再使用 AVSG match 的输出。
kv, rope = session.copy_into(
    layer_id=layer_id,
    full_q_actual_seq=cu_q_lens,
    mapping=stage_two_copy_mapping,
    count=stage_two_copy_count,
)

# 停止新调用后：Client 先 close，通知 Server，再由 Server close。
session.close()
# server.close() 由 Server 在确认 Client 关闭后执行。
```

`local_comm_res` 在 Python 构造器中是 **JSON 内容字符串**；smoke 的 `--local-comm-res-file` 才是文件路径。构造器强制 AIV direct 开启、UBMEM 关闭，并设置 `AivJettyNum`。

`RemoteCache` 描述 Server full cache 的地址、字节范围、block size、维度和字节 stride。Server 从本地 Tensor 生成，Client 从 manifest 重建；它不包含实际数据或 channel。

`prepare_layer()` 检查源/目标布局、通过 HIXL GetAddr 核对精确远端注册地址，然后打包：

```text
远端 full 地址/布局 + 本地 selection 地址/范围 + channel 指针列表
    → 24 个 uint64 固定头 + N 个 uint64 channel
    → 上传为本地 NPU int64 context Tensor
```

上传的是上下文元数据，不是 full cache。底层算子为：

```python
torch.ops.avsg_urma.copy_(
    selection_rope, selection_kv, q, mapping, counts,
    context, scratch, status, sync, timing, transfer_cores,
)
```

该算子原位写入，底层返回 `(descriptor_ns, transfer_ns)` 两个 Host 阶段耗时（未开启 Host 打点时为 0）。Session 对外仍返回原绑定的 KV/RoPE Tensor，并负责资源保活、加锁和失败状态管理。内部 schema 末尾增加 `measure_host=False`；更新后需要重编译扩展并重启 Python。

## 5. 数据约束与生命周期

| 项目 | 当前要求 |
|---|---|
| full cache | rank-3 `[blocks, block_size, dim]`，正 stride，末维连续，远端 Device HBM |
| selection | rank-3、连续、地址固定，KV/RoPE 布局匹配 |
| dtype | FP16 / BF16 / INT8 / FP8（float8_e4m3fn、float8_e5m2）原始字节传输，不做量化或尺度转换 |
| 维度 | KV ≤ 1024，RoPE ≤ 64；block size 为 2 的幂 |
| mapping | INT32，`[*counts.shape, entries, 2]`；末维 `[src,dst]` |
| counts | INT32，每组有效条数；同次调用不允许重复目标 dst |
| q | INT32 累积长度；最后一个值参与计算有效 group 数 |
| channels | 1～32，两端配置一致；实际使用导出的 channel 数 |
| transfer_cores | Client 传输核数，默认等于实际 channel 数；软件允许 channel 数～64，仍须符合目标硬件可用资源 |
| jetty_strategy | `auto` 默认；`exclusive` 要求核数等于 jetty 数，`shared` 要求核数大于 jetty 数 |
| batch_descriptors | 保留原参数名，当前表示 HIXL SIMT lanes：8 / 32 / 64，默认 32 |

### 多 core 共享 jetty 与 DevBW

```python
session = AvsgUrmaSession(
    local_engine=client_engine, device=client_device,
    local_comm_res=client_resource_json,
    channels=4, transfer_cores=8, jetty_strategy="shared",
    batch_descriptors=32, measure_devbw=True,
)
# 其余 bind/connect/prepare/copy_into 顺序不变。
# 成功 copy_into 后读取；默认不打点时返回 None。
stats = session.last_transfer_stats
```

`auto` 根据实际导出的 channel 数分流：核数相等走独占，大于时走共享。`shared` 经 HIXL 112 字节 ABI 的 `reserved[6]=1` 开启，4096 字节同步区地址传入 `[7..8]`，每次调用前清零。组内成员填各自 SQ 切片，leader 统一发布、等待完成。组批宽 `groupSize*lanes` 必须满足 SQ 深度限制，HIXL 在设备侧检查。

分组使用 `step=ceil(cores/jettys)`，实际组数为 `ceil(cores/step)`。8 核/4 jetty 是每组 2 核；5 核/4 jetty 实际只形成 3 个组。因此不能总把配置 jetty 数当作有效打点行数。

开启 `measure_devbw=True` 时，分配 32×16 个 int64 的 timing 区，经 `reserved[3..4]` 传给 HIXL，每次调用清零。独占模式各核写自己的行，共享模式只由 leader 按 jetty 写行。读取后校验错误状态和 descriptor 覆盖数，再按 HIXL benchmark 的 SIMT URMA 口径换算：

```text
payload_bytes = token数 × (kv_dim + rope_dim) × element_bytes
DeviceTime/us = max(有效行 completion槽[1]) / 1000
DevBW/GiB/s   = payload_bytes / (DeviceTime/us × 1e-6) / 1024³
```

沿用该平台 benchmark 的 1 ns/tick 配置。分母为 HIXL Start 到 Drain 后 Finish 的窗口；共享模式起点位于部分初始化之后，独占模式起点更早。它不包含 descriptor 生成、Host 校验、传输后缓存维护或 timing 回读，也不是跨核绝对时间戳的 max-end/min-start 或纯链路时间。打点会增加回读开销，默认关闭。

`last_transfer_stats` 包含 `strategy`、`jettys`、`active_jettys`、`transfer_cores`、`lanes`、`payload_bytes`、`descriptors`、`completion_us`、`issue_us`、`publish_count`、`devbw_gib_s` 和 `valid`。零工作量返回 `valid=False`、零字节、零带宽；失败清除旧样本并禁用 Session，不复用上次结果。

### Host copy_into 计时与带宽

Session 可独立开启 `measure_hostbw=True`；默认关闭，smoke 固定开启。`time.perf_counter_ns()` 的起点在进入 `copy_into()`、获取锁之前，终点在发布统计并返回之前：包含锁等待、首次 scratch 扩容、上下文/映射检查、描述符生成、HIXL 传输、缓存维护、同步、状态回读，以及开启后的 DevBW 回读。

输入构造、外部 ready 握手、输出精度校验不在该窗口内。分子仅为本轮实际拷贝的有效载荷，不计完整 full/selection 容量或 descriptor 元数据：

```text
host_copy_us = (end_ns - begin_ns) / 1000
host_bandwidth_gib_s = payload_bytes / ((end_ns - begin_ns) × 1e-9) / 1024³
```

新增两段 C++ `steady_clock` 墙钟计时，并通过底层算子的返回值传给 Session，无需增加 Device→Host timing Tensor 拷贝：

| 字段 / smoke 列 | 起点 | 终点 |
|---|---|---|
| `host_descriptor_us` / `HostDescriptor/us` | 描述符生成 RunOpApiV2 下发前 | 生成 kernel 与流同步完成、status 回读、全部生成状态/覆盖数检查通过后 |
| `host_transfer_us` / `HostTransfer/us` | 传输 RunOpApiV2 下发前 | HIXL READ/Drain、缓存维护 kernel、流同步返回后 |
| `host_other_us` / `HostOther/us` | 差额 | HostCopy 减去上述两段耗时 |

传输分段不包含随后单独的错误标志 D2H 回读/检查；这部分属于 HostOther。HostOther 还包括分段之外的分配、前置校验、Python 和统计开销。两段均包含调度与队列等待；描述符阶段的同步也可能等待此前排队的缓冲区清零等操作，不能等同于纯描述符 kernel 时间。

两处 RunOpApiV2 必须显式传 `sync=true`。其默认 false 在 torch_npu task queue 开启时可能仅完成入队就返回；即使回调内部同步，也不能保证外层计时结束前已执行。`9284379` 中未传该参数，可能导致 HostTransfer 只有几微秒，而实际等待落入随后 `status.cpu()`、误归入 HostOther。更新后重编 native 扩展再比较 HostTransfer 与 DeviceTime；显式同步也会计入框架同步开销。

零传输仍记录生成/确认时间，传输分段为 0。`HOSTSUMMARY` 输出各分段耗时的算术平均值；总 Host 带宽仍按总有效字节/总 Host 时间计算。总时间与阶段时间使用各自单调 Host 时钟，只对持续时间进行换算，不混用两个时钟的绝对时间戳。

统计存入同一 `last_transfer_stats`，只开 Host 时不包含 Device 字段。两种计时都关闭时仍为 None。零传输 Host 耗时可能非零，但带宽为 0。开启 DevBW 增加 Host 回读开销，比较时应保持相同打点配置；首次调用的扩容开销可通过预热排除。

原先串行 GetList 的其他批量值不再适用。scratch 按全部候选 mapping 预留完整 descriptor 列表，有 RoPE 时每 token 两条、每条 24 字节，按 128 字节补齐，并在容量不足时增长。

### 描述符生成：多 lane 组装与批量写 GM

生成阶段固定使用 32 lanes，每个 tile 处理最多 32 个有效 token；它与传输阶段的 `batch_descriptors`、`transfer_cores` 独立。每个 lane 从累计 count 定位自己的 mapping，计算地址，在自己的 UB 区间写一条 KV descriptor；启用 RoPE 时再写一条 RoPE descriptor。每条仍为 24 字节，排列仍为 `KV0, RoPE0, KV1, RoPE1, ...`。

生成核按完整 tile 连续分工：将 `ceil(total_tokens/32)` 个 tile 按商和余数分给 channel 数个核，前面的核多处理一个 tile，空闲核写零完成数。每个完整 tile 无 RoPE 为 768 B、有 RoPE 为 1536 B，均为 128 B 的整数倍，因此核间写区边界对齐，不共享同一条 GM cache line。

每个生成核预留一个 1536 B UB buffer。AIV 驱动先派发 VF 并行填 UB，通过 `V_MTE3` 的 SetFlag/WaitFlag 等待，再一次 `DataCopyPad` 写入 GM，通过 `MTE3_S` 等待 DMA 完成后才复用 UB 或报告生成完成。最后一个 tile 按有效 descriptor 数精确写入，单 token 尾批为 24 B / 48 B，列表内部不补齐。这里经过 UB 的是描述符，KV/RoPE 有效载荷仍走 URMA 的 HBM→HBM。

描述符 DMA 全部完成后，每核复用 UB 组装完整 128 B status 行：包含 code、phase、生成 token 数，并清零传输错误字和保留字段。通过 `S_MTE3` 等待标量 UB 写入，再一次 DMA 发布整行，最后用 `MTE3_S` 等待完成；空工作核、context 校验失败分支也发布完整行。取消原先的临时校验状态、逐字段标量 GM 覆盖和单独的 `status.zero_()`。这不是宣称 128 B DMA 对并发读取具备原子性；Host 仍在流同步结束后回读所有核状态，只有全部为 `code=0, phase=5` 且覆盖数正确才下发 HIXL。

默认 smoke 每轮 2048 token、有 RoPE：生成 64 个 tile，共 64 次 DMA，写入 96 KiB 描述符。Python 接口、context ABI v2、Host 阶段检查以及 HIXL 传输策略保持不变，需要重新编译 native 扩展。该改动尚未实现双缓冲或 AICube 重叠；实际收益需在目标机比较 `HostDescriptor`、`HostCopy`，不能仅凭 GM 写次数判断。

无 RoPE 时 register/bind 传 `None`。若模型 full cache 有可消去的单例 head 轴，可在确认语义一致后使用 `squeeze(-2)` 转为 rank-3。

多个层可顺序共享 Session，但必须在 Connect 前注册/绑定全部层，再逐层 prepare。`publish_layout()` 会同步当前生产者写入并封闭注册，不会自动发送网络消息，也不保证之后每轮更新已就绪。

调用前由 Host 检查映射、地址范围和 context 所有权；描述符生成状态检查通过后才提交 READ。传输成功依据 HIXL 内部 Drain、流同步和错误字节检查；生成 token 数只是描述符覆盖统计，不是独立 CQ 完成计数。

失败后 Session 标记为 poisoned，不自动重试可能部分提交的列表。应丢弃本轮 selection/match 状态，关闭 Session 并按模型恢复策略重建。源 Tensor 必须存活到 Client 完成并关闭 channel 之后。

## 6. LocalCommRes 生成

使用同版本 HIXL 附带的 `hixl_tool`，在实际设备环境中生成。下面以具有 UB CTP 连通性的本地用户 Device 0、1 为例，编号和协议应按实际环境替换：

```bash
cd "$HIXL_SOURCE_DIR"
bash build.sh --aiv --examples

# 若无关 benchmark 链接失败，可在已有配置的 build 中只构建工具。
cmake --build build --target hixl_tool --parallel 8
export LD_LIBRARY_PATH="$HIXL_SOURCE_DIR/build/src/hixl:${LD_LIBRARY_PATH}"

mkdir -p "$HOME/avsg_comm_res"
./build/scripts/tools/hixl_tool/hixl_tool local_comm_res \
  --device_id 0,1 --protocol_desc ub_ctp:device \
  --output "$HOME/avsg_comm_res"
```

文件名为 `local_comm_res_<用户Device ID>_<物理Device ID>.json`，以实际输出为准。跨机在两台机器分别生成，不能复制对端的 endpoint 配置。容器需要可访问设备、urma_admin 和实际硬件拓扑。

Host IP/端口只解决控制与建链地址；设备数据路径由通信资源与硬件互联决定。生成其他协议的 JSON 不等于当前 AIV 传输后端已验证支持该协议。

## 7. 本机双卡 / 跨机双卡 smoke

先设置脚本绝对路径，并从源码包目录之外运行：

```bash
export SMOKE="$RECIPES_ROOT/ops/ascendc/avsg_urma/examples/d2d_smoke.py"
cd /tmp
```

先启动 Server，再启动 Client。以下变量均需填写实际值：

```bash
# Server：full cache 持有端
python "$SMOKE" server \
  --device "$SERVER_DEVICE" --local-engine "$SERVER_ENGINE" \
  --local-comm-res-file "$SERVER_COMM_RES" --channels "$CHANNELS" \
  --control-host "$CONTROL_HOST" --control-port "$CONTROL_PORT"

# Client：READ 发起端
python "$SMOKE" client \
  --device "$CLIENT_DEVICE" --local-engine "$CLIENT_ENGINE" \
  --peer-engine "$SERVER_ENGINE" \
  --local-comm-res-file "$CLIENT_COMM_RES" --channels "$CHANNELS" \
  --control-host "$CONTROL_HOST" --control-port "$CONTROL_PORT" \
  --copy-mode single-kernel
```

单 kernel 模式两端使用相同的 `CHANNELS`，例如设为 4。客户端保持默认 32 lanes，可省略 `--transfer-cores`，也可显式指定 `--transfer-cores "$CHANNELS" --jetty-strategy exclusive`。将客户端模式改成 `aclgraph` 或 `npugraph-ex` 可测试图重放。追加 `--check-write` 可测试 WRITE 后的 READ。每次运行重新启动 smoke 服务端和客户端。

要测旧两阶段路径和它的带宽统计，将客户端模式改为 `--copy-mode baseline`。以下多 jetty 参数也用于此模式。

共享模式示例：Server 保持 `--channels 4`，Client 在原参数后加：

```bash
--channels 4 --transfer-cores 8 --jetty-strategy shared --lanes 32 --devbw --warmup 2
```

独占模式使用 `--channels 4 --transfer-cores 4 --jetty-strategy exclusive`。新增核数/策略/打点/预热参数只由 Client 消费，不传给 Server 的 Python 构造器。默认不传时仍为独占、不打点。

`--warmup` 增加同样进行精度校验的预热轮数，默认 0。随后仍测量 8 轮，逐轮输出 `DEVRESULT`，最后 `DEVSUMMARY` 使用总字节数/设备时间总和计算聚合 DevBW，排除预热和零工作量调用，不对带宽简单求算术平均。

Host 统计不需要 `--devbw`：每轮输出 `HOSTRESULT` 的 `HostCopy/us`、`Bandwidth/GiB/s`，最终输出 `HOSTSUMMARY`，按总有效字节/Host 时间总和计算，排除预热与零传输。

两端新增数据参数 `--full-tokens 8192 --selection-tokens 4096 --copy-tokens 2048 --dtype float8_e4m3fn`，这些也是默认值。full/selection token 数需为 128 的倍数，copy token 数不超过两者。两端 dtype/full 配置应一致；可选择 `float8_e5m2`。

| 数据 | 默认形状，1 字节/元素 | 大小 |
|---|---|---|
| full KV + RoPE | `[64,128,512]` + `[64,128,64]` | 4.5 MiB |
| selection KV + RoPE | `[32,128,512]` + `[32,128,64]` | 2.25 MiB |
| 每轮传输 | 2048 × (512 + 64) 字节 | 1.125 MiB |

每轮 4096 条 descriptor，占 96 KiB；8 轮合计 9 MiB 有效载荷。默认 32 lanes、单核/jetty、队列足够时有 128 批；默认尺寸没有短尾，可传 `--copy-tokens 2049` 单独覆盖尾批。需要框架支持 FP8 dtype view 和 NPU 存储；不做 FP8 算术、量化或类型回退。

| 参数 | 本机双卡示例 | 跨机双卡示例 |
|---|---|---|
| Device | Server 0、Client 1 | A 的 0、B 的 0 |
| Server engine | 本机 Host IP:20011 | A 的 Host IP:20011 |
| Client engine | 本机 Host IP:20012 | B 的 Host IP:20012 |
| peer-engine | 等于 Server engine | 等于 A 的 Server engine |
| control-host | 本机可绑定、可达的 Host IP | A 可绑定且 B 可达的 Host IP |
| control-port | 两端相同，如 17011 | 两端相同，如 17011 |
| 通信资源文件 | 分别使用本机 Device 0/1 的文件 | 各自机器、各自 Device 的文件 |

上述编号和端口仅为填写示例，不代表你的机器配置。engine 端口和控制端口应分开且可用。先令两端 `CHANNELS=1` 跑通，再验证所需的多 channel 配置；可见设备映射也必须与资源文件对应。

### Smoke 具体检查什么

- Server 构造默认 FP8 full KV `[64,128,512]` 和 RoPE `[64,128,64]`。CPU 生成展平元素编号 `%123` 的 uint8 位模式，上传 NPU 后 view 为 FP8；这些位模式在两种 FP8 编码下均为有限值。
- Client 分配 FP8 selection KV `[32,128,512]` 和 RoPE `[32,128,64]`，字节初值 `0x80`，注册后通知 Server。
- 收到 manifest 后 Connect、prepare；执行 8 轮，每轮手工生成 2048 个 `src=(arange(2048)*17+step)%8192`，目标为 slot 0～2047。
- 每轮将输出 view 为 uint8 后回读到 CPU，按原始字节精确比较；检查 slot 2048～4095 仍为 `0x80`，避免 FP8 转 float 导致舍入或 NaN 比较干扰。
- 每轮追加一次 `count=0` 的合法调用；零调用后没有额外再比较全部输出。
- Client 先关闭 Session，再发 `closed`；Server 收到后才退出并释放源资源。

TCP 只传布局和握手，KV 数据走 URMA。测试没有执行真实 top-k 或 AVSG match，映射是人工合成的；也没有 CPU 传输回退。它不覆盖整网精度、真实 cache 更新协议、性能、AICube 或流水重叠。

## 8. CPU 验证与设备验证边界

```bash
cd "$RECIPES_ROOT/ops/ascendc/avsg_urma"
PYTHONPATH=. python -m unittest discover -s tests -v

# 已设置 HIXL_SOURCE_DIR 时，额外以真实 checkout 的 launcher/参数头编译适配层。
# 该 Host ABI 测试不需要 CANN，但需要 g++。
HIXL_SOURCE_DIR="$HIXL_SOURCE_DIR" PYTHONPATH=. python -m unittest discover -s tests -p test_pipeline.py -v

g++ -std=c++17 -Wall -Wextra -Werror -Iinclude \
  tests/abi_check.cpp -o /tmp/avsg_urma_abi
/tmp/avsg_urma_abi
```

设置 `HIXL_SOURCE_DIR` 时会运行真实 checkout 参数头的 ABI 检查；本次以 HIXL `avsg/5ac7f35` 验证，共 21 项 CPU 测试通过，覆盖 1/2 字节描述符、FP8 布局、Host/Device 带宽与阶段分解、参数映射、共享分组、错误与零工作量等。描述符用例新增多 lane UB 组装、DMA 精确长度及事件调用顺序、1/31/32/33 token 边界、145 token 跨核多 tile 与未使用区域哨兵检查。CPU stub 不验证硬件事件和 cache 可见性；测试不包含 Torch 原生绑定/CANN kernel 编译、FP8 NPU Tensor 操作或真实传输。

CPU 测试没有执行 HIXL 设备 kernel，不能验证 ASC 编译、SQ/CQ 行为、Drain 错误路径、cache/fence 语义或真实 NIC 完成。两阶段版本的目标 CANN 编译、双端 smoke 和 profiling 仍需在 Ascend Linux 上完成。提交前执行仓库要求的 `pre-commit run --all-files`；若本机缺少该工具，应明确记录未执行。

## 9. 常见错误定位

| 现象 | 检查方向 |
|---|---|
| 找不到 `HIXL_LIBRARY` | `HIXL_LIBRARY_DIR` 必须指向实际包含库文件的目录，而非源码根目录 |
| `ge::GRAPH_FAILED` 重定义 | 确认已包含 SDK/框架独立 OBJECT target 修复；bridge 编译命令不应出现 torch_npu include；检查全局 CPATH/CXXFLAGS |
| 对象 `incompatible with aicorelinux` | 最终产物是 Host `.so`；检查 ASC driver 链接命令包含 `-shared`，勿直接使用设备 ld 链接 Host 对象 |
| `libtorch_npu.so: undefined symbol` | 先单独导入 torch/torch_npu，确认 Python、框架版本和实际库路径；例如 Conda 3.10 不应误加载系统 3.11 的框架副本 |
| 源码目录 `cannot import _native` | 检查包 `__file__`、`_native*.so` 和安装位置；源码包可能遮蔽已安装包 |
| `GetAivChannelPointers` 符号缺失 | HIXL 头文件、重建库及实际加载库是否一致 |
| build.sh 最后只有 `collect2` / Error 1 | 真正链接错误在前面的 undefined reference / cannot find 行；可单独构建 hixl_tool 继续生成资源 |
| context ABI 不匹配 | 重编译扩展、重启进程、重新创建 Session；不要复用 v1 context |
| `Descriptor generation failed ... code=0, phase=1` | 成功应为 `0/5`，校验失败应为 `100/1`；`0/1` 不是完整终态。已将状态改为每核一次完整行 DMA 发布；需重编 native 并重启进程，仍失败时保留 generated_tokens/total_tokens、轮次、完整日志和实际加载的扩展路径。仅凭该组合不能确定硬件 cache/写入顺序或旧二进制等具体原因 |
| Server 的 `receive/exact` 报 EOFError | 控制连接被对端关闭；若 Client 先报生成异常，应先处理 Client 原始错误。该轮尚未提交 HIXL READ；关闭失败 Session 后重启两端，不直接复用继续测 |

用以下命令保留完整构建日志和实际链接命令：

```bash
python -m pip install --no-build-isolation --no-cache-dir -v . 2>&1 | tee /tmp/avsg_build.log
find build -path '*/CMakeFiles/_native.dir/link.txt' -print -exec cat {} \;
```

不要通过修改 CANN 系统头、预定义 GE include guard 或关闭警告来隐藏头文件冲突。诊断时保留出错源文件、完整符号和 previous-definition 路径。

## 10. 性能比较与后续 AICube

当前版本是后续实验的顺序基线：保留完整 descriptor 生成、Host 校验、HIXL 传输、缓存维护和同步。它与旧版 `315fc2a` 的 kernel 数和 Host 往返次数不同，不能只比较 HIXL kernel 耗时就认定端到端更快。

比较时保持相同设备对、源布局、mapping/count、dtype、channel 数和 payload，分别记录描述符生成、传输、缓存维护、Host 端 `copy_into` 墙钟耗时，并同时验证结果。

后续 AICube tile 流水版本需要增加按计算消费顺序的 tile 计划、缓冲复用、READY/FREE 代数协议和传输完成后的数据可见性保证。旧两阶段路径通过完整 `HixlAivSimtDevGetList` 入口下发 READ，并提供多 core 共享 jetty。单 kernel 路径复用 HIXL 内部的独占 jetty 提交与回收逻辑，提供 `torch.compile` 和 ACLGraph 的 smoke 入口，设备编译及图重放仍待目标机验证。

## 11. 源码索引

| 文件 | 职责 |
|---|---|
| [session.py](avsg_urma/session.py) | Python 资源管理、注册、建链、prepare 与 copy_into |
| [protocol.py](avsg_urma/protocol.py) | RemoteCache、context ABI、完整 descriptor 工作区容量 |
| [binding.cpp](csrc/binding.cpp) | Torch schema、Host 校验、两阶段下发与状态检查 |
| [avsg_urma.asc](csrc/avsg_urma.asc) | 描述符生成、目标缓存维护 kernel |
| [hixl_transfer.cpp](csrc/hixl_transfer.cpp) | 标量参数 → HixlAivOpParam → HixlLaunchAivSimt |
| [hixl_bridge.cpp](csrc/hixl_bridge.cpp) | HIXL 初始化、注册、Connect、channel 导出与 GetAddr |
| [avsg_urma_abi.h](include/avsg_urma_abi.h) | Host/Device 共享布局 |
| [d2d_smoke.py](examples/d2d_smoke.py) | 两进程合成数据传输与精确验证 |
| [test_pipeline.py](tests/test_pipeline.py) | 真实 descriptor/adapter 的 CPU 编译回归 |
