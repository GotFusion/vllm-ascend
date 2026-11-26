# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Two-process, real remote-HBM READ smoke test. No CPU/mock transport fallback."""
import argparse
import json
import socket
import struct

import torch
import torch_npu  # noqa: F401

from avsg_urma import AvsgUrmaCacheServer, AvsgUrmaSession, RemoteCache
from avsg_urma.protocol import STATUS_BYTES

BLOCK_SIZE = 128
KV_DIM = 512
ROPE_DIM = 64
SENTINEL_BYTE = 0x80


def full_cache(tokens, dim, dtype):
    # Transport test: finite positive FP8 encodings, not a quantization test.
    # Upload bytes, then reinterpret on NPU; no FP8 arithmetic kernel required.
    raw = (torch.arange(tokens * dim, dtype=torch.int64) % 123).to(torch.uint8)
    return raw.reshape(tokens // BLOCK_SIZE, BLOCK_SIZE, dim).to("npu").view(dtype)


def selection_cache(tokens, dim, dtype):
    return torch.full((tokens // BLOCK_SIZE, BLOCK_SIZE, dim), SENTINEL_BYTE,
                      dtype=torch.uint8, device="npu").view(dtype)


def receive(sock):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise EOFError("peer closed control connection")
            data.extend(chunk)
        return data
    size = struct.unpack("!I", exact(4))[0]
    if size > 1024 * 1024:
        raise ValueError("control message too large")
    return json.loads(exact(size))


def send(sock, obj):
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def server(args, options):
    with AvsgUrmaCacheServer(**options) as endpoint:
        dtype = getattr(torch, args.dtype)
        kv = full_cache(args.full_tokens, KV_DIM, dtype)
        rope = full_cache(args.full_tokens, ROPE_DIM, dtype)
        endpoint.register_layer(0, kv, rope)
        manifest = endpoint.publish_layout()  # also waits for the producer writes
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.control_host, args.control_port))
            listener.listen(1)
            with listener.accept()[0] as sock:
                sock.settimeout(120)
                # Client sends READY only after registering its selection storage.
                if receive(sock) != {"state": "registered"}:
                    raise RuntimeError("invalid registration barrier")
                send(sock, manifest)
                if receive(sock) != {"state": "closed"}:
                    raise RuntimeError("client failed before completing the READs")
        print("Remote full HBM remained alive until the client closed its channels.")


def client(args, options):
    if not args.peer_engine:
        raise ValueError("client requires --peer-engine")
    with socket.create_connection((args.control_host, args.control_port), timeout=120) as sock:
        with AvsgUrmaSession(**options, transfer_cores=args.transfer_cores,
                             jetty_strategy=args.jetty_strategy, batch_descriptors=args.lanes,
                             measure_devbw=args.devbw, measure_hostbw=True) as endpoint:
            dtype = getattr(torch, args.dtype)
            if args.check_write:
                kv = selection_cache(2 * args.selection_tokens, KV_DIM, dtype)
                rope = selection_cache(2 * args.selection_tokens, ROPE_DIM, dtype)
                _, _, source_kv, source_rope = endpoint.bind_staging(
                    0, "write", kv, rope, args.selection_tokens // BLOCK_SIZE)
                for source, dim in ((source_kv, KV_DIM), (source_rope, ROPE_DIM)):
                    source.view(torch.uint8).copy_(full_cache(args.selection_tokens, dim, dtype).view(torch.uint8))
            else:
                kv = selection_cache(args.selection_tokens, KV_DIM, dtype)
                rope = selection_cache(args.selection_tokens, ROPE_DIM, dtype)
                endpoint.bind_selection(0, kv, rope)
            send(sock, {"state": "registered"})
            layout = RemoteCache(**receive(sock)["0"])
            endpoint.connect(args.peer_engine)
            endpoint.prepare_layer(0, layout)
            if layout.blocks * layout.block_size != args.full_tokens:
                raise ValueError("server/client --full-tokens mismatch")
            n = args.copy_tokens
            q = torch.tensor([1], dtype=torch.int32, device="npu")
            if args.copy_mode != "baseline":
                check_single_kernel(args, endpoint, q, n)
                endpoint.close()
                send(sock, {"state": "closed"})
                return
            count = torch.tensor([[n]], dtype=torch.int32, device="npu")
            samples = []
            for step in range(args.warmup + 8):
                src = (torch.arange(n) * 17 + step) % (layout.blocks * layout.block_size)
                mapping = torch.stack((src, torch.arange(n)), -1).reshape(1, 1, n, 2).to("npu", torch.int32)
                result, result_rope = endpoint.copy_into(layer_id=0, full_q_actual_seq=q,
                                                         mapping=mapping, count=count)
                stats = endpoint.last_transfer_stats
                for output, dim in ((result, KV_DIM), (result_rope, ROPE_DIM)):
                    actual = output.view(torch.uint8).cpu().reshape(-1, dim)
                    expected = ((src[:, None] * dim + torch.arange(dim)) % 123).to(torch.uint8)
                    torch.testing.assert_close(actual[:n], expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual[n:], torch.full_like(actual[n:], SENTINEL_BYTE),
                                               rtol=0, atol=0)
                if stats is not None and step >= args.warmup:
                    samples.append(stats)
                    print(f"HOSTRESULT|step={step - args.warmup}|strategy={stats['strategy']}"
                          f"|cores={stats['transfer_cores']}|jettys={stats['jettys']}"
                          f"|active_jettys={stats['active_jettys']}|lanes={stats['lanes']}"
                          f"|dtype={args.dtype}|tokens={n}|bytes={stats['payload_bytes']}"
                          f"|HostCopy/us={stats['host_copy_us']:.3f}"
                          f"|HostDescriptor/us={stats['host_descriptor_us']:.3f}"
                          f"|HostTransfer/us={stats['host_transfer_us']:.3f}"
                          f"|HostOther/us={stats['host_other_us']:.3f}"
                          f"|Bandwidth/GiB/s={stats['host_bandwidth_gib_s']:.3f}", flush=True)
                    if args.devbw:
                        print(f"DEVRESULT|step={step - args.warmup}|strategy={stats['strategy']}"
                              f"|cores={stats['transfer_cores']}|jettys={stats['jettys']}"
                              f"|active_jettys={stats['active_jettys']}|lanes={stats['lanes']}"
                              f"|bytes={stats['payload_bytes']}|DeviceTime/us={stats['completion_us']:.3f}"
                              f"|DevBW/GiB/s={stats['devbw_gib_s']:.3f}", flush=True)
                # Zero copy is a valid call and must not reuse stale status.
                endpoint.copy_into(layer_id=0, full_q_actual_seq=q, mapping=mapping, count=torch.zeros_like(count))
                zero = endpoint.last_transfer_stats
                assert not zero["valid"] and zero["payload_bytes"] == 0 and zero["host_bandwidth_gib_s"] == 0
                if args.devbw:
                    assert zero["devbw_gib_s"] == 0
            if samples:
                total_bytes = sum(sample["payload_bytes"] for sample in samples)
                host_us = sum(sample["host_copy_us"] for sample in samples)
                host_bw = total_bytes / (host_us * 1e-6) / (1024 ** 3)
                descriptor_us = sum(sample["host_descriptor_us"] for sample in samples) / len(samples)
                transfer_us = sum(sample["host_transfer_us"] for sample in samples) / len(samples)
                other_us = sum(sample["host_other_us"] for sample in samples) / len(samples)
                print(f"HOSTSUMMARY|samples={len(samples)}|warmup={args.warmup}"
                      f"|MeanHostDescriptor/us={descriptor_us:.3f}|MeanHostTransfer/us={transfer_us:.3f}"
                      f"|MeanHostOther/us={other_us:.3f}"
                      f"|Bandwidth/GiB/s={host_bw:.3f}|scope=copy_into_wall", flush=True)
                if args.devbw:
                    total_us = sum(sample["completion_us"] for sample in samples)
                    bw = total_bytes / (total_us * 1e-6) / (1024 ** 3)
                    print(f"DEVSUMMARY|samples={len(samples)}|warmup={args.warmup}"
                          f"|DevBW/GiB/s={bw:.3f}|scope=HIXL_transfer_only", flush=True)
        send(sock, {"state": "closed"})  # endpoint closed before remote source is freed
    print(f"PASS: 8 remote {args.dtype} READ steps, KV/RoPE byte-exact match, untouched slots and zero-copy calls.")


def check_single_kernel(args, endpoint, q, n):
    rope, kv, context, scratch, seen, status = endpoint.prepare_graph_copy(layer_id=0, mapping_capacity=n)
    status_words = STATUS_BYTES // 8
    status_rows = status.reshape(args.channels, status_words)
    mapping = torch.zeros((1, 1, n, 2), dtype=torch.int32, device=q.device)
    count = torch.zeros((1, 1), dtype=torch.int32, device=q.device)
    write = endpoint.prepare_graph_write(layer_id=0, source="write", mapping_capacity=n) if args.check_write else None
    write_mapping = torch.zeros_like(mapping) if write is not None else None

    def copy():
        write_status = None
        if write is not None:
            source_rope, source_kv, write_context, write_scratch, write_seen, write_status, remote_slots = write
            torch.ops.avsg_urma.write_graph_(source_rope, source_kv, q, write_mapping, count,
                                           write_context, write_scratch, write_seen, write_status, remote_slots)
        torch.ops.avsg_urma.copy_graph_(rope, kv, q, mapping, count, context, scratch, seen, status, write_status)

    if args.copy_mode == "aclgraph":
        copy()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            copy()
        submit = graph.replay
    elif args.copy_mode == "npugraph-ex":
        submit = torch.compile(copy, backend="npugraph_ex", fullgraph=True, dynamic=False,
                               options={"clone_input": False})
    else:
        submit = copy

    local = torch.arange(n)
    # Interleave destinations across descriptor shards, including adjacent KV/RoPE slots.
    dest = torch.cat((local[::2], local[1::2])) if args.channels > 1 else local
    for step in range(args.warmup + 8):
        src = (local + step) % args.full_tokens if args.check_write else (local * 17 + step) % args.full_tokens
        mapping.copy_(torch.stack((src, dest), -1).reshape_as(mapping))
        if write_mapping is not None:
            write_mapping.copy_(torch.stack((src, local), -1).reshape_as(write_mapping))
        count.fill_(n)
        status.fill_(-1)
        submit()
        if write is not None:
            assert endpoint.check_graph_copy(write[-2]) == n
        assert endpoint.check_graph_copy(status) == n
        active = sum(tokens > 0 for tokens in status_rows[:, 2].cpu().tolist())
        assert active == min(args.channels, (n + 31) // 32)
        for output, dim in ((kv, KV_DIM), (rope, ROPE_DIM)):
            actual = output.view(torch.uint8).cpu().reshape(-1, dim)
            expected = (((local if args.check_write else src)[:, None] * dim + torch.arange(dim)) % 123).to(torch.uint8)
            torch.testing.assert_close(actual[dest], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual[n:], torch.full_like(actual[n:], SENTINEL_BYTE), rtol=0, atol=0)

        # Reset the targets so an accidental READ on a zero-count replay is observable.
        for output in (kv, rope):
            output.view(torch.uint8).fill_(SENTINEL_BYTE)
        count.zero_()
        status.fill_(-1)
        submit()
        assert endpoint.check_graph_copy(status) == 0
        for output in (kv, rope):
            actual = output.view(torch.uint8).cpu()
            torch.testing.assert_close(actual, torch.full_like(actual, SENTINEL_BYTE), rtol=0, atol=0)

    invalid_mapping = mapping if write_mapping is None else write_mapping
    invalid_mapping[0, 0, -1, 0] = args.full_tokens
    count.fill_(n)
    status.fill_(-1)
    submit()
    expected_code = 101 if write is None else 102  # kBadMapping or failed prerequisite WRITE
    assert status_rows[:, :3].cpu().tolist() == [[expected_code, 1, 0]] * args.channels
    if write is not None:
        write_rows = write[-2].reshape(args.channels, status_words)
        assert write_rows[:, :3].cpu().tolist() == [[101, 1, 0]] * args.channels
    for output in (kv, rope):
        actual = output.view(torch.uint8).cpu()
        torch.testing.assert_close(actual, torch.full_like(actual, SENTINEL_BYTE), rtol=0, atol=0)
    print(f"PASS: {args.copy_mode}, {args.channels} exclusive jettys, byte-exact KV/RoPE, zero count and invalid mapping.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--local-engine", required=True)
    parser.add_argument("--peer-engine")
    parser.add_argument("--local-comm-res-file", required=True)
    parser.add_argument("--control-host", required=True)
    parser.add_argument("--control-port", type=int, required=True)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--transfer-cores", type=int, help="client transfer grid; default is actual channel count")
    parser.add_argument("--jetty-strategy", choices=("auto", "exclusive", "shared"), default="auto")
    parser.add_argument("--lanes", type=int, choices=(8, 32, 64), default=32)
    parser.add_argument("--devbw", action="store_true", help="client reads HIXL device timing after every call")
    parser.add_argument("--copy-mode", choices=("baseline", "single-kernel", "aclgraph", "npugraph-ex"),
                        default="baseline", help="client execution path")
    parser.add_argument("--check-write", action="store_true", help="client tests WRITE followed by READ")
    parser.add_argument("--warmup", type=int, default=0, help="extra verified steps excluded from DevBW summary")
    parser.add_argument("--full-tokens", type=int, default=8192)
    parser.add_argument("--selection-tokens", type=int, default=4096)
    parser.add_argument("--copy-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("float8_e4m3fn", "float8_e5m2"), default="float8_e4m3fn")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.role == "client" and args.copy_mode != "baseline" and args.devbw:
        parser.error("--devbw is available for the baseline copy only")
    if args.role == "client" and args.copy_mode != "baseline" and (
            args.jetty_strategy == "shared" or args.transfer_cores not in (None, args.channels) or args.lanes != 32):
        parser.error("single-kernel modes require one AIV per jetty and --lanes 32")
    if args.role == "client" and args.check_write and args.copy_mode == "baseline":
        parser.error("--check-write requires a single-kernel copy mode")
    if (args.full_tokens <= 0 or args.selection_tokens <= 0 or
            args.full_tokens % BLOCK_SIZE or args.selection_tokens % BLOCK_SIZE or
            not 0 < args.copy_tokens <= min(args.full_tokens, args.selection_tokens)):
        parser.error("full/selection tokens must be positive multiples of 128; copy tokens must fit both")
    torch.npu.set_device(args.device)
    with open(args.local_comm_res_file, encoding="utf-8") as file:
        comm_res = file.read()
    options = dict(local_engine=args.local_engine, device=args.device,
                   local_comm_res=comm_res, channels=args.channels)
    (server if args.role == "server" else client)(args, options)


if __name__ == "__main__":
    main()
