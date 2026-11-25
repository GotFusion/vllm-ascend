# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Launch the peer cache service or run the three-layer AVSG example."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import signal
import threading


def load_local_comm_res(resource, directory):
    """Return the JSON string expected by HIXL from an object or a file path."""
    if isinstance(resource, dict):
        return json.dumps(resource)
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("local_comm_res must be a JSON object or a nonempty JSON file path")

    path = directory / resource
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load local_comm_res from {path}: {error}") from error
    if not isinstance(content, dict):
        raise ValueError(f"local_comm_res file {path} must contain a JSON object")
    return json.dumps(content)


def random_causal_topk(rng, position, width):
    """Include the current token to exercise WRITE -> READ on every decode."""
    history = rng.sample(range(position), min(position, width - 1))
    indices = history + [position]
    rng.shuffle(indices)
    return indices + [-1] * (width - len(indices))


def load_match(ops_library):
    """Load the vLLM Ascend Match binding before the first capture."""
    import torch

    if ops_library is None:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Build and install vLLM Ascend with Ascend 950 custom kernels")
    else:
        library = str(ops_library.resolve(strict=True))
        if hasattr(torch.ops._C_ascend, "npu_avsg_match") and library not in torch.ops.loaded_libraries:
            raise RuntimeError("Another AvsgMatch binding is already loaded; run the example in a fresh process")
        torch.ops.load_library(library)
    name = "_C_ascend::npu_avsg_match"
    if not hasattr(torch.ops._C_ascend, "npu_avsg_match") or not torch._C._dispatch_has_kernel_for_dispatch_key(
        name, "PrivateUse1"
    ):
        raise RuntimeError(f"The selected library must register the NPU kernel for {name}")


def serve(config, directory):
    local_comm_res = load_local_comm_res(config["local_comm_res"], directory)

    import torch
    import torch_npu  # noqa: F401
    from avsg_urma import AvsgUrmaCacheServer

    from .model import BLOCK_SIZE, FULL_BLOCKS, NUM_LAYERS, TOKEN_DIM, FP8_DTYPE

    device = config["device"]
    torch.npu.set_device(device)
    manifest = directory / config["manifest"]
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())

    with AvsgUrmaCacheServer(
        local_engine=config["local_engine"], device=device,
        local_comm_res=local_comm_res, channels=config.get("channels", 1),
    ) as server:
        caches = [torch.empty((FULL_BLOCKS, BLOCK_SIZE, TOKEN_DIM),
                              dtype=FP8_DTYPE, device=f"npu:{device}") for _ in range(NUM_LAYERS)]
        for layer, cache in enumerate(caches):
            server.register_layer(layer, cache)
        # HIXL AIV requires equal registration counts on both endpoints.
        # The client also registers one shared Prefill staging buffer and one Decode staging buffer per layer.
        for _ in range(1 + NUM_LAYERS):
            server.register_auxiliary_cache(caches[0].new_empty((1, BLOCK_SIZE, TOKEN_DIM)))
        layout = json.dumps(server.publish_layout(), indent=2)
        with manifest.open("x", encoding="utf-8") as output:
            output.write(layout)
        try:
            print(f"Remote KV ready on npu:{device}. Manifest: {manifest}", flush=True)
            done.wait()
        finally:
            manifest.unlink(missing_ok=True)


def check_transfer(session, status, operation):
    try:
        return session.check_graph_copy(status)
    except RuntimeError as error:
        raise RuntimeError(f"{operation}: {error}") from error


def check_prefill(session, model, length):
    for layer, args in enumerate(model.prefill_writes):
        written = check_transfer(session, args[-2], f"Prefill WRITE layer {layer}")
        if written != length:
            raise RuntimeError(f"Prefill layer {layer}: wrote {written}, expected {length}")


def check_decode(session, model, position, topk, remapped, misses, reference, *, cold_selection):
    import torch

    from .model import TOKEN_DIM, selection_slots, token_rows

    transferred = []
    width = topk.shape[-1]
    minimum = width if cold_selection else 1
    for layer, (write, read) in enumerate(zip(model.decode_writes, model.reads)):
        step = f"Decode position {position}, layer {layer}"
        if check_transfer(session, write[-2], f"{step}, WRITE") != 1:
            raise RuntimeError(f"Decode layer {layer}: expected one fresh KV row on the remote side")
        copied = check_transfer(session, read[-1], f"{step}, READ")
        if copied != misses[layer].item():
            raise RuntimeError(f"Decode layer {layer}: URMA READ count differs from Match count")
        if not minimum <= copied <= width:
            raise RuntimeError(f"{step}: expected {minimum}..{width} READ tokens, got {copied}")
        transferred.append(copied)
        if reference is not None:
            stage = write[1].view(torch.uint8).reshape(-1, TOKEN_DIM)
            reference[layer][position].copy_(stage[0])
            # Decode inputs have exactly `width` valid picks. Attention would mask a negative remap.
            if (remapped[layer] < 0).any().item():
                raise RuntimeError(f"{step}: Match masked a valid TopK entry")
            slots = selection_slots(remapped[layer], model.selection_table)
            actual = token_rows(read[1], slots).cpu()
            expected = token_rows(reference[layer], topk[layer]).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"Layer {layer} KV payload")
    return transferred


def run(args, config, directory):
    local_comm_res = load_local_comm_res(config["local_comm_res"], directory)

    import torch
    import torch_npu  # noqa: F401
    from avsg_urma import AvsgUrmaSession, RemoteCache

    from .model import (
        AvsgSparseModel, BLOCK_SIZE, FULL_BLOCKS, NUM_LAYERS, TOKEN_DIM, FP8_DTYPE, TOPK,
    )

    capacity = FULL_BLOCKS * BLOCK_SIZE
    if args.prompt_length < TOPK or args.steps < 1 or args.prompt_length + args.steps > capacity:
        raise ValueError(f"Require prompt_length >= {TOPK}, steps >= 1 and prompt_length + steps <= {capacity}")
    device_index = config["device"]
    torch.npu.set_device(device_index)
    device = torch.device("npu", device_index)
    load_match(args.ops_library)

    manifest_path = directory / config["remote_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {str(layer) for layer in range(NUM_LAYERS)}:
        raise ValueError("Remote manifest must describe the three example layers")
    remote_layers = tuple(RemoteCache(**manifest[str(layer)]) for layer in range(NUM_LAYERS))
    for remote in remote_layers:
        if (remote.blocks, remote.block_size, remote.kv_dim, remote.rope_dim, remote.dtype) != (
            FULL_BLOCKS, BLOCK_SIZE, TOKEN_DIM, 0, "float8_e4m3fn",
        ):
            raise ValueError("Remote cache layout differs from the example's FP8 KV-only layout")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    with AvsgUrmaSession(
        local_engine=config["local_engine"], device=device_index,
        local_comm_res=local_comm_res, channels=config.get("channels", 1),
    ) as session, torch.no_grad(), ExitStack() as cleanup:
        cleanup.callback(torch.npu.synchronize)
        model = AvsgSparseModel(device, session, remote_layers).eval()
        session.connect(config["peer_engine"], timeout_ms=config.get("timeout_ms", 30000))
        model.prepare_transfers(session)
        reference = None
        if args.verify:
            reference = [torch.empty((capacity, TOKEN_DIM), dtype=torch.uint8, device=device)
                         for _ in range(NUM_LAYERS)]
        prompt = (torch.randn(args.prompt_length, TOKEN_DIM).mul_(0.5)
                  .to(device=device, dtype=torch.bfloat16).to(FP8_DTYPE))
        prompt_topk = torch.tensor(
            [[random_causal_topk(rng, pos, TOPK) for pos in range(args.prompt_length)]
             for _ in range(NUM_LAYERS)],
            dtype=torch.int32, device=device,
        )
        token = model.prefill(prompt, prompt_topk, reference)
        check_prefill(session, model, args.prompt_length)
        print(f"Prefill: {args.prompt_length} tokens written per layer")

        position = torch.empty((1,), dtype=torch.int32, device=device)
        topk = torch.empty((NUM_LAYERS, 1, TOPK), dtype=torch.int32, device=device)
        decode = model.decode
        graph = None

        for pos in range(args.prompt_length, args.prompt_length + args.steps):
            # Update fixed-address graph inputs outside the decode graph.
            position.fill_(pos)
            topk.copy_(torch.tensor([[random_causal_topk(rng, pos, TOPK)] for _ in range(NUM_LAYERS)],
                                    dtype=torch.int32, device=device))
            if args.mode == "aclgraph" and graph is None:
                # Warm up the exact input, then start the first replay with cold Match state.
                _, warmup_indices, warmup_counts = decode(token, position, topk)
                check_decode(session, model, pos, topk, warmup_indices, warmup_counts, reference, cold_selection=True)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    graph_outputs = decode(token, position, topk)
                for state, read in zip(model.match_status, model.reads):
                    state.fill_(-1)
                    # Erase warmup payloads so --verify observes the READ issued by the first replay.
                    read[1].view(torch.uint8).fill_(0x7F)  # FP8 E4M3FN NaN
            for write, read in zip(model.decode_writes, model.reads):
                write[-2].fill_(-1)
                read[-1].fill_(-1)
            if graph is None:
                next_token, remapped, misses = decode(token, position, topk)
            else:
                graph.replay()
                next_token, remapped, misses = graph_outputs
            transferred = check_decode(
                session, model, pos, topk, remapped, misses, reference, cold_selection=pos == args.prompt_length,
            )
            token.view(torch.uint8).copy_(next_token.view(torch.uint8))
            print(f"Decode position {pos}: per-layer READ tokens {transferred}", flush=True)
        del decode, graph
        verified = " and byte-exact selected KV" if args.verify else ""
        print(f"PASS: {args.mode}, {args.steps} decode steps{verified}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("serve", help="register the Full Cache on the remote NPU")
    server.add_argument("--config", type=Path, required=True)
    client = subparsers.add_parser("run", help="run Prefill and Decode on the model NPU")
    client.add_argument("--config", type=Path, required=True)
    client.add_argument("--mode", choices=("aclgraph", "eager"), default="aclgraph")
    client.add_argument("--ops-library", type=Path, help="load a vLLM Ascend AvsgMatch shared library directly")
    client.add_argument("--prompt-length", type=int, default=384)
    client.add_argument("--steps", type=int, default=8)
    client.add_argument("--seed", type=int, default=42)
    client.add_argument("--verify", action="store_true", help="compare every selected KV row byte-for-byte")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "serve":
        serve(config, args.config.parent)
    else:
        run(args, config, args.config.parent)


if __name__ == "__main__":
    main()
