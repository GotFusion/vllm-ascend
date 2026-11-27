# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Hold a GLM model rank's full KV/RoPE in peer NPU HBM."""

import argparse
import json
import signal
import threading
from contextlib import ExitStack
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from avsg_urma import AvsgUrmaCacheServer


def close_gather_peer(peer, *, failure=None):
    """Emit one session-level evidence record, including failed teardown."""
    observed = dict(peer.stats)
    record = {
        "event": "kvgather_peer_summary",
        "status": "failed",
        "healthy_before_close": False,
        "observed_before_close": observed,
    }
    if failure is not None:
        record["session_error"] = f"{type(failure).__name__}: {failure}"
    try:
        try:
            peer.check_health()
            record["healthy_before_close"] = True
        finally:
            peer.close()
        if failure is None:
            record["status"] = "closed"
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # "closed" describes lifecycle completion, not benchmark acceptance.
        print(json.dumps(record, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    layout = config["cache"]
    manifest = args.config.parent / config["manifest"]
    dtype = getattr(torch, layout["dtype"])
    device = config["device"]
    torch.npu.set_device(device)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())

    with (
        AvsgUrmaCacheServer(
            local_engine=config["local_engine"],
            device=device,
            local_comm_res=json.dumps(config["local_comm_res"]),
            channels=config.get("channels", 1),
        ) as server,
        ExitStack() as stack,
    ):
        shape = (layout["layers"], layout["blocks"], layout["block_size"])
        kv = torch.zeros((*shape, layout["kv_dim"]), dtype=dtype, device=f"npu:{device}")
        rope = torch.zeros((*shape, layout["rope_dim"]), dtype=dtype, device=f"npu:{device}")
        server.register_layers(0, kv, rope)
        payload = json.dumps(server.publish_layout(), indent=2)
        gather_peer = None
        if "kvgather" in config:
            # Both transports register these exact allocations. HIXL continues
            # to publish PD/new-token rows; HCOMM reads them for selection.
            from ascend_kv_gather import KVGatherPeer, parse_endpoint

            gather_peer = KVGatherPeer(
                kv, rope, device=device, **parse_endpoint(config["kvgather"], args.config.parent)
            )
            stack.push(lambda exc_type, exc, traceback: close_gather_peer(gather_peer, failure=exc))
            # start() binds the socket synchronously and connects HCOMM on its
            # background thread. Publish the manifest before waiting for D.
            gather_peer.start()
        with manifest.open("x", encoding="utf-8") as output:
            output.write(payload)
        try:
            print(f"Registered remote HBM for {layout['layers']} layers. Manifest: {manifest}", flush=True)
            while not done.wait(1):
                if gather_peer is not None:
                    gather_peer.check_health()
        finally:
            manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
