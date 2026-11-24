# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Hold a GLM model rank's full KV/RoPE in peer NPU HBM."""

import argparse
import json
from pathlib import Path
import signal
import threading

import torch
import torch_npu  # noqa: F401
from avsg_urma import AvsgUrmaCacheServer


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

    with AvsgUrmaCacheServer(local_engine=config["local_engine"], device=device,
                            local_comm_res=json.dumps(config["local_comm_res"]),
                            channels=config.get("channels", 1)) as server:
        for layer in range(layout["layers"]):
            shape = (layout["blocks"], layout["block_size"])
            kv = torch.zeros((*shape, layout["kv_dim"]), dtype=dtype, device=f"npu:{device}")
            rope = torch.zeros((*shape, layout["rope_dim"]), dtype=dtype, device=f"npu:{device}")
            server.register_layer(layer, kv, rope)
        payload = json.dumps(server.publish_layout(), indent=2)
        with manifest.open("x", encoding="utf-8") as output:
            output.write(payload)
        try:
            print(f"Registered remote HBM for {layout['layers']} layers. Manifest: {manifest}", flush=True)
            done.wait()
        finally:
            manifest.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
