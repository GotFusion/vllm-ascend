# SPDX-License-Identifier: Apache-2.0
"""Two-NPU test of repeated native KV/RoPE gather followed by a Torch consumer.

Run in a private environment with exactly the two selected cards exposed.
This is an operator test with deterministic tensors, not a model benchmark.
"""

import argparse
import hashlib
import json
import multiprocessing
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from ascend_kv_gather import KVGatherClient, KVGatherPeer

LAYERS = 2
BLOCK_SIZE = 128
BLOCKS = 257
SOURCE_ROWS = BLOCKS * BLOCK_SIZE
SELECTION_ROWS = 8192
KV_DIM = 512
ROPE_DIM = 64


def pattern(layer, rows, dim, generation):
    # All integer values remain exactly representable in BF16, including the
    # peer update. Rows above the source example's 20,000-row limit are covered.
    columns = torch.arange(dim, dtype=torch.int64)
    return ((rows[:, None] * 17 + columns + layer * 31) % 127 + generation).to(torch.bfloat16)


def peer_worker(pipe, device, library, cores, timeout):
    peer = None
    try:
        torch.npu.set_device(device)
        rows = torch.arange(SOURCE_ROWS, dtype=torch.int64)
        banks = [
            torch.stack([pattern(layer, rows, dim, 0) for layer in range(LAYERS)])
            .reshape(LAYERS, BLOCKS, BLOCK_SIZE, dim)
            .to(f"npu:{device}")
            for dim in (KV_DIM, ROPE_DIM)
        ]
        torch.npu.synchronize()
        peer = KVGatherPeer(
            *banks, host="127.0.0.1", port=0, device=device, library=library, cores=cores, timeout=timeout
        ).start()
        pipe.send({"port": peer.port})
        while True:
            if not pipe.poll(timeout):
                raise TimeoutError("smoke supervisor stopped responding")
            command = pipe.recv()
            if command == "update":
                # Executed between completed gathers. Reusing the receive
                # buffers after a prior Torch read checks receiver visibility.
                for bank in banks:
                    bank.add_(1)
                torch.npu.synchronize()
                pipe.send({"updated": True})
            elif command == "close":
                peer.check_health()
                stats = peer.stats
                peer.close()
                peer = None
                pipe.send({"closed": True, "stats": stats})
                return
            else:
                raise ValueError(f"unknown supervisor command {command!r}")
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
        raise
    finally:
        if peer is not None:
            peer.close()
        pipe.close()


def receive(pipe, timeout):
    if not pipe.poll(timeout):
        raise TimeoutError("timed out waiting for the owned peer process")
    result = pipe.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def exercise(client, selections, pipe, device, timeout, rounds):
    expected = [torch.full((SELECTION_ROWS, dim), -1, dtype=torch.bfloat16) for dim in (KV_DIM, ROPE_DIM)]
    sizes = (1, 3, 13, 31, 128, 2048, SELECTION_ROWS, 0)
    cases = []
    generation = 0
    for step in range(rounds):
        if step and step % len(sizes) == 0:
            pipe.send("update")
            assert receive(pipe, timeout) == {"updated": True}
            generation += 1
        count = sizes[step % len(sizes)]
        layer = step % LAYERS
        sources = (torch.arange(count, dtype=torch.int64) * 7919 + 19 + step * 13) % SOURCE_ROWS
        destinations = (torch.arange(count, dtype=torch.int64) * 5 + step * 7) % SELECTION_ROWS
        # Leave deliberate poison outside the valid prefix. Match owns only
        # count rows; padding must never become a transfer or an error.
        mapping = torch.full((1, 1, SELECTION_ROWS, 2), -999, dtype=torch.int32)
        mapping[0, 0, :count, 0] = sources.to(torch.int32)
        mapping[0, 0, :count, 1] = destinations.to(torch.int32)
        client.gather(
            layer,
            torch.tensor([1], dtype=torch.int32, device=device),
            mapping.to(device),
            torch.tensor([[count]], dtype=torch.int32, device=device),
        )
        for tensor, reference, dim in zip(selections, expected, (KV_DIM, ROPE_DIM)):
            reference.index_copy_(0, destinations, pattern(layer, sources, dim, generation))
            # Full selection comparison checks fresh values and untouched hits.
            torch.testing.assert_close(tensor.view(SELECTION_ROWS, dim).cpu(), reference, rtol=0, atol=0)
        cases.append({"step": step, "layer": layer, "rows": count, "generation": generation})
    return cases


def run(args):
    if args.rounds < 16:
        raise ValueError("at least 16 rounds are required to test updated source data and cursor reuse")
    context = multiprocessing.get_context("spawn")
    pipe, child_pipe = context.Pipe()
    child = context.Process(target=peer_worker, args=(child_pipe, 0, args.library, args.cores, args.timeout))
    child.start()
    child_pipe.close()
    client = None
    report = None
    try:
        ready = receive(pipe, args.timeout)
        torch.npu.set_device(1)
        device = torch.device("npu:1")
        selections = tuple(
            torch.full((SELECTION_ROWS // BLOCK_SIZE, BLOCK_SIZE, dim), -1, dtype=torch.bfloat16, device=device)
            for dim in (KV_DIM, ROPE_DIM)
        )
        client = KVGatherClient(
            *selections,
            host="127.0.0.1",
            port=ready["port"],
            device=device,
            library=args.library,
            cores=args.cores,
            timeout=args.timeout,
        )
        cases = exercise(client, selections, pipe, device, args.timeout, args.rounds)
        stats = client.stats
        client.close()
        client = None
        pipe.send("close")
        peer_report = receive(pipe, args.timeout)
        child.join(args.timeout)
        if child.exitcode != 0:
            raise RuntimeError(f"owned peer exit code {child.exitcode}")
        assert stats["transfer_requests"] == sum(case["rows"] > 0 for case in cases)
        assert stats["transferred_rows"] == sum(case["rows"] for case in cases)
        assert peer_report["stats"]["native_transfers"] == 2 * stats["transfer_requests"]
        library = Path(stats["native_library_path"])
        report = {
            "status": "PASS",
            "scope": "two-NPU deterministic operator tensors; not model accuracy",
            "source_rows": SOURCE_ROWS,
            "layers": LAYERS,
            "cores": args.cores,
            "cases": cases,
            "client": stats,
            "peer": peer_report["stats"],
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            if child.is_alive():
                child.terminate()
                child.join(10)
                if child.is_alive():
                    child.kill()
                    child.join(10)
            pipe.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="explicit libascend_kv_gather.so path; defaults to installed package")
    parser.add_argument("--cores", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(args), indent=2)
    if args.output is not None:
        args.output.write_text(result + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
