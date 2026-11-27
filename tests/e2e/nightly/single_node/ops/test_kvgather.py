# SPDX-License-Identifier: Apache-2.0
"""Requires the separately built ascend-kv-gather package and two idle 950 NPUs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("cores", [1, 2])
def test_kvgather_repeated_native_transfers(tmp_path, cores):
    pytest.importorskip("ascend_kv_gather")
    if torch.npu.device_count() != 2 or "950" not in torch.npu.get_device_name(0):
        pytest.skip("expose exactly two idle Ascend 950 cards for this operator test")
    root = Path(__file__).resolve().parents[5]
    output = tmp_path / "native-result.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "csrc/ascend_kv_gather/examples/d2d_smoke.py"),
            "--cores",
            str(cores),
            "--rounds",
            "32",
            "--output",
            str(output),
        ],
        check=True,
        timeout=1200,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["peer"]["native_transfers"] == 56
    assert result["client"]["transfer_requests"] == 28
