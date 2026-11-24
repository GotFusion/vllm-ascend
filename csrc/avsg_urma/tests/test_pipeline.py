# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""CPU compilation checks for real descriptor code and Host launcher adapter."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CXX = shutil.which("g++")


@unittest.skipUnless(CXX, "g++ is required for the pipeline regression")
class PipelineTest(unittest.TestCase):
    def compile_run(self, *sources, includes=()):
        with tempfile.TemporaryDirectory(prefix="avsg_pipeline_") as temp:
            binary = Path(temp) / ("check.exe" if os.name == "nt" else "check")
            compile_result = subprocess.run(
                [CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 *("-I" + str(path) for path in includes),
                 "-I" + str(ROOT / "include"), "-I" + str(ROOT / "tests/stubs"),
                 *(str(ROOT / source) for source in sources), "-o", str(binary)],
                capture_output=True, text=True,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS:", run.stdout)

    def test_compact_descriptor_generation(self):
        self.compile_run("tests/kernel_cpu_test.cpp")

    def test_hixl_launcher_adapter(self):
        self.compile_run("csrc/hixl_transfer.cpp", "tests/transfer_cpu_test.cpp")

    @unittest.skipUnless(os.environ.get("HIXL_SOURCE_DIR"), "set HIXL_SOURCE_DIR for real launcher-header ABI check")
    def test_hixl_launcher_adapter_with_checkout_headers(self):
        hixl = Path(os.environ["HIXL_SOURCE_DIR"])
        # Real launcher/parameter headers precede stubs, without CANN or Torch.
        self.compile_run("csrc/hixl_transfer.cpp", "tests/transfer_cpu_test.cpp",
                         includes=(hixl / "src", hixl / "include"))
