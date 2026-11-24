# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compile the real HIXL bridge against conflicting fake SDK header roots."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUBS = ROOT / "tests" / "stubs"
CXX = shutil.which("g++")


@unittest.skipUnless(CXX, "g++ is required for the C++ include-path regression")
class HeaderIsolationTest(unittest.TestCase):
    def test_framework_include_leak_reproduces_ge_redefinition(self):
        result = subprocess.run(
            [CXX, "-std=c++17", "-fsyntax-only", "-I" + str(ROOT / "include"),
             "-I" + str(STUBS / "framework"), "-I" + str(STUBS / "cann"), "-I" + str(STUBS),
             str(ROOT / "csrc" / "hixl_bridge.cpp")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GRAPH_FAILED", result.stderr)
        self.assertIn("redefinition", result.stderr)

    def test_sdk_only_bridge_compiles_links_and_runs(self):
        with tempfile.TemporaryDirectory(prefix="avsg_bridge_") as temp:
            binary = Path(temp) / ("bridge.exe" if os.name == "nt" else "bridge")
            result = subprocess.run(
                [CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I" + str(ROOT / "include"),
                 "-I" + str(STUBS / "cann"), "-I" + str(STUBS),
                 str(ROOT / "csrc" / "hixl_bridge.cpp"), str(ROOT / "tests" / "bridge_cpu_test.cpp"),
                 "-o", str(binary)], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("PASS:", run.stdout)


if __name__ == "__main__":
    unittest.main()
