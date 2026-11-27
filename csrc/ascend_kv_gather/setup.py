# SPDX-License-Identifier: Apache-2.0
"""Build the standalone CANN library without compiling against Torch headers."""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class AscBuild(build_ext):
    def get_ext_filename(self, ext_name):
        # This library exports a C ABI and is loaded with ctypes, not PyInit_*.
        return str(Path(*ext_name.split(".")).with_suffix(".so"))

    def build_extension(self, ext):
        cann = os.environ.get("ASCEND_HOME_PATH")
        if not cann:
            raise RuntimeError("Source the target CANN set_env.sh before building KV gather")
        source = Path(__file__).resolve().parent
        target = Path(self.get_ext_fullpath(ext.name)).resolve()
        build = Path(self.build_temp).resolve() / "ascend_kv_gather"
        build.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DCMAKE_PREFIX_PATH={cann}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DOUTPUT_DIRECTORY={target.parent}",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )
        subprocess.check_call(["cmake", "--build", str(build), "--parallel", os.environ.get("MAX_JOBS", "4")])
        if not target.is_file():
            raise RuntimeError(f"CMake did not produce the expected shared library: {target}")


setup(
    name="ascend-kv-gather",
    version="0.1.0",
    license_files=["vendor/ascend-kv-gather/LICENSE"],
    packages=find_packages(),
    python_requires=">=3.10",
    ext_modules=[Extension("ascend_kv_gather.libascend_kv_gather", sources=[])],
    cmdclass={"build_ext": AscBuild},
)
