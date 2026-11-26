# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Build the independent eager-only ASC/Torch extension on an Ascend Linux host."""
import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class AscBuild(build_ext):
    def build_extension(self, ext):
        import torch
        import torch_npu
        required = ("HIXL_SOURCE_DIR", "HIXL_LIBRARY_DIR", "ASCEND_HOME_PATH")
        missing = [key for key in required if not os.environ.get(key)]
        if missing:
            raise RuntimeError(f"Missing build configuration: {', '.join(missing)}")
        source = Path(__file__).resolve().parent
        target = Path(self.get_ext_fullpath(ext.name)).resolve()
        build = Path(self.build_temp).resolve() / "avsg_urma"
        build.mkdir(parents=True, exist_ok=True)
        args = ["cmake", "-S", str(source), "-B", str(build),
                f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path};{os.environ['ASCEND_HOME_PATH']}",
                f"-DPython3_EXECUTABLE={sys.executable}",
                f"-DTORCH_NPU_DIR={Path(torch_npu.__file__).parent}",
                f"-DOUTPUT_DIRECTORY={target.parent}", "-DCMAKE_BUILD_TYPE=Release"]
        args.extend(f"-D{key}={os.environ[key]}" for key in required[:2])
        subprocess.check_call(args)
        subprocess.check_call(["cmake", "--build", str(build), "--parallel", os.environ.get("MAX_JOBS", "4")])
        # CMake produces a plain _native.so, which CPython can import; setuptools tracks the ABI suffix.
        built = target.parent / "_native.so"
        if built != target:
            built.rename(target)


setup(name="avsg-urma", version="0.1.0", packages=find_packages(), python_requires=">=3.10",
      ext_modules=[Extension("avsg_urma._native", sources=[])], cmdclass={"build_ext": AscBuild})
