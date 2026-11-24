# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.20)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "HIXL SIMT architecture")
set(CMAKE_ASC_STANDARD 17)
find_package(ASC REQUIRED)
project(avsg_urma LANGUAGES CXX ASC)
if(NOT CMAKE_ASC_ARCHITECTURES STREQUAL "dav-3510")
    message(FATAL_ERROR "This HIXL SIMT implementation requires dav-3510")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)
if(NOT EXISTS "${HIXL_SOURCE_DIR}/src/hixl/aiv/primitives/hixl_aiv_simt_urma.h" OR
   NOT EXISTS "${HIXL_SOURCE_DIR}/src/ops/hixl_kernel/aiv/simt/hixl_aiv_op_simt_urma.h")
    message(FATAL_ERROR "HIXL_SOURCE_DIR must be the patched hixl_0327 checkout")
endif()
file(READ "${HIXL_SOURCE_DIR}/include/hixl/hixl.h" HIXL_PUBLIC_HEADER)
if(NOT HIXL_PUBLIC_HEADER MATCHES "GetAivChannelPointers")
    message(FATAL_ERROR "Apply patches/hixl_channel_export.patch and rebuild HIXL first")
endif()
set(CANN "$ENV{ASCEND_HOME_PATH}")
if(NOT EXISTS "${CANN}/include/acl/acl.h")
    message(FATAL_ERROR "Source the CANN setenv script (ASCEND_HOME_PATH is required)")
endif()
find_library(HIXL_LIBRARY NAMES cann_hixl hixl HINTS "${HIXL_LIBRARY_DIR}" REQUIRED NO_DEFAULT_PATH)
find_library(TORCH_NPU_LIBRARY torch_npu HINTS "${TORCH_NPU_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
set(TIK_HINTS
    "${CANN}/compiler/tikcpp/tikcfw" "${CANN}/tools/tikcpp/tikcfw"
    "${CANN}/tikcpp/tikcfw" "${CANN}/include/tikcpp/tikcfw"
    "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw")
find_path(KERNEL_OPERATOR_DIR kernel_operator.h HINTS ${TIK_HINTS} REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Source-file separation alone is insufficient: HIXL's indirect graph/* includes
# must never resolve against torch_npu's vendored GE headers. Do not link Torch
# targets to this object library (their INTERFACE_INCLUDE_DIRECTORIES propagate).
add_library(avsg_hixl_host OBJECT csrc/hixl_bridge.cpp csrc/hixl_transfer.cpp)
target_include_directories(avsg_hixl_host PRIVATE include "${HIXL_SOURCE_DIR}/include" "${HIXL_SOURCE_DIR}/src"
    "${CANN}/include" "${CANN}/pkg_inc"
    "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/include" "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc")
target_compile_options(avsg_hixl_host PRIVATE ${TORCH_CXX_FLAGS_LIST})

# Framework headers use their own bundled ACL/GE definitions, without CANN/HIXL
# include roots. Keep the same C++ ABI flags on both sides of the PIMPL boundary.
add_library(avsg_torch_host OBJECT csrc/binding.cpp)
target_include_directories(avsg_torch_host PRIVATE include ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_DIR}/include" "${TORCH_NPU_DIR}/include/third_party/acl/inc" "${TORCH_NPU_DIR}/..")
target_link_libraries(avsg_torch_host PRIVATE Python3::Module)
target_compile_options(avsg_torch_host PRIVATE ${TORCH_CXX_FLAGS_LIST})

# ASC's documented heterogeneous-library path is SHARED. Some ASC package
# versions derive their shared flags from the C language (not enabled here), so
# also pass -shared explicitly: these objects belong in a host DSO, not in an
# aicorelinux device link. The ASC object already contains the packaged kernel.
set(HIXL_SIMT_LAUNCH "${HIXL_SOURCE_DIR}/cmake/aiv_launch/hixl_aiv_simt_launch.asc")
if(NOT EXISTS "${HIXL_SIMT_LAUNCH}")
    message(FATAL_ERROR "HIXL_SOURCE_DIR must provide the raw ASC SIMT launcher")
endif()
add_library(_native SHARED csrc/avsg_urma.asc "${HIXL_SIMT_LAUNCH}"
    $<TARGET_OBJECTS:avsg_hixl_host> $<TARGET_OBJECTS:avsg_torch_host>)
set_source_files_properties(csrc/avsg_urma.asc "${HIXL_SIMT_LAUNCH}"
    PROPERTIES LANGUAGE ASC COMPILE_DEFINITIONS USE_SIMT)
target_include_directories(_native PRIVATE include "${HIXL_SOURCE_DIR}/include" "${HIXL_SOURCE_DIR}/src"
    "${HIXL_SOURCE_DIR}/src/hixl/aiv/primitives"
    "${HIXL_SOURCE_DIR}/src/ops/hixl_kernel/aiv/simt" "${CANN}/include" "${CANN}/pkg_inc"
    "${CANN}/asc/include/adv_api" "${KERNEL_OPERATOR_DIR}" "${KERNEL_OPERATOR_DIR}/interface"
    "${KERNEL_OPERATOR_DIR}/impl" "${KERNEL_OPERATOR_DIR}/lib"
    "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/include" "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc")
target_compile_options(_native PRIVATE ${TORCH_CXX_FLAGS_LIST}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    $<$<COMPILE_LANGUAGE:ASC>:--cce-auto-infer-kernel-type=false>
    $<$<COMPILE_LANGUAGE:ASC>:-I${HIXL_SOURCE_DIR}/include>)
target_link_directories(_native PRIVATE "${CANN}/lib64" "${CANN}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_options(_native PRIVATE -shared)
target_link_libraries(_native PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${TORCH_NPU_LIBRARY}
    ${HIXL_LIBRARY} Python3::Module ascendcl acl_rt runtime dl pthread)
set_target_properties(_native PROPERTIES PREFIX "" LINKER_LANGUAGE ASC
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}" BUILD_RPATH "${HIXL_LIBRARY_DIR};${TORCH_NPU_DIR}/lib")
