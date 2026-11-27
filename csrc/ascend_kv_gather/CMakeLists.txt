# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.20)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "Ascend 950 target")
set(CMAKE_ASC_STANDARD 17)
find_package(ASC REQUIRED)
project(ascend_kv_gather LANGUAGES CXX ASC)
if(NOT CMAKE_ASC_ARCHITECTURES STREQUAL "dav-3510")
    message(FATAL_ERROR "The imported SDK vector WQE builder requires dav-3510")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(cann "$ENV{ASCEND_HOME_PATH}")
if(NOT EXISTS "${cann}/include/acl/acl.h")
    message(FATAL_ERROR "Source the target CANN set_env.sh before configuring")
endif()
set(vendor "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ascend-kv-gather/aiv")
if(NOT EXISTS "${vendor}/vector_builder.h" OR NOT EXISTS "${vendor}/protocol.h")
    message(FATAL_ERROR "The pinned ascend-kv-gather vendor sources are missing")
endif()
add_library(ascend_kv_gather SHARED native/session.asc native/gather.asc)
target_include_directories(ascend_kv_gather PRIVATE native "${vendor}"
    "${cann}/include" "${cann}/include/hccl" "${cann}/pkg_inc"
    "${cann}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${cann}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/hcomm"
    "${cann}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    "${cann}/asc/include/comm_api" "${cann}/asc/include/adv_api")
target_compile_options(ascend_kv_gather PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>)
target_link_directories(ascend_kv_gather PRIVATE
    "${cann}/lib64" "${cann}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
target_link_options(ascend_kv_gather PRIVATE -shared)
target_link_libraries(ascend_kv_gather PRIVATE hcomm hccl ascendcl acl_rt runtime pthread)
set_target_properties(ascend_kv_gather PROPERTIES LINKER_LANGUAGE ASC SKIP_BUILD_RPATH TRUE)
if(OUTPUT_DIRECTORY)
    set_target_properties(ascend_kv_gather PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}")
endif()
