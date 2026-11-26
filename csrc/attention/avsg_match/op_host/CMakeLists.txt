# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE avsg_match_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME AvsgMatch
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)
add_modules_sources(
    OPTYPE avsg_match ACLNNTYPE aclnn_exclude
    OP_API_INDEPENDENT ON OP_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../op_api"
)
