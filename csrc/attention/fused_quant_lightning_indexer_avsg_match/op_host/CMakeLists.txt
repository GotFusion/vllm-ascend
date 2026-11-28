# SPDX-License-Identifier: Apache-2.0
if (NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE fused_quant_lightning_indexer_avsg_match_def.cpp)
endif()
add_ops_compile_options(
    OP_NAME FusedQuantLightningIndexerAvsgMatch
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE fused_quant_lightning_indexer_avsg_match ACLNNTYPE aclnn)
endif()
