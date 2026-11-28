// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/DeviceGuard.h>
#include "avsg_match_outputs.h"

namespace vllm_ascend::avsg {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_quant_lightning_indexer_avsg_match(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &block_table, const at::Tensor &hot_block_table, const at::Tensor &status)
{
    const c10::DeviceGuard guard(query.device());
    auto outputs = prepare_avsg_match_outputs(query, key, weights, query_dequant_scale, key_dequant_scale,
                                             actual_seq_lengths_query, actual_seq_lengths_key,
                                             block_table, hot_block_table, status);
    auto &[slots, hits, mapping, counts] = outputs;
    EXEC_NPU_CMD(aclnnFusedQuantLightningIndexerAvsgMatch, query, key, weights,
                 query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key,
                 block_table, hot_block_table, status, slots, hits, mapping, counts);
    return outputs;
}

}  // namespace vllm_ascend::avsg
