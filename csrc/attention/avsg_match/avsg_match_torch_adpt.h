// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

namespace vllm_ascend {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> avsg_match_outputs(
    const at::Tensor& selection_table, const at::Tensor& selection_status,
    const at::Tensor& topk_indices, const at::Tensor& full_table,
    const at::Tensor& kv_lengths, const at::Tensor& query_ends,
    int64_t selection_block_size, int64_t kv_block_size)
{
    for (const auto* tensor : {&selection_table, &selection_status, &topk_indices,
                               &full_table, &kv_lengths, &query_ends}) {
        TORCH_CHECK(tensor->device() == topk_indices.device() &&
                    tensor->scalar_type() == at::kInt && tensor->is_contiguous(),
                    "AVSG match requires contiguous int32 tensors on one device");
    }
    TORCH_CHECK(topk_indices.dim() == 3 || topk_indices.dim() == 4,
                "AVSG match requires TND or BSND top-k indices");
    TORCH_CHECK(kv_lengths.dim() == 1 && query_ends.sizes() == kv_lengths.sizes(),
                "AVSG match requires a KV length and cumulative query end for each request");
    const auto requests = kv_lengths.size(0);
    TORCH_CHECK(full_table.dim() == 2 && full_table.size(0) == requests &&
                selection_table.dim() == 2 && selection_table.size(0) >= requests &&
                selection_status.dim() == 2 && selection_status.size(0) >= requests,
                "AVSG match requires block tables and selection status for each request");
    TORCH_CHECK(kv_block_size > 0 && selection_block_size > 0, "AVSG block sizes must be positive");
    const auto status_slots = selection_status.size(1) / 2;
    TORCH_CHECK(selection_status.size(1) % 2 == 0 && status_slots % kv_block_size == 0 &&
                status_slots / kv_block_size == selection_table.size(1),
                "AVSG match requires two int32 status words per selection slot");

    // Each mapping row has an aligned number of (source, destination) pairs.
    constexpr int64_t mapping_alignment = 4;
    constexpr int64_t mapping_fields = 2;
    const auto topk = topk_indices.size(-1);
    TORCH_CHECK(topk > 0, "AVSG match requires a nonempty top-k dimension");
    auto counts_shape = topk_indices.sizes().vec();
    counts_shape.pop_back();
    auto mapping_shape = counts_shape;
    mapping_shape.push_back((topk + mapping_alignment - 1) / mapping_alignment * mapping_alignment);
    mapping_shape.push_back(mapping_fields);
    return {at::empty_like(kv_lengths), at::empty(mapping_shape, topk_indices.options()),
            at::empty(counts_shape, topk_indices.options())};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> avsg_match(
    const at::Tensor& selection_table, const at::Tensor& selection_status,
    const at::Tensor& topk_indices, const at::Tensor& full_table,
    const at::Tensor& kv_lengths, const at::Tensor& query_ends,
    int64_t selection_block_size, int64_t kv_block_size)
{
    const c10_npu::OptionalNPUGuard device_guard(topk_indices.device());
    auto outputs = avsg_match_outputs(selection_table, selection_status, topk_indices,
                                      full_table, kv_lengths, query_ends, selection_block_size, kv_block_size);
    EXEC_NPU_CMD(aclnnAvsgMatch, selection_table, selection_status, topk_indices,
                 full_table, kv_lengths, query_ends, selection_block_size, kv_block_size,
                 std::get<0>(outputs), std::get<1>(outputs), std::get<2>(outputs));
    return outputs;
}

} // namespace vllm_ascend
