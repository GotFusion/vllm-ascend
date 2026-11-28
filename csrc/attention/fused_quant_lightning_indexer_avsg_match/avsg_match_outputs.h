// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <tuple>

namespace vllm_ascend::avsg {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> prepare_avsg_match_outputs(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &block_table, const at::Tensor &hot_block_table, const at::Tensor &status)
{
    constexpr int64_t heads = 32;
    constexpr int64_t head_dim = 128;
    constexpr int64_t page_size = 128;
    constexpr int64_t topk = 2048;
    constexpr int64_t status_words = 2;
    constexpr int64_t hot_pages_4k = 4096 / page_size;
    constexpr int64_t hot_pages_8k = 8192 / page_size;
    constexpr int64_t mtp3_queries = 4;
    // Recipes reserves logical token 0xffffff as the empty-slot sentinel.
    constexpr int64_t max_full_pages = 0xffffff / page_size;

    TORCH_CHECK(query.scalar_type() == at::kFloat8_e4m3fn && key.scalar_type() == at::kFloat8_e4m3fn,
                "query and key must have FP8 E4M3FN dtype");
    TORCH_CHECK(query.dim() == 3 && query.size(1) == heads && query.size(2) == head_dim,
                "query must have shape [T,32,128]");
    TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(1) == page_size &&
                key.size(2) == 1 && key.size(3) == head_dim, "key must have shape [pages,128,1,128]");
    TORCH_CHECK(weights.scalar_type() == at::kBFloat16 && weights.sizes() == query.sizes().slice(0, 2),
                "weights must be BF16 [T,32]");
    TORCH_CHECK(query_dequant_scale.scalar_type() == at::kFloat && query_dequant_scale.sizes() == weights.sizes(),
                "query_dequant_scale must be FP32 [T,32]");
    TORCH_CHECK(key_dequant_scale.scalar_type() == at::kFloat && key_dequant_scale.sizes() == key.sizes().slice(0, 3),
                "key_dequant_scale must be FP32 [pages,128,1]");

    for (const auto *tensor : {&actual_seq_lengths_query, &actual_seq_lengths_key, &block_table,
                               &hot_block_table, &status}) {
        TORCH_CHECK(tensor->scalar_type() == at::kInt, "lengths, page tables and status must be int32");
    }
    TORCH_CHECK(actual_seq_lengths_key.dim() == 1 && actual_seq_lengths_key.numel() > 0 &&
                actual_seq_lengths_query.sizes() == actual_seq_lengths_key.sizes(), "lengths must have shape [B]");
    const int64_t batch = actual_seq_lengths_key.size(0);
    const int64_t rows = query.size(0);
    TORCH_CHECK(hot_block_table.dim() == 2 && hot_block_table.size(0) == batch,
                "hot_block_table must have shape [B,hot_pages]");
    const int64_t hot_pages = hot_block_table.size(1);
    const bool single_query = rows == batch && (hot_pages == hot_pages_4k || hot_pages == hot_pages_8k);
    const bool multi_query = rows / batch == mtp3_queries && rows % batch == 0 && hot_pages == hot_pages_8k;
    TORCH_CHECK(single_query || multi_query, "supported shapes are Q=1 with 4K/8K slots, or Q=4 with 8K slots");
    const int64_t min_full_pages = (topk + (rows / batch - 1) + page_size - 1) / page_size;
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == batch &&
                block_table.size(1) >= min_full_pages && block_table.size(1) <= max_full_pages,
                "full block table width must cover top-k and fit AVSG token IDs");
    TORCH_CHECK(status.dim() == 2 && status.size(0) == batch && status.size(1) == status_words * hot_pages * page_size,
                "status must have shape [B,2*capacity]");
    for (const auto *tensor : {&query, &key, &weights, &query_dequant_scale, &key_dequant_scale,
                               &actual_seq_lengths_query, &actual_seq_lengths_key, &block_table, &hot_block_table,
                               &status}) {
        TORCH_CHECK(tensor->device() == query.device() && tensor->is_contiguous(),
                    "all inputs must be contiguous and on the same device");
        if (tensor != &status && !query.is_meta()) {
            at::assert_no_overlap(status, *tensor);
        }
    }

    auto options = query.options().dtype(at::kInt);
    auto slots = at::empty({rows, 1, topk}, options);
    auto hits = at::empty({batch}, options);
    auto mapping = at::empty({rows, 1, topk, 2}, options);
    auto counts = at::empty({rows, 1}, options);
    return {slots, hits, mapping, counts};
}

}  // namespace vllm_ascend::avsg
