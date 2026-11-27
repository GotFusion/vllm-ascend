// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "context.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ascend_kv_gather {
struct Region {
    std::uint64_t base;
    std::uint64_t bytes;
};

inline std::uint32_t ValidateTransfer(const Region* regions, std::size_t region_count,
                                      std::uint64_t source, std::uint64_t destination,
                                      std::uint32_t row_bytes, const std::uint64_t* indices,
                                      std::uint32_t count)
{
    if (count == 0 || count > kMaxRows || indices == nullptr) {
        throw std::invalid_argument("gather requires a nonnull index list with 1-8192 rows");
    }
    if (row_bytes != 128 && row_bytes != 1024) {
        throw std::invalid_argument("gather expects BF16 RoPE or BF16 main KV rows");
    }
    const auto max = std::numeric_limits<std::uint64_t>::max();
    const auto output_bytes = std::uint64_t{count} * row_bytes;
    if (destination == 0 || destination > max - output_bytes) {
        throw std::invalid_argument("gather destination address overflow");
    }
    for (std::uint32_t bank = 0; bank < region_count; ++bank) {
        const auto& region = regions[bank];
        if (source < region.base || source - region.base >= region.bytes) continue;
        const auto offset = source - region.base;
        const auto expected_bytes = bank == 0 ? 1024U : 128U;
        if (row_bytes != expected_bytes || offset % row_bytes != 0) {
            throw std::invalid_argument("gather source does not match its KV/RoPE row layout");
        }
        const auto remaining = region.bytes - offset;
        if (remaining < row_bytes) throw std::invalid_argument("gather source has no complete row");
        for (std::uint32_t index = 0; index < count; ++index) {
            // Division precedes multiplication, so an adversarial uint64 index
            // cannot wrap around into a valid-looking registered address.
            if (indices[index] > (remaining - row_bytes) / row_bytes) {
                throw std::invalid_argument("gather source row exceeds its registered bank");
            }
        }
        return bank;
    }
    throw std::invalid_argument("gather source is outside registered banks");
}
}  // namespace ascend_kv_gather
