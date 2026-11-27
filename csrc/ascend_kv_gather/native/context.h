// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace ascend_kv_gather {

constexpr std::uint32_t kMaxRows = 8192;
constexpr std::uint32_t kMaxCores = 8;
constexpr std::uint32_t kSges = 13;
constexpr std::uint32_t kBatchWqes = 64;
constexpr std::uint32_t kWindow = 16;
constexpr std::int32_t kStreamTimeoutMs = 30000;

// Queue cursors are persisted by SDK Unlock in the channel's GM state. Only
// our completion and error status lives here, separately for each bank/core.
struct alignas(128) CoreState {
    std::uint64_t completed;
    std::int32_t error;
    std::uint32_t reserved;
    std::uint8_t padding[112];
};
static_assert(sizeof(CoreState) == 128);

struct TransferContext {
    std::uint64_t channel;
    std::uint64_t source;
    std::uint64_t destination;
    std::uint64_t indices;
    std::uint64_t state;
    std::uint64_t sequence;
    std::uint32_t rows;
    std::uint32_t row_bytes;
    std::uint32_t cores;
    std::uint32_t core;
};
static_assert(sizeof(TransferContext) == 64);

}  // namespace ascend_kv_gather
