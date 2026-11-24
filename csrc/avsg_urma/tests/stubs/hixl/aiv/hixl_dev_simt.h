/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * CPU transport fake for executing the real partition/descriptor kernel code.
 */
#pragma once
#include <cstring>
#include <vector>
namespace hixl::aiv {
constexpr uint32_t HIXL_DEV_OK = 0;
struct HixlDevXferDesc { uint64_t remote, local; uint32_t len, reserved; };
struct HixlDevChannelInfo { uint32_t sqDepth, cqDepth, remoteBufCnt, maxBatchCnt, maxInlineLen, reserved; };
namespace simt {
inline uint32_t sqDepth = 8, failSubmit = 0, failDrain = 0;
inline std::vector<uint64_t> submittedChannels;
inline uint64_t submits = 0, drains = 0, largestBatch = 0;
inline uint32_t hixlDevQueryChannelInfo(uint64_t, HixlDevChannelInfo *info)
{
    *info = {sqDepth, 8, 2, 64, 8, 0};
    return 0;
}
inline uint32_t hixlDevGetList(uint64_t channel, const HixlDevXferDesc *desc, uint32_t n)
{
    ++submits;
    submittedChannels.push_back(channel);
    if (n > largestBatch) largestBatch = n;
    if (failSubmit) return failSubmit;
    for (uint32_t i = 0; i < n; ++i) {
        if (desc[i].reserved != 0) return 1;
        std::memcpy(reinterpret_cast<void *>(desc[i].local), reinterpret_cast<void *>(desc[i].remote), desc[i].len);
    }
    return 0;
}
inline uint32_t hixlDevDrain(uint64_t) { ++drains; return failDrain; }
} // namespace simt
} // namespace hixl::aiv
