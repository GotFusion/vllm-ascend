/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

namespace kv_gather_aiv {
namespace vr = AscendC::Reg;
using Vec64 = vr::RegTensor<uint64_t>;

__simd_callee__ inline void VecStoreField(
    __ubuf__ uint64_t* dst, Vec64& bases, uint64_t field, Vec64& values, vr::MaskReg& mask)
{
    Vec64 indices;
    vr::Adds(indices, bases, field, mask);
    vr::Scatter<uint64_t, uint64_t>(dst, values, indices, mask);
}

__simd_callee__ inline void VecStoreConstant(
    __ubuf__ uint64_t* dst, Vec64& bases, uint64_t field, uint64_t value, vr::MaskReg& mask)
{
    Vec64 values;
    vr::Duplicate(values, value, mask);
    VecStoreField(dst, bases, field, values, mask);
}

template <bool cqePerWqe = false>
__aicore__ inline void FillVectorWqes(
    __ubuf__ uint64_t* dst, __ubuf__ uint64_t* idxBase, uint32_t count, uint32_t sges, uint32_t head,
    uint64_t localBase, uint64_t remoteBase, uint32_t bytes, bool signalLast,
    const AscendC::UbcBatchHandle& batch)
{
    constexpr auto cfg = AscendC::URMA_DEFAULT_CFG;
    const uint64_t bb = (48U + 16U * static_cast<uint64_t>(sges) + 63U) / 64U;
    const uint64_t words = bb * 8U;
    const uint64_t depth = batch.sqContext.contextInfo.ubJfs.sqDepth;
    constexpr uint64_t flags = cfg.odr | (static_cast<uint64_t>(cfg.fence) << 3U) |
                               (static_cast<uint64_t>(cfg.se) << 4U);
    constexpr uint64_t opcode = static_cast<uint64_t>(AscendC::HcommUrmaOpCode::WRITE);
    constexpr uint64_t header0 = (flags << 16U) | (1ULL << 28U) | (1ULL << 29U) | (opcode << 40U) |
                            (cqePerWqe ? (1ULL << 21U) : 0ULL);
    const uint64_t header1 = (batch.remoteInfo.tpId & 0xFFFFFFU) |
                             (static_cast<uint64_t>(sges) << 24U) |
                             (static_cast<uint64_t>(batch.remoteInfo.tokenId & 0xFFFFFU) << 32U);
    const uint64_t eidLow = batch.remoteInfo.remoteEidLow;
    const uint64_t eidHigh = batch.remoteInfo.remoteEidHigh;
    const uint64_t token = batch.remoteInfo.tokenValue;
    const uint64_t completionLane = signalLast && !cqePerWqe ? static_cast<uint64_t>(count - 1U) : 32ULL;
    const uint16_t sgeCount = static_cast<uint16_t>(sges);
    const uint16_t wordCount = static_cast<uint16_t>(words);
    const uint16_t firstPaddingWord = static_cast<uint16_t>(6U + 2U * sges);

    __VEC_SCOPE__
    {
        vr::RegTensor<int64_t> signedLanes;
        vr::Arange(signedLanes, int64_t{0});
        auto& lanes = reinterpret_cast<Vec64&>(signedLanes);
        uint32_t active = count;
        vr::MaskReg mask = vr::UpdateMask<uint64_t>(active);
        Vec64 bases, heads, phase, depthReg, zero, owner0, owner1, values;
        vr::Muls(bases, lanes, words, mask);
        vr::Muls(heads, lanes, bb, mask);
        vr::Adds(heads, heads, static_cast<uint64_t>(head), mask);
        vr::Duplicate(depthReg, depth, mask);
        vr::And(phase, heads, depthReg, mask);
        vr::Duplicate(zero, uint64_t{0}, mask);
        vr::MaskReg ownerMask;
        vr::Compare<uint64_t, AscendC::CMPMODE::EQ>(ownerMask, phase, zero, mask);
        vr::Duplicate(owner0, header0, mask);
        vr::Duplicate(owner1, header0 | (1ULL << 31U), mask);
        vr::Select(values, owner1, owner0, ownerMask);
        Vec64 lastLane, signaled;
        vr::Duplicate(lastLane, completionLane, mask);
        vr::MaskReg lastMask;
        vr::Compare<uint64_t, AscendC::CMPMODE::EQ>(lastMask, lanes, lastLane, mask);
        vr::Adds(signaled, values, uint64_t{1} << 21U, mask);
        vr::Select(values, signaled, values, lastMask);
        VecStoreField(dst, bases, 0U, values, mask);
        VecStoreConstant(dst, bases, 1U, header1, mask);
        VecStoreConstant(dst, bases, 2U, eidLow, mask);
        VecStoreConstant(dst, bases, 3U, eidHigh, mask);
        VecStoreConstant(dst, bases, 4U, token, mask);

        vr::Muls(values, lanes, static_cast<uint64_t>(sges) * bytes, mask);
        vr::Adds(values, values, remoteBase, mask);
        VecStoreField(dst, bases, 5U, values, mask);

        for (uint16_t sge = 0U; sge < sgeCount; ++sge) {
            Vec64 idxReg, blockReg, addrReg;
            vr::Muls(idxReg, lanes, static_cast<uint64_t>(sges), mask);
            vr::Adds(idxReg, idxReg, static_cast<uint64_t>(sge), mask);
            vr::Gather<uint64_t, uint64_t>(blockReg, idxBase, idxReg, mask);
            vr::Muls(addrReg, blockReg, static_cast<uint64_t>(bytes), mask);
            vr::Adds(addrReg, addrReg, localBase, mask);
            VecStoreConstant(dst, bases, 6U + 2U * sge, static_cast<uint64_t>(bytes), mask);
            VecStoreField(dst, bases, 7U + 2U * sge, addrReg, mask);
        }
        for (uint16_t field = firstPaddingWord; field < wordCount; ++field) {
            VecStoreConstant(dst, bases, field, uint64_t{0}, mask);
        }
    }
}

// Choose the policy outside the SIMD scope. The default specialization keeps
// the verified constant header; a runtime boolean is not captured by the scope.
__aicore__ inline void DispatchVectorWqes(
    __ubuf__ uint64_t* dst, __ubuf__ uint64_t* idx, uint32_t count, uint32_t sges, uint32_t head,
    uint64_t local, uint64_t remote, uint32_t bytes, bool signalLast,
    const AscendC::UbcBatchHandle& batch, bool every)
{
    if (every) FillVectorWqes<true>(dst, idx, count, sges, head, local, remote, bytes, signalLast, batch);
    else FillVectorWqes<false>(dst, idx, count, sges, head, local, remote, bytes, signalLast, batch);
}

} // namespace kv_gather_aiv
