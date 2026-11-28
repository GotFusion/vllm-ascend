/*
 * This file is part of the Ascend LI/AVSG operator.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See ../../LICENSE.
 */
#ifndef VLLM_ASCEND_AVSG_MATCH_EPILOGUE_H
#define VLLM_ASCEND_AVSG_MATCH_EPILOGUE_H

#include "kernel_operator.h"

namespace AvsgMatchNs {
// These fields are local scalar parameters for the recipes device implementation;
// no second CANN tiling buffer or kernel dispatch is needed.
struct AvsgMatchTilingData {
    static constexpr int64_t selTopKBlockSize = 1;
    static constexpr int64_t fullKvBlockSize = 128;
    static constexpr int32_t fullKvBlockSizeLog2 = 7;
    static constexpr int64_t selKvBlockSize = fullKvBlockSize;
    static constexpr int64_t selKvBlockSizeLog2 = fullKvBlockSizeLog2;
    static constexpr int64_t headnum = 1;
    static constexpr int64_t topk = 2048;

    int64_t stageOneUsedCoreNum;
    int64_t stageOneMainCoreBsLoopNum;
    int64_t stageOneTailCoreBsLoopNum;
    int64_t fullMaxBlockNum;
    int64_t selMaxBlockNum;
    int64_t batchsize;
    int64_t t;
    int64_t statesize;
    int64_t ubSize;
};
} // namespace AvsgMatchNs

// Adapted from cann-recipes-infer_sjy@d107a3e106866344c05c6ba58948ab1f196c6169:
// read and write the GE status input/output addresses separately.
#include "avsg_match_split_bs_reuse_vec.h"

namespace AvsgMatchNs {
// All producer AIVs must finish writing sparseIndices and synchronize before the
// caller resets the LI pipe and enters here. Each active AIV owns complete requests
// so that all query rows of a request share the same persistent state and hit count.
__aicore__ inline void RunAvsgMatch(
    AscendC::TPipe* pipe, GM_ADDR hotBlockTable, GM_ADDR status, GM_ADDR statusOut, GM_ADDR sparseIndices,
    GM_ADDR fullBlockTable, GM_ADDR keyLengths, GM_ADDR cumulativeQueryLengths, GM_ADDR hits, GM_ADDR copyMapping, GM_ADDR copyCount,
    uint32_t batch, uint32_t queryRows, uint32_t fullPagesPerRequest, uint32_t capacity, uint32_t ubBytes,
    uint32_t vectorCores)
{
    if (batch == 0 || vectorCores == 0) {
        return;
    }
    const uint32_t requestsPerCore = (batch - 1) / vectorCores + 1;
    const uint32_t activeCores = (batch - 1) / requestsPerCore + 1;
    if (AscendC::GetBlockIdx() >= activeCores) {
        return;
    }

    AvsgMatchTilingData config;
    config.stageOneUsedCoreNum = activeCores;
    config.stageOneMainCoreBsLoopNum = requestsPerCore;
    config.stageOneTailCoreBsLoopNum = batch - (activeCores - 1) * requestsPerCore;
    config.fullMaxBlockNum = fullPagesPerRequest;
    config.selMaxBlockNum = capacity / AvsgMatchTilingData::selKvBlockSize;
    config.batchsize = batch;
    config.t = queryRows;
    config.statesize = capacity;
    config.ubSize = ubBytes;

    AvsgMatchSplitBsReuseVec match(pipe, &config);
    match.Init(hotBlockTable, status, statusOut, sparseIndices, fullBlockTable, keyLengths, cumulativeQueryLengths, hits,
               copyMapping, copyCount);
    match.Process();
}
} // namespace AvsgMatchNs
#endif // VLLM_ASCEND_AVSG_MATCH_EPILOGUE_H
