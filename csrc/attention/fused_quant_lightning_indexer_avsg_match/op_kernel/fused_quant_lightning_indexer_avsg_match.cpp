/*
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See ../LICENSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// The OPC-generated fused tiling record contains the fields read by the CANN LI kernel.
using QLITilingData = FusedQuantLightningIndexerAvsgMatchTilingData;

#include "qli/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h"
#include "avsg_match/avsg_match_epilogue.h"

namespace {
using namespace AscendC;
using Indexer = QLIKernel::QuantLightningIndexerKernel<
    QLICommon::QLIType<fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t, true,
                       QLICommon::LI_LAYOUT::TND, QLICommon::LI_LAYOUT::PA_BSND,
                       bfloat16_t, float32_t, float32_t, uint16_t>>;

constexpr uint32_t EMPTY_AVSG_TOKEN = 0x00ffffffU;

// All MIX participants make the same decision before entering the LI handshakes.
// The narrow decode ABI requires a full top-k on every causal query row. Passing
// a padded LI index (-1) into the recipes hash lookup would address an invalid page.
__aicore__ inline bool ValidDecodeLengths(GM_ADDR queryEnds, GM_ADDR keyLengths, const QLITilingData& shape)
{
    GlobalTensor<int32_t> ends;
    GlobalTensor<int32_t> lengths;
    ends.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryEnds), shape.bSize);
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(keyLengths), shape.bSize);
    const uint32_t rowsPerRequest = shape.tSize / shape.bSize;
    const uint32_t minKeyLength = shape.sparseCount + rowsPerRequest - 1;
    for (uint32_t request = 0; request < shape.bSize; ++request) {
        const uint32_t keyLength = static_cast<uint32_t>(lengths.GetValue(request));
        const uint32_t queryEnd = static_cast<uint32_t>(ends.GetValue(request));
        if (queryEnd != (request + 1) * rowsPerRequest || keyLength < minKeyLength ||
            keyLength > shape.s2Size || keyLength > EMPTY_AVSG_TOKEN) {
            return false;
        }
    }
    return true;
}

__aicore__ inline void MarkInvalidCounts(GM_ADDR result, uint32_t count)
{
    GlobalTensor<int32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(result), count);
    for (uint32_t idx = 0; idx < count; ++idx) {
        output.SetValue(idx, -1);
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(output);
}
} // namespace

extern "C" __global__ __aicore__ void fused_quant_lightning_indexer_avsg_match(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale, GM_ADDR keyScale,
    GM_ADDR queryEnds, GM_ADDR keyLengths, GM_ADDR fullBlockTable, GM_ADDR hotBlockTable, GM_ADDR status,
    GM_ADDR slots, GM_ADDR hits, GM_ADDR copyMapping, GM_ADDR copyCount, GM_ADDR statusOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace AscendC;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(FusedQuantLightningIndexerAvsgMatchTilingData, shape, tiling);
    TPipe pipe;
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    if (!ValidDecodeLengths(queryEnds, keyLengths, shape)) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                MarkInvalidCounts(hits, shape.bSize);
                MarkInvalidCounts(copyCount, shape.tSize);
            }
        }
        return;
    }

    {
        Indexer indexer;
        indexer.Init(query, key, weights, queryScale, keyScale, queryEnds, keyLengths, fullBlockTable,
                     slots, GetUserWorkspace(workspace), &shape, &pipe);
        indexer.Process();
    }

    // A request's query rows can have different producer AIVs. Publish every
    // final top-k write before its batch owner rewrites those rows with KV slots.
    if ASCEND_IS_AIV {
        const auto event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(event);
        WaitFlag<HardEvent::MTE3_S>(event);
    }
    SyncAll<false>(); // All launched Cube and Vector cores, including idle LI workers.
    if ASCEND_IS_AIV {
        pipe.Reset();
        AvsgMatchNs::RunAvsgMatch(&pipe, hotBlockTable, status, statusOut, slots, fullBlockTable, keyLengths,
                                  queryEnds, hits, copyMapping, copyCount, shape.bSize, shape.tSize,
                                  shape.maxBlockNumPerBatch, shape.capacity, shape.ubBytes, shape.vectorCores);
    }
}
