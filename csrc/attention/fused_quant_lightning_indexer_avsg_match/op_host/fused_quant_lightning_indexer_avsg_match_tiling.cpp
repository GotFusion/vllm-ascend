// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstdint>

#include "fused_quant_lightning_indexer_avsg_match_tiling.h"
#include "log/log.h"
#include "operator_contract.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
using namespace avsg_match;

constexpr int64_t kSmallCache = 4096;
constexpr int64_t kLargeCache = 8192;
constexpr int64_t kInvalidToken = 0x00ffffff;  // Reserved by the AVSG packed status word.
constexpr uint32_t kCausalMask = 3;
constexpr uint32_t kQueriesPerCubeTile = 4;
constexpr uint32_t kVectorsPerCube = 2;
constexpr uint64_t kUbAlignment = 32;

uint64_t Align32(uint64_t count)
{
    return (count + kUbAlignment - 1) / kUbAlignment * kUbAlignment;
}

// Match holds the selected IDs, counts, packed status and page tables in UB
// alongside nested probe, eviction and compaction scratch. Each table length is
// padded to 32 *entries* by the AVSG vector kernel.
uint64_t MatchUbBytes(uint64_t fullPages, uint64_t capacity, uint64_t requestsPerCore, uint64_t queries)
{
    constexpr uint64_t kScratchBytesPerTopKItem = 64;
    return kUbAlignment + kTopK * sizeof(int32_t) + Align32(requestsPerCore * sizeof(int32_t)) +
           Align32(queries * requestsPerCore * sizeof(int32_t)) + capacity * 2 * sizeof(int32_t) +
           (Align32(fullPages) + Align32(capacity / kPage)) * sizeof(int32_t) + kTopK * kScratchBytesPerTopKItem;
}

ge::graphStatus Tile(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    constexpr std::array<size_t, NumInputs> ranks{3, 4, 2, 2, 3, 1, 1, 2, 2, 2};
    constexpr std::array<ge::DataType, NumInputs> types{
        ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
    for (size_t i = 0; i < NumInputs; ++i) {
        const auto* input = context->GetInputShape(i);
        const auto* desc = context->GetInputDesc(i);
        OP_CHECK_IF(input == nullptr || desc == nullptr || input->GetStorageShape().GetDimNum() != ranks[i] ||
                        desc->GetDataType() != types[i],
                    OP_LOGE(context->GetNodeName(), "Input %zu has an invalid rank or dtype.", i),
                    return ge::GRAPH_FAILED);
    }
    const auto shape = [context](Input input) -> const gert::Shape& {
        return context->GetInputShape(input)->GetStorageShape();
    };
    const int64_t rows = shape(Query).GetDim(0);
    const int64_t batch = shape(QueryEnds).GetDim(0);
    const int64_t pages = shape(Key).GetDim(0);
    const int64_t fullPages = shape(FullTable).GetDim(1);
    const int64_t hotPages = shape(HotTable).GetDim(1);
    OP_CHECK_IF(batch <= 0 || rows <= 0 || rows > INT32_MAX || rows % batch != 0 ||
                    (rows / batch != 1 && rows / batch != 4),
                OP_LOGE(context->GetNodeName(), "Expected one or four query rows per request."), return ge::GRAPH_FAILED);
    const int64_t queries = rows / batch;
    OP_CHECK_IF((hotPages != kSmallCache / kPage && hotPages != kLargeCache / kPage) ||
                    (queries == 4 && hotPages != kLargeCache / kPage),
                OP_LOGE(context->GetNodeName(), "Expected a 4K or 8K cache; four queries require 8K slots."),
                return ge::GRAPH_FAILED);
    const int64_t capacity = hotPages * kPage;
    const int64_t minPages = (kTopK + queries - 1 + kPage - 1) / kPage;
    OP_CHECK_IF(pages <= 0 || pages > (static_cast<int64_t>(INT32_MAX) + 1) / kPage ||
                    fullPages < minPages || fullPages > kInvalidToken / kPage || rows * fullPages > UINT32_MAX,
                OP_LOGE(context->GetNodeName(), "Invalid page counts for the causal top-k and AVSG address range."),
                return ge::GRAPH_FAILED);

    const gert::Shape rowHeads({rows, kHeads});
    OP_CHECK_IF(shape(Query) != gert::Shape({rows, kHeads, kDim}) || shape(Key) != gert::Shape({pages, kPage, 1, kDim}) ||
                    shape(Weights) != rowHeads || shape(QueryScale) != rowHeads ||
                    shape(KeyScale) != gert::Shape({pages, kPage, 1}) || shape(KeyLengths) != shape(QueryEnds) ||
                    shape(FullTable).GetDim(0) != batch || shape(HotTable).GetDim(0) != batch ||
                    shape(Status) != gert::Shape({batch, 2 * capacity}),
                OP_LOGE(context->GetNodeName(), "Inconsistent FP8 indexer, page table or AVSG status shapes."),
                return ge::GRAPH_FAILED);
    const auto expected = OutputShapes(rows, batch, shape(Status));
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto* output = context->GetOutputShape(i);
        const auto* desc = context->GetOutputDesc(i);
        OP_CHECK_IF(output == nullptr || desc == nullptr || output->GetStorageShape() != expected[i] ||
                        desc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(context->GetNodeName(), "Output %zu has an invalid shape or dtype.", i),
                    return ge::GRAPH_FAILED);
    }

    auto* info = context->GetPlatformInfo();
    OP_CHECK_IF(info == nullptr, OP_LOGE(context->GetNodeName(), "Missing platform information."),
                return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(info);
    const uint32_t aic = platform.GetCoreNumAic();
    const uint32_t aiv = platform.GetCoreNumAiv();
    OP_CHECK_IF(aic == 0 || aiv == 0, OP_LOGE(context->GetNodeName(), "Missing mixed AIC/AIV cores."),
                return ge::GRAPH_FAILED);
    const uint32_t blocks = platform.CalcTschBlockDim(aiv, aic, aiv);
    OP_CHECK_IF(blocks == 0 || blocks > aic || blocks > aiv / kVectorsPerCube,
                OP_LOGE(context->GetNodeName(), "Invalid AIC:2AIV launch geometry."), return ge::GRAPH_FAILED);
    const uint32_t vectorCores = blocks * kVectorsPerCube;
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint64_t perCore = (batch + vectorCores - 1) / vectorCores;
    OP_CHECK_IF(ub > UINT32_MAX || MatchUbBytes(fullPages, capacity, perCore, queries) > ub,
                OP_LOGE(context->GetNodeName(), "Page tables and AVSG scratch exceed per-core UB."),
                return ge::GRAPH_FAILED);

    const uint32_t maxLength = static_cast<uint32_t>(fullPages * kPage);
    FusedQuantLightningIndexerAvsgMatchTilingData tile;
    tile.set_bSize(batch);
    tile.set_tSize(rows);
    tile.set_n2Size(1);
    tile.set_gSize(kHeads);
    tile.set_s1Size(queries);
    tile.set_s2Size(maxLength);
    tile.set_sparseCount(kTopK);
    tile.set_keyStride0(kPage * kDim);
    tile.set_keyDequantScaleStride0(kPage);
    tile.set_usedCoreNum(blocks);
    tile.set_blockSize(kPage);
    tile.set_maxBlockNumPerBatch(fullPages);
    tile.set_sparseMode(kCausalMask);
    tile.set_capacity(capacity);
    tile.set_ubBytes(static_cast<uint32_t>(ub));
    tile.set_vectorCores(vectorCores);

    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(raw == nullptr || raw->GetCapacity() < tile.GetDataSize() || workspace == nullptr,
                OP_LOGE(context->GetNodeName(), "Missing tiling or workspace buffer."), return ge::GRAPH_FAILED);
    // One FP16 score tile per cube core. The page-aligned context already
    // includes the padding required by QLI's 128-column score tile.
    workspace[0] = platform.GetLibApiWorkSpaceSize() +
                   static_cast<size_t>(blocks) * kQueriesPerCubeTile * maxLength * sizeof(uint16_t);
    tile.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(tile.GetDataSize());
    context->SetBlockDim(blocks);
    context->SetTilingKey(0);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_OPTILING(FusedQuantLightningIndexerAvsgMatch).Tiling(Tile);
}  // namespace optiling
