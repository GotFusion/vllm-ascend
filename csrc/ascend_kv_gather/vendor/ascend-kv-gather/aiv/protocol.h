#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// ASC defines __NPU_ARCH__ while compiling Host launch code too. Only the
// transfer translation unit opts into AI Core helper annotations.
#if defined(KV_GATHER_AIV_DEVICE)
#define KV_AIV_FN __aicore__
#else
#define KV_AIV_FN
#endif

namespace kv_gather_aiv {

constexpr std::uint32_t kMaxBlocks = 20000;
constexpr std::uint32_t kMaxCores = 8;
constexpr std::uint32_t kPackedRopeOffset = 512;
constexpr std::uint32_t kPackedScaleOffset = kPackedRopeOffset + 64 * 2;
constexpr std::uint32_t kPackedBlockBytes = kPackedScaleOffset + 4 * 4;
#if defined(KV_RECV_SCATTER_656)
constexpr std::uint32_t kBlockBytes = kPackedBlockBytes;
#else
constexpr std::uint32_t kBlockBytes = 512;
#endif
constexpr std::uint32_t kMaxSges = 13;
constexpr std::uint32_t kMaxBatchWqes = 64;
constexpr std::uint32_t kBatchBufferBytes = 16384;
constexpr std::uint32_t kIndexScratchBytes = (kMaxBatchWqes * kMaxSges + 4) * 8;
constexpr std::uint32_t kServiceWaitSeconds = 300;
constexpr std::uint64_t kStopRequest = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t kDataOffset = 128;
constexpr std::uint64_t kPayloadBytes = std::uint64_t{kMaxBlocks} * kBlockBytes;
constexpr std::uint64_t kAllocationBytes = kDataOffset + kPayloadBytes + 128;
// DataCopy stages whole 32-byte groups, including the final partial group.
constexpr std::uint32_t kRequestAllocationBytes = (2 + kMaxBlocks + 4) * 8;
constexpr std::uint8_t kGuard = 0xa5;
constexpr std::uint8_t kSentinel = 0x5a;

enum class Builder : std::uint32_t { Scalar, Vector, Simt, SimtSgl };
enum class SimtIndexRefresh : std::uint32_t { Element, Line128 };

// Single-core SIMT SGL starts with the parameters already tested by the probe.
KV_AIV_FN inline constexpr bool ValidSimtSgl(std::uint32_t cores, std::uint32_t sges,
                                           std::uint32_t batchWqes, std::uint32_t window)
{
    return cores == 1 && sges >= 1 && sges <= 5 && batchWqes >= 1 && batchWqes <= 32 &&
           window >= 1 && window <= 16;
}

struct alignas(128) SimtQueueSample {
    std::uint64_t requestId;
    std::uint64_t channel;
    std::uint64_t sqAddress;
    std::uint64_t cqAddress;
    std::uint64_t headBefore;
    std::uint64_t headAfter;
    std::uint32_t sqTailBefore;
    std::uint32_t cqTailBefore;
    std::uint32_t sqTailAfter;
    std::uint32_t cqTailAfter;
    std::uint32_t sqDepth;
    std::uint32_t cqDepth;
    std::uint32_t wqeSize;
    std::int32_t status;
    std::uint8_t padding[48];
};
static_assert(sizeof(SimtQueueSample) == 128);

// AIV scalar owns this separate allocation; never share the SIMT-owned queue
// record's cache lines. These are whole-call envelopes, not VF-dispatch-only time.
struct alignas(128) SimtProfileSample {
    std::uint64_t requestId;
    std::uint64_t appendVfCycles;
    std::uint64_t appendStatusCycles;
    std::uint64_t drainVfCycles;
    std::uint64_t drainStatusCycles;
    std::uint64_t auditCycles;
    std::uint64_t constructCycles;
    std::uint64_t dataCycles;
    std::uint32_t appendCalls;
    std::uint32_t drainCalls;
    std::uint32_t auditCalls;
    std::uint8_t padding[52];
};
static_assert(sizeof(SimtProfileSample) == 128 && alignof(SimtProfileSample) == 128);

struct BatchPlan {
    std::uint32_t wqes;
    std::uint32_t indices;
    std::uint32_t uniformWqes;
    std::uint32_t tailSges;
    std::uint32_t queueBlocks;
};

struct CoreRange {
    std::uint32_t first;
    std::uint32_t count;
};

// Partition whole WQEs so only the global final WQE has a partial SGE list.
KV_AIV_FN inline constexpr CoreRange Partition(std::uint32_t blocks, std::uint32_t sges,
                                              std::uint32_t cores, std::uint32_t core)
{
    const auto wqes = (blocks + sges - 1) / sges;
    const auto extra = wqes % cores;
    const auto firstWqe = core * (wqes / cores) + (core < extra ? core : extra);
    const auto count = wqes / cores + (core < extra ? 1U : 0U);
    const auto first = firstWqe * sges < blocks ? firstWqe * sges : blocks;
    const auto end = (firstWqe + count) * sges < blocks ? (firstWqe + count) * sges : blocks;
    return {first, end - first};
}

KV_AIV_FN inline constexpr std::uint32_t WqeBlocks(std::uint32_t sges)
{
    return (48 + 16 * sges + 63) / 64;
}

// Inputs are checked by the Host and both Device control entries.
KV_AIV_FN inline constexpr BatchPlan PlanBatch(std::uint32_t remaining, std::uint32_t sges, std::uint32_t batchWqes)
{
    const auto needed = (remaining + sges - 1) / sges;
    const auto count = needed < batchWqes ? needed : batchWqes;
    const auto capacity = count * sges;
    const auto indices = remaining < capacity ? remaining : capacity;
    const auto tail = indices - (count - 1) * sges;
    const auto uniform = tail == sges ? count : count - 1;
    return {count, indices, uniform, tail,
            uniform * WqeBlocks(sges) + (uniform < count ? WqeBlocks(tail) : 0)};
}

KV_AIV_FN inline constexpr bool ValidTransfer(std::uint32_t blocks, std::uint32_t sges,
                                   std::uint32_t batchWqes, std::uint32_t window)
{
    return blocks > 0 && blocks <= kMaxBlocks && sges > 0 && sges <= kMaxSges &&
           batchWqes > 0 && batchWqes <= kMaxBatchWqes && window > 0 && window <= 25;
}

inline std::uint8_t PatternByte(std::uint32_t block, std::uint32_t offset)
{
    std::uint32_t state = block * 0x9e3779b1U + 0x12345678U;
    state += offset * 0x85ebca6bU;
    state ^= state >> 13;
    state *= 0xc2b2ae35U;
    state ^= state >> 16;
    return static_cast<std::uint8_t>(state);
}

// The packed fixture stores finite E4M3FN, BF16 and positive FP32 scales.
// Emit little-endian bytes directly so the host and AICPU use the same oracle.
inline std::uint8_t BlockValue(std::uint32_t block, std::uint32_t offset)
{
    if constexpr (kBlockBytes == 512) {
        return PatternByte(block, offset);
    }
    if (offset < kPackedRopeOffset) {
        const auto bits = PatternByte(block, offset);
        return (bits & 0x7fU) == 0x7fU ? static_cast<std::uint8_t>(bits ^ 1U) : bits;
    }
    if (offset < kPackedScaleOffset) {
        const auto lane = (offset - kPackedRopeOffset) / 2U;
        const auto bits = 0x3e80U | (PatternByte(block, kPackedRopeOffset + lane) & 0x7fU);
        return static_cast<std::uint8_t>(bits >> (8U * (offset % 2U)));
    }
    const auto group = (offset - kPackedScaleOffset) / 4U;
    const auto bits = 0x3d000000U | (std::uint32_t{PatternByte(block, kPackedScaleOffset + group)} << 15U);
    return static_cast<std::uint8_t>(bits >> (8U * (offset % 4U)));
}

struct TransferSample {
    std::uint64_t requestId;
    std::uint64_t dataCycles;
    std::uint64_t indexDcciCycles;
    std::uint64_t constructCycles;
    std::uint32_t wqes;
    std::uint32_t batches;
    std::uint32_t drains;
    std::uint32_t cqes;
};

struct alignas(128) Command {
    volatile std::uint64_t requestId;
    volatile std::uint32_t blocks;
    volatile std::uint32_t stop;
    volatile std::uint64_t scatterRelease;
    std::uint8_t padding[104];
};

struct alignas(128) TransferStatus {
    volatile std::uint64_t done;
    volatile std::uint64_t ready;
    volatile std::int32_t error;
    std::uint32_t reserved;
    TransferSample sample;
    volatile std::uint64_t sqAddress;
    volatile std::uint64_t cqAddress;
    volatile std::uint32_t sqDepth;
    volatile std::uint32_t cqDepth;
    volatile std::uint32_t phase;
    volatile std::uint32_t nextIndex;
    volatile std::uint32_t completedWqes;
    volatile std::uint32_t completedBatches;
    volatile std::uint32_t completedDrains;
    std::uint32_t reservedProgress;
    volatile std::uint64_t activeRequest;
};

struct alignas(128) Control {
    volatile std::uint32_t ready;
    volatile std::uint32_t stopped;
    volatile std::uint64_t seq;
    volatile std::uint64_t resultReady;
    volatile std::uint64_t consumed;
    volatile std::uint64_t slotFree;
    volatile std::uint32_t verifyOk;
    volatile std::int32_t error;
    volatile std::uint64_t accepted;
    volatile std::uint64_t sendNs;
    volatile std::uint64_t readyNs;
    volatile std::uint64_t verifyNs;
    volatile std::uint64_t freeNs;
    volatile std::uint64_t verifyDone;
    volatile std::uint32_t phase;
    volatile std::uint32_t firstBadBlock;
    volatile std::uint32_t firstBadByte;
    volatile std::uint32_t firstBadKv;
    volatile std::uint32_t firstBadActual;
    volatile std::uint32_t firstBadExpected;
    volatile std::uint64_t scatterDone;
};

struct alignas(128) Mailbox {
    Command command;
    TransferStatus transfer;
    Control control;
};

static_assert(sizeof(TransferSample) == 48);
static_assert(sizeof(Command) == 128 && sizeof(TransferStatus) == 128 && sizeof(Control) == 128);
static_assert(offsetof(Mailbox, transfer) == 128 && offsetof(Mailbox, control) == 256);
static_assert(sizeof(Mailbox) == 384);

struct TransferContext {
    std::uint64_t channel;
    std::uint64_t localBuffer;
    std::uint64_t remoteBuffer;
    std::uint64_t requestBuffer;
    std::uint64_t mailbox;
    Builder builder;
    std::uint32_t sges;
    std::uint32_t batchWqes;
    std::uint32_t window;
    std::uint32_t cores;
    std::uint32_t core;
    std::uint32_t cqePerWqe;
    std::uint64_t queueSamples;
    std::uint32_t sampleCapacity;
    std::uint32_t debugProgress;
    std::uint64_t profileSamples;
    std::uint32_t profileSimt;
    SimtIndexRefresh simtIndexRefresh;
};
static_assert(sizeof(TransferContext) == 104 && offsetof(TransferContext, debugProgress) == 84);
static_assert(offsetof(TransferContext, profileSamples) == 88 && offsetof(TransferContext, profileSimt) == 96);
static_assert(offsetof(TransferContext, simtIndexRefresh) == 100);

} // namespace kv_gather_aiv

#undef KV_AIV_FN
