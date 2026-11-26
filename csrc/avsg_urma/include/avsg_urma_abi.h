/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef AVSG_URMA_ABI_H
#define AVSG_URMA_ABI_H
#include <cstddef>
#include <cstdint>

namespace avsg_urma {
constexpr uint64_t kMagic = 0x4156534755524d41ULL;
constexpr uint64_t kVersion = 2;
constexpr uint64_t kMaxChannels = 64;
constexpr uint64_t kMaxGraphChannels = 32;
constexpr uint64_t kMaxBatch = 64;
constexpr uint64_t kStatusBytes = 128;
constexpr uint64_t kInfoBytes = 128;
// All strides are bytes. channels[] immediately follows this fixed header.
struct alignas(8) Context {
    uint64_t magic, version, bytes, generation;
    uint64_t channels, batch, elementBytes, blockSize;
    uint64_t kvDim, ropeDim, hasRope, dtype;
    uint64_t remoteKv, remoteKvBytes, kvStride0, kvStride1;
    uint64_t remoteRope, remoteRopeBytes, ropeStride0, ropeStride1;
    uint64_t localKv, localKvBytes, localRope, localRopeBytes;
};
constexpr uint64_t kHeaderWords = 24;
static_assert(sizeof(Context) == kHeaderWords * sizeof(uint64_t), "Context ABI changed");
static_assert(offsetof(Context, localKv) == 20 * sizeof(uint64_t), "Context layout changed");
// Per-core generation status: [code, phase, generated_tokens, reserved...], 128B.
// Transport success is checked separately; generation is not a CQ counter.
enum Phase : uint64_t { kValidate = 1, kSubmit = 2, kDrain = 3, kComplete = 4 };
constexpr uint64_t kDescriptorsReady = 5;
// The HIXL kernel writes a byte only on failure. Reserve word 3 of status row 0;
// descriptor generation writes words 0..2 and clears word 3 before transport.
constexpr uint64_t kTransportErrorOffset = 3 * sizeof(uint64_t);
constexpr uint64_t kBadContext = 100;
constexpr uint64_t kBadMapping = 101;
constexpr uint64_t kBadQueue = 102;
// Graph validation detail in status word 4; word 3 is reserved for transport.
enum GraphContextDetail : uint32_t {
    kContextValid = 0, kContextHeader = 1, kContextLayout = 2, kContextKvAddress = 3,
    kContextKvBytes = 4, kContextRope = 5, kContextLocalSlots = 6, kContextRemoteRegion = 7,
};
// Plain POD copied into the raw ASC launcher. No pointers to host stack storage.
struct LaunchArgs {
    uint64_t context, mapping, counts, scratch, status;
    uint64_t groups, entries, total, channels;
};
struct GraphLaunchArgs {
    LaunchArgs copy;
    uint64_t q, qLength, groupsCapacity, heads;
    uint64_t seen, selectionSlots, localKv, localKvBytes, localRope, localRopeBytes;
    uint64_t remoteSlots = 0;
    uint64_t writeStatus = 0;
    uint64_t seenStrideWords = 0;
    bool write = false;
};
} // namespace avsg_urma
#endif
