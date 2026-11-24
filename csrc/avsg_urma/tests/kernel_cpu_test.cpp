/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#define AVSG_URMA_CPU_TEST
#include "../csrc/avsg_urma.asc"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

using namespace avsg_urma;
namespace fake = hixl::aiv::simt;

template <typename T> void Run(bool rope, const std::vector<int32_t> &counts)
{
    constexpr uint64_t cores = 3, entries = 61, groups = 6;
    assert(counts.size() == groups);
    std::vector<uint64_t> storage(kHeaderWords + cores, 0);
    auto &ctx = *reinterpret_cast<Context *>(storage.data());
    ctx.magic = kMagic; ctx.version = kVersion; ctx.bytes = storage.size() * 8;
    ctx.channels = cores; ctx.batch = 4; ctx.elementBytes = sizeof(T); ctx.blockSize = 4;
    ctx.kvDim = 8; ctx.ropeDim = rope ? 4 : 0; ctx.hasRope = rope;
    ctx.kvStride0 = 128; ctx.kvStride1 = 24;
    ctx.ropeStride0 = 80; ctx.ropeStride1 = 16;
    std::vector<T> src(8 * 128 / sizeof(T)), srcRope(8 * 80 / sizeof(T));
    std::iota(src.begin(), src.end(), 1);
    std::iota(srcRope.begin(), srcRope.end(), 1000);
    const T sentinel = static_cast<T>(-1);
    std::vector<T> dst(512 * 8, sentinel), dstRope(512 * 4, sentinel);
    ctx.remoteKv = reinterpret_cast<uint64_t>(src.data()); ctx.remoteKvBytes = src.size() * sizeof(T);
    ctx.remoteRope = reinterpret_cast<uint64_t>(srcRope.data()); ctx.remoteRopeBytes = srcRope.size() * sizeof(T);
    ctx.localKv = reinterpret_cast<uint64_t>(dst.data()); ctx.localKvBytes = dst.size() * sizeof(T);
    ctx.localRope = reinterpret_cast<uint64_t>(dstRope.data()); ctx.localRopeBytes = dstRope.size() * sizeof(T);
    storage[kHeaderWords] = 111; storage[kHeaderWords + 1] = 222; storage[kHeaderWords + 2] = 333;
    std::vector<int32_t> mapping(groups * entries * 2, -1);
    uint64_t token = 0;
    std::vector<T> expected = dst, expectedRope = dstRope;
    for (uint64_t g = 0; g < groups; ++g) {
        for (int32_t i = 0; i < counts[g]; ++i) {
            const uint64_t s = (token * 7 + 3) % 32, d = token * 2;
            mapping[(g * entries + i) * 2] = s;
            mapping[(g * entries + i) * 2 + 1] = d;
            std::copy_n(src.begin() + (s / 4 * 128 + s % 4 * 24) / sizeof(T), 8, expected.begin() + d * 8);
            if (rope) std::copy_n(srcRope.begin() + (s / 4 * 80 + s % 4 * 16) / sizeof(T), 4,
                                 expectedRope.begin() + d * 4);
            ++token;
        }
    }
    const uint64_t descriptors = token * (rope ? 2 : 1);
    // Extra sentinel descriptor detects writes beyond the compact list.
    std::vector<uint64_t> scratch((descriptors + 1) * 3, UINT64_MAX);
    std::vector<uint64_t> status(cores * kStatusBytes / 8, UINT64_MAX);
    LaunchArgs args{reinterpret_cast<uint64_t>(storage.data()), reinterpret_cast<uint64_t>(mapping.data()),
                    reinterpret_cast<uint64_t>(counts.data()), reinterpret_cast<uint64_t>(scratch.data()),
                    reinterpret_cast<uint64_t>(status.data()), groups, entries, token, cores};
    fake::submits = fake::drains = fake::largestBatch = 0;
    fake::submittedChannels.clear();
    assert(AscendC::copyPhase == 0);
    // Reuse the same allocations across iterations, including step 4. Poison
    // old phase/error/reserved fields to require a fresh full-row publication.
    for (uint32_t round = 0; round < 8; ++round) {
        std::fill(status.begin(), status.end(), UINT64_MAX);
        for (uint32_t core = 0; core < cores; ++core) {
            status[core * 16] = 0;
            status[core * 16 + 1] = kValidate;
        }
        AscendC::copies.clear();
        uint64_t covered = 0;
        for (uint32_t core = 0; core < cores; ++core) {
            const auto range = BuildDescriptors(args, core);
            assert(range.first == covered);
            assert(range.count == 0 || range.first % kDescriptorTileTokens == 0);
            covered += range.count;
        }
        assert(covered == token);
        for (uint32_t core = 0; core < cores; ++core) {
            assert(status[core * 16] == 0 && status[core * 16 + 1] == kDescriptorsReady);
            for (uint32_t word = 3; word < 16; ++word) assert(status[core * 16 + word] == 0);
        }
    }
    assert(AscendC::copyPhase == 0);
    assert(AscendC::copies.size() == (token + kDescriptorTileTokens - 1) / kDescriptorTileTokens + cores);
    uint64_t copiedBytes = 0;
    std::vector<uint32_t> statusWrites(cores, 0);
    for (const auto &copy : AscendC::copies) {
        const auto statusBase = reinterpret_cast<uintptr_t>(status.data());
        if (copy.destination >= statusBase && copy.destination < statusBase + status.size() * 8) {
            const auto offset = copy.destination - statusBase;
            assert(offset % kStatusBytes == 0 && copy.bytes == kStatusBytes);
            assert(copy.producer == AscendC::HardEvent::S_MTE3);
            ++statusWrites[offset / kStatusBytes];
            continue;
        }
        assert(copy.producer == AscendC::HardEvent::V_MTE3);
        const auto offset = copy.destination - reinterpret_cast<uintptr_t>(scratch.data());
        assert(offset % 128 == 0);
        assert(copy.bytes > 0 && copy.bytes <= kDescriptorTileBytes);
        const auto first = offset / (sizeof(HixlDevXferDesc) * (rope ? 2 : 1));
        assert(copy.bytes == std::min<uint64_t>(32, token - first) * (rope ? 2 : 1) * sizeof(HixlDevXferDesc));
        copiedBytes += copy.bytes;
    }
    assert(std::all_of(statusWrites.begin(), statusWrites.end(), [](uint32_t n) { return n == 1; }));
    assert(copiedBytes == descriptors * sizeof(HixlDevXferDesc));
    assert(std::all_of(dst.begin(), dst.end(), [sentinel](T x) { return x == sentinel; }));
    assert(fake::submits == 0 && fake::drains == 0); // Generation never transports.
    auto *list = reinterpret_cast<hixl::aiv::HixlDevXferDesc *>(scratch.data());
    for (uint64_t i = 0; i < descriptors; ++i) {
        assert(list[i].reserved == 0 && list[i].len == (i % 2 == 1 && rope ? 4U : 8U) * sizeof(T));
        // CPU stand-in for a separate consumer; real HIXL is tested on device.
        std::memcpy(reinterpret_cast<void *>(list[i].local), reinterpret_cast<void *>(list[i].remote), list[i].len);
    }
    for (uint32_t core = 0; core < cores; ++core) FinishDescriptors(args, core);
    assert(dst == expected && dstRope == expectedRope);
    for (uint64_t i = descriptors * 3; i < scratch.size(); ++i) assert(scratch[i] == UINT64_MAX);
    uint64_t completed = 0;
    for (uint64_t core = 0; core < cores; ++core) {
        assert(status[core * 16] == 0 && status[core * 16 + 1] == kDescriptorsReady);
        completed += status[core * 16 + 2];
    }
    assert(completed == token);
    // Zero workload: fresh completion status, no data mutation and no submission.
    args.total = 0;
    const auto before = fake::submits;
    const auto copiesBefore = AscendC::copies.size();
    BuildDescriptors(args, 0);
    FinishDescriptors(args, 0);
    assert(fake::submits == before && status[2] == 0 && status[1] == kDescriptorsReady);
    assert(AscendC::copies.size() == copiesBefore + 1); // Only the final status DMA.
    assert(AscendC::copies.back().destination == reinterpret_cast<uintptr_t>(status.data()));
    assert(AscendC::copies.back().bytes == kStatusBytes);
    args.total = token;
    // More cores than tokens must produce fresh zero-work rows, without holes.
    args.total = 1;
    for (uint32_t core = 0; core < cores; ++core) BuildDescriptors(args, core);
    assert(status[2] == 1 && status[18] == 0 && status[34] == 0);

    // Graph replay can reuse input and bitmap addresses while values change.
    ctx.channels = cores;
    ctx.bytes = sizeof(Context) + cores * sizeof(uint64_t);
    ctx.generation = 1;
    ctx.batch = 32;
    std::vector<int32_t> q{0, 2};
    constexpr uint64_t seenStride = 32;
    std::vector<uint32_t> seen(cores * seenStride, UINT32_MAX);
    auto replayCounts = counts;
    GraphLaunchArgs graph{
        args, reinterpret_cast<uint64_t>(q.data()), q.size(), groups, 3,
        reinterpret_cast<uint64_t>(seen.data()), 512, ctx.localKv, ctx.localKvBytes,
        rope ? ctx.localRope : 0, rope ? ctx.localRopeBytes : 0};
    graph.seenStrideWords = seenStride;
    graph.copy.counts = reinterpret_cast<uint64_t>(replayCounts.data());
    for (uint32_t core = 0; core < cores; ++core) {
        assert(ValidateGraphCopy(graph, core) == 0 && graph.copy.total == token && graph.copy.groups == groups);
        assert(seen[core * seenStride] != UINT32_MAX);
        assert(seen[core * seenStride + 16] == UINT32_MAX);
    }
    q[1] = 0;
    assert(ValidateGraphCopy(graph) == 0 && graph.copy.total == 0);
    q[1] = 2;
    assert(ValidateGraphCopy(graph) == 0 && graph.copy.total == token);
    const auto count0 = replayCounts[0];
    replayCounts[0] = entries + 1;
    assert(ValidateGraphCopy(graph) == kBadMapping);
    const auto first = std::find_if(counts.begin(), counts.end(), [](int32_t count) { return count > 0; });
    const auto firstEntry = (first - counts.begin()) * entries * 2;
    replayCounts[0] = 1;
    mapping[0] = mapping[firstEntry];
    mapping[1] = mapping[firstEntry + 1];
    for (uint32_t core = 0; core < cores; ++core) {
        assert(ValidateGraphCopy(graph, core) == kBadMapping); // Reject before any shard submits.
    }
    replayCounts[0] = count0;
    const auto oldSrc = mapping[firstEntry];
    mapping[firstEntry] = 32;
    assert(ValidateGraphCopy(graph) == kBadMapping);
    mapping[firstEntry] = oldSrc;
    q[0] = 3;
    assert(ValidateGraphCopy(graph) == kBadMapping);
    q[0] = 0;
    graph.write = true;
    graph.remoteSlots = 32;
    std::vector<int32_t> writeMap{7, 0, 8, 0};
    std::vector<int32_t> writeCount{2};
    q[1] = 1;
    graph.heads = graph.groupsCapacity = 1;
    graph.copy.entries = 2;
    graph.copy.mapping = reinterpret_cast<uint64_t>(writeMap.data());
    graph.copy.counts = reinterpret_cast<uint64_t>(writeCount.data());
    assert(ValidateGraphCopy(graph) == 0 && graph.copy.total == 2);
    writeMap[2] = 7;
    assert(ValidateGraphCopy(graph) == kBadMapping); // Duplicated remote WRITE destination.
    writeMap[2] = 8;
    graph.remoteSlots = 8;
    assert(ValidateGraphCopy(graph) == kBadMapping);
    graph.remoteSlots = 32;
    ++graph.localKv;
    assert(ValidateGraphCopy(graph) == kBadContext); // Never write an unregistered clone.

    for (uint32_t core = 0; core < cores; ++core) {
        status[core * 16] = 0;
        status[core * 16 + 1] = kComplete;
    }
    assert(WriteCompleted(status.data(), cores));
    status[(cores - 1) * 16] = kBadMapping;
    assert(!WriteCompleted(status.data(), cores));

    ctx.magic = 0;
    BuildDescriptors(args, 0);
    assert(status[0] == kBadContext && status[1] == kValidate && status[2] == 0);
    for (uint32_t word = 3; word < 16; ++word) assert(status[word] == 0);
}
int main()
{
    for (const auto &counts : {std::vector<int32_t>{0, 19, 61, 0, 53, 12},
                               std::vector<int32_t>{0, 1, 0, 0, 0, 0},
                               std::vector<int32_t>{0, 19, 12, 0, 0, 0},
                               std::vector<int32_t>{0, 19, 13, 0, 0, 0},
                               std::vector<int32_t>{0, 19, 14, 0, 0, 0}}) {
        Run<uint16_t>(true, counts);
        Run<uint16_t>(false, counts);
        Run<uint8_t>(true, counts); // FP8/INT8 use identical one-byte transport semantics.
        Run<uint8_t>(false, counts);
    }
    std::cout << "PASS: repeated full-row status DMA, multi-lane UB fill, exact DMA tails, compact descriptors "
                 "(CPU only, not HIXL transport)\n";
}
