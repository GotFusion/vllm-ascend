/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <torch/extension.h>
#include <torch/library.h>
#include <pybind11/stl.h>
#include <c10/core/DeviceGuard.h>
// Use the ACL types provided by the framework's own headers below. Including
// CANN's umbrella acl.h here can reintroduce incompatible GE definitions.
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "hixl_bridge.h"
#include "avsg_urma_abi.h"

namespace avsg_urma {
extern "C" void LaunchAvsgBuildDescriptors(void *stream, LaunchArgs args);
extern "C" void LaunchAvsgFinishDescriptors(void *stream, LaunchArgs args);
extern "C" void LaunchAvsgCopyUrma(void *stream, GraphLaunchArgs args);
extern "C" void LaunchAvsgTransfer(void *stream, uint64_t channels, uint64_t descriptors,
                                   uint64_t error, uint32_t count, uint32_t jettys, uint32_t cores,
                                   uint32_t lanes, uint64_t sync, uint64_t timing);
using Tensor = at::Tensor;

void CheckAcl(aclError st, const char *name)
{
    TORCH_CHECK(st == ACL_SUCCESS, name, " failed: ", st);
}
void CheckHixl(uint32_t st, const char *name)
{
    TORCH_CHECK(st == 0, name, " failed: ", st);
}
void NpuTensor(const Tensor &t, c10::Device device)
{
    TORCH_CHECK(t.defined() && t.device() == device && t.is_contiguous(),
                "Expected a contiguous Tensor on ", device);
}
uint64_t Span(const Tensor &t)
{
    TORCH_CHECK(t.numel() > 0 && t.layout() == at::kStrided, "Cannot register empty/non-strided storage");
    uint64_t span = t.element_size();
    for (int64_t i = 0; i < t.dim(); ++i) {
        TORCH_CHECK(t.stride(i) > 0, "Only positive strides can be registered");
        const auto step = static_cast<uint64_t>(t.stride(i)) * t.element_size();
        TORCH_CHECK(static_cast<uint64_t>(t.size(i) - 1) <= (UINT64_MAX - span) / step,
                    "Storage span overflows");
        span += (t.size(i) - 1) * step;
    }
    TORCH_CHECK(span <= t.storage().nbytes() - t.storage_offset() * t.element_size(),
                "Tensor view exceeds its storage");
    return span;
}
bool Covers(uint64_t base, uint64_t size, uint64_t addr, uint64_t bytes)
{
    return addr >= base && addr - base <= size && bytes <= size - (addr - base);
}

Context ReadContext(const Tensor &context)
{
    TORCH_CHECK(context.scalar_type() == at::kLong && context.dim() == 1 &&
                context.numel() > kHeaderWords && context.numel() <= kHeaderWords + kMaxChannels,
                "Invalid context Tensor");
    auto cpu = context.cpu();
    Context ctx;
    std::memcpy(&ctx, cpu.const_data_ptr(), sizeof(ctx));
    TORCH_CHECK(ctx.magic == kMagic && ctx.version == kVersion && ctx.generation != 0 &&
                ctx.bytes == context.nbytes() && ctx.channels == context.numel() - kHeaderWords &&
                ctx.batch >= 2 && ctx.batch <= kMaxBatch, "Invalid context ABI");
    TORCH_CHECK(ctx.blockSize && !(ctx.blockSize & (ctx.blockSize - 1)) &&
                ctx.kvDim > 0 && ctx.kvDim <= 1024 && ctx.ropeDim <= 64 &&
                ctx.hasRope <= 1 && ctx.hasRope == (ctx.ropeDim != 0) && ctx.dtype <= 4 &&
                ctx.elementBytes == (ctx.dtype >= 2 ? 1 : 2), "Invalid cache layout");
    TORCH_CHECK(ctx.kvStride1 >= ctx.kvDim * ctx.elementBytes &&
                ctx.kvStride0 / ctx.blockSize >= ctx.kvStride1 &&
                (!ctx.hasRope || (ctx.ropeStride1 >= ctx.ropeDim * ctx.elementBytes &&
                 ctx.ropeStride0 / ctx.blockSize >= ctx.ropeStride1)), "Invalid source strides");
    for (auto region : {std::make_pair(ctx.remoteKv, ctx.remoteKvBytes),
                        std::make_pair(ctx.localKv, ctx.localKvBytes)}) {
        TORCH_CHECK(region.first && region.second && region.first <= UINT64_MAX - region.second,
                    "Invalid KV address range");
    }
    if (ctx.hasRope) {
        TORCH_CHECK(ctx.remoteRope && ctx.remoteRopeBytes && ctx.localRope && ctx.localRopeBytes &&
                    ctx.remoteRope <= UINT64_MAX - ctx.remoteRopeBytes &&
                    ctx.localRope <= UINT64_MAX - ctx.localRopeBytes, "Invalid RoPE address range");
    }
    std::set<uint64_t> unique;
    const auto words = reinterpret_cast<const uint64_t *>(cpu.const_data_ptr());
    for (uint64_t i = 0; i < ctx.channels; ++i) {
        TORCH_CHECK(words[kHeaderWords + i] && unique.insert(words[kHeaderWords + i]).second,
                    "Channels must be nonzero and exclusive");
    }
    return ctx;
}

void CheckSelection(const Context &ctx, const Tensor &kv, const Tensor &rope)
{
    const auto dtype = ctx.dtype == 0 ? at::kHalf :
                       ctx.dtype == 1 ? at::kBFloat16 :
                       ctx.dtype == 2 ? at::kChar :
                       ctx.dtype == 3 ? at::ScalarType::Float8_e4m3fn : at::ScalarType::Float8_e5m2;
    TORCH_CHECK(kv.dim() == 3 && kv.size(1) == ctx.blockSize && kv.size(2) == ctx.kvDim &&
                kv.scalar_type() == dtype && reinterpret_cast<uint64_t>(kv.data_ptr()) == ctx.localKv &&
                kv.nbytes() == ctx.localKvBytes, "Selection KV no longer matches its registration");
    if (ctx.hasRope) {
        TORCH_CHECK(rope.dim() == 3 && rope.size(0) == kv.size(0) && rope.size(1) == ctx.blockSize &&
                    rope.size(2) == ctx.ropeDim && rope.scalar_type() == dtype &&
                    reinterpret_cast<uint64_t>(rope.data_ptr()) == ctx.localRope &&
                    rope.nbytes() == ctx.localRopeBytes, "Selection RoPE no longer matches its registration");
    } else {
        TORCH_CHECK(rope.numel() == 0, "Absent RoPE requires an empty device Tensor");
    }
}

void CheckTokenRegion(uint64_t src, uint64_t blockSize, uint64_t base, uint64_t bytes,
                      uint64_t stride0, uint64_t stride1, uint64_t row)
{
    const uint64_t block = src / blockSize, offset = src % blockSize;
    TORCH_CHECK(block <= bytes / stride0 && offset <= bytes / stride1, "Source index is out of range");
    const uint64_t off0 = block * stride0, off1 = offset * stride1;
    TORCH_CHECK(off1 <= bytes && off0 <= bytes - off1 && row <= bytes - off0 - off1,
                "Source token exceeds registered storage");
    TORCH_CHECK(Covers(base, bytes, base + off0 + off1, row), "Invalid source address");
}

void CheckMappingShape(const Tensor &q, const Tensor &mapping, const Tensor &counts)
{
    TORCH_CHECK(q.scalar_type() == at::kInt && q.dim() == 1 && q.numel() > 0 &&
                mapping.scalar_type() == at::kInt && counts.scalar_type() == at::kInt && counts.dim() >= 1,
                "Query lengths, mapping and counts must be INT32 with valid ranks");
    TORCH_CHECK(mapping.dim() == counts.dim() + 2 && mapping.size(-1) == 2 && mapping.size(-2) > 0,
                "Mapping must have shape [*counts.shape, entries, 2]");
    for (int64_t i = 0; i < counts.dim(); ++i) {
        TORCH_CHECK(mapping.size(i) == counts.size(i), "Mapping/count shape mismatch");
    }
}

LaunchArgs ValidateMapping(const Context &ctx, const Tensor &q, const Tensor &mapping, const Tensor &counts)
{
    CheckMappingShape(q, mapping, counts);
    auto qc = q.cpu(), cc = counts.cpu(), mc = mapping.cpu();
    const auto qp = qc.const_data_ptr<int32_t>();
    int32_t prev = 0;
    for (int64_t i = 0; i < q.numel(); ++i) {
        TORCH_CHECK(qp[i] >= prev, "Query lengths must be nonnegative cumulative lengths");
        prev = qp[i];
    }
    const uint64_t groups = static_cast<uint64_t>(prev) * counts.size(-1);
    TORCH_CHECK(groups <= counts.numel(), "Query/head count exceeds mapping capacity");
    const uint64_t entries = mapping.size(-2), slots = ctx.localKvBytes / (ctx.kvDim * ctx.elementBytes);
    const auto cp = cc.const_data_ptr<int32_t>(), mp = mc.const_data_ptr<int32_t>();
    uint64_t total = 0;
    std::set<int32_t> destinations;
    for (uint64_t group = 0; group < groups; ++group) {
        TORCH_CHECK(cp[group] >= 0 && static_cast<uint64_t>(cp[group]) <= entries, "Invalid copy count");
        for (int32_t i = 0; i < cp[group]; ++i) {
            const int32_t src = mp[(group * entries + i) * 2], dst = mp[(group * entries + i) * 2 + 1];
            TORCH_CHECK(src >= 0 && dst >= 0 && static_cast<uint64_t>(dst) < slots &&
                        destinations.insert(dst).second, "Invalid or duplicate destination slot");
            CheckTokenRegion(src, ctx.blockSize, ctx.remoteKv, ctx.remoteKvBytes,
                             ctx.kvStride0, ctx.kvStride1, ctx.kvDim * ctx.elementBytes);
            if (ctx.hasRope) {
                CheckTokenRegion(src, ctx.blockSize, ctx.remoteRope, ctx.remoteRopeBytes,
                                 ctx.ropeStride0, ctx.ropeStride1, ctx.ropeDim * ctx.elementBytes);
            }
        }
        total += cp[group];
    }
    return {0, reinterpret_cast<uint64_t>(mapping.data_ptr()), reinterpret_cast<uint64_t>(counts.data_ptr()),
            0, 0, groups, entries, total, ctx.channels};
}

std::tuple<int64_t, int64_t> Copy(const Tensor &rope, const Tensor &kv, const Tensor &q,
          const Tensor &mapping, const Tensor &counts,
          const Tensor &context, const Tensor &scratch, const Tensor &status,
          const Tensor &sync, const Tensor &timing, int64_t transferCores, bool measureHost)
{
    c10::DeviceGuard guard(kv.device());
    TORCH_CHECK(kv.device().type() == c10::DeviceType::PrivateUse1, "An NPU selection Tensor is required");
    for (const auto &t : {kv, rope, q, mapping, counts, context, scratch, status, sync, timing}) NpuTensor(t, kv.device());
    const auto ctx = ReadContext(context);
    CheckSelection(ctx, kv, rope);
    TORCH_CHECK(ctx.channels <= 32 && (ctx.batch == 8 || ctx.batch == 32 || ctx.batch == 64),
                "HIXL launcher requires <=32 channels and lanes (batch_descriptors) in {8,32,64}");
    TORCH_CHECK(transferCores >= static_cast<int64_t>(ctx.channels) && transferCores <= 64,
                "Transfer cores must be between actual jetty count and 64");
    const bool shared = transferCores > static_cast<int64_t>(ctx.channels);
    TORCH_CHECK(sync.scalar_type() == at::kByte &&
                (shared ? sync.numel() == 4096 : sync.numel() == 0), "Invalid HIXL multi-core sync buffer");
    TORCH_CHECK(timing.scalar_type() == at::kLong && (timing.numel() == 0 || timing.numel() == 32 * 16),
                "Timing must be empty or HIXL's 32 x 16 int64 slots");
    TORCH_CHECK((!shared || reinterpret_cast<uint64_t>(sync.data_ptr()) % 128 == 0) &&
                (!timing.numel() || reinterpret_cast<uint64_t>(timing.data_ptr()) % 128 == 0),
                "Sync/timing buffers must be 128-byte aligned");
    LaunchArgs args = ValidateMapping(ctx, q, mapping, counts);
    const uint64_t perToken = ctx.hasRope ? 2 : 1;
    TORCH_CHECK(args.total <= UINT32_MAX / perToken, "HIXL descriptor count exceeds uint32");
    const uint64_t descriptorCount = args.total * perToken;
    const uint64_t scratchBytes = descriptorCount * 24;
    TORCH_CHECK(scratch.scalar_type() == at::kByte && scratch.nbytes() >= scratchBytes &&
                status.scalar_type() == at::kLong && status.numel() == ctx.channels * kStatusBytes / 8,
                "Invalid descriptor scratch/status buffer");
    TORCH_CHECK(reinterpret_cast<uint64_t>(scratch.data_ptr()) % 128 == 0 &&
                reinterpret_cast<uint64_t>(status.data_ptr()) % 128 == 0,
                "Scratch and status must be 128-byte aligned");
    args.context = reinterpret_cast<uint64_t>(context.data_ptr());
    args.scratch = reinterpret_cast<uint64_t>(scratch.data_ptr());
    args.status = reinterpret_cast<uint64_t>(status.data_ptr());
    // Generation publishes every full status row (including the transport-error
    // word) once via UB->GM DMA. Avoid a separate cached fill of the same rows.
    // Each launch starts a fresh barrier generation and a fresh timing sample.
    // Stream order ensures these fills finish before the HIXL kernel starts.
    if (shared) sync.zero_();
    if (timing.numel()) timing.zero_();
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto build = [args, stream]() -> int {
        LaunchAvsgBuildDescriptors(stream, args);
        return static_cast<int>(aclrtSynchronizeStream(stream));
    };
    using HostClock = std::chrono::steady_clock;
    const auto descriptorBegin = measureHost ? HostClock::now() : HostClock::time_point{};
    // RunOpApiV2 may enqueue the callback and return before it runs. Waiting
    // inside the callback alone does not make the caller's timing synchronous.
    at_npu::native::OpCommand::RunOpApiV2("avsg_build_descriptors", build, true);
    // Never issue a READ from a partially generated/invalid list. This baseline
    // intentionally has a Host check between the two stages.
    auto sc = status.cpu();
    const auto sp = reinterpret_cast<const uint64_t *>(sc.const_data_ptr());
    uint64_t done = 0;
    for (uint64_t core = 0; core < ctx.channels; ++core) {
        const auto row = sp + core * kStatusBytes / 8;
        TORCH_CHECK(row[0] == 0 && row[1] == kDescriptorsReady, "Descriptor generation failed on core ", core,
                    ", code=", row[0], ", phase=", row[1], ", generated_tokens=", row[2],
                    ", expected_phase=", kDescriptorsReady, ", total_tokens=", args.total,
                    "; discard AVSG match state and close Session");
        done += row[2];
    }
    TORCH_CHECK(done == args.total, "Descriptor token coverage mismatch");
    const int64_t descriptorNs = measureHost ?
        std::chrono::duration_cast<std::chrono::nanoseconds>(HostClock::now() - descriptorBegin).count() : 0;
    if (descriptorCount == 0) return {descriptorNs, 0};
    const uint64_t syncAddr = shared ? reinterpret_cast<uint64_t>(sync.data_ptr()) : 0;
    const uint64_t timingAddr = timing.numel() ? reinterpret_cast<uint64_t>(timing.data_ptr()) : 0;
    auto transfer = [args, stream, descriptorCount, ctx, transferCores, syncAddr, timingAddr]() -> int {
        LaunchAvsgTransfer(stream, args.context + sizeof(Context), args.scratch,
                           args.status + kTransportErrorOffset, static_cast<uint32_t>(descriptorCount),
                           static_cast<uint32_t>(args.channels), static_cast<uint32_t>(transferCores),
                           static_cast<uint32_t>(ctx.batch), syncAddr, timingAddr);
        // Maintenance only; the preceding HIXL kernel performs the READ+Drain.
        LaunchAvsgFinishDescriptors(stream, args);
        return static_cast<int>(aclrtSynchronizeStream(stream));
    };
    const auto transferBegin = measureHost ? HostClock::now() : HostClock::time_point{};
    at_npu::native::OpCommand::RunOpApiV2("avsg_copy_urma", transfer, true);
    // End after the callback's post-maintenance stream synchronization, before
    // the separate error-flag D2H read/check below.
    const int64_t transferNs = measureHost ?
        std::chrono::duration_cast<std::chrono::nanoseconds>(HostClock::now() - transferBegin).count() : 0;
    auto completed = status.cpu();
    const auto transportError = reinterpret_cast<const uint8_t *>(completed.const_data_ptr())[kTransportErrorOffset];
    // The generation counter is not a CQ completion counter. Transport success
    // relies on HIXL's internal Drain, stream completion and this error flag.
    TORCH_CHECK(transportError == 0, "HIXL SIMT READ failed, code=", static_cast<unsigned>(transportError),
                "; discard AVSG match state and close Session (no automatic retry)");
    return {descriptorNs, transferNs};
}

void TransferGraph(const Tensor &rope, const Tensor &kv, const Tensor &q, const Tensor &mapping,
                   const Tensor &counts, const Tensor &context, const Tensor &scratch,
                   const Tensor &seen, const Tensor &status, int64_t remoteSlots,
                   const c10::optional<Tensor> &writeStatus)
{
    c10::DeviceGuard guard(kv.device());
    for (const auto &t : {kv, rope, q, mapping, counts, context, scratch, seen, status}) NpuTensor(t, kv.device());
    CheckMappingShape(q, mapping, counts);
    TORCH_CHECK(kv.dim() == 3 && kv.numel() > 0 &&
                (rope.numel() == 0 || (rope.dim() == 3 && rope.size(0) == kv.size(0) &&
                 rope.size(1) == kv.size(1) && rope.scalar_type() == kv.scalar_type())),
                "Invalid Selection KV/RoPE layout");
    TORCH_CHECK(context.scalar_type() == at::kLong && context.dim() == 1 && context.numel() > kHeaderWords &&
                context.numel() <= kHeaderWords + kMaxGraphChannels,
                "Graph transfer requires a prepared context with 1 to 32 exclusive jettys");
    const uint64_t channels = context.numel() - kHeaderWords;
    const uint64_t slots = kv.numel() / kv.size(2);
    const uint64_t checkedSlots = remoteSlots > 0 ? remoteSlots : slots;
    const uint64_t seenWords = (checkedSlots + 31) / 32;
    constexpr uint64_t wordsPerLine = kStatusBytes / sizeof(uint32_t);
    const uint64_t seenStride = channels == 1 ? seenWords : (seenWords + wordsPerLine - 1) / wordsPerLine * wordsPerLine;
    const uint64_t statusWords = channels * kStatusBytes / sizeof(uint64_t);
    const uint64_t maxDescriptors = (mapping.numel() / 2) * (rope.numel() ? 2 : 1);
    TORCH_CHECK(maxDescriptors <= UINT32_MAX && scratch.scalar_type() == at::kByte &&
                scratch.nbytes() >= maxDescriptors * 24 &&
                seen.scalar_type() == at::kInt && static_cast<uint64_t>(seen.numel()) >= channels * seenStride &&
                status.scalar_type() == at::kLong && status.numel() == statusWords,
                "Invalid static descriptor scratch, destination bitmap or status");
    auto addr = [](const Tensor &t) { return reinterpret_cast<uint64_t>(t.data_ptr()); };
    TORCH_CHECK(addr(scratch) % 128 == 0 && addr(status) % 128 == 0,
                "Scratch and status must be 128-byte aligned");
    TORCH_CHECK(channels == 1 || addr(seen) % 128 == 0, "Multi-jetty destination bitmap must be 128-byte aligned");
    if (writeStatus.has_value()) {
        NpuTensor(*writeStatus, kv.device());
        TORCH_CHECK(writeStatus->scalar_type() == at::kLong && writeStatus->numel() == statusWords &&
                    addr(*writeStatus) % 128 == 0 && addr(*writeStatus) != addr(status),
                    "Invalid prerequisite WRITE status");
    }
    GraphLaunchArgs args{
        {addr(context), addr(mapping), addr(counts), addr(scratch), addr(status), 0,
         static_cast<uint64_t>(mapping.size(-2)), 0, channels},
        addr(q), static_cast<uint64_t>(q.numel()), static_cast<uint64_t>(counts.numel()),
        static_cast<uint64_t>(counts.size(-1)), addr(seen), slots, addr(kv), kv.nbytes(),
        addr(rope), rope.nbytes(), static_cast<uint64_t>(remoteSlots),
        writeStatus.has_value() ? addr(*writeStatus) : 0, seenStride, remoteSlots > 0};
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Keep tensors alive while the host task queue owns the launch callback.
    auto tensors = std::make_tuple(rope, kv, q, mapping, counts, context, scratch, seen, status, writeStatus);
    auto launch = [args, stream, tensors]() -> int {
        (void)tensors;
        LaunchAvsgCopyUrma(stream, args);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(args.write ? "avsg_write_urma_graph" : "avsg_copy_urma_graph", launch);
}

void CopyGraph(const Tensor &rope, const Tensor &kv, const Tensor &q, const Tensor &mapping,
               const Tensor &counts, const Tensor &context, const Tensor &scratch,
               const Tensor &seen, const Tensor &status, const c10::optional<Tensor> &writeStatus)
{
    TransferGraph(rope, kv, q, mapping, counts, context, scratch, seen, status, 0, writeStatus);
}

void WriteGraph(const Tensor &rope, const Tensor &kv, const Tensor &q, const Tensor &mapping,
                const Tensor &counts, const Tensor &context, const Tensor &scratch,
                const Tensor &seen, const Tensor &status, int64_t remoteSlots)
{
    TORCH_CHECK(remoteSlots > 0 && remoteSlots <= INT32_MAX, "Invalid number of remote WRITE slots");
    TransferGraph(rope, kv, q, mapping, counts, context, scratch, seen, status, remoteSlots, c10::nullopt);
}

void CopyGraphMeta(const Tensor &, const Tensor &, const Tensor &q, const Tensor &mapping,
                   const Tensor &counts, const Tensor &, const Tensor &, const Tensor &, const Tensor &,
                   const c10::optional<Tensor> &)
{
    CheckMappingShape(q, mapping, counts);
}

void WriteGraphMeta(const Tensor &, const Tensor &, const Tensor &q, const Tensor &mapping,
                    const Tensor &counts, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t)
{
    CheckMappingShape(q, mapping, counts);
}

class Endpoint {
public:
    Endpoint(const std::string &local, int device, const std::map<std::string, std::string> &options)
        : device_(c10::DeviceType::PrivateUse1, device)
    {
        c10::DeviceGuard guard(device_);
        CheckHixl(engine_.Initialize(local, options), "Hixl::Initialize");
        open_ = true;
    }
    ~Endpoint()
    {
        if (open_) {
            try { Close(); } catch (...) { engine_.Finalize(); }
        }
    }
    std::pair<uint64_t, uint64_t> Register(const Tensor &tensor)
    {
        c10::DeviceGuard guard(device_);
        TORCH_CHECK(open_ && !connected_ && tensor.device() == device_, "Register before Connect on the owner Device");
        const uint64_t base = reinterpret_cast<uint64_t>(tensor.data_ptr()), bytes = Span(tensor);
        TORCH_CHECK(base <= UINT64_MAX - bytes, "Registration address overflows");
        for (const auto &reg : registrations_) {
            TORCH_CHECK(base + bytes <= reg.base || reg.base + reg.bytes <= base, "Overlapping registrations");
        }
        Registration reg{tensor, base, bytes};
        CheckHixl(engine_.RegisterMem(base, bytes), "Hixl::RegisterMem");
        registrations_.push_back(reg);
        return {base, bytes};
    }
    std::vector<uint64_t> Connect(const std::string &peer, int timeout)
    {
        c10::DeviceGuard guard(device_);
        TORCH_CHECK(open_ && !connected_ && timeout > 0, "Endpoint is closed/already connected or timeout invalid");
        peer_ = peer;
        CheckHixl(engine_.Connect(peer, timeout), "Hixl::Connect");
        connected_ = true;
        CheckHixl(engine_.Channels(peer, channels_), "GetAivChannelPointers");
        TORCH_CHECK(!channels_.empty() && channels_.size() <= kMaxChannels &&
                    std::set<uint64_t>(channels_.begin(), channels_.end()).size() == channels_.size(),
                    "No usable exclusive AIV channels");
        return channels_;
    }
    uint64_t RemoteAddress(uint64_t base, uint64_t bytes)
    {
        c10::DeviceGuard guard(device_);
        TORCH_CHECK(open_ && connected_ && base && bytes, "Exact registered RVA required after Connect");
        uint64_t addr = 0;
        CheckHixl(engine_.RemoteAddress(peer_, base, bytes, addr), "Hixl::GetAddr");
        return addr;
    }
    void Validate(const Tensor &context)
    {
        c10::DeviceGuard guard(device_);
        TORCH_CHECK(open_ && connected_, "Endpoint is not connected");
        NpuTensor(context, device_);
        const auto ctx = ReadContext(context);
        TORCH_CHECK(ctx.channels == channels_.size(), "Context channel count differs from owner");
        auto cpu = context.cpu();
        const auto words = reinterpret_cast<const uint64_t *>(cpu.const_data_ptr());
        TORCH_CHECK(std::equal(channels_.begin(), channels_.end(), words + kHeaderWords), "Foreign channel context");
        bool kv = false, rope = !ctx.hasRope;
        for (const auto &reg : registrations_) {
            TORCH_CHECK(reinterpret_cast<uint64_t>(reg.tensor.data_ptr()) == reg.base && Span(reg.tensor) == reg.bytes,
                        "Registered Tensor storage changed");
            kv |= reg.base == ctx.localKv && reg.bytes == ctx.localKvBytes;
            rope |= reg.base == ctx.localRope && reg.bytes == ctx.localRopeBytes;
        }
        TORCH_CHECK(kv && rope, "Selection not owned by this endpoint");
    }
    void Close()
    {
        if (!open_) return;
        c10::DeviceGuard guard(device_);
        CheckAcl(aclrtSynchronizeDevice(), "aclrtSynchronizeDevice");
        if (connected_) {
            CheckHixl(engine_.Disconnect(peer_), "Hixl::Disconnect");
            connected_ = false;
        }
        // Finalize also destroys incoming-only connections before dropping registered Tensor owners.
        engine_.Finalize();
        open_ = false;
        registrations_.clear();
        channels_.clear();
    }
private:
    struct Registration { Tensor tensor; uint64_t base, bytes; };
    c10::Device device_;
    HixlBridge engine_;
    std::vector<Registration> registrations_;
    std::vector<uint64_t> channels_;
    std::string peer_;
    bool open_ = false, connected_ = false;
};
} // namespace avsg_urma

TORCH_LIBRARY(avsg_urma, m)
{
    m.def("copy_(Tensor(a!) rope, Tensor(b!) kv, Tensor q, Tensor mapping, Tensor counts, "
          "Tensor(c!) context, Tensor(d!) scratch, Tensor(e!) status, "
          "Tensor(f!) sync, Tensor(g!) timing, int transfer_cores, bool measure_host=False) -> (int, int)");
    m.def("copy_graph_(Tensor(a!) rope, Tensor(b!) kv, Tensor q, Tensor mapping, Tensor counts, "
          "Tensor context, Tensor(c!) scratch, Tensor(d!) seen, Tensor(e!) status, Tensor? write_status=None) -> ()");
    m.def("write_graph_(Tensor rope, Tensor kv, Tensor q, Tensor mapping, Tensor counts, "
          "Tensor context, Tensor(a!) scratch, Tensor(b!) seen, Tensor(c!) status, int remote_slots) -> ()");
}
TORCH_LIBRARY_IMPL(avsg_urma, PrivateUse1, m)
{
    m.impl("copy_", &avsg_urma::Copy);
    m.impl("copy_graph_", &avsg_urma::CopyGraph);
    m.impl("write_graph_", &avsg_urma::WriteGraph);
}
TORCH_LIBRARY_IMPL(avsg_urma, Meta, m)
{
    m.impl("copy_graph_", &avsg_urma::CopyGraphMeta);
    m.impl("write_graph_", &avsg_urma::WriteGraphMeta);
}
PYBIND11_MODULE(_native, m)
{
    pybind11::class_<avsg_urma::Endpoint>(m, "Endpoint")
        .def(pybind11::init<const std::string &, int, const std::map<std::string, std::string> &>())
        .def("register_tensor", &avsg_urma::Endpoint::Register)
        .def("connect", &avsg_urma::Endpoint::Connect)
        .def("remote_address", &avsg_urma::Endpoint::RemoteAddress)
        .def("validate_context", &avsg_urma::Endpoint::Validate)
        .def("close", &avsg_urma::Endpoint::Close);
}
