/*
 * Adapted from the s29 950PR probe and cann/asc-comm examples.
 * See ../SOURCES.md and ../LICENSE for source and license details.
 */
#include "hcomm/hcomm.h"
#include "kernel_operator.h"
#include "c_api/cache_ctrl/cache_ctrl.h"

#define KV_GATHER_AIV_DEVICE 1
#include "protocol.h"
#include "vector_builder.h"
#if defined(KV_GATHER_ENABLE_SDK_SIMT)
#include "simt_builder.h"
#endif

namespace kv_gather_aiv {

constexpr auto kCqe = AscendC::URMA_DEFAULT_CFG;
constexpr auto kNoCqe = []() constexpr {
    auto config = AscendC::URMA_DEFAULT_CFG;
    config.cqe = 0;
    return config;
}();

__aicore__ inline void DcciRange(__gm__ void* base, std::uint32_t bytes)
{
    const auto address = reinterpret_cast<std::uint64_t>(base);
    const auto end = address + bytes;
    // Cover both 64-byte and 128-byte cache-line hypotheses used by s29.
    for (auto line = address & ~std::uint64_t{127}; bytes != 0 && line < end; line += 64) {
        __asc_aicore::asc_dcci_single(reinterpret_cast<__gm__ void*>(line));
    }
}

class Transfer {
public:
    __aicore__ inline void Init(GM_ADDR context, AscendC::TPipe& pipe)
    {
        context_ = reinterpret_cast<__gm__ TransferContext*>(context) + AscendC::GetBlockIdx();
        mailbox_ = reinterpret_cast<__gm__ Mailbox*>(context_->mailbox);
        activeRequest_ = 0;
        pipe.InitBuffer(wqeBuffer_, kBatchBufferBytes);
        pipe.InitBuffer(indexBuffer_, kIndexScratchBytes);
#if defined(KV_GATHER_ENABLE_SDK_SIMT)
        pipe.InitBuffer(simtStatusBuffer_, 32 * sizeof(int32_t));
#endif
        wqes_ = wqeBuffer_.Get<uint8_t>();
        indices_ = indexBuffer_.Get<uint64_t>();
#if defined(KV_GATHER_ENABLE_SDK_SIMT)
        simtStatuses_ = simtStatusBuffer_.Get<int32_t>();
#endif
    }

    __aicore__ inline void Run()
    {
        auto batch = hcomm_.MakeBatchHandle(
            static_cast<AscendC::MultiChannelHandle>(context_->channel), wqes_, kBatchBufferBytes);
        auto destination = reinterpret_cast<GM_ADDR>(context_->remoteBuffer) + kDataOffset;
        auto& peer = hcomm_.GetHandleRef(batch, 0U, destination);
        mailbox_->transfer.sqAddress = peer.sqContext.contextInfo.ubJfs.sqVa;
        mailbox_->transfer.cqAddress = peer.cqContext.contextInfo.ubJfc.scqVa;
        mailbox_->transfer.sqDepth = peer.sqContext.contextInfo.ubJfs.sqDepth;
        mailbox_->transfer.cqDepth = peer.cqContext.contextInfo.ubJfc.cqDepth;
        TransferSample empty{};
        Progress(1, 0, empty);
        mailbox_->transfer.ready = 1;
        DcciRange(&mailbox_->transfer, sizeof(TransferStatus));
        std::uint64_t last = 0;
        for (;;) {
            DcciRange(&mailbox_->command, sizeof(Command));
            if (mailbox_->command.stop != 0) {
                return;
            }
            const auto id = mailbox_->command.requestId;
            if (id == last) {
                continue;
            }
            const auto blocks = mailbox_->command.blocks;
            if (id != last + 1 || !ValidTransfer(blocks, context_->sges, context_->batchWqes, context_->window)) {
                Fail(-1);
                return;
            }
            const auto range = Partition(blocks, context_->sges, context_->cores, context_->core);
            activeRequest_ = id;
            Progress(10, range.first, empty);
            const auto indexAddress = context_->requestBuffer + 16;
            const auto beforeDcci = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            DcciRange(reinterpret_cast<__gm__ void*>(indexAddress + range.first * 8), range.count * 8);
            const auto afterDcci = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            TransferSample sample{};
            sample.requestId = id;
            sample.indexDcciCycles = afterDcci - beforeDcci;
            int error;
#if defined(KV_GATHER_ENABLE_SDK_SIMT)
            if (context_->builder == Builder::Simt) error = WriteSimt(peer.channelHandle, range, indexAddress, sample);
            else
#endif
            error = WriteRequest(batch, peer, range, indexAddress, sample);
            if (error != AscendC::HCOMM_SUCCESS) {
                Fail(error);
                return;
            }
            sample.dataCycles = static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - afterDcci;
            // Publish sample payload before its done flag, across all status lines.
            mailbox_->transfer.sample.requestId = id;
            mailbox_->transfer.sample.dataCycles = sample.dataCycles;
            mailbox_->transfer.sample.indexDcciCycles = sample.indexDcciCycles;
            mailbox_->transfer.sample.constructCycles = sample.constructCycles;
            mailbox_->transfer.sample.wqes = sample.wqes;
            mailbox_->transfer.sample.batches = sample.batches;
            mailbox_->transfer.sample.drains = sample.drains;
            mailbox_->transfer.sample.cqes = sample.cqes;
            DcciRange(&mailbox_->transfer, sizeof(TransferStatus));
            Progress(20, range.first + range.count, sample);
            mailbox_->transfer.done = id;
            DcciRange(&mailbox_->transfer, sizeof(TransferStatus));
            last = id;
        }
    }

private:
    using Hcomm = AscendC::Hcomm<AscendC::COMM_PROTOCOL_UB_CTP>;

#if defined(KV_GATHER_ENABLE_SDK_SIMT)
    __aicore__ inline int SimtStatus(std::uint32_t lanes)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        for (std::uint32_t lane = 0; lane < lanes; ++lane) {
            const auto status = simtStatuses_.GetValue(lane);
            if (status != 0) {
                return status;
            }
        }
        return 0;
    }

    __aicore__ inline int WriteSimt(std::uint64_t channel, CoreRange range,
                                   std::uint64_t indexAddress, TransferSample& sample)
    {
        if (context_->sges != 1 || context_->batchWqes > 32 || context_->cqePerWqe != 1) {
            return -1;
        }
        auto* workspace = reinterpret_cast<__ubuf__ uint8_t*>(wqes_.GetPhyAddr());
        auto* statuses = reinterpret_cast<__ubuf__ int32_t*>(simtStatuses_.GetPhyAddr());
        std::uint32_t inWindow = 0;
        const auto end = range.first + range.count;
        for (auto first = range.first; first < end;) {
            const auto remaining = end - first;
            const auto lanes = remaining < context_->batchWqes ? remaining : context_->batchWqes;
            const auto begin = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            asc_vf_call<SimtAppend>(dim3(lanes), channel,
                reinterpret_cast<__gm__ uint8_t*>(context_->localBuffer + kDataOffset),
                reinterpret_cast<__gm__ uint8_t*>(context_->remoteBuffer + kDataOffset + std::uint64_t{first} * kBlockBytes),
                reinterpret_cast<__gm__ uint64_t*>(indexAddress) + first, workspace, statuses);
            auto status = SimtStatus(lanes);
            if (status != 0) {
                return status;
            }
            sample.constructCycles += static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - begin;
            sample.wqes += lanes;
            sample.cqes += lanes;
            ++sample.batches;
            ++inWindow;
            first += lanes;
            if (inWindow == context_->window || first == end) {
                asc_vf_call<SimtDrain>(dim3(1), channel, workspace, statuses);
                status = SimtStatus(1);
                if (status != 0) {
                    return status;
                }
                ++sample.drains;
                inWindow = 0;
            }
        }
        return 0;
    }
#endif

    __aicore__ inline void Fail(std::int32_t error)
    {
        mailbox_->transfer.error = error;
        DcciRange(&mailbox_->transfer, sizeof(TransferStatus));
    }

    __aicore__ inline void Progress(uint32_t phase, uint32_t next, const TransferSample& sample)
    {
        if (context_->debugProgress == 0) return;
        mailbox_->transfer.activeRequest = activeRequest_;
        mailbox_->transfer.nextIndex = next;
        mailbox_->transfer.completedWqes = sample.wqes;
        mailbox_->transfer.completedBatches = sample.batches;
        mailbox_->transfer.completedDrains = sample.drains;
        mailbox_->transfer.phase = phase;
        DcciRange(&mailbox_->transfer, sizeof(TransferStatus));
    }

    __aicore__ inline int WriteRequest(AscendC::UbcMultiBatchHandle& batch, AscendC::UbcBatchHandle& peer,
                                      CoreRange range, std::uint64_t indexAddress, TransferSample& sample)
    {
        std::uint32_t offset = range.first;
        const auto blocks = range.first + range.count;
        std::uint32_t outstandingBatches = 0;
        const auto sqDepth = peer.sqContext.contextInfo.ubJfs.sqDepth;
        const auto cqDepth = peer.cqContext.contextInfo.ubJfc.cqDepth;
        if (sqDepth < 2 || (sqDepth & (sqDepth - 1)) != 0 || cqDepth < 2) {
            return -1;
        }
        while (offset < blocks) {
            const auto plan = PlanBatch(blocks - offset, context_->sges, context_->batchWqes);
            const auto sqUsed = peer.cursor.sqHead - peer.cursor.sqTail;
            const auto cqUsed = peer.cursor.cqHead - peer.cursor.cqTail;
            if (plan.queueBlocks > peer.buffer.bufferCapacity || sqUsed >= sqDepth ||
                plan.queueBlocks >= sqDepth - sqUsed || cqUsed >= cqDepth ||
                (context_->cqePerWqe ? plan.wqes : 1U) >= cqDepth - cqUsed) {
                return -1;
            }
            const auto start = static_cast<std::uint64_t>(AscendC::GetSystemCycle());
            Progress(11, offset, sample);
            int error;
            if (context_->builder == Builder::Vector) {
                error = BuildVector(peer, indexAddress, offset, plan);
            } else {
                error = BuildScalar(peer, indexAddress, offset, plan);
            }
            if (error != AscendC::HCOMM_SUCCESS) {
                return error;
            }
            Progress(12, offset, sample);
            error = hcomm_.BatchCommit(batch);
            if (error != AscendC::HCOMM_SUCCESS) {
                return error;
            }
            sample.constructCycles += static_cast<std::uint64_t>(AscendC::GetSystemCycle()) - start;
            sample.wqes += plan.wqes;
            sample.cqes += context_->cqePerWqe ? plan.wqes : 1;
            ++sample.batches;
            ++outstandingBatches;
            offset += plan.indices;
            Progress(13, offset, sample);
            if (outstandingBatches == context_->window || offset == blocks) {
                Progress(14, offset, sample);
                error = hcomm_.Drain(batch);
                if (error != AscendC::HCOMM_SUCCESS) {
                    return error;
                }
                ++sample.drains;
                Progress(15, offset, sample);
                outstandingBatches = 0;
            }
        }
        return AscendC::HCOMM_SUCCESS;
    }

    __aicore__ inline int BuildScalar(AscendC::UbcBatchHandle& batch, std::uint64_t indexAddress,
                                     std::uint32_t offset, const BatchPlan& plan)
    {
        auto* sourceIndices = reinterpret_cast<__gm__ uint64_t*>(indexAddress);
        for (std::uint32_t wqe = 0; wqe < plan.wqes; ++wqe) {
            const auto count = wqe + 1 == plan.wqes ? plan.tailSges : context_->sges;
            AscendC::BufDesc descriptors[kMaxSges];
            for (std::uint32_t sge = 0; sge < count; ++sge) {
                const auto block = sourceIndices[offset + wqe * context_->sges + sge];
                descriptors[sge].addr = reinterpret_cast<GM_ADDR>(context_->localBuffer) +
                                        kDataOffset + block * kBlockBytes;
                descriptors[sge].len = kBlockBytes;
            }
            auto* destination = reinterpret_cast<GM_ADDR>(context_->remoteBuffer) + kDataOffset +
                                std::uint64_t{offset + wqe * context_->sges} * kBlockBytes;
            const auto error = context_->cqePerWqe || wqe + 1 == plan.wqes ?
                hcomm_.WriteNbi<kCqe>(batch, destination, descriptors, count) :
                hcomm_.WriteNbi<kNoCqe>(batch, destination, descriptors, count);
            if (error != AscendC::HCOMM_SUCCESS) {
                return error;
            }
        }
        return AscendC::HCOMM_SUCCESS;
    }

    __aicore__ inline int BuildVector(AscendC::UbcBatchHandle& batch, std::uint64_t indexAddress,
                                     std::uint32_t offset, const BatchPlan& plan)
    {
        // Core partitions may begin between 32-byte boundaries. Stage an aligned
        // span and skip its prefix, including the request header for offset zero.
        const auto firstAddress = indexAddress + std::uint64_t{offset} * 8;
        const auto alignedAddress = firstAddress & ~std::uint64_t{31};
        const auto prefix = static_cast<std::uint32_t>((firstAddress - alignedAddress) / 8);
        const auto padded = (plan.indices + prefix + 3) & ~std::uint32_t{3};
        AscendC::GlobalTensor<uint64_t> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(alignedAddress), padded);
        AscendC::DataCopy(indices_, source, padded);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        auto* wqes = reinterpret_cast<__ubuf__ uint64_t*>(wqes_.GetPhyAddr());
        auto* indices = reinterpret_cast<__ubuf__ uint64_t*>(indices_.GetPhyAddr()) + prefix;
        const auto fullBlocks = WqeBlocks(context_->sges);
        const auto previous = batch.cursor.preSqCnt;
        const auto head = batch.cursor.sqHead + previous;
        const auto local = context_->localBuffer + kDataOffset;
        const auto remote = context_->remoteBuffer + kDataOffset + std::uint64_t{offset} * kBlockBytes;
        for (std::uint32_t first = 0; first < plan.uniformWqes; first += 32) {
            const auto remaining = plan.uniformWqes - first;
            const auto count = remaining < 32 ? remaining : 32;
            DispatchVectorWqes(wqes + (previous + first * fullBlocks) * 8,
                           indices + first * context_->sges, count, context_->sges, head + first * fullBlocks,
                           local, remote + std::uint64_t{first} * context_->sges * kBlockBytes,
                           kBlockBytes, first + count == plan.wqes, batch, context_->cqePerWqe);
        }
        if (plan.uniformWqes < plan.wqes) {
            const auto first = plan.uniformWqes;
            DispatchVectorWqes(wqes + (previous + first * fullBlocks) * 8,
                           indices + first * context_->sges, 1, plan.tailSges, head + first * fullBlocks,
                           local, remote + std::uint64_t{first} * context_->sges * kBlockBytes,
                           kBlockBytes, true, batch, context_->cqePerWqe);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        batch.cursor.preSqCnt = previous + plan.queueBlocks;
        batch.cursor.cqHead += context_->cqePerWqe ? plan.wqes : 1;
        return AscendC::HCOMM_SUCCESS;
    }

    Hcomm hcomm_;
    __gm__ TransferContext* context_;
    __gm__ Mailbox* mailbox_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> wqeBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECIN> indexBuffer_;
    AscendC::LocalTensor<uint8_t> wqes_;
    AscendC::LocalTensor<uint64_t> indices_;
#if defined(KV_GATHER_ENABLE_SDK_SIMT)
    AscendC::TBuf<AscendC::TPosition::VECOUT> simtStatusBuffer_;
    AscendC::LocalTensor<int32_t> simtStatuses_;
#endif
    uint64_t activeRequest_;
};

} // namespace kv_gather_aiv

extern "C" __vector__ __global__ __aicore__ void KvGatherAivTransferKernel(GM_ADDR context)
{
    AscendC::TPipe pipe;
    kv_gather_aiv::Transfer transfer;
    transfer.Init(context, pipe);
    transfer.Run();
}
