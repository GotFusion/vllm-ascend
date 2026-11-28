/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file avsg_match_split_bs_reuse_vec.h
 * \brief
 */
#ifndef AVSG_MATCH_SPLIT_BS_REUSE_VEC_H
#define AVSG_MATCH_SPLIT_BS_REUSE_VEC_H

#include "kernel_operator.h"
#include "avsg_match_vector.h"
#include <cstdint>

namespace AvsgMatchNs {
using namespace AscendC;

constexpr int32_t SORT_ELEMENTS_PER_REPEAT = 32;
constexpr int32_t INVALID_SLOT = -1;
constexpr int32_t UB_BLOCK_ALIGN_BYTES = 32;

template <typename U>
__aicore__ inline U CeilAlign(U a, U b)
{
    return (a + b - 1) / b * b;
}

// ============================================================
// TempBufferCursor: UB 临时缓冲区分配器
// ============================================================
struct TempBufferCursor {
    LocalTensor<int32_t> buf;
    int64_t remainingBytes;

    __aicore__ inline TempBufferCursor(LocalTensor<int32_t> buffer, int64_t bytes)
        : buf(buffer), remainingBytes(bytes) {}

    // Allocate count elements with 32-byte UB alignment.
    template<typename T>
    __aicore__ inline LocalTensor<T> Alloc(int64_t count) {
        int64_t bytes = count * sizeof(T);
        bytes = CeilAlign(bytes, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        ASSERT_MSG(bytes <= remainingBytes,
                   "TempBufferCursor: insufficient bytes, need %ld, have %ld \n",
                   bytes, remainingBytes);
        auto allocated = buf.template ReinterpretCast<T>();
        Advance(bytes);
        return allocated;
    }

    // 创建快照，子函数可安全修改，不影响原 cursor
    __aicore__ inline TempBufferCursor Fork() const {
        return TempBufferCursor(buf, remainingBytes);
    }

    // Reuse scratch at the current cursor without advancing it.
    __aicore__ inline LocalTensor<int32_t>& Current() { return buf; }
    __aicore__ inline const LocalTensor<int32_t>& Current() const { return buf; }

private:
    __aicore__ inline void Advance(int64_t bytes) {
        // 向上对齐到 int32 边界，保证后续分配地址对齐
        int64_t alignedBytes = (bytes + sizeof(int32_t) - 1) / sizeof(int32_t) * sizeof(int32_t);
        buf = buf[alignedBytes / sizeof(int32_t)];
        remainingBytes -= alignedBytes;
    }
};

class AvsgMatchSplitBsReuseVec
{
public:
    __aicore__ inline AvsgMatchSplitBsReuseVec(TPipe* pipe, const AvsgMatchTilingData* tiling,
                                          avsg_fusion::LocalTopkRow* localRow)
        : pipe_(pipe), tiling_(tiling), localRow_(localRow)
    {}

    __aicore__ inline void Init(
        GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status, GM_ADDR selection_kv_block_status_out,
        GM_ADDR selection_topk_indices,
        GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq, GM_ADDR full_q_actual_seq,
        GM_ADDR selection_kv_actual_seq, GM_ADDR stage_two_copy_mapping, GM_ADDR stage_two_copy_count, uint32_t worker)
    {

        blkIdx_ = worker;
        if (blkIdx_ >= tiling_->stageOneUsedCoreNum) {
            return;
        }

        // stage1
        stageOneLoopNum_ = (blkIdx_ == tiling_->stageOneUsedCoreNum - 1) ? tiling_->stageOneTailCoreBsLoopNum : tiling_->stageOneMainCoreBsLoopNum;

        // Apply Local Buff
        selTopKIdxUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->topk * sizeof(int32_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        if (localRow_ == nullptr) {
            pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUb_);
        }

        selKvActSeqUb_ = CeilAlign(static_cast<int64_t>(stageOneLoopNum_ * sizeof(int32_t)), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        // One batch of status: [statesize, 2] int32 words.
        selBlockStatUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->statesize * sizeof(uint64_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        offsetOutputUb_ = CeilAlign(
            static_cast<int64_t>(tiling_->topk  * sizeof(uint64_t)),
            static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));

        int64_t totalUb = CeilAlign(static_cast<int64_t> (tiling_->ubSize - UB_BLOCK_ALIGN_BYTES), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        tmpbuffUb_ = CeilAlign(static_cast<int64_t>(totalUb - selTopKIdxUb_ - selKvActSeqUb_), static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES)); // Bytes

        if (localRow_ != nullptr) {
            localRow_->scratch.InitBuffer(workBuf_, selKvActSeqUb_ + tmpbuffUb_);
        } else {
            pipe_->InitBuffer(workBuf_, selKvActSeqUb_ + tmpbuffUb_);
        }

        // For GM_Address
        // [batchsize, statesize * 2], initialized to -1 for an empty cache.
        selKvBlockStatusGm_.SetGlobalBuffer(
            (__gm__ int32_t*)selection_kv_block_status + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum * tiling_->statesize * HASH_ENTRY_FIELDS);
        selKvBlockStatusOutGm_.SetGlobalBuffer(
            (__gm__ int32_t*)selection_kv_block_status_out + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum * tiling_->statesize * HASH_ENTRY_FIELDS);
        // [batchsize, seq, headnum, topk] token粒度=groupid * selection_topk_block_size
        selTopKIndicesGm_.SetGlobalBuffer((__gm__ int32_t*)selection_topk_indices);

        // [batchsize] 每个batch实际的seq长度
        fullKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_actual_seq + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum);
        // [batchsize] Cumulative query lengths (TND batch boundaries).
        fullQActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)full_q_actual_seq);

        // [batchsize] Reused token counts accumulated across queries.
        selKvActualSeqGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_actual_seq + blkIdx_ * tiling_->stageOneMainCoreBsLoopNum);
        // [query_token, head, aligned_topk, 2], each valid entry is {src_token, dst_token}.
        stageOneCopyMappingGM_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_mapping);
        // [query_token, head], only the active prefix selected by full_q_actual_seq is consumed.
        mismatchNumArrayGm_.SetGlobalBuffer((__gm__ int32_t*)stage_two_copy_count);

        selKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)selection_kv_block_table);
        fullKvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t*)full_kv_block_table);
    }

    __aicore__ inline void Process()
    {
        if (blkIdx_ >= tiling_->stageOneUsedCoreNum) {
            return;
        }

        StageOne();
    }

private:
    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    // ============================================================
    // 哈希表字段访问器
    // ============================================================
    // word0: signed record[31:24] | unsigned tokenId[23:0].
    // word1: fingerprint | cacheSlot; independent of the cache record.
    // Token 0xFFFFFF is the invalid sentinel; decode it back to -1.
    __aicore__ inline void StripRecord(LocalTensor<int32_t>& dst,
        LocalTensor<int32_t>& packedIdx, int32_t nums)
    {
        AscendC::ShiftLeft(dst, packedIdx, RECORD_BIT_WIDTH, nums);
        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> tmp = dst.template ReinterpretCast<uint32_t>();
        AscendC::ShiftRight(tmp, tmp, static_cast<uint32_t>(RECORD_BIT_WIDTH), nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackEntry(LocalTensor<int32_t>& index,
        LocalTensor<int32_t>& record, int32_t nums)
    {
        AscendC::ShiftLeft(record, record, RECORD_SHIFT_BITS, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(index.template ReinterpretCast<uint16_t>(),
            index.template ReinterpretCast<uint16_t>(),
            record.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ============================================================
    // 模运算
    // ============================================================

    // 矢量重载 — TempBufferCursor 入参 → PIPE_V
    __aicore__ inline void BitwiseModPowerOfTwo(LocalTensor<int32_t>& vector,
        int32_t modValue, TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> modMaskLocal = cursor.Alloc<int32_t>(nums);
        AscendC::Duplicate(modMaskLocal, static_cast<int32_t>(modValue - 1), nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::And(vector.template ReinterpretCast<uint16_t>(),
            vector.template ReinterpretCast<uint16_t>(),
            modMaskLocal.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeFingerprintFromTopkId(
        LocalTensor<int32_t>& result, LocalTensor<int32_t>& topkId,
        TempBufferCursor cursor, bool copyInput, int32_t nums)
    {
        if (copyInput) {
            AscendC::Adds(result, topkId, 0, nums);
            AscendC::PipeBarrier<PIPE_V>();
        }
        int32_t fingerprintMod = HASH_INDEX_SIGN_BIT / tiling_->statesize;
        BitwiseModPowerOfTwo(result, fingerprintMod, cursor.Fork(), nums);
        AscendC::Muls(result, result,
            static_cast<int32_t>(tiling_->statesize), nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EncodeFingerprintIntoPackedIndex(
        LocalTensor<int32_t>& packedIdx, LocalTensor<int32_t>& topkId,
        TempBufferCursor cursor, int32_t nums)
    {
        LocalTensor<int32_t> fingerprint = cursor.Alloc<int32_t>(nums);
        ComputeFingerprintFromTopkId(
            fingerprint, topkId, cursor.Fork(), true, nums);
        AscendC::Or(packedIdx.template ReinterpretCast<uint16_t>(),
            packedIdx.template ReinterpretCast<uint16_t>(),
            fingerprint.template ReinterpretCast<uint16_t>(),
            nums * sizeof(int32_t) / sizeof(uint16_t));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline int64_t GetActualQSeqLen(int64_t bsIdx)
    {
        return fullQActualSeqGm_.GetValue(bsIdx);
    }

    __aicore__ inline LocalTensor<int32_t> AcquireTopKIndices(int64_t globalOffset)
    {
        if (localRow_ != nullptr) {
            return localRow_->tokens;
        }
        int64_t gmOffset = globalOffset * tiling_->topk;
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selTopKIdxLocal, selTopKIndicesGm_[gmOffset], dataCopyParams, dataCopyPadParams);
        selTopKIdxQue_.EnQue(selTopKIdxLocal);
        return selTopKIdxQue_.DeQue<int32_t>();
    }

    __aicore__ inline void ReleaseTopKIndices(LocalTensor<int32_t>& ids)
    {
        if (localRow_ == nullptr) {
            selTopKIdxQue_.FreeTensor(ids);
        }
    }

    __aicore__ inline void CopyInBlockStatus(int64_t loopIdx, LocalTensor<int32_t>& selBlockStatLocal)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyExtParams dataCopyParamsSt{
            static_cast<uint16_t>(1), static_cast<uint32_t>(selBlockStatUb_), 0, 0, 0};
        DataCopyPad(selBlockStatLocal, selKvBlockStatusGm_[loopIdx * selBlockStatUb_/sizeof(int32_t)], dataCopyParamsSt, dataCopyPadParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void LoadBlockTables(TempBufferCursor& cursor, int32_t globalBatchId,
                                           LocalTensor<int32_t>& localFullBlkTable,
                                           LocalTensor<int32_t>& localSelBlkTable)
    {
        int64_t fullBlkTableSize = CeilAlign(tiling_->fullMaxBlockNum, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        int64_t selBlkTableSize = CeilAlign(tiling_->selMaxBlockNum, static_cast<int64_t>(UB_BLOCK_ALIGN_BYTES));
        localFullBlkTable = cursor.Alloc<int32_t>(fullBlkTableSize);
        localSelBlkTable = cursor.Alloc<int32_t>(selBlkTableSize);

        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyExtParams fullCopyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->fullMaxBlockNum * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(localFullBlkTable,
                    fullKvBlockTableGm_[globalBatchId * tiling_->fullMaxBlockNum],
                    fullCopyParams, padParams);
        DataCopyExtParams selCopyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->selMaxBlockNum * sizeof(int32_t)),
            0, 0, 0};
        DataCopyPad(localSelBlkTable,
                    selKvBlockTableGm_[globalBatchId * tiling_->selMaxBlockNum],
                    selCopyParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void GatherPackedTokenFromHash(LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& hashIdx, LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t nums){

        LocalTensor<int32_t> topkidindices = cursor.Alloc<int32_t>(nums);
        AscendC::Maxs(topkidindices, hashIdx, 0, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(topkidindices, topkidindices, static_cast<int32_t> (HASH_ENTRY_FIELDS * sizeof(int32_t)), nums);
        AscendC::PipeBarrier<PIPE_V>(); // 偏移的单位为Bytes, 每个元素占4个Byte,还得乘4
        AscendC::Gather(hashRecordId, hashMapping, topkidindices.template ReinterpretCast<uint32_t>(), (uint32_t)0, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareEq(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& selTopKIdxLocal, int32_t nums){
        AscendC::Duplicate(result.template ReinterpretCast<int32_t>(), 0, nums/AscendC::AscendCUtils::GetBitSize(sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(result, hashRecordId, selTopKIdxLocal, AscendC::CMPMODE::EQ, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareNe(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, LocalTensor<int32_t>& selTopKIdxLocal, int32_t nums){
        CompareEq(result, hashRecordId, selTopKIdxLocal, nums);
        AscendC::Not(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), nums/AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareEq(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, int32_t src1Scalar, int32_t nums){
        AscendC::Duplicate(result.template ReinterpretCast<int32_t>(), 0, nums/AscendC::AscendCUtils::GetBitSize(sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(result, hashRecordId, src1Scalar, AscendC::CMPMODE::EQ, nums);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CompareNe(LocalTensor<uint8_t>& result, LocalTensor<int32_t>& hashRecordId, int32_t src1Scalar, int32_t nums){
        CompareEq(result, hashRecordId, src1Scalar, nums);
        AscendC::Not(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), nums/AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void IgnoreInvalidTopk(LocalTensor<uint8_t> result, LocalTensor<int32_t>& selTopKIdxLocal, TempBufferCursor cursor, int32_t nums){
        LocalTensor<uint8_t> validMask = cursor.Alloc<uint8_t>(nums);
        CompareNe(validMask, selTopKIdxLocal, INVALID_SLOT, nums);
        AscendC::And(result.template ReinterpretCast<uint16_t>(), result.template ReinterpretCast<uint16_t>(), validMask.template ReinterpretCast<uint16_t>(), nums / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename T>
    __aicore__ inline void Scatter(LocalTensor<T>& value, LocalTensor<int32_t>& indicesLocal,
        LocalTensor<T>& dstLocal, uint64_t validNum)
    {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        if (validNum == 0) {
            return;
        }
        constexpr uint32_t shift = sizeof(T) == sizeof(int64_t) ? 3 : 2;
        auto offsets = indicesLocal.template ReinterpretCast<uint32_t>();
        AscendC::ShiftLeft(offsets, offsets, shift, static_cast<int32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
        // Submit destinations directly to SIMD Scatter in this architecture branch.
        AscendC::Scatter(dstLocal, value, offsets, 0, static_cast<uint32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(offsets, offsets, shift, static_cast<int32_t>(validNum));
        AscendC::PipeBarrier<PIPE_V>();
#else
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t i = 0; i < validNum; ++i) {
            dstLocal.SetValue(indicesLocal.GetValue(i), value.GetValue(i));
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
#endif
    }

    __aicore__ inline void ProcessMissMaskAndUpdateMatchTopkId(
        LocalTensor<uint8_t> mismatchMask, LocalTensor<int32_t>& tokens, LocalTensor<int32_t>& state,
        TempBufferCursor cursor, uint64_t& misses, uint64_t& hits, int32_t count, bool refreshHits)
    {
        LocalTensor<int32_t> slots = cursor.Alloc<int32_t>(count);
        LocalTensor<uint8_t> matchMask = cursor.Alloc<uint8_t>(count);
        ProbeHashTable<AvsgProbeMode::Match>(slots, matchMask, tokens, state, cursor.Fork(), count);

        LocalTensor<int32_t> compact = cursor.Alloc<int32_t>(count);
        uint64_t hitCount = 0;
        GatherByBitMask(compact, matchMask, slots, hitCount, count);
        if (refreshHits && hitCount != 0) {
            RefreshRecord(compact, state, cursor.Fork(), MAX_RECORD_LIFETIME, hitCount);
        }
        hits += hitCount;
        AscendC::Select(tokens.template ReinterpretCast<float>(), matchMask, slots.template ReinterpretCast<float>(),
                        tokens.template ReinterpretCast<float>(), AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
        AscendC::Not(mismatchMask.template ReinterpretCast<uint16_t>(), matchMask.template ReinterpretCast<uint16_t>(),
                     count / AscendC::AscendCUtils::GetBitSize(sizeof(uint16_t)));
        PipeBarrier<PIPE_V>();
        IgnoreInvalidTopk(mismatchMask, tokens, cursor.Fork(), count);
        GatherByBitMask(compact, mismatchMask, tokens, misses, count);
    }

    __aicore__ inline void RefreshMatchStatus(LocalTensor<int32_t>& hashMapping, TempBufferCursor cursor, int32_t globalTOffset, int32_t numSeqs){

        for(int32_t seqId = 0; seqId < numSeqs; seqId++){
            TempBufferCursor chunkCursor = cursor.Fork();
            int32_t globalIndex = (globalTOffset+seqId) * tiling_->headnum;
            LocalTensor<int32_t> selTopKIdLocal = AcquireTopKIndices(globalIndex);

            LocalTensor<uint8_t> matchMask = chunkCursor.Alloc<uint8_t>(tiling_->topk);
            LocalTensor<int32_t> slots = chunkCursor.Alloc<int32_t>(tiling_->topk);
            ProbeHashTable<AvsgProbeMode::Match>(slots, matchMask, selTopKIdLocal, hashMapping,
                                                chunkCursor.Fork(), tiling_->topk);
            uint64_t validNum = 0;
            LocalTensor<int32_t> hitSlots = chunkCursor.Alloc<int32_t>(tiling_->topk);
            GatherByBitMask(hitSlots, matchMask, slots, validNum, tiling_->topk);
            RefreshRecord(hitSlots, hashMapping, chunkCursor.Fork(), MAX_RECORD_LIFETIME, validNum);
            ReleaseTopKIndices(selTopKIdLocal);
        }

    }

    template <AvsgProbeMode mode>
    __aicore__ inline void ProbeHashTable(LocalTensor<int32_t>& indices, LocalTensor<uint8_t> mask,
        LocalTensor<int32_t>& tokens, LocalTensor<int32_t>& state, TempBufferCursor cursor, int32_t count)
    {
        static_assert(AvsgMatchTilingData::topk % AVSG_RECORD_LANES == 0);
        LocalTensor<int32_t> flags = cursor.Alloc<int32_t>(count);
        PipeBarrier<PIPE_V>();
        ProbeAvsgTable<mode>((__ubuf__ uint32_t*)indices.GetPhyAddr(), (__ubuf__ uint32_t*)flags.GetPhyAddr(),
                               (__ubuf__ uint32_t*)tokens.GetPhyAddr(), (__ubuf__ uint32_t*)state.GetPhyAddr(),
                               static_cast<uint16_t>(count / AVSG_RECORD_LANES), tiling_->statesize);
        PipeBarrier<PIPE_V>();
        CompareEq(mask, flags, 1, count);
        IgnoreInvalidTopk(mask, tokens, cursor.Fork(), count);
    }

    __aicore__ inline void GatherByBitMask(LocalTensor<int32_t>& result, LocalTensor<uint8_t>& mask, LocalTensor<int32_t>& src, uint64_t& validNum, int32_t totalNum){
        uint8_t src0BlockStride = 1; // 每个Block内的元素偏移
        uint16_t repeatTimes = 1; //总共totalNum个元素，共需要处理多少次
        uint16_t src0RepeatStride = 1; // 64个元素的 int32_t 是256字节，每个blocksize是32， 256个字节就是占了8个blocks，
        uint8_t src1RepeatStride = 1;//每次处理64个元素, pattern占了64位， 一共是两个blocks
        AscendC::GatherMask(result, src, mask.template ReinterpretCast<uint32_t>(), true,  static_cast<uint32_t>(totalNum), { src0BlockStride, repeatTimes, src0RepeatStride, src1RepeatStride}, validNum);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecrementRecordCounters(LocalTensor<int32_t>& state, int64_t slots)
    {
        PipeBarrier<PIPE_V>();
        ScanAvsgRecords<AvsgRecordScan::Decay>((__ubuf__ int32_t*)state.GetPhyAddr(), nullptr,
                                             static_cast<uint16_t>(slots / AVSG_RECORD_LANES));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RefreshRecord(LocalTensor<int32_t>& emptyIdx, LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, int32_t refreshValue, uint64_t totalNum){
        LocalTensor<int32_t> value = cursor.Alloc<int32_t>(CeilAlign(static_cast<int32_t>(totalNum), SORT_ELEMENTS_PER_REPEAT));
        LocalTensor<int32_t> key = cursor.Alloc<int32_t>(CeilAlign(static_cast<int32_t>(totalNum), SORT_ELEMENTS_PER_REPEAT));
        GatherPackedTokenFromHash(value, emptyIdx, statusLocal, cursor.Fork(), totalNum);
        StripRecord(value, value, totalNum);
        AscendC::Adds(value, value, refreshValue << RECORD_SHIFT_BITS, totalNum); // 刷新record
        // Cache word offset: slot * 2
        AscendC::Muls(key, emptyIdx, static_cast<int32_t>(HASH_ENTRY_FIELDS), totalNum);
        AscendC::PipeBarrier<PIPE_V>();
        Scatter(value, key, statusLocal, totalNum);
    }

    __aicore__ inline void UpdateHashStatus(LocalTensor<int32_t>& topkId, LocalTensor<int32_t>& emptyIdx, LocalTensor<int32_t>& statusLocal, TempBufferCursor cursor, uint64_t uniqueNum, uint64_t totalNum){
        // 计算hash
        LocalTensor<int32_t> hashLocal = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<uint8_t> emptyMask = cursor.Alloc<uint8_t>(totalNum);
        // Only compacted misses participate in insertion. The consumer reads
        // exactly uniqueNum predicates, so rounding the last register is safe.
        const int32_t probeCount = CeilAlign(static_cast<int32_t>(uniqueNum), static_cast<int32_t>(AVSG_RECORD_LANES));
        ProbeHashTable<AvsgProbeMode::Empty>(hashLocal, emptyMask, topkId, statusLocal, cursor.Fork(), probeCount);

        // 构建InterLevel
        uint64_t insertNum = 0;
        LocalTensor<int32_t> insertId = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> insertIdx = cursor.Alloc<int32_t>(totalNum);

        // Cache update touches only word0, leaving the bucket word intact.
        StripRecord(insertIdx, topkId, uniqueNum);
        AscendC::Duplicate(insertId, static_cast<int32_t>(MAX_RECORD_LIFETIME), uniqueNum);
        PipeBarrier<PIPE_V>();
        PackEntry(insertIdx, insertId, uniqueNum);
        AscendC::Muls(insertId, emptyIdx, static_cast<int32_t>(HASH_ENTRY_FIELDS), uniqueNum);
        PipeBarrier<PIPE_V>();
        Scatter(insertIdx, insertId, statusLocal, uniqueNum);

        LocalTensor<int32_t> insertTopkId = cursor.Alloc<int32_t>(totalNum);
        GatherByBitMask(insertTopkId, emptyMask, topkId, insertNum, uniqueNum);
        GatherByBitMask(insertIdx, emptyMask, emptyIdx, insertNum, uniqueNum);
        EncodeFingerprintIntoPackedIndex(insertIdx, insertTopkId, cursor.Fork(), insertNum);
        GatherByBitMask(insertId, emptyMask, hashLocal, insertNum, uniqueNum);
        AscendC::Muls(insertId, insertId, static_cast<int32_t>(HASH_ENTRY_FIELDS), insertNum);
        PipeBarrier<PIPE_V>();
        AscendC::Adds(insertId, insertId, 1, insertNum);
        PipeBarrier<PIPE_V>();
        Scatter(insertIdx, insertId, statusLocal, insertNum);

    }

    template <AvsgRecordScan mode>
    __aicore__ inline uint32_t CollectVictims(LocalTensor<int32_t>& slots, LocalTensor<int32_t>& state,
                                            uint16_t repeats)
    {
        PipeBarrier<PIPE_V>();
        ScanAvsgRecords<mode>((__ubuf__ int32_t*)state.GetPhyAddr(), (__ubuf__ int32_t*)slots.GetPhyAddr(), repeats);
        // Squeeze stores the byte count in AR. Read it before the next VF may reuse AR.
        const uint32_t count = static_cast<uint32_t>(AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t);
        PipeBarrier<PIPE_V>();
        return count;
    }

    __aicore__ inline void CalEmptyIdx(LocalTensor<int32_t>& slots, LocalTensor<int32_t>& state,
                                     uint32_t required, uint32_t& unusedUpperBound)
    {
        const uint16_t repeats = static_cast<uint16_t>(tiling_->statesize / AVSG_RECORD_LANES);
        if (unusedUpperBound >= required) {
            // Until MTP tail cleanup, a row can only consume unused slots.
            unusedUpperBound = CollectVictims<AvsgRecordScan::Unused>(slots, state, repeats);
            if (unusedUpperBound >= required) {
                unusedUpperBound -= required;
                return;
            }
        }
        uint32_t expired = CollectVictims<AvsgRecordScan::Expired>(slots, state, repeats);
        while (expired < required) {
            expired = CollectVictims<AvsgRecordScan::DecayAndSelectExpired>(slots, state, repeats);
        }
    }

    __aicore__ inline void BlockTableLookupVec(
        LocalTensor<int32_t>& idxLocal,
        LocalTensor<int32_t>& localBlkTable,
        LocalTensor<int32_t>& blockTableIdx,
        LocalTensor<int32_t>& blockNumIdx,
        TempBufferCursor cursor,
        int32_t blockSize,
        int32_t blockSizeLog2,
        int32_t maxBlockNum,
        uint64_t count)
    {
        AscendC::Muls(idxLocal, idxLocal, static_cast<int32_t>(tiling_->selTopKBlockSize), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(blockTableIdx, idxLocal, blockSizeLog2, count);
        AscendC::PipeBarrier<PIPE_V>();
        BitwiseModPowerOfTwo(idxLocal, blockSize, cursor.Fork(), count);

        AscendC::Maxs(blockTableIdx, blockTableIdx, 0, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(blockTableIdx, blockTableIdx, maxBlockNum - 1, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(blockTableIdx, blockTableIdx, static_cast<int32_t>(sizeof(int32_t)), count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Gather(blockNumIdx, localBlkTable,
                        blockTableIdx.template ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(0), count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(blockNumIdx, blockNumIdx, blockSize, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(idxLocal, idxLocal, blockNumIdx, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeCopyMappingVec(
        LocalTensor<int32_t>& missTopkId,
        LocalTensor<int32_t>& emptyIdxLocal,
        LocalTensor<int32_t>& localFullBlkTable,
        LocalTensor<int32_t>& localSelBlkTable,
        TempBufferCursor cursor,
        uint64_t mismatchNum,
        int64_t totalNum)
    {
        LocalTensor<int32_t> blockTableIdx = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> blockNumIdx = cursor.Alloc<int32_t>(totalNum);

        BlockTableLookupVec(missTopkId, localFullBlkTable, blockTableIdx, blockNumIdx, cursor,
            static_cast<int32_t>(tiling_->fullKvBlockSize),
            static_cast<int32_t>(tiling_->fullKvBlockSizeLog2),
            static_cast<int32_t>(tiling_->fullMaxBlockNum),
            mismatchNum);

        BlockTableLookupVec(emptyIdxLocal, localSelBlkTable, blockTableIdx, blockNumIdx, cursor,
            static_cast<int32_t>(tiling_->selKvBlockSize),
            static_cast<int32_t>(tiling_->selKvBlockSizeLog2),
            static_cast<int32_t>(tiling_->selMaxBlockNum),
            mismatchNum);
    }

    __aicore__ inline void ProcessMisMatchTopkId(LocalTensor<uint8_t>& mismatchMask, LocalTensor<int32_t>& selTopKIdLocal, LocalTensor<int32_t>& statusLocal, LocalTensor<int32_t>& emptyIdxLocal, LocalTensor<int32_t>& localFullBlkTable, LocalTensor<int32_t>& localSelBlkTable, TempBufferCursor cursor, int64_t bshIdx, uint64_t mismatchNum, int64_t totalNum){

        LocalTensor<int32_t> missTopkId = cursor.Alloc<int32_t>(totalNum);
        LocalTensor<int32_t> missTopkIdx = cursor.Current();
        PipeBarrier<PIPE_ALL>();
        AscendC::CreateVecIndex(missTopkId, 0, totalNum);
        PipeBarrier<PIPE_V>();

        GatherByBitMask(missTopkIdx, mismatchMask, missTopkId, mismatchNum, totalNum);
        GatherByBitMask(missTopkId, mismatchMask, selTopKIdLocal, mismatchNum, totalNum);

        // UpdateSelTopkIdx
        Scatter(emptyIdxLocal, missTopkIdx, selTopKIdLocal, mismatchNum); // missTopKIdx用完释放
        // UpdateHashMapping
        UpdateHashStatus(missTopkId, emptyIdxLocal, statusLocal, cursor.Fork(), mismatchNum, totalNum);

        ComputeCopyMappingVec(missTopkId, emptyIdxLocal, localFullBlkTable, localSelBlkTable, cursor.Fork(), mismatchNum, totalNum);

        // Export physical token indices using block sizes supplied by host tiling.
        LocalTensor<int32_t> topkIdToIdxMap = cursor.Alloc<int32_t>(offsetOutputUb_ / sizeof(int32_t));
        PipeBarrier<PIPE_V>();
        WriteAvsgPlanPairs((__ubuf__ int32_t*)topkIdToIdxMap.GetPhyAddr(), (__ubuf__ int32_t*)missTopkId.GetPhyAddr(),
                           (__ubuf__ int32_t*)emptyIdxLocal.GetPhyAddr(), static_cast<uint32_t>(mismatchNum),
                           static_cast<uint16_t>(totalNum / AVSG_RECORD_LANES));

        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        CopyOutStageOneCopyMapping(bshIdx, topkIdToIdxMap);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void ClearMtpTokens(LocalTensor<int32_t>& state, int32_t firstInvalid, int64_t slots)
    {
        PipeBarrier<PIPE_V>();
        InvalidateAvsgTail((__ubuf__ int32_t*)state.GetPhyAddr(), static_cast<uint16_t>(slots / AVSG_RECORD_LANES),
                           firstInvalid);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StageOne(){

        LocalTensor<int32_t> selKvActSeqLocal = workBuf_.Get<int32_t>();
        freeBufLocal_ = selKvActSeqLocal[selKvActSeqUb_ / sizeof(int32_t)];
        TempBufferCursor rootCursor(freeBufLocal_, tmpbuffUb_); // Per-core scratch cursor.

        AscendC::Duplicate(selKvActSeqLocal, 0, selKvActSeqUb_ / sizeof(int32_t));
        AscendC::Duplicate(freeBufLocal_, 0, tmpbuffUb_ / sizeof(int32_t));
        PipeBarrier<PIPE_ALL>();

        // 在 batch 循环前，一次性将当前核负责的元素范围
        int32_t firstBatchId = blkIdx_ * tiling_->stageOneMainCoreBsLoopNum;
        int32_t lastBatchId = firstBatchId + stageOneLoopNum_ - 1;
        int64_t gmStart = (firstBatchId == 0) ? 0 : GetActualQSeqLen(firstBatchId - 1) * tiling_->headnum;
        int64_t gmEnd = GetActualQSeqLen(lastBatchId) * tiling_->headnum;
        LocalTensor<int32_t> mismatchNumLocal = rootCursor.Alloc<int32_t>(gmEnd - gmStart);

        int64_t curFullKvSeqLen = 0;
        int64_t curFullQSeqLen = 0;

        for (int64_t loopIdx = 0; loopIdx < stageOneLoopNum_; loopIdx++) {

            curFullKvSeqLen = fullKvActualSeqGm_.GetValue(loopIdx);

            int32_t globalBatchId = (blkIdx_ * tiling_->stageOneMainCoreBsLoopNum + loopIdx);

            int64_t cumQSeqLen = GetActualQSeqLen(globalBatchId);
            int64_t tOffset = (globalBatchId == 0) ? 0 : GetActualQSeqLen(globalBatchId - 1);
            curFullQSeqLen = cumQSeqLen - tOffset;
            int64_t curFullKvSeqModify = curFullKvSeqLen - curFullQSeqLen;

            TempBufferCursor cursor = rootCursor.Fork(); // 每个batch重新创建cursor

            LocalTensor<int32_t> statusLocal = cursor.Alloc<int32_t>(selBlockStatUb_ / sizeof(int32_t));
            CopyInBlockStatus(loopIdx, statusLocal);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            DecrementRecordCounters(statusLocal, tiling_->statesize);

            // Reserve later rows first. Row zero reserves its own hits while looking
            // up output slots, before any eviction. Reservation changes only lifetimes.
            if (curFullQSeqLen > 1) {
                RefreshMatchStatus(statusLocal, cursor.Fork(), tOffset + 1, curFullQSeqLen - 1);
            }

            // block table 按 batch 索引，对本 batch 所有 seq/head 相同，循环外加载一次即可
            LocalTensor<int32_t> localFullBlkTable, localSelBlkTable;
            LoadBlockTables(cursor, globalBatchId, localFullBlkTable, localSelBlkTable);

            uint32_t unusedSlotUpperBound = static_cast<uint32_t>(tiling_->statesize);
            uint64_t reusedNum = 0;
            for (int64_t seqIdx = 0; seqIdx < curFullQSeqLen; seqIdx++) {
                for (int64_t headIdx = 0; headIdx < tiling_->headnum; headIdx++) {
                    TempBufferCursor seqCursor = cursor.Fork();

                    int32_t globalIndex = (tOffset + seqIdx) * tiling_->headnum + headIdx;
                    LocalTensor<int32_t> selTopKIdLocal = AcquireTopKIndices(globalIndex);

                    uint64_t mismatchNum = 0;
                    LocalTensor<uint8_t> mismatchMask = seqCursor.Alloc<uint8_t>(tiling_->topk);
                    ProcessMissMaskAndUpdateMatchTopkId(mismatchMask, selTopKIdLocal, statusLocal, seqCursor.Fork(), mismatchNum, reusedNum, tiling_->topk, seqIdx == 0);

                    if (mismatchNum > 0) {
                        // The scan may emit every slot. Only its first K entries stay
                        // live while subsequent miss processing reuses the rest of scratch.
                        auto victimCursor = seqCursor.Fork();
                        LocalTensor<int32_t> emptyIdxLocal = seqCursor.Alloc<int32_t>(tiling_->topk);
                        if (curFullKvSeqLen > poolSize_){
                            auto candidates = victimCursor.Alloc<int32_t>(tiling_->statesize);
                            CalEmptyIdx(candidates, statusLocal, static_cast<uint32_t>(mismatchNum), unusedSlotUpperBound);
                        }
                        else{
                            GatherByBitMask(emptyIdxLocal, mismatchMask, selTopKIdLocal, mismatchNum, tiling_->topk);
                        }

                        ProcessMisMatchTopkId(mismatchMask, selTopKIdLocal, statusLocal, emptyIdxLocal, localFullBlkTable, localSelBlkTable, seqCursor.Fork(), globalIndex, mismatchNum, tiling_->topk);
                    } else {
                        // Miss handling already writes a full row padded with -1.
                        WriteEmptyCopyMapping(seqCursor.Fork(), globalIndex);
                    }
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    mismatchNumLocal.SetValue(globalIndex - gmStart, static_cast<int32_t>(mismatchNum));
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    CopyOutTopKIndices(globalIndex, selTopKIdLocal);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                    ReleaseTopKIndices(selTopKIdLocal);
                }
            }
            selKvActSeqLocal.SetValue(loopIdx, reusedNum);
            ClearMtpTokens(statusLocal, curFullKvSeqModify + 1, tiling_->statesize);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            CopyOutState(loopIdx, statusLocal);
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        CopyOutMismatchNum(gmStart, mismatchNumLocal, (gmEnd - gmStart) * sizeof(int32_t));
        CopyOutSelKvActSeq(selKvActSeqLocal);
    }

    __aicore__ inline void CopyOutState(int64_t loopIdx, LocalTensor<int32_t>& writeOutBlockStatLocal){
        DataCopyExtParams dataCopyParBlkStat{
        static_cast<uint16_t>(1), static_cast<uint32_t>(selBlockStatUb_), 0, 0, 0};
        DataCopyPad(selKvBlockStatusOutGm_[loopIdx * selBlockStatUb_ / sizeof(int32_t)], writeOutBlockStatLocal, dataCopyParBlkStat);
    }

    __aicore__ inline void CopyOutSelKvActSeq(LocalTensor<int32_t>& selKvActSeqLocal){
        DataCopyExtParams dataCopyParSeqLen{
            static_cast<uint16_t>(1), static_cast<uint32_t>(stageOneLoopNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selKvActualSeqGm_, selKvActSeqLocal, dataCopyParSeqLen);
    }

    __aicore__ inline void CopyOutMismatchNum(int64_t gmStart, LocalTensor<int32_t>& mismatchNumUb, int64_t ubBytes){
        DataCopyExtParams dataCopyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(ubBytes), 0, 0, 0};
        DataCopyPad(mismatchNumArrayGm_[gmStart], mismatchNumUb, dataCopyParams);
    }

    __aicore__ inline void CopyOutStageOneCopyMapping(int64_t bshIdx, LocalTensor<int32_t>& stageOneAddressMappingLocal){
        int64_t gmOffset = bshIdx * offsetOutputUb_ / sizeof(int32_t);
        DataCopyExtParams dataCopyParAddrMapping{
        static_cast<uint16_t>(1), static_cast<uint32_t>(offsetOutputUb_), 0, 0, 0};
        DataCopyPad(stageOneCopyMappingGM_[gmOffset], stageOneAddressMappingLocal, dataCopyParAddrMapping);
    }

    __aicore__ inline void WriteEmptyCopyMapping(TempBufferCursor cursor, int64_t row)
    {
        int64_t words = offsetOutputUb_ / sizeof(int32_t);
        LocalTensor<int32_t> mapping = cursor.Alloc<int32_t>(words);
        AscendC::Duplicate(mapping, INVALID_SLOT, words);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        CopyOutStageOneCopyMapping(row, mapping);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void CopyOutTopKIndices(int64_t globalOffset, LocalTensor<int32_t>& selTopKIdLocal){
        int64_t gmOffset = globalOffset * tiling_->topk;
        DataCopyExtParams dataCopyTopkIdx{
            static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(selTopKIndicesGm_[gmOffset], selTopKIdLocal, dataCopyTopkIdx);

    }

private:
    TPipe* pipe_;
    const AvsgMatchTilingData* tiling_;
    avsg_fusion::LocalTopkRow* localRow_;

    // Stage One
    int32_t stageOneLoopNum_ = 0;

    int32_t blkIdx_ = -1;

    int32_t selTopKIdxUb_ = 0;
    int32_t selBlockStatUb_ = 0;
    int32_t selKvActSeqUb_ = 0;
    int32_t tmpbuffUb_ = 0;
    int32_t offsetOutputUb_ = 0;
    int32_t poolSize_ = tiling_->selMaxBlockNum * tiling_->selKvBlockSize;

    constexpr static int64_t HASH_ENTRY_FIELDS = 2;
    constexpr static int64_t MAX_RECORD_LIFETIME = 8; // (0~127)
    constexpr static int32_t HASH_INDEX_MASK = 0x00FFFFFF;
    constexpr static int32_t HASH_INDEX_SIGN_BIT = 0x00800000;

    // 哈希表 entry 位域常量 — 统一管理 record 位宽与偏移
    // entry: [record:8 | tokenId:24], [fingerprint | cacheSlot].
    // Reinitialize persisted tables when migrating from the old layout.
    int32_t RECORD_BIT_WIDTH = AscendC::AscendCUtils::GetBitSize(sizeof(int8_t));    // 8
    int32_t RECORD_SHIFT_BITS = AscendC::AscendCUtils::GetBitSize(sizeof(int32_t))   // 32
                                - RECORD_BIT_WIDTH;                                    // = 24

    GlobalTensor<int32_t> selKvBlockTableGm_;
    GlobalTensor<int32_t> selKvBlockStatusGm_;
    GlobalTensor<int32_t> selKvBlockStatusOutGm_;
    GlobalTensor<int32_t> selTopKIndicesGm_;
    GlobalTensor<int32_t> fullKvBlockTableGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> selKvActualSeqGm_;

    GlobalTensor<int32_t> stageOneCopyMappingGM_;
    GlobalTensor<int32_t> mismatchNumArrayGm_;

    LocalTensor<int32_t> freeBufLocal_;

    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TBuf<QuePosition::VECCALC> workBuf_;

};

} // namespace AvsgMatchNs
#endif // AVSG_MATCH_SPLIT_BS_REUSE_VEC_H
