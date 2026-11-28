/*
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * See ../../LICENSE.
 */
#ifndef VLLM_ASCEND_AVSG_MATCH_VECTOR_H
#define VLLM_ASCEND_AVSG_MATCH_VECTOR_H

#include "kernel_operator.h"

namespace AvsgMatchNs {
constexpr uint32_t AVSG_RECORD_LANES = 64;
constexpr uint32_t AVSG_WORDS_PER_RECORD = 2;
constexpr int32_t AVSG_TOKEN_MASK = 0x00ffffff;

// Each register pair spans 64 complete {packed token, hash entry} records.
// Both supported capacities are exact multiples of the vector width.
__simd_vf__ inline void DecayAvsgRecords(__ubuf__ int32_t *table, uint16_t repeats)
{
    using namespace AscendC;
    Reg::MaskReg all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<int32_t> tokenMask, minAge, minPackedAge;
    Reg::Duplicate(tokenMask, AVSG_TOKEN_MASK);
    Reg::Duplicate(minAge, -2);
    Reg::Duplicate(minPackedAge, -2 * (1 << 24));

    for (uint16_t i = 0; i < repeats; ++i) {
        __ubuf__ int32_t *record = table + i * AVSG_RECORD_LANES * AVSG_WORDS_PER_RECORD;
        Reg::RegTensor<int32_t> packed, bucket, age, minBucket;
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_DINTLV_B32>(packed, bucket, record);
        Reg::ShiftRights(age, packed, static_cast<int16_t>(24), all);
        Reg::Adds(age, age, -1, all);
        Reg::Max(age, age, minAge, all);
        Reg::ShiftLefts(age, age, static_cast<int16_t>(24), all);
        Reg::And(packed, packed, tokenMask, all);
        Reg::Or(packed, packed, age, all);
        // The recipes implementation also clamps the non-decremented high
        // byte of word 1. Preserve the low 24 bits for every input bit pattern.
        Reg::And(minBucket, bucket, tokenMask, all);
        Reg::Or(minBucket, minBucket, minPackedAge, all);
        Reg::Max(bucket, bucket, minBucket, all);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_INTLV_B32>(record, packed, bucket, all);
    }
}

__simd_vf__ inline void InvalidateAvsgTail(__ubuf__ int32_t *table, uint16_t repeats, int32_t firstInvalid)
{
    using namespace AscendC;
    Reg::MaskReg all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<int32_t> tokenMask, bound;
    Reg::Duplicate(tokenMask, AVSG_TOKEN_MASK);
    Reg::Duplicate(bound, firstInvalid);

    for (uint16_t i = 0; i < repeats; ++i) {
        __ubuf__ int32_t *record = table + i * AVSG_RECORD_LANES * AVSG_WORDS_PER_RECORD;
        Reg::RegTensor<int32_t> packed, bucket, tokenPlusOne;
        Reg::MaskReg pastEnd;
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_DINTLV_B32>(packed, bucket, record);
        // The low 24 bits of (word + 1) are token + 1, or zero for an
        // already empty slot. Preserve that slot's original lifetime byte.
        Reg::And(tokenPlusOne, packed, tokenMask, all);
        Reg::Adds(tokenPlusOne, tokenPlusOne, 1, all);
        Reg::And(tokenPlusOne, tokenPlusOne, tokenMask, all);
        Reg::Compare<int32_t, CMPMODE::GT>(pastEnd, tokenPlusOne, bound, all);
        Reg::Select(packed, tokenMask, packed, pastEnd);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_INTLV_B32>(record, packed, bucket, all);
    }
}
enum class AvsgProbeMode { Match, Empty };
constexpr uint32_t AVSG_HASH_MULTIPLIER = 2039;
constexpr uint32_t AVSG_HASH_STRIDE_MULTIPLIER = 7;
constexpr uint16_t AVSG_HASH_PROBES = 4;

// Compute four bounded double-hash probes per token. Only read the table here:
// insertion still scatters all winners together after selecting their buckets.
template <AvsgProbeMode mode>
__simd_vf__ inline void ProbeAvsgBuckets(__ubuf__ uint32_t *buckets, __ubuf__ uint32_t *flags,
                                         __ubuf__ uint32_t *tokens, __ubuf__ uint32_t *table,
                                         uint16_t repeats, uint32_t capacity)
{
    using namespace AscendC;
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> slotMask, tokenMask, strideMask, fingerprintMask, fingerprintBits, zero, one;
    Reg::Duplicate(slotMask, capacity - 1);
    Reg::Duplicate(strideMask, capacity - 2);
    Reg::Duplicate(tokenMask, static_cast<uint32_t>(AVSG_TOKEN_MASK));
    Reg::Duplicate(zero, 0u);
    Reg::Duplicate(one, 1u);
    if constexpr (mode == AvsgProbeMode::Empty) {
        Reg::Duplicate(fingerprintBits, 0x007fffffu);
        Reg::Duplicate(fingerprintMask, static_cast<uint32_t>(AVSG_TOKEN_MASK) & ~(capacity - 1));
    }
    for (uint16_t i = 0; i < repeats; ++i) {
        Reg::RegTensor<uint32_t> token, bucket, stride;
        Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_NORM>(token, tokens + i * AVSG_RECORD_LANES);
        Reg::Muls(bucket, token, AVSG_HASH_MULTIPLIER, all);
        Reg::And(bucket, bucket, slotMask, all);
        Reg::Muls(stride, token, AVSG_HASH_STRIDE_MULTIPLIER, all);
        Reg::And(stride, stride, strideMask, all);
        Reg::Or(stride, stride, one, all);
        Reg::MaskReg found;
        for (uint16_t probe = 0; probe < AVSG_HASH_PROBES; ++probe) {
            Reg::RegTensor<uint32_t> offset, entry, packed, value;
            Reg::ShiftLefts(offset, bucket, static_cast<int16_t>(1), all);
            Reg::Adds(offset, offset, 1u, all);
            Reg::Gather(entry, table, offset, all);
            Reg::And(offset, entry, slotMask, all);
            Reg::ShiftLefts(offset, offset, static_cast<int16_t>(1), all);
            Reg::Gather(packed, table, offset, all);
            if constexpr (mode == AvsgProbeMode::Match) {
                // Decode the reserved token to UINT32_MAX before comparing.
                Reg::Adds(value, packed, 1u, all);
                Reg::And(value, value, tokenMask, all);
                Reg::Sub(value, value, one, all);
                Reg::Compare<uint32_t, CMPMODE::EQ>(found, value, token, all);
            } else {
                // A stale fingerprint or an expired resident allows insertion.
                // Only bits 0..22 of token * capacity form a fingerprint.
                Reg::RegTensor<uint32_t> expired;
                Reg::Muls(value, packed, capacity, all);
                Reg::And(value, value, fingerprintBits, all);
                Reg::And(entry, entry, fingerprintMask, all);
                Reg::Xor(value, value, entry, all);
                Reg::ShiftRights(expired, packed, static_cast<int16_t>(31), all);
                Reg::Or(value, value, expired, all);
                Reg::Compare<uint32_t, CMPMODE::NE>(found, value, zero, all);
            }
            if (probe + 1 < AVSG_HASH_PROBES) {
                Reg::Add(offset, bucket, stride, all);
                Reg::And(offset, offset, slotMask, all);
                Reg::Select(bucket, bucket, offset, found);
            }
        }
        Reg::RegTensor<uint32_t> valid;
        Reg::Select(valid, one, zero, found);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(buckets + i * AVSG_RECORD_LANES, bucket, all);
        Reg::StoreAlign<uint32_t, Reg::StoreDist::DIST_NORM>(flags + i * AVSG_RECORD_LANES, valid, all);
    }
}

// Each output pair is {physical source, physical destination}. The public
// plan includes all K pairs, with -1 in both words of every unused pair.
__simd_vf__ inline void WriteAvsgPlanPairs(__ubuf__ int32_t *out, __ubuf__ int32_t *source,
                                            __ubuf__ int32_t *destination, uint32_t count, uint16_t chunks)
{
    using namespace AscendC;
    Reg::RegTensor<int32_t> src, dst, empty;
    Reg::MaskReg all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(empty, -1);
    uint16_t full = static_cast<uint16_t>(count / AVSG_RECORD_LANES);
    uint16_t i = 0;
    for (; i < full; ++i) {
        const uint32_t offset = i * AVSG_RECORD_LANES;
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(src, source + offset);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(dst, destination + offset);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_INTLV_B32>(out + 2 * offset, src, dst, all);
    }
    uint32_t tail = count % AVSG_RECORD_LANES;
    if (tail != 0) {
        const uint32_t offset = i * AVSG_RECORD_LANES;
        Reg::MaskReg valid = Reg::UpdateMask<int32_t>(tail);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(src, source + offset);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_NORM>(dst, destination + offset);
        Reg::Select(src, src, empty, valid);
        Reg::Select(dst, dst, empty, valid);
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_INTLV_B32>(out + 2 * offset, src, dst, all);
        ++i;
    }
    for (; i < chunks; ++i) {
        const uint32_t offset = i * AVSG_RECORD_LANES;
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_INTLV_B32>(out + 2 * offset, empty, empty, all);
    }
}
} // namespace AvsgMatchNs
#endif // VLLM_ASCEND_AVSG_MATCH_VECTOR_H
