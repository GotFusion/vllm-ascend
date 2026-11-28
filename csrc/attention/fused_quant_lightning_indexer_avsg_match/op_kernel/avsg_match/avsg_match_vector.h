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
} // namespace AvsgMatchNs
#endif // VLLM_ASCEND_AVSG_MATCH_VECTOR_H
