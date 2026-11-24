/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * CPU simulation only; these definitions are never part of a device build.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_callee__
#define __simt_vf__
#define __launch_bounds__(n)
struct ThreadIndex { uint32_t x = 0, y = 0, z = 0; };
inline ThreadIndex threadIdx;
inline void st_dev(uint64_t value, uint64_t *dst, uint32_t) { std::memcpy(dst, &value, sizeof(value)); }
namespace AscendC {
enum class TPosition { VECCALC };
enum class HardEvent { V_MTE3, MTE3_S, S_MTE3 };
struct CopyRecord { uintptr_t destination; uint32_t bytes; HardEvent producer; };
inline std::vector<CopyRecord> copies;
inline uint32_t copyPhase = 0;
inline HardEvent copyProducer = HardEvent::V_MTE3;
template <HardEvent event> inline void SetFlag(uint32_t id)
{
    assert(id == (event == HardEvent::V_MTE3 ? 0U : event == HardEvent::S_MTE3 ? 2U : 1U));
    assert(copyPhase == (event == HardEvent::MTE3_S ? 3U : 0U));
    if (event != HardEvent::MTE3_S) copyProducer = event;
    ++copyPhase;
}
template <HardEvent event> inline void WaitFlag(uint32_t)
{
    assert(copyPhase == (event == HardEvent::MTE3_S ? 4U : 1U));
    copyPhase = event == HardEvent::MTE3_S ? 0U : 2U;
}
template <typename T> struct LocalTensor {
    T *data;
    T *GetPhyAddr() const { return data; }
    void SetValue(uint32_t index, T value) { assert(copyPhase == 0); data[index] = value; }
};
template <TPosition> struct TBuf {
    std::vector<uint64_t> storage;
    template <typename T> LocalTensor<T> Get() { return {reinterpret_cast<T *>(storage.data())}; }
};
struct TPipe {
    template <TPosition position> void InitBuffer(TBuf<position> &buffer, uint32_t bytes)
    { buffer.storage.resize((bytes + 7) / 8); }
};
template <typename T> struct GlobalTensor {
    T *data;
    void SetGlobalBuffer(T *ptr) { data = ptr; }
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen, srcStride, dstStride, rsv;
};
template <typename T> inline void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src,
                                             const DataCopyExtParams &params)
{
    assert(copyPhase == 2 && params.blockCount == 1 && params.srcStride == 0 && params.dstStride == 0);
    copies.push_back({reinterpret_cast<uintptr_t>(dst.data), params.blockLen, copyProducer});
    std::memcpy(dst.data, src.data, params.blockLen);
    copyPhase = 3;
}
}
struct ulonglong2 { uint64_t x, y; };
inline ulonglong2 make_ulonglong2(uint64_t x, uint64_t y) { return {x, y}; }
inline void asc_stwt(ulonglong2 *dst, ulonglong2 value) { std::memcpy(dst, &value, sizeof(value)); }
inline void asc_stwt(uint64_t *dst, uint64_t value) { std::memcpy(dst, &value, sizeof(value)); }
inline void asc_threadfence() {}
inline void asc_dcci_single(void *) {}
