// CPU model of the actual SIMD builder's integer/mask operations, not ASC ISA.
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#define __aicore__
#define __simd_callee__
#define __ubuf__
#define __VEC_SCOPE__
namespace AscendC {
enum class CMPMODE { EQ };
enum class HcommUrmaOpCode : uint64_t { WRITE = 3 };
struct Config { uint32_t odr = 5, fence = 1, se = 0; };
constexpr Config URMA_DEFAULT_CFG{};
struct UbcBatchHandle {
    struct { struct { struct { uint32_t sqDepth = 32768; } ubJfs; } contextInfo; } sqContext;
    struct { uint32_t tpId = 123, tokenId = 456; uint64_t remoteEidLow = 789, remoteEidHigh = 987, tokenValue = 654; } remoteInfo;
};
namespace Reg {
template<class T> struct RegTensor { std::array<T, 32> data{}; };
struct MaskReg { std::array<bool, 32> data{}; };
inline void Arange(RegTensor<int64_t>& dst, int64_t first) { for (unsigned i = 0; i < 32; ++i) dst.data[i] = first + i; }
template<class T> MaskReg UpdateMask(uint32_t& n) { MaskReg m; for (unsigned i = 0; i < 32; ++i) m.data[i] = i < n; return m; }
template<class T, class U> void Duplicate(RegTensor<T>& d, U x, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d.data[i] = static_cast<T>(x); }
template<class T, class U> void Adds(RegTensor<T>& d, const RegTensor<T>& a, U b, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d.data[i] = a.data[i] + static_cast<T>(b); }
template<class T, class U> void Muls(RegTensor<T>& d, const RegTensor<T>& a, U b, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d.data[i] = a.data[i] * static_cast<T>(b); }
template<class T> void And(RegTensor<T>& d, const RegTensor<T>& a, const RegTensor<T>& b, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d.data[i] = a.data[i] & b.data[i]; }
template<class T, CMPMODE> void Compare(MaskReg& d, const RegTensor<T>& a, const RegTensor<T>& b, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) d.data[i] = m.data[i] && a.data[i] == b.data[i]; }
template<class T> void Select(RegTensor<T>& d, const RegTensor<T>& a, const RegTensor<T>& b, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) d.data[i] = m.data[i] ? a.data[i] : b.data[i]; }
template<class T, class I> void Scatter(T* d, const RegTensor<T>& a, const RegTensor<I>& ids, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d[ids.data[i]] = a.data[i]; }
template<class T, class I> void Gather(RegTensor<T>& d, T* a, const RegTensor<I>& ids, const MaskReg& m) { for (unsigned i = 0; i < 32; ++i) if (m.data[i]) d.data[i] = a[ids.data[i]]; }
}
}
#include "vector_builder.h"
int main()
{
    // uint64_t is unsigned long on LP64 Linux, unsigned long long on this Mac.
    // Explicitly exercise unlike scalar/register types on either platform.
    uint32_t one = 1;
    auto mask = AscendC::Reg::UpdateMask<unsigned long>(one);
    AscendC::Reg::RegTensor<unsigned long> mixed;
    AscendC::Reg::Duplicate(mixed, 1ULL << 31U, mask);
    AscendC::Reg::Adds(mixed, mixed, 3ULL, mask);
    assert(mixed.data[0] == (1UL << 31U) + 3UL);
    AscendC::UbcBatchHandle batch;
    constexpr uint64_t local = 0x120000040000ULL, remote = 0x120000080000ULL;
    for (unsigned sges : {1U, 3U, 5U, 13U}) for (unsigned count : {1U, 7U, 31U, 32U})
    for (unsigned head : {0U, 32764U, 65532U}) for (bool signal : {false, true}) for (bool every : {false, true}) {
        const auto bb = (48 + 16 * sges + 63) / 64;
        const auto words = bb * 8;
        std::array<uint64_t, 32 * 32> output;
        std::array<uint64_t, 32 * 13> indices;
        output.fill(UINT64_MAX);
        for (unsigned i = 0; i < indices.size(); ++i) indices[i] = (i * 7919 + 7) % 20000;
        kv_gather_aiv::DispatchVectorWqes(output.data(), indices.data(), count, sges, head,
                                         local, remote, 512, signal, batch, every);
        for (unsigned lane = 0; lane < count; ++lane) {
            const bool cqe = every || (signal && lane + 1 == count);
            const uint64_t owner = ((head + lane * bb) & 32768U) == 0 ? 1ULL << 31 : 0;
            assert(output[lane * words] == ((13ULL << 16) | (1ULL << 28) | (1ULL << 29) |
                   (3ULL << 40) | owner | (cqe ? 1ULL << 21 : 0)));
            assert(output[lane * words + 1] == (123ULL | (uint64_t{sges} << 24) | (456ULL << 32)));
            assert(output[lane * words + 2] == 789 && output[lane * words + 3] == 987 && output[lane * words + 4] == 654);
            assert(output[lane * words + 5] == remote + uint64_t{lane} * sges * 512);
            for (unsigned sge = 0; sge < sges; ++sge) {
                assert(output[lane * words + 6 + sge * 2] == 512);
                assert(output[lane * words + 7 + sge * 2] == local + indices[lane * sges + sge] * 512);
            }
            for (unsigned word = 6 + sges * 2; word < words; ++word) assert(output[lane * words + word] == 0);
        }
        for (unsigned word = count * words; word < output.size(); ++word) assert(output[word] == UINT64_MAX);
    }
    std::puts("vector_builder_test PASS (CPU integer/mask model; ASC code generation unmodeled)");
}
