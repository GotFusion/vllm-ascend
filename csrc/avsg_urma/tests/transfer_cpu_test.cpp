/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cassert>
#include <iostream>
#include "ops/hixl_kernel/aiv/hixl_aiv_launch.h"

extern "C" void LaunchAvsgTransfer(void *, uint64_t, uint64_t, uint64_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, uint64_t, uint64_t);
static unsigned calls = 0;
extern "C" void HixlLaunchAivSimt(void *stream, const HixlAivOpParam *p, uint32_t blocks)
{
    ++calls;
    assert(stream == reinterpret_cast<void *>(0x1234));
    assert(p->channel == 0x1000 && p->op_desc_list_addr == 0x2000);
    const uint64_t error = static_cast<uint64_t>(p->reserved[0]) |
                           (static_cast<uint64_t>(p->reserved[1]) << 32U);
    assert(error == 0xfedcba9876543210ULL);
    assert(p->list_num == 194 && p->is_get == 1 && p->mem_type == HIXL_AIV_MEM_TYPE_URMA);
    const bool shared = calls == 2;
    assert(p->compute_mode == 1 && p->reserved[2] == (shared ? 4U : 3U));
    assert(blocks == (shared ? 8U : 3U) && p->reserved[5] == 32);
    assert(p->reserved[6] == (shared ? 1U : 0U));
    const uint64_t timing = p->reserved[3] | (static_cast<uint64_t>(p->reserved[4]) << 32U);
    const uint64_t sync = p->reserved[7] | (static_cast<uint64_t>(p->reserved[8]) << 32U);
    assert(timing == (shared ? 0xf123456789abc000ULL : 0ULL));
    assert(sync == (shared ? 0xf987654321abc000ULL : 0ULL));
    for (unsigned i = 9; i < 20; ++i) {
        assert(p->reserved[i] == 0);
    }
}
int main()
{
    LaunchAvsgTransfer(reinterpret_cast<void *>(0x1234), 0x1000, 0x2000, 0xfedcba9876543210ULL,
                       194, 3, 3, 32, 0, 0);
    LaunchAvsgTransfer(reinterpret_cast<void *>(0x1234), 0x1000, 0x2000, 0xfedcba9876543210ULL,
                       194, 4, 8, 32, 0xf987654321abc000ULL, 0xf123456789abc000ULL);
    assert(calls == 2);
    std::cout << "PASS: real Host adapter forwards channel table, list, READ mode and launch configuration\n";
}
