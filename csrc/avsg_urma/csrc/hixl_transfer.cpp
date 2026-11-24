/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <cstddef>
#include "ops/hixl_kernel/aiv/hixl_aiv_launch.h"

// avsg/a11ce32 uses the 112-byte reserved[] ABI, not the named-field ABI
// from other development branches. Fail clearly if a different header wins.
static_assert(sizeof(HixlAivOpParam) == 112 && offsetof(HixlAivOpParam, reserved) == 32,
              "AVSG requires HIXL avsg's 112-byte reserved[20] launcher ABI");

// C++/Torch boundary uses only scalar types. HIXL's launcher and kernel are
// compiled from HIXL_SOURCE_DIR, keeping its parameter ABI in one source tree.
extern "C" void LaunchAvsgTransfer(void *stream, uint64_t channels, uint64_t descriptors,
                                   uint64_t error, uint32_t count, uint32_t jettys, uint32_t cores,
                                   uint32_t lanes, uint64_t sync, uint64_t timing)
{
    HixlAivOpParam param{};
    param.channel = channels;
    param.op_desc_list_addr = descriptors;
    param.list_num = count;
    param.mem_type = HIXL_AIV_MEM_TYPE_URMA;
    param.is_get = 1;
    param.compute_mode = 1; // hixl::AIV_COMPUTE_SIMT; avoid GE-bearing hixl_types.h here.
    param.reserved[0] = static_cast<uint32_t>(error);
    param.reserved[1] = static_cast<uint32_t>(error >> 32U);
    param.reserved[2] = jettys;
    param.reserved[3] = static_cast<uint32_t>(timing);
    param.reserved[4] = static_cast<uint32_t>(timing >> 32U);
    param.reserved[5] = lanes;
    param.reserved[6] = cores > jettys ? 1U : 0U;
    param.reserved[7] = static_cast<uint32_t>(sync);
    param.reserved[8] = static_cast<uint32_t>(sync >> 32U);
    HixlLaunchAivSimt(stream, &param, cores);
}
