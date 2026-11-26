/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once
#include <cstdint>
constexpr uint32_t HIXL_AIV_MEM_TYPE_URMA = 0;
struct HixlAivOpParam {
    uint64_t channel, op_desc_list_addr;
    uint32_t list_num, mem_type, compute_mode, is_get;
    uint32_t reserved[20] = {};
};
extern "C" void HixlLaunchAivSimt(void *, const HixlAivOpParam *, uint32_t);
