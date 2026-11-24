/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * CPU simulation only; require the same explicit include as the device build.
 */
#pragma once
namespace cooperative_groups {
struct ThreadBlock { uint32_t thread_rank() const { return threadIdx.x; } };
inline ThreadBlock this_thread_block() { return {}; }
}
