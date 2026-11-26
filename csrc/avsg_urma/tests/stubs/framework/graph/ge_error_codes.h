/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 * Simulated framework header with an independent GE definition/include guard.
 */
#pragma once
#include <cstdint>
namespace ge {
using graphStatus = uint32_t;
constexpr graphStatus GRAPH_FAILED = 0xffffffffU;
} // namespace ge
