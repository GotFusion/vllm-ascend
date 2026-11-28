// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "exe_graph/runtime/shape.h"

namespace avsg_match {
enum Input : size_t { Query, Key, Weights, QueryScale, KeyScale, QueryEnds, KeyLengths, FullTable, HotTable, Status, NumInputs };
constexpr int64_t kPage = 128;
constexpr int64_t kTopK = 2048;
constexpr int64_t kHeads = 32;
constexpr int64_t kDim = 128;
constexpr size_t kNumOutputs = 5;

inline std::array<gert::Shape, kNumOutputs> OutputShapes(int64_t rows, int64_t batch, const gert::Shape& status)
{
    return {gert::Shape({rows, 1, kTopK}), gert::Shape({batch}), gert::Shape({rows, 1, kTopK, 2}),
            gert::Shape({rows, 1}), status};
}
}  // namespace avsg_match
