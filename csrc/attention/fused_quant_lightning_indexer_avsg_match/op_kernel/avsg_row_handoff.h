// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "kernel_operator.h"

namespace avsg_fusion {

// Only the scratch pool is reset between LI and Match. The selected row lives
// in a separate TPipe allocation until the request's Match outputs are written.
struct LocalTopkRow {
    static constexpr uint32_t BUFFER_COUNT = 12;
    static constexpr uint32_t UB_ALIGNMENT = 32;
    AscendC::TBufPool<AscendC::QuePosition::VECCALC, BUFFER_COUNT> scratch;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage;
    AscendC::LocalTensor<int32_t> tokens;

    __aicore__ inline void Init(AscendC::TPipe& pipe, uint32_t topk, uint32_t ubBytes)
    {
        const uint32_t rowBytes = topk * sizeof(int32_t);
        pipe.InitBuffer(storage, rowBytes);
        tokens = storage.Get<int32_t>();
        pipe.InitBufPool(scratch, ubBytes - rowBytes - UB_ALIGNMENT);
    }
};

} // namespace avsg_fusion
