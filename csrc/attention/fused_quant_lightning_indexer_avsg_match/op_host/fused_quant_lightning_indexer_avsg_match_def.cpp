// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class FusedQuantLightningIndexerAvsgMatch : public OpDef {
public:
    explicit FusedQuantLightningIndexerAvsgMatch(const char* name) : OpDef(name)
    {
        const auto input = [this](const char* name, ge::DataType dtype) {
            Input(name).ParamType(REQUIRED).DataType({dtype}).Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        };
        input("query", ge::DT_FLOAT8_E4M3FN);
        input("key", ge::DT_FLOAT8_E4M3FN);
        input("weights", ge::DT_BF16);
        input("query_scale", ge::DT_FLOAT);
        input("key_scale", ge::DT_FLOAT);
        input("cumulative_query_lengths", ge::DT_INT32);
        input("key_lengths", ge::DT_INT32);
        input("full_block_table", ge::DT_INT32);
        input("hot_block_table", ge::DT_INT32);
        input("status", ge::DT_INT32);

        // Matching input/output names declare the mutable status tensor to ACLNN.
        for (const char* output : {"slots", "hits", "mapping", "counts", "status"}) {
            Output(output).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
        }
        AICore().AddConfig("ascend950");
    }
};
OP_ADD(FusedQuantLightningIndexerAvsgMatch);
}  // namespace ops
