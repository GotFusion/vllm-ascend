// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
#include "operator_contract.h"

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* query = context->GetInputShape(avsg_match::Query);
    const auto* queryEnds = context->GetInputShape(avsg_match::QueryEnds);
    const auto* status = context->GetInputShape(avsg_match::Status);
    if (query == nullptr || queryEnds == nullptr || status == nullptr || query->GetDimNum() != 3 ||
        queryEnds->GetDimNum() != 1 || status->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const auto shapes = avsg_match::OutputShapes(query->GetDim(0), queryEnds->GetDim(0), *status);
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto* output = context->GetOutputShape(i);
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = shapes[i];
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < avsg_match::kNumOutputs; ++i) {
        context->SetOutputDataType(i, ge::DT_INT32);
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace
IMPL_OP_INFERSHAPE(FusedQuantLightningIndexerAvsgMatch).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ops
