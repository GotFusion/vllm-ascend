# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


@pytest.fixture(scope="module")
def prepare_outputs(tmp_path_factory):
    # The CPU UT job has no CANN runtime. Compile only the actual ATen Meta
    # implementation, so invalid shapes exercise the same checks as NPU launches.
    build = tmp_path_factory.mktemp("avsg_meta")
    source = build / "meta.cpp"
    source.write_text(
        '#include "avsg_match_outputs.h"\n'
        "#include <torch/library.h>\n"
        "TORCH_LIBRARY(avsg_meta_ut, m) {\n"
        '    m.def("prepare", &vllm_ascend::avsg::prepare_avsg_match_outputs);\n'
        "}\n"
    )
    op_dir = Path(__file__).resolve().parents[3] / "csrc/attention/fused_quant_lightning_indexer_avsg_match"
    load(
        name="vllm_avsg_meta_ut",
        sources=[str(source)],
        extra_include_paths=[str(op_dir)],
        build_directory=str(build),
        is_python_module=False,
    )
    return torch.ops.avsg_meta_ut.prepare


def meta_inputs(batch=1, queries=1, capacity=4096):
    rows = batch * queries
    pages = 512
    shapes = [
        (rows, 32, 128),
        (pages * batch, 128, 1, 128),
        (rows, 32),
        (rows, 32),
        (pages * batch, 128, 1),
        (batch,),
        (batch,),
        (batch, pages),
        (batch, capacity // 128),
        (batch, 2 * capacity),
    ]
    dtypes = [torch.float8_e4m3fn] * 2 + [torch.bfloat16] + [torch.float32] * 2 + [torch.int32] * 5
    return [torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in zip(shapes, dtypes)]


@pytest.mark.parametrize(("batch", "queries", "capacity"), [(1, 1, 4096), (8, 1, 8192), (8, 4, 8192)])
def test_avsg_output_contract(prepare_outputs, batch, queries, capacity):
    rows = batch * queries
    slots, hits, mapping, counts = prepare_outputs(*meta_inputs(batch, queries, capacity))
    assert [x.shape for x in (slots, hits, mapping, counts)] == [
        (rows, 1, 2048),
        (batch,),
        (rows, 1, 2048, 2),
        (rows, 1),
    ]
    assert all(x.dtype == torch.int32 and x.is_meta for x in (slots, hits, mapping, counts))


@pytest.mark.parametrize(
    ("index", "change", "message"),
    [
        (0, lambda x: x.to(torch.bfloat16), "FP8"),
        (2, lambda x: x.to(torch.float32), "weights"),
        (4, lambda x: x.squeeze(-1), "key_dequant_scale"),
        (5, lambda x: x.unsqueeze(0), "lengths"),
        (7, lambda x: x[:, :1], "full block table"),
        (9, lambda x: x[:, :-1], "status must have shape"),
        (9, lambda x: torch.empty_like(x).repeat_interleave(2, dim=1)[:, ::2], "contiguous"),
    ],
)
def test_avsg_rejects_malformed_tensors(prepare_outputs, index, change, message):
    inputs = meta_inputs()
    inputs[index] = change(inputs[index])
    with pytest.raises(RuntimeError, match=message):
        prepare_outputs(*inputs)


def test_avsg_rejects_four_queries_in_4k(prepare_outputs):
    with pytest.raises(RuntimeError, match="Q=4 with 8K"):
        prepare_outputs(*meta_inputs(1, 4, 4096))
