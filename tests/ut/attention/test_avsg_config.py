# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.ascend_config import (
    AVSG_DECODE_GRAPH_MAX_BLOCKS,
    AVSG_KERNEL_BLOCK_SIZE,
    AscendConfig,
    AvsgConfig,
    SparseKVOffloadConfig,
)


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048),
            dtype=torch.bfloat16,
            enforce_eager=True,
            enable_sleep_mode=False,
            max_model_len=AVSG_DECODE_GRAPH_MAX_BLOCKS * AVSG_KERNEL_BLOCK_SIZE,
        ),
        parallel_config=SimpleNamespace(
            world_size=2,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            splitting_ops=None,
            use_inductor_graph_partition=False,
            cudagraph_capture_sizes=[1],
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        kv_transfer_config=None,
    )


def test_avsg_is_opt_in_and_typed():
    assert not AvsgConfig().enabled
    AvsgConfig().validate(None, False)
    cfg = AscendConfig(
        sparse_kv_offload_config=SparseKVOffloadConfig(), avsg_config={"endpoint_file": "/endpoints/{rank}.json"}
    )
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        cfg.avsg_config.validate(_glm_config(), False)
    assert cfg.avsg_config.enabled
    with pytest.raises(ValidationError):
        AvsgConfig(endpoint_file="")


@pytest.mark.parametrize(
    "group,field,value,reason",
    [
        ("model_config", "enforce_eager", False, "eager"),
        ("cache_config", "cache_dtype", "fp8", "BF16"),
        ("parallel_config", "decode_context_parallel_size", 2, "unsharded"),
        ("attention_config", "hisparse_config", {}, "another KV backend"),
    ],
)
def test_avsg_rejects_incompatible_backend(group, field, value, reason):
    vc = _glm_config()
    setattr(getattr(vc, group), field, value)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match=reason):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(vc, False)


def test_avsg_requires_matching_hardware_and_distinct_endpoints():
    cfg = AvsgConfig(endpoint_file="/endpoints/0.json")
    with patch("vllm_ascend.ascend_config.is_950", return_value=False), pytest.raises(ValueError, match="950"):
        cfg.validate(_glm_config(), False)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match="rank"):
        cfg.validate(_glm_config(), False)


def test_avsg_rejects_kv_layer_parallelism():
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match="KVPP"):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(_glm_config(), False, kvpp_size=2)


@pytest.mark.parametrize("mode", list(CUDAGraphMode))
def test_avsg_graph_execution_modes(mode):
    cfg = AvsgConfig(endpoint_file="/endpoints/{rank}.json")
    vc = _glm_config()
    vc.model_config.enforce_eager = False
    vc.compilation_config.cudagraph_mode = mode
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        if mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY):
            cfg.validate(vc, False)
            vc.compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
        with pytest.raises(ValueError, match="VLLM_COMPILE.*PIECEWISE"):
            cfg.validate(vc, False)


def test_avsg_worker_uses_resolved_mla_split():
    cfg = AvsgConfig(endpoint_file="/endpoints/{rank}.json")
    vc = _glm_config()
    cfg.validate_worker_graph(vc)  # Eager has no FX graph.
    vc.model_config.enforce_eager = False
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    with pytest.raises(ValueError, match="mla_forward"):
        cfg.validate_worker_graph(vc)
    vc.compilation_config.splitting_ops = ["vllm::mla_forward"]
    cfg.validate_worker_graph(vc)
    vc.compilation_config.use_inductor_graph_partition = True
    with pytest.raises(ValueError, match="mla_forward"):
        cfg.validate_worker_graph(vc)
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    with pytest.raises(ValueError, match="model runner resolved"):
        cfg.validate_worker_graph(vc)


@pytest.mark.parametrize(
    "group,field,value,error",
    [
        ("compilation_config", "cudagraph_capture_sizes", [1, 2], "capture_sizes"),
        ("compilation_config", "decode_cudagraph_capture_request_sizes", [2], "capture_sizes"),
        ("model_config", "max_model_len", AVSG_DECODE_GRAPH_MAX_BLOCKS * AVSG_KERNEL_BLOCK_SIZE + 1, "pages"),
        ("parallel_config", "data_parallel_size", 2, "DP1"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP1"),
    ],
)
def test_avsg_restricts_full_decode_only(group, field, value, error):
    cfg = AvsgConfig(endpoint_file="/endpoints/{rank}.json")
    vc = _glm_config()
    vc.model_config.enforce_eager = False
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    cfg.validate_worker_graph(vc, CUDAGraphMode.FULL_DECODE_ONLY)
    setattr(getattr(vc, group), field, value)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match=error):
        cfg.validate(vc, False)
    with pytest.raises(ValueError, match=error):
        cfg.validate_worker_graph(vc, CUDAGraphMode.FULL_DECODE_ONLY)


def test_avsg_full_decode_rejects_worker_graph_downgrade():
    cfg = AvsgConfig(endpoint_file="/endpoints/{rank}.json")
    vc = _glm_config()
    vc.model_config.enforce_eager = False
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    vc.compilation_config.splitting_ops = ["vllm::mla_forward"]
    with pytest.raises(ValueError, match="model runner resolved"):
        cfg.validate_worker_graph(vc, CUDAGraphMode.FULL_DECODE_ONLY)


def test_avsg_rejects_xlite_and_speculation_in_piecewise():
    cfg = AvsgConfig(endpoint_file="/endpoints/{rank}.json")
    vc = _glm_config()
    vc.model_config.enforce_eager = False
    vc.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        with pytest.raises(ValueError, match="XLite"):
            cfg.validate(vc, False, xlite_enabled=True)
        vc.speculative_config = SimpleNamespace()
        with pytest.raises(ValueError, match="speculation"):
            cfg.validate(vc, False)
