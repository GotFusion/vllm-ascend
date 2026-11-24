# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from vllm_ascend.ascend_config import AscendConfig, AvsgConfig, SparseKVOffloadConfig


def _glm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048),
            dtype=torch.bfloat16,
            enforce_eager=True,
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(
            world_size=2, data_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
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
