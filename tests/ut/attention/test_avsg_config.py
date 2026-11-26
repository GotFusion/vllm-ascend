# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError

from vllm_ascend.ascend_config import AscendConfig, AvsgConfig, SparseKVOffloadConfig


def _glm_config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", index_topk=2048),
            dtype=torch.bfloat16,
            enforce_eager=True,
            enable_sleep_mode=False,
        ),
        parallel_config=SimpleNamespace(
            world_size=2,
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
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


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("use_v2_model_runner", False, "V2_MODEL_RUNNER"),
        ("kv_transfer_config", SimpleNamespace(), "SfaRemoteD2HConnector"),
        ("speculative_config", SimpleNamespace(), "speculation"),
        ("additional_config", {"enable_dsa_cp": True}, "enable_dsa_cp"),
    ],
)
def test_avsg_rejects_unsupported_release_modes(field, value, reason):
    vc = _glm_config()
    setattr(vc, field, value)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match=reason):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(vc, False)


def test_avsg_allows_flashcomm_with_full_attention_rows():
    vc = _glm_config()
    vc.additional_config = {"enable_flashcomm1": True}
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(vc, False)


def test_avsg_rejects_host_cache_offload():
    with (
        patch("vllm_ascend.ascend_config.is_950", return_value=True),
        pytest.raises(ValueError, match="another KV backend"),
    ):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(_glm_config(), True)


def _pd_config():
    vc = _glm_config()
    vc.kv_transfer_config = SimpleNamespace(
        kv_connector="SfaRemoteD2HConnector", kv_role="kv_consumer", kv_connector_extra_config={}
    )
    return vc


def test_avsg_accepts_pd_consumer_without_host_offload():
    with patch("vllm_ascend.ascend_config.is_950", return_value=True):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(_pd_config(), False)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("kv_connector", "MultiConnector", "single SfaRemoteD2HConnector"),
        ("kv_role", "kv_producer", "kv_consumer"),
        ("kv_role", "kv_both", "kv_consumer"),
        ("kv_connector_extra_config", {"transfer_backend": "mooncake"}, "MemFabric"),
        ("kv_connector_extra_config", {"memfabric_transfer_protocol": "sdma"}, "device_urma"),
        ("kv_connector_extra_config", {"use_layerwise": False}, "layerwise"),
        ("kv_connector_extra_config", {"use_layerwise": "true"}, "layerwise"),
    ],
)
def test_avsg_rejects_unsupported_pd_transfer(field, value, reason):
    vc = _pd_config()
    setattr(vc.kv_transfer_config, field, value)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match=reason):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(vc, False)


@pytest.mark.parametrize("field", ["tensor_parallel_size", "pipeline_parallel_size"])
def test_avsg_pd_requires_single_rank(field):
    vc = _pd_config()
    setattr(vc.parallel_config, field, 2)
    with patch("vllm_ascend.ascend_config.is_950", return_value=True), pytest.raises(ValueError, match="TP=1 and PP=1"):
        AvsgConfig(endpoint_file="/endpoints/{rank}.json").validate(vc, False)
