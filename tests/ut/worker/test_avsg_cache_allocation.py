# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    get_avsg_cache_block_limit,
)
from vllm_ascend.worker import worker as worker_module
from vllm_ascend.worker.v2 import attn_utils


@pytest.fixture(params=[torch.bfloat16, torch.int8])
def cache_specs(request):
    # Two main layers share one logical indexer; each manager page spans two kernel pages.
    main = AscendMLAAttentionSpec(
        block_size=256, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_peer=True
    )
    indexer = AscendSFAIndexerCacheSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=request.param,
        scale_dim=int(request.param == torch.int8),
        scale_dtype=torch.float16,
    )
    specs = {"main.0": main, "main.1": main, "indexer.0": indexer}
    group = KVCacheGroupSpec(list(specs), UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=specs))
    return specs, group


@pytest.mark.parametrize(("indexer_budget_blocks", "peer_pages", "limit"), [(9, 7, 3), (2, 9, 2)])
def test_worker_sizes_logical_pool_and_guards_overrides(
    monkeypatch, cache_specs, indexer_budget_blocks, peer_pages, limit
):
    specs, group = cache_specs
    cache_config = SimpleNamespace(
        num_gpu_blocks_override=None, get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=None), cache_config=cache_config),
        cache_config=cache_config,
        _avsg_endpoint_file="peer.json",
        avsg_transport=SimpleNamespace(remote_block_capacity=peer_pages, block_size=128),
        get_kv_cache_spec=lambda: specs,
    )
    monkeypatch.setattr(worker_module, "get_kv_cache_groups", lambda *_: [group])
    device_bytes = indexer_budget_blocks * specs["indexer.0"].page_size_bytes
    scale = worker_module.NPUWorker._scale_kv_cache_memory_for_multi_group
    assert scale(worker, device_bytes) == limit * sum(spec.page_size_bytes for spec in specs.values())
    cache_config.num_gpu_blocks_override = limit + 1
    with pytest.raises(ValueError, match="local indexer and peer permit"):
        scale(worker, device_bytes)


def test_peer_only_pipeline_rank_and_final_plan_guard(cache_specs):
    main = cache_specs[0]["main.0"]
    groups = [KVCacheGroupSpec(["main.0"], main)]
    assert get_avsg_cache_block_limit(groups, peer_blocks=7, kernel_block_size=128, device_bytes=0) == 3
    worker = SimpleNamespace(
        _avsg_endpoint_file="peer.json", avsg_transport=SimpleNamespace(remote_block_capacity=7, block_size=128)
    )
    with pytest.raises(ValueError, match="peer holds 3"):
        worker_module.NPUWorker.initialize_from_config(worker, SimpleNamespace(kv_cache_groups=groups, num_blocks=4))


@pytest.mark.parametrize("num_blocks", [2, 7])
def test_only_indexer_storage_grows_with_logical_block_count(monkeypatch, cache_specs, num_blocks):
    specs, group = cache_specs
    logical_bytes = num_blocks * sum(spec.page_size_bytes for spec in specs.values())
    runtime = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    plan = get_kv_cache_config_from_groups(runtime, [group], logical_bytes)
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: runtime)
    monkeypatch.setattr(attn_utils, "_get_attention_kv_cache_dims", lambda *_: (512, 64))
    raw = attn_utils._allocate_kv_cache(plan, shared_layers={}, device=torch.device("cpu"))
    backend = SimpleNamespace(get_kv_cache_shape=lambda *shape: shape)
    groups = [
        SimpleNamespace(
            kv_cache_group_id=0, kv_cache_spec=spec, layer_names=[name], kernel_block_size=None, backend=backend
        )
        for name, spec in specs.items()
    ]
    caches = attn_utils._reshape_kv_cache_v2(groups, raw, "auto", [128], {}, plan)
    assert plan.num_blocks == num_blocks
    for name in ("main.0", "main.1"):
        kv, rope = caches[name]
        assert kv.shape == (0, 128, 1, 512) and rope.shape == (0, 128, 1, 64)
        assert kv.untyped_storage().nbytes() == rope.untyped_storage().nbytes() == 0
    indexer_spec = specs["indexer.0"]
    key, *scales = caches["indexer.0"]
    assert key.shape == (2 * num_blocks, 128, 1, 128)
    assert key.dtype == indexer_spec.dtype
    assert key.untyped_storage().nbytes() == num_blocks * indexer_spec.page_size_bytes
    assert len(scales) == indexer_spec.scale_dim
    if scales:
        assert scales[0].shape == (2 * num_blocks, 128, 1, 1)
        assert scales[0].dtype == torch.float16
        assert key.untyped_storage().data_ptr() == scales[0].untyped_storage().data_ptr()
