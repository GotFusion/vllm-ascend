# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "avsg" / "glm53_scenario.py"
spec = importlib.util.spec_from_file_location("glm53_scenario", SCRIPT)
scenario_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenario_module)


@pytest.mark.parametrize("profile,input_tokens", [("smoke", 4096), ("128k", 131072)])
def test_prefix_scenario_preserves_load_and_uses_performance_mode(tmp_path, profile, input_tokens):
    baseline = scenario_module.make_scenario(
        "/models/glm53", "/data/test.jsonl", "http://localhost:20003", profile, tmp_path
    )
    avsg = scenario_module.make_scenario(
        "/models/glm53", "/data/test.jsonl", "http://localhost:20004", profile, tmp_path / "avsg"
    )
    for field in ("tokenizer", "corpus", "requests", "prefix_cache"):
        assert baseline[field] == avsg[field]
    assert baseline["requests"]["count"] == 5
    assert baseline["requests"]["input_length"]["value"] == input_tokens
    assert baseline["requests"]["output_length"]["value"] == baseline["aisbench"]["model"]["max_out_len"] == 32
    assert baseline["prefix_cache"]["target_hit_rate"] == 0.9
    assert baseline["aisbench"]["extra_args"]["--mode"] == "perf"
    assert baseline["service"]["model"] == "glm53"
    assert baseline["service"]["reset_url"] == "http://localhost:20003/reset_prefix_cache"
    assert baseline["run"]["output_dir"] != avsg["run"]["output_dir"]


@pytest.mark.parametrize("url", ["file:///tmp/model", "http://host:80/v1", "not-a-url", "http://host?token=x"])
def test_prefix_scenario_rejects_non_origin_url(tmp_path, url):
    with pytest.raises(ValueError, match="HTTP"):
        scenario_module.make_scenario("/models/glm53", "/data/test.jsonl", url, "smoke", tmp_path)
