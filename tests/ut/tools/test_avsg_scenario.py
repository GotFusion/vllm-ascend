# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
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


def test_pd_prefix_metrics_use_prefiller_and_reset_both_roles(tmp_path):
    scenario = scenario_module.make_scenario(
        "/models/glm53",
        "/data/test.jsonl",
        "http://localhost:20004/",
        "smoke",
        tmp_path,
        metrics_origin="http://localhost:20002/",
        reset_origin="http://localhost:20004/",
    )
    service = scenario["service"]
    assert service["inference_url"] == "http://localhost:20004/v1/completions"
    assert service["metrics_url"] == "http://localhost:20002/metrics"
    assert service["reset_url"] == "http://localhost:20004/reset_prefix_cache"
    assert not service["assume_empty_cache"]


@pytest.mark.parametrize("field", ["metrics_origin", "reset_origin"])
@pytest.mark.parametrize("url", ["", "file:///tmp/state", "http://host/metrics", "http://host?token=x"])
def test_prefix_scenario_validates_control_origins(tmp_path, field, url):
    with pytest.raises(ValueError, match=field):
        scenario_module.make_scenario(
            "/models/glm53", "/data/test.jsonl", "http://localhost:20004", "smoke", tmp_path, **{field: url}
        )


def test_pd_scenario_cli_writes_distinct_origins_and_full_length(tmp_path):
    output = tmp_path / "pd-128k.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "/models/glm53",
            "--corpus",
            "/data/test.jsonl",
            "--service-url",
            "http://localhost:20004",
            "--metrics-origin",
            "http://localhost:20002",
            "--reset-origin",
            "http://localhost:20004",
            "--profile",
            "128k",
            "--output-dir",
            str(tmp_path / "results"),
            "--scenario",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scenario = json.loads(output.read_text())
    assert scenario["service"]["metrics_url"] == "http://localhost:20002/metrics"
    assert scenario["service"]["reset_url"] == "http://localhost:20004/reset_prefix_cache"
    assert scenario["requests"]["input_length"]["value"] + scenario["requests"]["output_length"]["value"] == 131104
    assert "131104" in result.stdout
