# SPDX-License-Identifier: Apache-2.0
"""Write a GLM-5.3 ais-bench-prefix-cache scenario without contacting a server."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

BLOCK_SIZE = 128
OUTPUT_TOKENS = 32
PROFILE_INPUT_TOKENS = {"smoke": 4096, "128k": 131072}


def normalize_origin(origin: str, name: str) -> str:
    parsed = urlparse(origin)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an HTTP(S) origin, for example http://127.0.0.1:20003")
    return origin.rstrip("/")


def make_scenario(
    model_path: str,
    corpus_path: str,
    service_url: str,
    profile: str,
    output_dir: Path,
    *,
    metrics_origin: str | None = None,
    reset_origin: str | None = None,
) -> dict:
    service_url = normalize_origin(service_url, "service_url")
    metrics_origin = normalize_origin(metrics_origin if metrics_origin is not None else service_url, "metrics_origin")
    reset_origin = normalize_origin(reset_origin if reset_origin is not None else service_url, "reset_origin")
    input_tokens = PROFILE_INPUT_TOKENS[profile]
    return {
        "schema_version": "1.0",
        "run": {
            "run_id": f"glm53-avsg-{profile}",
            "random_seed": 42,
            "output_dir": str(output_dir.resolve()),
            "overwrite": False,
        },
        "tokenizer": {"path": model_path, "block_size": BLOCK_SIZE, "revision": None, "trust_remote_code": True},
        "corpus": {"path": corpus_path, "field": "question", "selection": {"mode": "random"}},
        "requests": {
            "count": 5,
            "input_length": {"mode": "fixed", "value": input_tokens},
            "output_length": {"mode": "fixed", "value": OUTPUT_TOKENS},
        },
        "output": {"output_key": None},
        "prefix_cache": {
            "mode": "warmup",
            "target_hit_rate": 0.9,
            "seed_blocks": 1,
            "minimum_non_shared_length": BLOCK_SIZE,
            "groups": {"count": 1, "assignment": {"mode": "uniform"}, "overrides": {}},
            "order": {"strategy": "interleave"},
        },
        "service": {
            "inference_url": service_url + "/v1/completions",
            "metrics_url": metrics_origin + "/metrics",
            "reset_url": reset_origin + "/reset_prefix_cache",
            "model": "glm53",
            "dp_size": 1,
            "assume_empty_cache": False,
            "engine_label_map": {},
            "timeout_seconds": 3600,
            "api_key": "",
            "poll_interval_seconds": 5.0,
        },
        "validation": {"target_warning_pp": 1.0, "actual_warning_pp": 5.0},
        "aisbench": {
            "config": str(Path(__file__).with_name("prefix_cache_perf.py").resolve()),
            "work_dir": str((output_dir / "aisbench").resolve()),
            "extra_args": {"--num-warmups": 0, "--mode": "perf"},
            "dataset": {
                "abbr": None,
                "input_columns": ["question", "max_out_len"],
                "output_column": "answer",
                "prompt_template": "{question}",
                "pred_role": "BOT",
            },
            "model": {
                "abbr": None,
                "attr": "service",
                "stream": True,
                "max_out_len": OUTPUT_TOKENS,
                "retry": 2,
                "batch_size": 2,
                "generation_kwargs": {"temperature": 0, "ignore_eos": True},
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Tokenizer directory for the tested GLM-5.3 checkpoint")
    parser.add_argument("--corpus", required=True, help="GSM8K test.jsonl, as in the supplied validation log")
    parser.add_argument("--service-url", default="http://127.0.0.1:20003")
    parser.add_argument("--metrics-origin", help="HTTP(S) origin for prefix metrics; use the P server for PD")
    parser.add_argument("--reset-origin", help="HTTP(S) origin for cache reset; the layerwise PD proxy resets P and D")
    parser.add_argument("--profile", choices=PROFILE_INPUT_TOKENS, default="smoke")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    args = parser.parse_args()
    try:
        scenario = make_scenario(
            args.model,
            args.corpus,
            args.service_url,
            args.profile,
            args.output_dir,
            metrics_origin=args.metrics_origin,
            reset_origin=args.reset_origin,
        )
    except ValueError as error:
        parser.error(str(error))
    args.scenario.parent.mkdir(parents=True, exist_ok=True)
    with args.scenario.open("x", encoding="utf-8") as output:
        json.dump(scenario, output, ensure_ascii=False, indent=2)
        output.write("\n")
    minimum_length = PROFILE_INPUT_TOKENS[args.profile] + OUTPUT_TOKENS
    print(f"Wrote {args.scenario}; service max-model-len must be at least {minimum_length}.")


if __name__ == "__main__":
    main()
