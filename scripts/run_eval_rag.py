#!/usr/bin/env python3
"""Run RAG output compliance evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval_rag import run_rag_eval, write_rag_eval_result
from src.utils.io import load_yaml


def main() -> None:
    eval_config_path = Path("configs/evaluation.yaml")
    result = run_rag_eval(
        eval_config_path=eval_config_path,
        inference_config_path=Path("configs/inference.yaml"),
        qdrant_config_path=Path("configs/qdrant.yaml"),
    )

    eval_cfg = load_yaml(eval_config_path)
    root = eval_cfg.get("evaluation", eval_cfg)
    section = root.get("rag", root)
    output_path = Path(str(section["output_path"]))
    write_rag_eval_result(result, output_path)

    print("=== run_eval_rag ===")
    results = result.get("results", [])
    if not isinstance(results, list):
        raise SystemExit("Invalid evaluation result format: `results` must be a list")

    for item in results:
        if not isinstance(item, dict):
            continue
        model_label = item.get("model_label", "<unknown>")
        model_name = item.get("model_name", "<unknown>")
        condition = item.get("condition", "<unknown>")
        status = item.get("status", "ok")
        print(f"model_label: {model_label}")
        print(f"model_name: {model_name}")
        print(f"condition: {condition}")
        print(f"status: {status}")
        if status != "ok":
            print(f"error: {item.get('error', '')}")
            continue
        print(f"total_count: {item['total_count']}")
        print(f"successful_count: {item['successful_count']}")
        print(f"match_count: {item['match_count']}")
        print(f"match_rate: {item['match_rate']:.4f}")
        print(f"repeat_per_query: {item['repeat_per_query']}")
        if "vlm_call_interval_seconds" in item:
            print(f"vlm_call_interval_seconds: {item['vlm_call_interval_seconds']}")
        if "queries" in item:
            print(f"query_details: {len(item['queries'])}")

    print(f"output_path: {output_path}")


if __name__ == "__main__":
    main()
