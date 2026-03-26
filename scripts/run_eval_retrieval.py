#!/usr/bin/env python3
"""Run minimal retrieval-success evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.eval_retrieval import run_retrieval_eval, write_eval_result
from src.utils.io import load_yaml


def main() -> None:
    eval_config_path = Path("configs/evaluation.yaml")
    result = run_retrieval_eval(
        eval_config_path=eval_config_path,
        qdrant_config_path=Path("configs/qdrant.yaml"),
    )

    eval_cfg = load_yaml(eval_config_path)
    root = eval_cfg.get("evaluation", eval_cfg)
    section = root.get("retrieval", root)
    output_path = Path(str(section["output_path"]))
    write_eval_result(result, output_path)

    print("=== run_eval_retrieval ===")
    print(f"total_count: {result['total_count']}")
    print(f"hit_count: {result['hit_count']}")
    print(f"hit_rate: {result['hit_rate']:.4f}")
    print(f"top_k: {result['top_k']}")
    print(f"repeat_per_query: {result['repeat_per_query']}")
    print(f"output_path: {output_path}")
    if "queries" in result:
        print(f"query_details: {len(result['queries'])}")


if __name__ == "__main__":
    main()
