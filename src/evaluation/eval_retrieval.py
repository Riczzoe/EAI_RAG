"""Minimal retrieval-success evaluation over curated JSONL queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from src.qdrant import LocalQdrantConfig, LocalQdrantStore
from src.utils.io import load_yaml


@dataclass(frozen=True)
class EvalConfig:
    retrieval_eval_path: Path
    top_k: int
    repeat_per_query: int
    output_path: Path
    include_query_details: bool


def run_retrieval_eval(
    eval_config_path: Path = Path("configs/evaluation.yaml"),
    qdrant_config_path: Path = Path("configs/qdrant.yaml"),
) -> dict[str, object]:
    """Run retrieval-success evaluation and return summary metrics."""
    eval_cfg = _load_eval_config(eval_config_path)
    queries = _load_eval_queries(eval_cfg.retrieval_eval_path)
    store = _init_qdrant_store(qdrant_config_path)

    total_count = len(queries) * eval_cfg.repeat_per_query
    hit_count = 0
    query_details: list[dict[str, object]] = []

    for query in queries:
        for repeat_index in range(1, eval_cfg.repeat_per_query + 1):
            results = store.search_text(query_text=query["question"], top_k=eval_cfg.top_k)
            retrieved_synset_ids = [
                item["synset_id"]
                for item in results
                if isinstance(item.get("synset_id"), str) and item["synset_id"].strip()
            ]
            hit = query["target_synset_id"] in retrieved_synset_ids
            if hit:
                hit_count += 1

            if eval_cfg.include_query_details:
                query_details.append(
                    {
                        "query_id": query["query_id"],
                        "repeat_index": repeat_index,
                        "question": query["question"],
                        "target_synset_id": query["target_synset_id"],
                        "retrieved_synset_ids": retrieved_synset_ids,
                        "hit": hit,
                    }
                )

    hit_rate = (hit_count / total_count) if total_count > 0 else 0.0
    summary: dict[str, object] = {
        "total_count": total_count,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "top_k": eval_cfg.top_k,
        "repeat_per_query": eval_cfg.repeat_per_query,
    }
    if eval_cfg.include_query_details:
        summary["queries"] = query_details
    return summary


def write_eval_result(result: Mapping[str, object], output_path: Path) -> None:
    """Write evaluation result JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_eval_config(path: Path) -> EvalConfig:
    raw = load_yaml(path)
    root = raw.get("evaluation") if isinstance(raw.get("evaluation"), Mapping) else raw
    if not isinstance(root, Mapping):
        raise ValueError("Invalid evaluation config format")
    section = root.get("retrieval") if isinstance(root.get("retrieval"), Mapping) else root

    required = ["retrieval_eval_path", "top_k", "output_path"]
    for key in required:
        if key not in section:
            raise ValueError(f"Missing required config key: {key}")

    top_k = int(section["top_k"])
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    repeat_per_query = int(section.get("repeat_per_query", 1))
    if repeat_per_query <= 0:
        raise ValueError("repeat_per_query must be > 0")

    return EvalConfig(
        retrieval_eval_path=Path(str(section["retrieval_eval_path"])),
        top_k=top_k,
        repeat_per_query=repeat_per_query,
        output_path=Path(str(section["output_path"])),
        include_query_details=bool(section.get("include_query_details", False)),
    )


def _load_eval_queries(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    queries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in eval dataset at line {line_number}: {exc.msg}") from exc

            if not isinstance(item, Mapping):
                raise ValueError(f"Eval dataset line {line_number} must be a JSON object")

            parsed = {
                "query_id": _require_non_empty_str(item, "query_id", line_number),
                "question": _require_non_empty_str(item, "question", line_number),
                "target_synset_id": _require_non_empty_str(item, "target_synset_id", line_number),
            }
            queries.append(parsed)

    if not queries:
        raise ValueError("Evaluation dataset contains zero valid queries")
    return queries


def _init_qdrant_store(qdrant_config_path: Path) -> LocalQdrantStore:
    raw = load_yaml(qdrant_config_path)
    section = raw.get("qdrant")
    if not isinstance(section, Mapping):
        raise ValueError("`qdrant` section is required in configs/qdrant.yaml")

    required = ["storage_path", "collection_name", "embedding_model", "distance"]
    for key in required:
        if key not in section:
            raise ValueError(f"Missing required qdrant config key: qdrant.{key}")

    store = LocalQdrantStore(
        LocalQdrantConfig(
            storage_path=Path(str(section["storage_path"])),
            collection_name=str(section["collection_name"]),
            recreate_collection=False,
            embedding_model=str(section["embedding_model"]),
            distance=str(section["distance"]),
            batch_size=int(section.get("batch_size", 32)),
        )
    )

    if not store.collection_exists():
        raise RuntimeError("Qdrant collection does not exist. Run sync before evaluation.")
    if store.points_count() <= 0:
        raise RuntimeError("Qdrant collection is empty. Run sync before evaluation.")
    return store


def _require_non_empty_str(item: Mapping[str, object], key: str, line_number: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Eval dataset line {line_number} has invalid `{key}`")
    return value.strip()
