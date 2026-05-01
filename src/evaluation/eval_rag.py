"""RAG output compliance evaluation against target_command."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
import time
from pathlib import Path

from src.rag.rag import RAGRunner
from src.utils.io import load_yaml


_VLM_CALL_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class RagEvalModel:
    label: str
    model_name: str


@dataclass(frozen=True)
class RagEvalConfig:
    rag_eval_path: Path
    conditions: list[str]
    models: list[RagEvalModel]
    repeat_per_query: int
    output_path: Path
    include_query_details: bool


def run_rag_eval(
    eval_config_path: Path = Path("configs/evaluation.yaml"),
    inference_config_path: Path = Path("configs/inference.yaml"),
    qdrant_config_path: Path = Path("configs/qdrant.yaml"),
) -> dict[str, object]:
    """Run RAG compliance evaluation for all configured model/condition pairs."""
    eval_cfg = _load_rag_eval_config(
        eval_config_path,
        inference_config_path=inference_config_path,
    )
    queries = _load_eval_queries(eval_cfg.rag_eval_path)
    base_runtime_config = _build_runtime_config(inference_config_path, qdrant_config_path)

    results: list[dict[str, object]] = []
    has_vlm_call_started = [False]

    for model in eval_cfg.models:
        for condition in eval_cfg.conditions:
            runtime_config = _build_model_runtime_config(
                base_runtime_config=base_runtime_config,
                model=model,
                condition=condition,
            )
            try:
                runner = RAGRunner(runtime_config)
                results.append(
                    _evaluate_one_condition(
                        runner=runner,
                        queries=queries,
                        condition=condition,
                        model=model,
                        repeat_per_query=eval_cfg.repeat_per_query,
                        include_query_details=eval_cfg.include_query_details,
                        has_vlm_call_started=has_vlm_call_started,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "model_label": model.label,
                        "model_name": model.model_name,
                        "condition": condition,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

    return {"results": results}


def write_rag_eval_result(result: Mapping[str, object], output_path: Path) -> None:
    """Write RAG evaluation result JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _evaluate_one_condition(
    *,
    runner: RAGRunner,
    queries: list[dict[str, str]],
    condition: str,
    model: RagEvalModel,
    repeat_per_query: int,
    include_query_details: bool,
    has_vlm_call_started: list[bool] | None = None,
) -> dict[str, object]:
    match_count = 0
    successful_count = 0
    query_details: list[dict[str, object]] = []
    if has_vlm_call_started is None:
        has_vlm_call_started = [False]

    for query in queries:
        question = query["question"]
        target_command = query["target_command"]

        for repeat_index in range(1, repeat_per_query + 1):
            if has_vlm_call_started[0]:
                time.sleep(_VLM_CALL_INTERVAL_SECONDS)
            else:
                has_vlm_call_started[0] = True

            try:
                rag_result = runner.run_rag(query_text=question, condition=condition)
                model_output = _extract_model_output(rag_result).strip()
            except Exception:
                continue

            matched = model_output == target_command
            successful_count += 1
            if matched:
                match_count += 1

            if include_query_details:
                query_details.append(
                    {
                        "model_label": model.label,
                        "model_name": model.model_name,
                        "query_id": query["query_id"],
                        "question": question,
                        "target_command": target_command,
                        "model_output": model_output,
                        "matched": matched,
                        "condition": condition,
                        "repeat_index": repeat_index,
                        "status": "ok",
                    }
                )

    summary: dict[str, object] = {
        "model_label": model.label,
        "model_name": model.model_name,
        "condition": condition,
        "status": "ok",
        "total_count": successful_count,
        "successful_count": successful_count,
        "match_count": match_count,
        "match_rate": (match_count / successful_count) if successful_count > 0 else 0.0,
        "repeat_per_query": repeat_per_query,
        "vlm_call_interval_seconds": _VLM_CALL_INTERVAL_SECONDS,
    }
    if include_query_details:
        summary["queries"] = query_details
    return summary


def _extract_model_output(rag_result: Mapping[str, object]) -> str:
    response = rag_result.get("response")
    if not isinstance(response, str):
        raise ValueError("RAG response must be a string")
    return response


def _load_rag_eval_config(
    path: Path,
    inference_config_path: Path = Path("configs/inference.yaml"),
) -> RagEvalConfig:
    raw = load_yaml(path)
    root = raw.get("evaluation") if isinstance(raw.get("evaluation"), Mapping) else raw
    if not isinstance(root, Mapping):
        raise ValueError("Invalid evaluation config format")

    section = root.get("rag") if isinstance(root.get("rag"), Mapping) else root
    if not isinstance(section, Mapping):
        raise ValueError("Missing evaluation.rag section")

    required = ["rag_eval_path", "conditions", "repeat_per_query", "output_path"]
    for key in required:
        if key not in section:
            raise ValueError(f"Missing required config key: {key}")

    conditions_raw = section["conditions"]
    if not isinstance(conditions_raw, list) or not conditions_raw:
        raise ValueError("conditions must be a non-empty list")

    conditions: list[str] = []
    for index, value in enumerate(conditions_raw, start=1):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"conditions[{index}] must be a non-empty string")
        conditions.append(value.strip())

    models = _load_eval_models(section, inference_config_path)

    repeat_per_query = int(section["repeat_per_query"])
    if repeat_per_query <= 0:
        raise ValueError("repeat_per_query must be > 0")

    return RagEvalConfig(
        rag_eval_path=Path(str(section["rag_eval_path"])),
        conditions=conditions,
        models=models,
        repeat_per_query=repeat_per_query,
        output_path=Path(str(section["output_path"])),
        include_query_details=bool(section.get("include_query_details", False)),
    )


def _build_runtime_config(inference_config_path: Path, qdrant_config_path: Path) -> dict[str, object]:
    inference_raw = load_yaml(inference_config_path)
    qdrant_raw = load_yaml(qdrant_config_path)

    inference_cfg = inference_raw.get("inference")
    if not isinstance(inference_cfg, Mapping):
        raise ValueError("`inference` section is required in configs/inference.yaml")

    qdrant_cfg = qdrant_raw.get("qdrant")
    if not isinstance(qdrant_cfg, Mapping):
        raise ValueError("`qdrant` section is required in configs/qdrant.yaml")

    qdrant_runtime_cfg = dict(qdrant_cfg)
    qdrant_runtime_cfg["recreate_collection"] = False

    return {
        "inference": dict(inference_cfg),
        "qdrant": qdrant_runtime_cfg,
    }


def _build_model_runtime_config(
    *,
    base_runtime_config: Mapping[str, object],
    model: RagEvalModel,
    condition: str,
) -> dict[str, object]:
    inference_cfg = base_runtime_config.get("inference")
    qdrant_cfg = base_runtime_config.get("qdrant")
    if not isinstance(inference_cfg, Mapping):
        raise ValueError("Base runtime config is missing inference section")
    if not isinstance(qdrant_cfg, Mapping):
        raise ValueError("Base runtime config is missing qdrant section")

    runtime_inference_cfg = dict(inference_cfg)
    runtime_inference_cfg["model_name"] = model.model_name
    runtime_inference_cfg["condition"] = condition

    return {
        "inference": runtime_inference_cfg,
        "qdrant": dict(qdrant_cfg),
    }


def _load_eval_models(
    rag_section: Mapping[str, object],
    inference_config_path: Path,
) -> list[RagEvalModel]:
    models_raw = rag_section.get("models")
    if models_raw is None:
        inference_raw = load_yaml(inference_config_path)
        inference_cfg = inference_raw.get("inference")
        if not isinstance(inference_cfg, Mapping):
            raise ValueError("`inference` section is required in configs/inference.yaml")
        model_name = _require_non_empty_config_str(inference_cfg, "inference.model_name")
        return [RagEvalModel(label=_model_label_from_name(model_name), model_name=model_name)]

    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("evaluation.rag.models must be a non-empty list when provided")

    models: list[RagEvalModel] = []
    seen_labels: set[str] = set()
    for index, raw_model in enumerate(models_raw, start=1):
        if not isinstance(raw_model, Mapping):
            raise ValueError(f"evaluation.rag.models[{index}] must be a mapping")
        model_name = _require_non_empty_config_str(
            raw_model,
            f"evaluation.rag.models[{index}].model_name",
        )
        label = _require_non_empty_config_str(
            raw_model,
            f"evaluation.rag.models[{index}].label",
        )
        if label in seen_labels:
            raise ValueError(f"Duplicate evaluation.rag.models label: {label}")
        seen_labels.add(label)
        models.append(RagEvalModel(label=label, model_name=model_name))
    return models


def _model_label_from_name(model_name: str) -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", model_name.strip()).strip("_").lower()
    return label or "model"


def _require_non_empty_config_str(item: Mapping[str, object], key: str) -> str:
    field_name = key.rsplit(".", maxsplit=1)[-1]
    value = item.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config key `{key}` must be a non-empty string")
    return value.strip()


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
                raise ValueError(
                    f"Invalid JSON in eval dataset at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(item, Mapping):
                raise ValueError(f"Eval dataset line {line_number} must be a JSON object")

            parsed = {
                "query_id": _require_non_empty_str(item, "query_id", line_number),
                "question": _require_non_empty_str(item, "question", line_number),
                "target_command": _require_non_empty_str(item, "target_command", line_number).strip(),
            }
            queries.append(parsed)

    if not queries:
        raise ValueError("Evaluation dataset contains zero valid queries")
    return queries


def _require_non_empty_str(item: Mapping[str, object], key: str, line_number: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Eval dataset line {line_number} has invalid `{key}`")
    return value.strip()
