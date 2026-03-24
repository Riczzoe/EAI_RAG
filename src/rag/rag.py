"""Thin RAG orchestration for one inference pass."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.models.vlm_interface import call_vlm
from src.qdrant import LocalQdrantConfig, LocalQdrantStore
from src.rag.context_builder import build_context


class RAGRunner:
    """Minimal RAG runner with a persistent Qdrant client."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._config = config
        self._inference_cfg = _read_inference_config(config)
        qdrant_cfg = _read_qdrant_config(config)

        self._store = LocalQdrantStore(
            LocalQdrantConfig(
                storage_path=Path(str(qdrant_cfg["storage_path"])),
                collection_name=str(qdrant_cfg["collection_name"]),
                recreate_collection=bool(qdrant_cfg.get("recreate_collection", False)),
                embedding_model=str(qdrant_cfg["embedding_model"]),
                distance=str(qdrant_cfg["distance"]),
                batch_size=int(qdrant_cfg.get("batch_size", 32)),
            )
        )
        self._store.ensure_collection()

    def run_rag(self, query_text: str, condition: str | None = None) -> dict[str, object]:
        """Run one RAG pass: retrieval -> context -> messages -> VLM."""
        if not query_text.strip():
            raise ValueError("query_text must be a non-empty string")

        if condition is None:
            selected_condition = str(self._inference_cfg["condition"])
        elif isinstance(condition, str) and condition.strip():
            selected_condition = condition.strip()
        else:
            raise ValueError("condition must be a non-empty string when provided")

        top_k = int(self._inference_cfg["top_k"])
        retrieved_entries = self._store.search_text(query_text=query_text, top_k=top_k)
        context = build_context(
            condition=selected_condition,
            retrieved_entries=retrieved_entries,
            query=query_text,
            config=self._inference_cfg,
        )
        messages = _build_messages(
            query_text=query_text,
            context=context,
            system_prompt=str(self._inference_cfg.get("system_prompt", "")),
        )
        response = call_vlm(messages=messages, model_name=str(self._inference_cfg["model_name"]))

        return {
            "condition": selected_condition,
            "messages": messages,
            "retrieved_entries": retrieved_entries,
            "context": context,
            "response": response,
        }


def run_rag(query_text: str, condition: str, config: Mapping[str, object]) -> dict[str, object]:
    """One-shot helper for backward compatibility."""
    runner = RAGRunner(config)
    return runner.run_rag(query_text=query_text, condition=condition)


def _build_messages(
    *,
    query_text: str,
    context: Mapping[str, object],
    system_prompt: str,
) -> list[dict[str, object]]:
    user_prompt = (
        "## Output rule (VERY IMPORTANT)\n"
        "- Output ONLY ONE single-line answer.\n"
        "- Use exactly one of the following formats:\n"
        "YES\n"
        "NO\n"
        "DESCRIBE: <brief image description>\n\n"
        "## Decision rule\n"
        "- Output YES if the target object is clearly visible.\n"
        "- Output NO if the target object is not visible.\n"
        "- Output DESCRIBE: <brief image description> if the target object is ambiguous, too broad, or unclear.\n\n"
        "## Target object\n"
        f"{query_text.strip()}"
    )

    text_items = context.get("text_items")
    if isinstance(text_items, list) and text_items:
        formatted = [
            f"{idx}. {item.strip()}"
            for idx, item in enumerate(text_items, start=1)
            if isinstance(item, str) and item.strip()
        ]
        if formatted:
            user_prompt += "\n\n## Object-related description\n" + "\n".join(formatted)

    user_content: list[dict[str, str]] = [{"text": user_prompt}]

    image_paths = context.get("image_paths")
    if isinstance(image_paths, list):
        for image_path in image_paths:
            if isinstance(image_path, str) and image_path.strip():
                user_content.append({"image": _to_file_uri(image_path)})

    messages: list[dict[str, object]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"text": system_prompt.strip()}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    return messages

def _to_file_uri(path_text: str) -> str:
    if path_text.startswith("file://"):
        return path_text
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path.as_uri()


def _read_inference_config(config: Mapping[str, object]) -> Mapping[str, object]:
    inference = config.get("inference")
    if isinstance(inference, Mapping):
        section = inference
    else:
        section = config

    required_keys = ["model_name", "condition", "top_k"]
    for key in required_keys:
        if key not in section:
            raise ValueError(f"Missing required inference config key: {key}")
    return section


def _read_qdrant_config(config: Mapping[str, object]) -> Mapping[str, object]:
    qdrant = config.get("qdrant")
    if not isinstance(qdrant, Mapping):
        raise ValueError("Missing `qdrant` section in config")

    required_keys = ["storage_path", "collection_name", "embedding_model", "distance"]
    for key in required_keys:
        if key not in qdrant:
            raise ValueError(f"Missing required qdrant config key: qdrant.{key}")
    return qdrant
