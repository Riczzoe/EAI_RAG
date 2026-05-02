"""Thin RAG orchestration for one inference pass."""

from __future__ import annotations

import base64
from collections.abc import Mapping
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from src.models.vlm_interface import call_vlm
from src.qdrant import QdrantStore, build_qdrant_store_config
from src.rag.context_builder import build_context
from src.rag.image_preprocess import resize_image_paths


class RAGRunner:
    """Minimal RAG runner with a persistent Qdrant client."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._config = config
        self._inference_cfg = _read_inference_config(config)
        qdrant_cfg = _read_qdrant_config(config)

        if bool(qdrant_cfg.get("recreate_collection", False)):
            raise ValueError(
                "qdrant.recreate_collection=true is not allowed in RAG inference. "
                "Run sync first, then set recreate_collection=false."
            )

        self._store = QdrantStore(
            build_qdrant_store_config(qdrant_cfg, recreate_collection=False)
        )
        if not self._store.collection_exists():
            raise RuntimeError(
                "Qdrant collection does not exist. Run KB sync before initializing RAGRunner."
            )
        if self._store.points_count() <= 0:
            raise RuntimeError(
                "Qdrant collection is empty. Run KB sync before initializing RAGRunner."
            )

    def run_rag(self, query_text: str, condition: str | None = None) -> dict[str, object]:
        """Run one RAG pass: dense retrieval -> context -> messages -> VLM."""
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
        context = _preprocess_context_images(
            context=context,
            inference_config=self._inference_cfg,
            condition=selected_condition,
        )
        messages = _build_messages(
            query_text=query_text,
            context=context,
            system_prompt=str(self._inference_cfg.get("system_prompt", "")),
        )
        response = call_vlm(messages=messages, inference_config=self._inference_cfg)

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


def _preprocess_context_images(
    *,
    context: Mapping[str, object],
    inference_config: Mapping[str, object],
    condition: str,
) -> dict[str, object]:
    processed_context = dict(context)
    if condition == "text_only":
        return processed_context

    image_paths = processed_context.get("image_paths")
    if not isinstance(image_paths, list) or not image_paths:
        return processed_context

    resize_config = inference_config.get("image_resize")
    if not isinstance(resize_config, Mapping) or not bool(resize_config.get("enabled", False)):
        return processed_context

    source_image_paths = [
        path.strip()
        for path in image_paths
        if isinstance(path, str) and path.strip()
    ]
    resized_image_paths = resize_image_paths(source_image_paths, resize_config)

    processed_context["source_image_paths"] = source_image_paths
    processed_context["image_paths"] = resized_image_paths
    processed_context["image_resize"] = dict(resize_config)
    return processed_context


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

    user_content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]

    image_paths = context.get("image_paths")
    if isinstance(image_paths, list):
        for image_path in image_paths:
            if isinstance(image_path, str) and image_path.strip():
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _to_image_data_url(image_path)},
                    }
                )

    messages: list[dict[str, object]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": system_prompt.strip(),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    return messages


def _to_image_data_url(path_text: str) -> str:
    if path_text.startswith("data:image/"):
        return path_text

    path = _resolve_local_image_path(path_text)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image MIME type for {path}: {mime_type}")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_local_image_path(path_text: str) -> Path:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("image path must be a non-empty string")

    normalized = path_text.strip()
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        if parsed.netloc and parsed.netloc not in {"localhost", "127.0.0.1"}:
            raise ValueError(f"Only local file:// image URIs are supported: {path_text}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(normalized).expanduser()

    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"RAG image not found: {path}")
    return path


def _read_inference_config(config: Mapping[str, object]) -> Mapping[str, object]:
    inference = config.get("inference")
    if isinstance(inference, Mapping):
        section = inference
    else:
        section = config

    required_keys = [
        "api_format",
        "model_name",
        "api_key_env",
        "base_url",
        "condition",
        "top_k",
    ]
    for key in required_keys:
        if key not in section:
            raise ValueError(f"Missing required inference config key: {key}")
    return section


def _read_qdrant_config(config: Mapping[str, object]) -> Mapping[str, object]:
    qdrant = config.get("qdrant")
    if not isinstance(qdrant, Mapping):
        raise ValueError("Missing `qdrant` section in config")

    required_keys = ["collection_name", "embedding_model", "distance"]
    for key in required_keys:
        if key not in qdrant:
            raise ValueError(f"Missing required qdrant config key: qdrant.{key}")
    return qdrant
