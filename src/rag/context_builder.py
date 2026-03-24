"""Merged context and condition routing RAG."""

from __future__ import annotations
from collections.abc import Callable, Mapping


def build_text_context(
    retrieved_entries: list[Mapping[str, object]],
    query: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Inject retrieved descriptions only."""
    max_text_items = _read_positive_int(config, "max_text_items", default=3)
    text_items: list[str] = []

    for entry in retrieved_entries:
        description = entry.get("description")
        if isinstance(description, str) and description.strip():
            text_items.append(description.strip())
        if len(text_items) >= max_text_items:
            break

    return {
        "query": query,
        "text_items": text_items,
        "image_paths": [],
    }


def build_image_context(
    retrieved_entries: list[Mapping[str, object]],
    query: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Inject retrieved KB images only."""
    max_images = _read_positive_int(config, "max_images", default=3)
    image_paths: list[str] = []

    for entry in retrieved_entries:
        paths = entry.get("image_paths")
        if not isinstance(paths, list):
            continue
        for path in paths:
            if isinstance(path, str) and path.strip():
                image_paths.append(path.strip())
            if len(image_paths) >= max_images:
                break
        if len(image_paths) >= max_images:
            break

    return {
        "query": query,
        "text_items": [],
        "image_paths": image_paths,
    }


def build_text_image_context(
    retrieved_entries: list[Mapping[str, object]],
    query: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Inject both descriptions and images from retrieval results."""
    text_context = build_text_context(retrieved_entries, query, config)
    image_context = build_image_context(retrieved_entries, query, config)
    return {
        "query": query,
        "text_items": text_context["text_items"],
        "image_paths": image_context["image_paths"],
    }


CONDITION_TO_FUNCTION: dict[str, Callable[[list[Mapping[str, object]], str, Mapping[str, object]], dict[str, object]]] = {
    "text_only": build_text_context,
    "image_only": build_image_context,
    "text_image": build_text_image_context,
}


def build_context(
    condition: str,
    retrieved_entries: list[Mapping[str, object]],
    query: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Dispatch to condition-specific context builder."""
    builder = CONDITION_TO_FUNCTION[condition]
    return builder(retrieved_entries, query, config)


def _read_positive_int(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{key} must be > 0")
    return parsed
