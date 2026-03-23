"""Sync processed KB entries into a local Qdrant store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from src.qdrant import LocalQdrantConfig, LocalQdrantStore
from src.utils.io import load_yaml


_POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "eai-rag/kb-entry")


@dataclass(frozen=True)
class QdrantSyncConfig:
    storage_path: Path
    collection_name: str
    entries_jsonl_path: Path
    recreate_collection: bool
    embedding_model: str
    distance: str
    batch_size: int
    fail_on_invalid_entry: bool


@dataclass(frozen=True)
class SyncStats:
    total_rows: int
    inserted_rows: int
    skipped_rows: int
    failed_rows: int


def get_qdrant_sync_config(config: Mapping[str, object]) -> QdrantSyncConfig:
    """Parse the qdrant section from YAML config."""
    qdrant_cfg = config.get("qdrant")
    if not isinstance(qdrant_cfg, Mapping):
        raise ValueError("`qdrant` section is required in configs/qdrant.yaml")

    required_keys = [
        "storage_path",
        "collection_name",
        "entries_jsonl_path",
        "recreate_collection",
        "embedding_model",
        "distance",
    ]
    for key in required_keys:
        if key not in qdrant_cfg:
            raise ValueError(f"Missing required config key: qdrant.{key}")

    batch_size = int(qdrant_cfg.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("qdrant.batch_size must be > 0")

    collection_name = str(qdrant_cfg["collection_name"]).strip()
    if not collection_name:
        raise ValueError("qdrant.collection_name must be a non-empty string")

    embedding_model = str(qdrant_cfg["embedding_model"]).strip()
    if not embedding_model:
        raise ValueError("qdrant.embedding_model must be a non-empty string")

    distance = str(qdrant_cfg["distance"]).strip()
    if not distance:
        raise ValueError("qdrant.distance must be a non-empty string")

    return QdrantSyncConfig(
        storage_path=Path(str(qdrant_cfg["storage_path"])),
        collection_name=collection_name,
        entries_jsonl_path=Path(str(qdrant_cfg["entries_jsonl_path"])),
        recreate_collection=bool(qdrant_cfg["recreate_collection"]),
        embedding_model=embedding_model,
        distance=distance,
        batch_size=batch_size,
        fail_on_invalid_entry=bool(qdrant_cfg.get("fail_on_invalid_entry", True)),
    )


def point_id_from_entry_id(entry_id: str) -> str:
    """Derive a stable Qdrant point id from entry_id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, entry_id))


def sync_entries_to_qdrant(config: QdrantSyncConfig) -> SyncStats:
    """Validate entries.jsonl and ingest them into a local Qdrant store."""
    if not config.entries_jsonl_path.exists() or not config.entries_jsonl_path.is_file():
        raise FileNotFoundError(f"entries.jsonl not found: {config.entries_jsonl_path}")

    store = LocalQdrantStore(
        LocalQdrantConfig(
            storage_path=config.storage_path,
            collection_name=config.collection_name,
            recreate_collection=config.recreate_collection,
            embedding_model=config.embedding_model,
            distance=config.distance,
            batch_size=config.batch_size,
        )
    )
    store.ensure_collection()

    total_rows = 0
    skipped_rows = 0

    documents: list[str] = []
    payloads: list[dict[str, object]] = []
    ids: list[str] = []

    with config.entries_jsonl_path.open("r", encoding="utf-8") as input_f:
        for line_number, raw_line in enumerate(input_f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            total_rows += 1
            try:
                document, payload, point_id = _parse_entry_line(line, line_number)
            except ValueError:
                if config.fail_on_invalid_entry:
                    raise
                skipped_rows += 1
                continue

            documents.append(document)
            payloads.append(payload)
            ids.append(point_id)

    inserted_rows = store.upsert_texts(documents=documents, payloads=payloads, ids=ids)
    failed_rows = 0

    return SyncStats(
        total_rows=total_rows,
        inserted_rows=inserted_rows,
        skipped_rows=skipped_rows,
        failed_rows=failed_rows,
    )


def run_from_config(config_path: Path) -> tuple[QdrantSyncConfig, SyncStats]:
    """Load config and run the local Qdrant sync flow."""
    config = load_yaml(config_path)
    parsed = get_qdrant_sync_config(config)
    stats = sync_entries_to_qdrant(parsed)
    return parsed, stats


def _parse_entry_line(line: str, line_number: int) -> tuple[str, dict[str, object], str]:
    try:
        entry = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc

    if not isinstance(entry, Mapping):
        raise ValueError(f"Line {line_number} must be a JSON object")

    entry_id = _require_non_empty_string(entry, "entry_id", line_number)
    synset_id = _require_non_empty_string(entry, "synset_id", line_number)
    class_name = _require_non_empty_string(entry, "class_name", line_number)
    description = _require_non_empty_string(entry, "description", line_number)
    image_paths = _require_image_paths(entry, line_number)

    payload = {
        "entry_id": entry_id,
        "synset_id": synset_id,
        "class_name": class_name,
        "description": description,
        "image_paths": image_paths,
        "num_images": len(image_paths),
    }
    return description, payload, point_id_from_entry_id(entry_id)


def _require_non_empty_string(
    entry: Mapping[str, object],
    field_name: str,
    line_number: int,
) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number} has invalid `{field_name}`")
    return value.strip()


def _require_image_paths(entry: Mapping[str, object], line_number: int) -> list[str]:
    image_paths = entry.get("image_paths")
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError(f"Line {line_number} has invalid `image_paths`")

    normalized: list[str] = []
    for image_path in image_paths:
        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError(f"Line {line_number} has malformed `image_paths`")
        normalized.append(image_path.strip())
    return normalized
