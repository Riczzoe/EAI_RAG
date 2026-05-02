"""Sync processed KB entries into a Qdrant store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from src.qdrant import QdrantStore, QdrantStoreConfig, build_qdrant_store_config
from src.utils.io import load_yaml


_POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "eai-rag/kb-entry")


@dataclass(frozen=True)
class QdrantSyncConfig:
    store_config: QdrantStoreConfig
    collection_name: str
    entries_jsonl_path: Path
    recreate_collection: bool
    embedding_model: str
    distance: str
    batch_size: int
    fail_on_invalid_entry: bool

    @property
    def mode(self) -> str:
        return self.store_config.mode

    @property
    def url(self) -> str | None:
        return self.store_config.url

    @property
    def storage_path(self) -> Path | None:
        return self.store_config.storage_path


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
        "collection_name",
        "entries_jsonl_path",
        "recreate_collection",
        "embedding_model",
        "distance",
    ]
    for key in required_keys:
        if key not in qdrant_cfg:
            raise ValueError(f"Missing required config key: qdrant.{key}")

    store_config = build_qdrant_store_config(qdrant_cfg)

    return QdrantSyncConfig(
        store_config=store_config,
        collection_name=store_config.collection_name,
        entries_jsonl_path=Path(str(qdrant_cfg["entries_jsonl_path"])),
        recreate_collection=store_config.recreate_collection,
        embedding_model=store_config.embedding_model,
        distance=store_config.distance,
        batch_size=store_config.batch_size,
        fail_on_invalid_entry=bool(qdrant_cfg.get("fail_on_invalid_entry", True)),
    )


def point_id_from_entry_id(entry_id: str) -> str:
    """Derive a stable Qdrant point id from entry_id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, entry_id))


def sync_entries_to_qdrant(config: QdrantSyncConfig) -> SyncStats:
    """Validate entries.jsonl and ingest them into a Qdrant store."""
    if not config.entries_jsonl_path.exists() or not config.entries_jsonl_path.is_file():
        raise FileNotFoundError(f"entries.jsonl not found: {config.entries_jsonl_path}")

    store = QdrantStore(config.store_config)
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
    """Load config and run the Qdrant sync flow."""
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
