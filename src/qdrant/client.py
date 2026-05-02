"""Dense semantic retrieval helpers for local or server Qdrant collections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class QdrantStoreConfig:
    mode: str
    url: str | None
    api_key_env: str | None
    storage_path: Path | None
    collection_name: str
    recreate_collection: bool
    embedding_model: str
    distance: str
    batch_size: int


class QdrantStore:
    """Manage dense vector upserts and semantic search in a Qdrant collection."""

    def __init__(self, config: QdrantStoreConfig) -> None:
        self.config = config
        if self.config.mode == "server":
            if not self.config.url:
                raise ValueError("qdrant.url is required when qdrant.mode is server")
            api_key = os.getenv(self.config.api_key_env) if self.config.api_key_env else None
            self._client = QdrantClient(url=self.config.url, api_key=api_key)
        elif self.config.mode == "local":
            if self.config.storage_path is None:
                raise ValueError("qdrant.storage_path is required when qdrant.mode is local")
            self.config.storage_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.config.storage_path))
        else:
            raise ValueError("qdrant.mode must be either 'server' or 'local'")

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def url(self) -> str | None:
        return self.config.url

    @property
    def storage_path(self) -> Path | None:
        return self.config.storage_path

    @property
    def collection_name(self) -> str:
        return self.config.collection_name

    def ensure_collection(self) -> None:
        """Create or recreate the target collection explicitly."""
        exists = self._client.collection_exists(collection_name=self.config.collection_name)

        if self.config.recreate_collection and exists:
            self._client.delete_collection(collection_name=self.config.collection_name)
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self._client.get_embedding_size(self.config.embedding_model),
                    distance=_parse_distance(self.config.distance),
                ),
            )

    def collection_exists(self) -> bool:
        """Check whether the target collection exists."""
        return self._client.collection_exists(collection_name=self.config.collection_name)

    def points_count(self) -> int:
        """Return current number of points in the target collection."""
        info = self._client.get_collection(collection_name=self.config.collection_name)
        count = getattr(info, "points_count", 0)
        return int(count) if count is not None else 0

    def upsert_texts(
        self,
        *,
        documents: list[str],
        payloads: list[dict[str, object]],
        ids: list[str],
    ) -> int:
        """Encode documents with the configured embedding model and upload dense vectors."""
        if not (len(documents) == len(payloads) == len(ids)):
            raise ValueError("documents, payloads, and ids must have the same length")

        if not documents:
            return 0

        inserted = 0
        batch_size = self.config.batch_size

        # Process documents in chunks to avoid sending everything in one request
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            batch_docs = documents[start:end]
            batch_payloads = payloads[start:end]
            batch_ids = ids[start:end]
            self._client.upload_collection(
                collection_name=self.config.collection_name,
                vectors=[
                    models.Document(text=document, model=self.config.embedding_model)
                    for document in batch_docs
                ],
                payload=batch_payloads,
                ids=batch_ids,
            )
            inserted += len(batch_docs)

        return inserted

    def search_text(self, query_text: str, top_k: int) -> list[dict[str, object]]:
        """Run dense semantic retrieval for a text query and return top-k KB entries."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        response = self._client.query_points(
            collection_name=self.config.collection_name,
            query=models.Document(text=query_text.strip(), model=self.config.embedding_model),
            limit=top_k,
            with_payload=True,
        )

        points = getattr(response, "points", [])
        results: list[dict[str, object]] = []
        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "entry_id": _to_string(payload.get("entry_id")),
                    "synset_id": _to_string(payload.get("synset_id")),
                    "class_name": _to_string(payload.get("class_name")),
                    "image_paths": _to_string_list(payload.get("image_paths")),
                    "description": _to_string(payload.get("description")),
                    "score": float(getattr(point, "score", 0.0)),
                }
            )
        # info = self._client.get_collection(collection_name=self.config.collection_name)
        # print(
        #     f"\n================ search_text ===========\ncollection info\n{info}\n"
        #     f"query_text:{query_text}\npoints:\n{points}"
        # )
        return results


def _parse_distance(distance: str):
    normalized = distance.strip().lower()
    mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclidean": models.Distance.EUCLID,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported qdrant distance: {distance!r}. Expected one of: cosine, dot, euclidean."
        ) from exc


def build_qdrant_store_config(
    qdrant_cfg: Mapping[str, object],
    *,
    recreate_collection: bool | None = None,
) -> QdrantStoreConfig:
    """Build a QdrantStoreConfig from a qdrant YAML section."""
    mode = str(qdrant_cfg.get("mode", "server")).strip().lower()
    if mode not in {"server", "local"}:
        raise ValueError("qdrant.mode must be either 'server' or 'local'")

    collection_name = _require_non_empty_config_str(qdrant_cfg, "collection_name")
    embedding_model = _require_non_empty_config_str(qdrant_cfg, "embedding_model")
    distance = _require_non_empty_config_str(qdrant_cfg, "distance")

    batch_size = int(qdrant_cfg.get("batch_size", 32))
    if batch_size <= 0:
        raise ValueError("qdrant.batch_size must be > 0")

    if recreate_collection is None:
        recreate = bool(qdrant_cfg.get("recreate_collection", False))
    else:
        recreate = recreate_collection

    url: str | None = None
    storage_path: Path | None = None
    if mode == "server":
        url = _require_non_empty_config_str(qdrant_cfg, "url")
    else:
        storage_path = Path(_require_non_empty_config_str(qdrant_cfg, "storage_path"))

    return QdrantStoreConfig(
        mode=mode,
        url=url,
        api_key_env=_optional_non_empty_config_str(qdrant_cfg, "api_key_env"),
        storage_path=storage_path,
        collection_name=collection_name,
        recreate_collection=recreate,
        embedding_model=embedding_model,
        distance=distance,
        batch_size=batch_size,
    )


def _require_non_empty_config_str(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"qdrant.{key} must be a non-empty string")
    return value.strip()


def _optional_non_empty_config_str(config: Mapping[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"qdrant.{key} must be a non-empty string when provided")
    return value.strip()


def _to_string(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _to_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        normalized = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
        return normalized
    return []


class LocalQdrantConfig(QdrantStoreConfig):
    """Backward-compatible config for embedded local Qdrant."""

    def __init__(
        self,
        *,
        storage_path: Path,
        collection_name: str,
        recreate_collection: bool,
        embedding_model: str,
        distance: str,
        batch_size: int,
    ) -> None:
        object.__setattr__(self, "mode", "local")
        object.__setattr__(self, "url", None)
        object.__setattr__(self, "api_key_env", None)
        object.__setattr__(self, "storage_path", storage_path)
        object.__setattr__(self, "collection_name", collection_name)
        object.__setattr__(self, "recreate_collection", recreate_collection)
        object.__setattr__(self, "embedding_model", embedding_model)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "batch_size", batch_size)


LocalQdrantStore = QdrantStore
