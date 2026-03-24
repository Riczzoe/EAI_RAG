"""Local Qdrant client wrapper for text-only KB ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass(frozen=True)
class LocalQdrantConfig:
    storage_path: Path
    collection_name: str
    recreate_collection: bool
    embedding_model: str
    distance: str
    batch_size: int


class LocalQdrantStore:
    """Manage a local Qdrant collection and text upserts."""

    def __init__(self, config: LocalQdrantConfig) -> None:
        self.config = config
        self.config.storage_path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.config.storage_path))

    @property
    def storage_path(self) -> Path:
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
        """Upload text documents with FastEmbed-backed Document vectors."""
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
        """Retrieve top-k KB entries by text query."""
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
