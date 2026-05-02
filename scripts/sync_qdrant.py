#!/usr/bin/env python3
"""Build a Qdrant store from processed KB entries."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kb.upsert_qdrant import run_from_config


def main() -> None:
    config_path = Path("configs/qdrant.yaml")
    try:
        qdrant_cfg, stats = run_from_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"sync_qdrant failed: {exc}") from exc

    print("=== sync_qdrant ===")
    print(f"Config: {config_path}")
    print(f"Mode: {qdrant_cfg.mode}")
    print(f"Collection: {qdrant_cfg.collection_name}")
    if qdrant_cfg.mode == "server":
        print(f"URL: {qdrant_cfg.url}")
    else:
        print(f"Storage path: {qdrant_cfg.storage_path}")
    print(f"Entries source: {qdrant_cfg.entries_jsonl_path}")
    print(f"Embedding model: {qdrant_cfg.embedding_model}")
    print(f"Total rows: {stats.total_rows}")
    print(f"Inserted rows: {stats.inserted_rows}")
    print(f"Skipped rows: {stats.skipped_rows}")
    print(f"Failed rows: {stats.failed_rows}")


if __name__ == "__main__":
    main()
