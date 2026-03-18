#!/usr/bin/env python3
"""Prepare KB class descriptions from Imagenet-Wiki article records."""

from __future__ import annotations

from pathlib import Path

from src.data.extract_description import run_from_config


def main() -> None:
    config_path = Path("configs/dataset.yaml")
    try:
        description_cfg, stats = run_from_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"prepare_description failed: {exc}") from exc
    total, written, skipped, truncated = stats

    print("=== prepare_description ===")
    print(f"Config: {config_path}")
    print(f"Input pickle: {description_cfg.pkl_path}")
    print(f"Output directory: {description_cfg.output_dir}")
    print(f"Article index: {description_cfg.article_index}")
    print(f"Max chars: {description_cfg.max_chars}")
    print(f"Total records: {total}")
    print(f"Written files: {written}")
    print(f"Skipped records: {skipped}")
    print(f"Truncated records: {truncated}")


if __name__ == "__main__":
    main()
