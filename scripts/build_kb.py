#!/usr/bin/env python3
"""Build processed KB entries from sampled images and descriptions."""

from __future__ import annotations

from pathlib import Path

from src.kb.build_kb import run_from_config


def main() -> None:
    kb_config_path = Path("configs/kb.yaml")
    try:
        kb_cfg, stats = run_from_config(kb_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"build_kb failed: {exc}") from exc

    print("=== build_kb ===")
    print(f"Config: {kb_config_path}")
    print(f"Sampled images dir: {kb_cfg.sampled_kb_images_dir}")
    print(f"Descriptions dir: {kb_cfg.kb_descriptions_dir}")
    print(f"Output entries: {kb_cfg.kb_entries_jsonl_path}")
    print(f"Total description files: {stats.total_description_files}")
    print(f"Written entries: {stats.written_entries}")
    print(f"Skipped missing image dirs: {stats.skipped_missing_image_dirs}")
    print(f"Skipped empty image dirs: {stats.skipped_empty_image_dirs}")
    print(f"Skipped empty descriptions: {stats.skipped_empty_descriptions}")


if __name__ == "__main__":
    main()
