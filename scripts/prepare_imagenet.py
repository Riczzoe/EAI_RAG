#!/usr/bin/env python3
"""Prepare ImageNet interim assets for this project."""

from __future__ import annotations

from pathlib import Path

from src.data.split_kb import run_from_config


def main() -> None:
    config_path = Path("configs/dataset.yaml")
    dataset_cfg, (total, sampled, skipped) = run_from_config(config_path)
    print("=== prepare_imagenet ===")
    print(f"Config: {config_path}")
    print(f"Source: {dataset_cfg.source_root}")
    print(f"Output: {dataset_cfg.output_root}")
    print(f"Total classes: {total}")
    print(f"Sampled classes: {sampled}")
    print(f"Skipped classes (empty): {skipped}")
    print(f"Images per class: {dataset_cfg.sample_count}")


if __name__ == "__main__":
    main()
