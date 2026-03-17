#!/usr/bin/env python3
"""Sample ImageNet images per class as KB candidates.

Output layout:
    data/interim/sampled_kb_images/<synset>(<class_name>)/<image_file>
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

from src.utils.io import load_yaml

@dataclass(frozen=True)
class DatasetConfig:
    source_root: Path
    mapping_path: Path
    output_root: Path
    seed: int
    clear_output: bool
    dry_run: bool
    sample_count: int

def load_synset_mapping(mapping_path: Path) -> dict[str, str]:
    synset_to_name: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            synset, names = parts
            primary_name = names.split(",")[0].strip()
            synset_to_name[synset] = primary_name
    return synset_to_name

def sanitize_class_name(class_name: str) -> str:
    # Keep names human-readable while removing filesystem-problematic chars.
    class_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", class_name)
    class_name = re.sub(r"\s+", " ", class_name).strip()
    return class_name if class_name else "unknown"

def get_dataset_config(config: Mapping[str, object]) -> DatasetConfig:
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("`dataset` section is required in configs/dataset.yaml")

    required_keys = [
        "imagenet_val_dir",
        "loc_synset_mapping_path",
        "sampled_kb_images_dir",
    ]
    for key in required_keys:
        if key not in dataset_cfg:
            raise ValueError(f"Missing required config key: dataset.{key}")

    only_keep_one = bool(dataset_cfg.get("only_keep_one_kb_reference", True))
    kb_images_per_class = int(dataset_cfg.get("kb_images_per_class", 1))
    sample_count = 1 if only_keep_one else kb_images_per_class
    if sample_count < 1:
        raise ValueError("dataset.kb_images_per_class must be >= 1")

    return DatasetConfig(
        source_root=Path(str(dataset_cfg["imagenet_val_dir"])),
        mapping_path=Path(str(dataset_cfg["loc_synset_mapping_path"])),
        output_root=Path(str(dataset_cfg["sampled_kb_images_dir"])),
        seed=int(dataset_cfg.get("random_seed", 42)),
        clear_output=bool(dataset_cfg.get("clear_output_dir", False)),
        dry_run=bool(dataset_cfg.get("dry_run", False)),
        sample_count=sample_count,
    )

def sample_k_per_class(
    cfg: DatasetConfig,
    synset_mapping: dict[str, str],
) -> tuple[int, int, int]:
    source_root = cfg.source_root
    output_root = cfg.output_root
    sample_count = cfg.sample_count
    seed = cfg.seed
    clear_output = cfg.clear_output
    dry_run = cfg.dry_run

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    if clear_output and output_root.exists() and not dry_run:
        shutil.rmtree(output_root)

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    rng = random.Random(seed)

    total = len(class_dirs)
    sampled = 0
    skipped = 0

    for class_dir in class_dirs:
        synset_id = class_dir.name
        class_name = sanitize_class_name(synset_mapping.get(synset_id, synset_id))
        target_class_dir = output_root / f"{synset_id}({class_name})"

        image_files = sorted([p for p in class_dir.iterdir() if p.is_file()], key=lambda p: p.name)
        if not image_files:
            skipped += 1
            continue

        if sample_count > len(image_files):
            selected_files = image_files
        elif sample_count == 1:
            selected_files = [rng.choice(image_files)]
        else:
            selected_files = rng.sample(image_files, sample_count)
        sampled += 1

        for selected in selected_files:
            if dry_run:
                print(f"[DRY-RUN] {synset_id} -> {target_class_dir.name}/{selected.name}")
                continue
            target_class_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, target_class_dir / selected.name)

    return total, sampled, skipped


def run_from_config(config_path: Path) -> tuple[DatasetConfig, tuple[int, int, int]]:
    config = load_yaml(config_path)
    dataset_cfg = get_dataset_config(config)
    mapping = load_synset_mapping(dataset_cfg.mapping_path)
    total, sampled, skipped = sample_k_per_class(
        cfg=dataset_cfg,
        synset_mapping=mapping,
    )
    return dataset_cfg, (total, sampled, skipped)
