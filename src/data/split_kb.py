#!/usr/bin/env python3
"""Sample ImageNet images per class as KB candidates, then move to processed KB images."""

from __future__ import annotations

from dataclasses import dataclass
import random
import shutil
from collections.abc import Mapping
from pathlib import Path

from src.utils.io import load_yaml

@dataclass(frozen=True)
class DatasetConfig:
    source_root: Path
    output_root: Path
    seed: int
    clear_output: bool
    dry_run: bool
    sample_count: int

def get_dataset_config(config: Mapping[str, object]) -> DatasetConfig:
    """Parse the dataset section from the loaded YAML config."""
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("`dataset` section is required in configs/dataset.yaml")

    required_keys = [
        "imagenet_val_dir",
        "sampled_kb_images_dir",
    ]
    for key in required_keys:
        if key not in dataset_cfg:
            raise ValueError(f"Missing required config key: dataset.{key}")

    # Decide how many images to keep for each class.
    only_keep_one = bool(dataset_cfg.get("only_keep_one_kb_reference", True))
    kb_images_per_class = int(dataset_cfg.get("kb_images_per_class", 1))
    sample_count = 1 if only_keep_one else kb_images_per_class
    if sample_count < 1:
        raise ValueError("dataset.kb_images_per_class must be >= 1")

    return DatasetConfig(
        source_root=Path(str(dataset_cfg["imagenet_val_dir"])),
        output_root=Path(str(dataset_cfg["sampled_kb_images_dir"])),
        seed=int(dataset_cfg.get("random_seed", 42)),
        clear_output=bool(dataset_cfg.get("clear_output_dir", False)),
        dry_run=bool(dataset_cfg.get("dry_run", False)),
        sample_count=sample_count,
    )

def sample_k_per_class(
    cfg: DatasetConfig,
) -> tuple[int, int, int]:
    """Sample k images from each class directory and copy them to the output."""
    source_root = cfg.source_root
    output_root = cfg.output_root
    sample_count = cfg.sample_count
    seed = cfg.seed
    clear_output = cfg.clear_output
    dry_run = cfg.dry_run

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    # Optionally clear the old output directory.
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
        target_class_dir = output_root / synset_id

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


def move_sampled_images_to_kb_dir(
    sampled_root: Path,
    kb_images_root: Path,
    *,
    dry_run: bool,
    clear_target: bool,
) -> tuple[int, int]:
    """Move all sampled class image folders into the processed KB images directory."""
    if not sampled_root.exists() or not sampled_root.is_dir():
        return 0, 0

    if clear_target and kb_images_root.exists() and not dry_run:
        shutil.rmtree(kb_images_root)

    class_dirs = sorted([p for p in sampled_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    moved_classes = 0
    moved_files = 0

    if not dry_run:
        kb_images_root.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        image_files = [p for p in class_dir.iterdir() if p.is_file()]
        moved_files += len(image_files)

        target_class_dir = kb_images_root / class_dir.name
        if dry_run:
            print(f"[DRY-RUN] move {class_dir} -> {target_class_dir}")
            moved_classes += 1
            continue

        if target_class_dir.exists():
            shutil.rmtree(target_class_dir)
        shutil.move(str(class_dir), str(target_class_dir))
        moved_classes += 1

    if not dry_run and sampled_root.exists() and not any(sampled_root.iterdir()):
        sampled_root.rmdir()

    return moved_classes, moved_files


def _read_kb_images_dir(kb_config_path: Path) -> Path:
    kb_config = load_yaml(kb_config_path)
    kb_cfg = kb_config.get("kb")
    if not isinstance(kb_cfg, Mapping):
        raise ValueError("`kb` section is required in configs/kb.yaml")
    if "kb_images_dir" not in kb_cfg:
        raise ValueError("Missing required config key: kb.kb_images_dir")
    return Path(str(kb_cfg["kb_images_dir"]))


def run_from_config(
    config_path: Path,
    kb_config_path: Path | None = None,
) -> tuple[DatasetConfig, tuple[int, int, int]]:
    config = load_yaml(config_path)
    dataset_cfg = get_dataset_config(config)
    total, sampled, skipped = sample_k_per_class(cfg=dataset_cfg)

    if kb_config_path is None:
        kb_config_path = Path("configs/kb.yaml")
    kb_images_dir = _read_kb_images_dir(kb_config_path)
    move_sampled_images_to_kb_dir(
        sampled_root=dataset_cfg.output_root,
        kb_images_root=kb_images_dir,
        dry_run=dataset_cfg.dry_run,
        clear_target=dataset_cfg.clear_output,
    )

    return dataset_cfg, (total, sampled, skipped)
