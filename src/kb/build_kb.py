#!/usr/bin/env python3
"""Build class-level KB entries from descriptions and sampled images."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from src.utils.io import load_yaml
from src.utils.synset_map import load_synset_mapping


@dataclass(frozen=True)
class KBConfig:
    sampled_kb_images_dir: Path
    kb_descriptions_dir: Path
    kb_entries_jsonl_path: Path
    overwrite_kb_entries: bool
    synset_mapping_path: Path
    project_root: Path

@dataclass(frozen=True)
class BuildKBStats:
    total_description_files: int
    written_entries: int
    skipped_missing_image_dirs: int
    skipped_empty_image_dirs: int
    skipped_empty_descriptions: int


def get_kb_config(
    kb_config: Mapping[str, object],
    dataset_config: Mapping[str, object],
    *,
    project_root: Path,
) -> KBConfig:
    """Parse kb and dataset configs for KB entry building."""
    kb_cfg = kb_config.get("kb")
    if not isinstance(kb_cfg, Mapping):
        raise ValueError("`kb` section is required in configs/kb.yaml")

    required_kb_keys = [
        "sampled_kb_images_dir",
        "kb_descriptions_dir",
        "kb_entries_jsonl_path",
        "overwrite_kb_entries",
    ]
    for key in required_kb_keys:
        if key not in kb_cfg:
            raise ValueError(f"Missing required config key: kb.{key}")

    dataset_cfg = dataset_config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("`dataset` section is required in configs/dataset.yaml")
    if "loc_synset_mapping_path" not in dataset_cfg:
        raise ValueError("Missing required config key: dataset.loc_synset_mapping_path")

    return KBConfig(
        sampled_kb_images_dir=Path(str(kb_cfg["sampled_kb_images_dir"])),
        kb_descriptions_dir=Path(str(kb_cfg["kb_descriptions_dir"])),
        kb_entries_jsonl_path=Path(str(kb_cfg["kb_entries_jsonl_path"])),
        overwrite_kb_entries=bool(kb_cfg["overwrite_kb_entries"]),
        synset_mapping_path=Path(str(dataset_cfg["loc_synset_mapping_path"])),
        project_root=project_root.resolve(),
    )

def _to_project_relative_posix(path: Path, project_root: Path) -> str:
    """return a project-relative posix path for serialization."""
    absolute_path = path.resolve()
    try:
        relative_path = absolute_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Path is outside project root: {absolute_path} (project root: {project_root})"
        ) from exc
    return relative_path.as_posix()

def build_kb_entries(config: KBConfig) -> BuildKBStats:
    """Build entries.jsonl from description files and sampled image directories."""
    if not config.overwrite_kb_entries:
        raise ValueError("`kb.overwrite_kb_entries: false` is not implemented in v1.")

    if not config.kb_descriptions_dir.exists():
        raise FileNotFoundError(f"Description directory not found: {config.kb_descriptions_dir}")

    # Resolve synset_id -> class_name once up front
    synset_mapping = load_synset_mapping(config.synset_mapping_path)

    # Use a stable ordering for reproducible output.
    description_files = sorted(
        [p for p in config.kb_descriptions_dir.glob("*.txt") if p.is_file()],
        key=lambda p: p.name,
    )

    config.kb_entries_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    written_entries = 0
    skipped_missing_image_dirs = 0
    skipped_empty_image_dirs = 0
    skipped_empty_descriptions = 0

    with config.kb_entries_jsonl_path.open("w", encoding="utf-8") as out_f:
        for description_path in description_files:
            synset_id = description_path.stem
            description = description_path.read_text(encoding="utf-8").strip()
            if not description:
                skipped_empty_descriptions += 1
                continue

            image_dir = config.sampled_kb_images_dir / synset_id
            if not image_dir.exists() or not image_dir.is_dir():
                skipped_missing_image_dirs += 1
                continue

            # Keep file ordering deterministic for downstream diffs/debugging.
            image_files = sorted([p for p in image_dir.iterdir() if p.is_file()], key=lambda p: p.name)
            if not image_files:
                skipped_empty_image_dirs += 1
                continue

            class_name = synset_mapping.get(synset_id)
            if class_name is None:
                raise ValueError(f"Missing synset mapping for synset_id: {synset_id}")

            image_paths = [
                _to_project_relative_posix(image_path, config.project_root)
                for image_path in image_files
            ]
            if not image_paths:
                skipped_empty_image_dirs += 1
                continue

            entry = {
                "entry_id": f"kb_{synset_id}",
                "synset_id": synset_id,
                "class_name": class_name,
                "description": description,
                "image_paths": image_paths,
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written_entries += 1

    return BuildKBStats(
        total_description_files=len(description_files),
        written_entries=written_entries,
        skipped_missing_image_dirs=skipped_missing_image_dirs,
        skipped_empty_image_dirs=skipped_empty_image_dirs,
        skipped_empty_descriptions=skipped_empty_descriptions,
    )

def run_from_config(
    kb_config_path: Path,
    dataset_config_path: Path | None = None,
) -> tuple[KBConfig, BuildKBStats]:
    """Load configs and build KB entries."""
    if dataset_config_path is None:
        dataset_config_path = Path("configs/dataset.yaml")

    kb_config = load_yaml(kb_config_path)
    dataset_config = load_yaml(dataset_config_path)
    parsed = get_kb_config(kb_config, dataset_config, project_root=Path.cwd())
    stats = build_kb_entries(parsed)
    return parsed, stats
