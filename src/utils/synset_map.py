"""Utilities for ImageNet synset-to-class-name mapping."""

from __future__ import annotations

from pathlib import Path
import re


def load_synset_mapping(mapping_path: Path) -> dict[str, str]:
    """Load synset-to-class-name mapping from a text file."""
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

            # Keep only the first class name before the comma.
            primary_name = names.split(",")[0].strip()
            synset_to_name[synset] = primary_name
    return synset_to_name


def sanitize_class_name(class_name: str) -> str:
    """Make class names safe and readable for filesystem paths."""
    # Remove filesystem-problematic characters.
    class_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", class_name)

    # Normalize repeated spaces.
    class_name = re.sub(r"\s+", " ", class_name).strip()

    # Fallback in case the name becomes empty after cleaning.
    return class_name if class_name else "unknown"
