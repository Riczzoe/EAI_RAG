#!/usr/bin/env python3
"""Extract one class description per wnid from Imagenet-Wiki records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

from src.utils.io import load_pickle, load_yaml


# Match one or more inline whitespace characters:
# spaces, tabs, form feeds, or vertical tabs.
# Note that this regex intentionally does NOT match newline characters,
# because paragraph boundaries are preserved until later processing.
_INLINE_SPACE_RE = re.compile(r"[ \t\f\v]+")

@dataclass(frozen=True)
class DescriptionConfig:
    pkl_path: Path
    output_dir: Path
    article_index: int
    max_chars: int
    clear_output_dir: bool


def get_description_config(config: Mapping[str, object]) -> DescriptionConfig:
    """Parse description-extraction config from dataset YAML."""
    dataset_cfg = config.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        raise ValueError("`dataset` section is required in configs/dataset.yaml")

    required_keys = [
        "imagenet_wiki_trainval_pkl_path",
        "kb_descriptions_dir",
        "description_article_index",
        "description_max_chars",
        "clear_kb_descriptions_dir",
    ]
    for key in required_keys:
        if key not in dataset_cfg:
            raise ValueError(f"Missing required config key: dataset.{key}")

    article_index = int(dataset_cfg["description_article_index"])
    if article_index < 0:
        raise ValueError("dataset.description_article_index must be >= 0")

    max_chars = int(dataset_cfg["description_max_chars"])
    if max_chars <= 0:
        raise ValueError("dataset.description_max_chars must be > 0")

    return DescriptionConfig(
        pkl_path=Path(str(dataset_cfg["imagenet_wiki_trainval_pkl_path"])),
        output_dir=Path(str(dataset_cfg["kb_descriptions_dir"])),
        article_index=article_index,
        max_chars=max_chars,
        clear_output_dir=bool(dataset_cfg["clear_kb_descriptions_dir"]),
    )


def _normalize_article(raw_text: str) -> list[str]:
    """Normalize newlines and inline spaces, preserving blank-line paragraph breaks."""

    # Convert all line endings to Unix style (\n), then strip leading/trailing
    # whitespace from the entire text block
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    lines: list[str] = []
    for line in text.split("\n"):
        # Collapse repeated inline whitespace inside each line into a single space.
        line = _INLINE_SPACE_RE.sub(" ", line).strip()
        lines.append(line)
    return lines


def _split_paragraphs(lines: list[str]) -> list[str]:
    """Split normalized lines into paragraphs using blank lines as separators."""
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        # An empty line indicates the end of the current paragraph.
        # Join all lines collected for the current paragraph into one single string.
        if line == "":
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue

        # Non-empty lines belong to the current paragraph.
        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    # Remove any accidental empty paragraphs.
    return [p for p in paragraphs if p]


def _is_short_title(paragraph: str) -> bool:
    """Heuristically decide whether a paragraph looks like a short title."""

    # Empty text is not considered a title.
    if not paragraph:
        return False

    # Heuristic rule:
    # - short total length
    # - small number of words
    # - does not end like a sentence
    #
    # This is not perfect, but it works reasonably well for skipping
    # title-like first paragraphs in article text.
    words = paragraph.split()
    ends_like_sentence = paragraph[-1] in ".!?;:"
    return len(paragraph) <= 80 and len(words) <= 12 and not ends_like_sentence


def extract_first_non_title_description(raw_text: str, max_chars: int) -> tuple[str, bool]:
    """Get the first non-title paragraph and apply char truncation."""
    lines = _normalize_article(raw_text)
    paragraphs = _split_paragraphs(lines)
    if not paragraphs:
        return "", False

    # If the first paragraph looks like a title, skip it and use the next one.
    # Otherwise, use the first paragraph directly.
    start_idx = 1 if _is_short_title(paragraphs[0]) else 0
    if start_idx >= len(paragraphs):
        return "", False

    description = paragraphs[start_idx].strip()
    if not description:
        return "", False

    truncated = False
    if len(description) > max_chars:
        description = description[:max_chars].strip()
        truncated = True

    return description, truncated


def _validate_records(data: object) -> list[Mapping[str, object]]:
    """Validate the pickle structure and convert it into a list of records."""

    # The top-level pickle object must be a mapping, for example:
    # {
    #   "some_key": {"wnid": "...", "articles": [...]},
    #   ...
    # }
    if not isinstance(data, Mapping):
        raise ValueError(
            "Imagenet-Wiki pickle top-level must be a mapping: Mapping[Any, Mapping[str, object]]."
        )
    if len(data) == 0:
        raise ValueError("Imagenet-Wiki pickle contains zero records.")

    records: list[Mapping[str, object]] = []
    for key, record in data.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Record for key {key!r} must be a mapping.")
        records.append(record)
    return records


def _find_duplicate_wnids(records: list[Mapping[str, object]]) -> list[str]:
    """Return all duplicate non-empty wnid values found in the records."""
    counter: Counter[str] = Counter()
    for record in records:
        wnid = record.get("wnid")
        if isinstance(wnid, str) and wnid.strip():
            counter[wnid.strip()] += 1

    # Keep only wnids that appear more than once.
    return sorted([wnid for wnid, count in counter.items() if count > 1])


def extract_descriptions(config: DescriptionConfig) -> tuple[int, int, int, int]:
    """Write one `<wnid>.txt` per record and return total/written/skipped/truncated."""
    raw_data = load_pickle(config.pkl_path)
    records = _validate_records(raw_data)

    # Ensure that no wnid appears more than once, because output files are
    # named "<wnid>.txt" and duplicates would overwrite each other.
    duplicates = _find_duplicate_wnids(records)
    if duplicates:
        sample = ", ".join(duplicates[:10])
        suffix = "..." if len(duplicates) > 10 else ""
        raise ValueError(
            f"Duplicate wnid values found ({len(duplicates)}): {sample}{suffix}. "
            "Output conflict is not allowed."
        )

    if config.clear_output_dir and config.output_dir.exists():
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    written = 0
    skipped = 0
    truncated = 0

    for record in records:
        wnid = record.get("wnid")
        if not isinstance(wnid, str) or not wnid.strip():
            skipped += 1
            continue
        wnid = wnid.strip()

        articles = record.get("articles")
        if not isinstance(articles, list) or len(articles) == 0:
            skipped += 1
            continue
        # Skip records when the configured article index is out of range.
        if config.article_index >= len(articles):
            skipped += 1
            continue

        article = articles[config.article_index]
        if not isinstance(article, str):
            skipped += 1
            continue

        description, is_truncated = extract_first_non_title_description(
            article,
            config.max_chars,
        )
        if not description:
            skipped += 1
            continue

        # Write the description to "<wnid>.txt".
        target_path = config.output_dir / f"{wnid}.txt"
        target_path.write_text(f"{description}\n", encoding="utf-8")
        written += 1
        if is_truncated:
            truncated += 1

    return total, written, skipped, truncated


def run_from_config(config_path: Path) -> tuple[DescriptionConfig, tuple[int, int, int, int]]:
    """Load config and run description extraction."""
    config = load_yaml(config_path)
    description_cfg = get_description_config(config)
    stats = extract_descriptions(description_cfg)
    return description_cfg, stats
