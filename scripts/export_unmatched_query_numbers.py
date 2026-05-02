#!/usr/bin/env python3
"""Append numeric query ids for unmatched RAG evaluation details to JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path


DEFAULT_INPUT_PATH = Path("outputs/evaluation/eval_rag_result.json")
DEFAULT_OUTPUT_PATH = Path("outputs/evaluation/unmatched_query_numbers.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append numeric query ids from unmatched eval_rag_result.json entries to JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input eval result JSON path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Write each query number only once, preserving first-seen order.",
    )
    args = parser.parse_args()

    result = _load_json(args.input)
    query_numbers = list(_iter_unmatched_query_numbers(result))
    if args.dedupe:
        query_numbers = _dedupe_preserve_order(query_numbers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "input_path": args.input.as_posix(),
        "unmatched_count": len(query_numbers),
        "query_numbers": query_numbers,
    }
    with args.output.open("a", encoding="utf-8") as output_f:
        output_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"input_path: {args.input}")
    print(f"output_path: {args.output}")
    print(f"unmatched_count: {len(query_numbers)}")


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Evaluation result JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Evaluation result root must be a JSON object: {path}")
    return data


def _iter_unmatched_query_numbers(result: Mapping[str, object]) -> Iterable[str]:
    for query in _iter_query_details(result):
        if query.get("matched") is not False:
            continue
        query_id = query.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            continue
        yield _extract_query_number(query_id)


def _iter_query_details(result: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    blocks = result.get("results")
    if not isinstance(blocks, list):
        blocks = result.get("conditions")
    if not isinstance(blocks, list):
        return

    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        queries = block.get("queries")
        if not isinstance(queries, list):
            continue
        for query in queries:
            if isinstance(query, Mapping):
                yield query


def _extract_query_number(query_id: str) -> str:
    match = re.search(r"(\d+)$", query_id.strip())
    if match is None:
        return query_id.strip()
    return match.group(1)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


if __name__ == "__main__":
    main()
