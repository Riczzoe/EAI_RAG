"""I/O helpers for project-wide file loading."""

from __future__ import annotations

import pickle
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load YAML file and return dict."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Enforce a dictionary-like root structure.
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")
    return data


def load_pickle(path: str | Path) -> object:
    """Load pickle file and return its Python object."""
    pkl_path = Path(path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    try:
        # Pickle files are binary, so they must be opened with "rb".
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    except EOFError as exc:
        raise ValueError(f"Pickle file is empty: {pkl_path}") from exc
    except pickle.UnpicklingError as exc:
        raise ValueError(f"Pickle file is not readable: {pkl_path}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to load pickle file: {pkl_path}") from exc
