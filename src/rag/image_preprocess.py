"""Image preprocessing helpers for RAG context images."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse


_SUPPORTED_ALGORITHMS = {"nearest", "bilinear", "bicubic", "lanczos"}
_SUPPORTED_OUTPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resize_image_paths(
    image_paths: list[str],
    resize_config: Mapping[str, object] | None,
) -> list[str]:
    """Resize context images according to config and return paths for VLM input."""
    if not image_paths:
        return []

    if resize_config is None or not bool(resize_config.get("enabled", False)):
        return image_paths

    width = _read_positive_int(resize_config, "width")
    height = _read_positive_int(resize_config, "height")
    algorithm = _read_algorithm(resize_config)
    output_dir = _read_output_dir(resize_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    resample_filter = _get_resample_filter(algorithm)
    resized_paths: list[str] = []

    for image_path in image_paths:
        source_path = _resolve_image_path(image_path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"RAG image not found: {source_path}")

        target_path = output_dir / _build_cache_filename(source_path)
        _resize_one_image(
            source_path=source_path,
            target_path=target_path,
            width=width,
            height=height,
            resample_filter=resample_filter,
        )
        resized_paths.append(target_path.as_posix())

    return resized_paths


def _resize_one_image(
    *,
    source_path: Path,
    target_path: Path,
    width: int,
    height: int,
    resample_filter: int,
) -> None:
    import cv2

    image = cv2.imread(source_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read RAG image with OpenCV: {source_path}")

    resized = cv2.resize(image, (width, height), interpolation=resample_filter)
    if not cv2.imwrite(target_path.as_posix(), resized):
        raise ValueError(f"Failed to write resized RAG image with OpenCV: {target_path}")


def _read_positive_int(config: Mapping[str, object], key: str) -> int:
    if key not in config:
        raise ValueError(f"image_resize.{key} is required when image resizing is enabled")

    try:
        value = int(config[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"image_resize.{key} must be a positive integer") from exc

    if value <= 0:
        raise ValueError(f"image_resize.{key} must be a positive integer")
    return value


def _read_algorithm(config: Mapping[str, object]) -> str:
    algorithm = str(config.get("algorithm", "bicubic")).strip().lower()
    if algorithm not in _SUPPORTED_ALGORITHMS:
        expected = ", ".join(sorted(_SUPPORTED_ALGORITHMS))
        raise ValueError(
            f"Unsupported image_resize.algorithm: {algorithm!r}. Expected one of: {expected}."
        )
    return algorithm


def _read_output_dir(config: Mapping[str, object]) -> Path:
    raw_output_dir = str(config.get("output_dir", "outputs/rag/resized_images")).strip()
    if not raw_output_dir:
        raise ValueError("image_resize.output_dir must be a non-empty path")
    return Path(raw_output_dir)


def _resolve_image_path(path_text: str) -> Path:
    if not isinstance(path_text, str) or not path_text.strip():
        raise ValueError("image path must be a non-empty string")

    normalized = path_text.strip()
    if normalized.startswith("file://"):
        parsed = urlparse(normalized)
        if parsed.netloc and parsed.netloc not in {"localhost", "127.0.0.1"}:
            raise ValueError(f"Only local file:// image URIs are supported: {path_text}")
        path = Path(unquote(parsed.path))
    else:
        path = Path(normalized).expanduser()

    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _build_cache_filename(source_path: Path) -> str:
    digest = hashlib.sha256(source_path.as_posix().encode("utf-8")).hexdigest()[:16]
    suffix = source_path.suffix.lower()
    if suffix not in _SUPPORTED_OUTPUT_SUFFIXES:
        suffix = ".jpg"
    return f"{source_path.stem}-{digest}{suffix}"


def _get_resample_filter(algorithm: str) -> int:
    import cv2

    return {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[algorithm]
