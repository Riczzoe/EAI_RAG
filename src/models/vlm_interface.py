"""OpenAI Chat Completions compatible VLM interface."""

from __future__ import annotations

from collections.abc import Mapping
import os

from openai import OpenAI


def call_vlm(
    *,
    messages: list[dict[str, object]],
    inference_config: Mapping[str, object],
) -> str:
    """Call a VLM through the OpenAI Chat Completions API format."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    api_format = _require_non_empty_config_str(inference_config, "api_format")
    if api_format != "openai_completions":
        raise ValueError(
            f"Unsupported inference.api_format: {api_format!r}. "
            "Expected: openai_completions."
        )

    model_name = _require_non_empty_config_str(inference_config, "model_name")
    api_key_env = _require_non_empty_config_str(inference_config, "api_key_env")
    base_url = _require_non_empty_config_str(inference_config, "base_url")

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable `{api_key_env}` is required for VLM calls")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise ValueError("OpenAI Chat Completions response content must be a string")
    return content


def _require_non_empty_config_str(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"inference.{key} must be a non-empty string")
    return value.strip()
