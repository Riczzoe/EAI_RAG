"""DashScope VLM interface (v1)."""

from __future__ import annotations
import os
import dashscope

def call_vlm(messages: list[dict[str, object]], model_name: str = "qwen-vl-plus") -> object:
    """Public entrypoint that routes calls by model_name."""
    if model_name.startswith("qwen"):
        return call_qwen(messages, model_name)
    raise ValueError(f"Unsupported model_name: {model_name!r}")

def call_qwen(messages: list[dict[str, object]], model_name: str = "qwen-vl-plus") -> object:
    """Call DashScope Qwen multimodal models."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
    response = dashscope.MultiModalConversation.call(
        api_key = os.getenv("DASHSCOPE_API_KEY")
        model=model_name,
        messages=messages,
    )
    status_code = getattr(response, "status_code", None)
    if status_code is not None and status_code != 200:
        message = getattr(response, "message", "unknown error")
        raise RuntimeError(f"DashScope call failed (status={status_code}): {message}")
    return response
