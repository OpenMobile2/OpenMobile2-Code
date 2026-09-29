"""
utils.py

Shared utility helpers used across the project.
Currently this module centralizes OpenAI-compatible client creation so the
main pipeline uses one consistent API endpoint and key resolution strategy.
"""

from __future__ import annotations

import json
import os
import threading

from openai import OpenAI


_thread_local = threading.local()

DEFAULT_BASE_URL = "http://<openai-compatible-host>/v1"
DEFAULT_STREAM_BASE_URL = "http://<openai-compatible-host>/v1"
DEFAULT_MODEL = "gemini-3.1-pro-preview"


def get_synthesis_model() -> str:
    """Model name for multimodal task synthesis / annotation."""
    return os.getenv("OPENAI_MODEL") or os.getenv("SYNTHESIS_MODEL") or DEFAULT_MODEL


def load_json(path: str | os.PathLike[str]):
    """Load JSON with UTF-8 (avoid Windows GBK defaults)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str | os.PathLike[str], *, indent: int = 2) -> None:
    """Write JSON with UTF-8 (avoid Windows GBK defaults / emoji crashes)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def get_client() -> OpenAI:
    """
    Return a thread-local default OpenAI-compatible client.
    """
    cfg = (
        os.getenv("OPENAI_API_KEY") or "EMPTY",
        os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL,
    )
    if getattr(_thread_local, "client", None) is None or getattr(_thread_local, "client_cfg", None) != cfg:
        api_key, base_url = cfg
        _thread_local.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        _thread_local.client_cfg = cfg
    return _thread_local.client


def get_qwen_stream_client() -> OpenAI:
    """
    Return a thread-local stream client for the optional qwen-style endpoint.
    """
    cfg = (
        os.getenv("QWEN_STREAM_API_KEY") or "EMPTY",
        os.getenv("QWEN_STREAM_BASE_URL") or DEFAULT_STREAM_BASE_URL,
    )
    if getattr(_thread_local, "qwen_stream_client", None) is None or getattr(_thread_local, "qwen_stream_client_cfg", None) != cfg:
        api_key, base_url = cfg
        _thread_local.qwen_stream_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        _thread_local.qwen_stream_client_cfg = cfg
    return _thread_local.qwen_stream_client
