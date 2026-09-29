"""Resolve the session backend for a model name."""

from __future__ import annotations

from android_world.agents.session.backends.base import SessionBackend
from android_world.agents.session.backends.deepseek_v4 import DeepSeekV4SessionBackend
from android_world.agents.session.backends.gemini import GeminiSessionBackend
from android_world.agents.session.backends.glm import GlmSessionBackend
from android_world.agents.session.backends.kimi_k3 import KimiK3SessionBackend
from android_world.agents.session.backends.qwen import QwenSessionBackend

_ORDERED_BACKENDS: tuple[type[SessionBackend], ...] = (
    DeepSeekV4SessionBackend,
    KimiK3SessionBackend,
    GeminiSessionBackend,
    GlmSessionBackend,
)


def resolve_session_backend(model_name: str | None) -> SessionBackend:
    for cls in _ORDERED_BACKENDS:
        if cls.matches(model_name):
            return cls()
    return QwenSessionBackend()
