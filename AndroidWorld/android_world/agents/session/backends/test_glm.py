"""GLM session backend: thinking.type + reasoning_effort, not Qwen enable_thinking."""

from __future__ import annotations

from android_world.agents.PROMPT import GLM_SESSION_SYSTEM_PROMPT
from android_world.agents.session.backends import resolve_session_backend
from android_world.agents.session.backends.gemini import GeminiSessionBackend
from android_world.agents.session.backends.glm import GlmSessionBackend
from android_world.agents.session.backends.qwen import QwenSessionBackend


def test_glm_model_names_match():
    for name in (
        "z-ai/glm-5.3-flash",
        "glm-5.3-flash",
        "GLM-4.5V",
        "chatglm-turbo",
    ):
        backend = resolve_session_backend(name)
        assert isinstance(backend, GlmSessionBackend), name


def test_gemini_does_not_match_glm():
    backend = resolve_session_backend("gemini-3.1-pro-preview")
    assert isinstance(backend, GeminiSessionBackend)


def test_qwen_stays_default():
    backend = resolve_session_backend("Qwen3.5-9B-0820")
    assert isinstance(backend, QwenSessionBackend)


def test_glm_create_kwargs_use_thinking_type_not_chat_template():
    backend = GlmSessionBackend()
    kwargs = backend.create_kwargs(
        model_name="z-ai/glm-5.3-flash",
        messages=[],
        enable_thinking=True,
        reasoning_effort="max",
    )
    extra = kwargs["extra_body"]
    assert extra["thinking"]["type"] == "enabled"
    assert extra["reasoning_effort"] == "max"
    assert "chat_template_kwargs" not in extra
    assert kwargs["tool_choice"] == "auto"


def test_glm_disable_thinking_falls_back_to_low_effort():
    backend = GlmSessionBackend()
    kwargs = backend.create_kwargs(
        model_name="glm-5.3-flash",
        messages=[],
        enable_thinking=False,
        reasoning_effort="max",
    )
    extra = kwargs["extra_body"]
    assert extra["thinking"]["type"] == "enabled"
    assert extra["reasoning_effort"] == "low"


def test_glm_tools_do_not_claim_square_999():
    kwargs = GlmSessionBackend().create_kwargs(
        model_name="glm-5.3-flash",
        messages=[],
        enable_thinking=True,
        reasoning_effort="high",
    )
    desc = kwargs["tools"][0]["function"]["description"]
    assert "999x999" not in desc
    assert "0–1000" in desc or "0-1000" in desc
    assert "999x999" not in GLM_SESSION_SYSTEM_PROMPT
