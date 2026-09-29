"""Thought: session backend — no native thinking, XML tools in system prompt."""

from __future__ import annotations

from types import SimpleNamespace

from android_world.agents.PROMPT import (
    QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT,
)
from android_world.agents.session.backends.qwen_thought import QwenThoughtSessionBackend
from android_world.agents.session.utils import (
    has_thought_prefix,
    normalize_session_assistant_content,
    normalize_thought_session_content,
)
from android_world.agents.session_runtimes import resolve_runtime


def test_thought_prompt_matches_training_sample():
    prompt = QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT
    assert "Put reasoning after `Thought:` BEFORE the function call" in prompt
    assert "Every assistant turn must be exactly: a `Thought:` paragraph" in prompt
    assert "<function=example_function_name>" in prompt
    assert "You are a mobile GUI agent." in prompt
    assert "<think>" not in prompt
    assert "Do NOT output <think>" not in prompt


def test_thought_backend_disables_native_thinking_and_tools():
    backend = QwenThoughtSessionBackend()
    prompts = backend.prompts("qwen35")
    assert prompts.runtime_name == "qwen35_thought_session"
    assert prompts.system == QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT
    kwargs = backend.create_kwargs(
        model_name="Qwen3.5-9B",
        messages=[],
        enable_thinking=True,
        reasoning_effort="max",
    )
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert "tools" not in kwargs
    assert backend.requires_text_thinking() is True


def test_thought_backend_parses_thought_plus_xml():
    backend = QwenThoughtSessionBackend()
    message = SimpleNamespace(
        content=(
            "Thought: Click Communities in the onboarding list.\n"
            "<tool_call>\n"
            "<function=mobile_use>\n"
            "<parameter=action>\n"
            "click\n"
            "</parameter>\n"
            "<parameter=coordinate>\n"
            "[120, 400]\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
        reasoning_content=None,
        tool_calls=None,
    )
    turn = backend.parse_turn(message, dialect="qwen35")
    assert turn.history_message["content"].startswith("Thought:")
    assert "<think>" not in turn.history_message["content"]
    assert "<tool_call>" in turn.history_message["content"]
    assert backend.has_required_thinking(turn.response_text) is True


def test_normalize_thought_does_not_wrap_in_think_tags():
    raw = (
        "Thought: Open Wikipedia.\n"
        "<tool_call>\n<function=mobile_use>\n"
        "<parameter=action>\nclick\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    thought_norm = normalize_thought_session_content(raw)
    session_norm = normalize_session_assistant_content(raw)
    assert thought_norm.startswith("Thought:")
    assert "<think>" not in thought_norm
    assert session_norm.startswith("<think>")


def test_has_thought_prefix():
    assert has_thought_prefix("Thought: click the icon\n<tool_call>")
    assert not has_thought_prefix("<tool_call>\n<function=mobile_use>")


def test_runtime_preset_resolves():
    spec = resolve_runtime("qwen35_thought_session")
    assert spec.agent_name == "qwen35_thought_session"
    assert "Thought:" in spec.description
