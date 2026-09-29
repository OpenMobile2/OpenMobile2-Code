"""Qwen session: ``Thought:`` in assistant text, no native thinking.

Matches the xml-in-system training sample (tools schema in the system prompt,
``Thought:`` then one Qwen XML ``<tool_call>``). ``enable_thinking`` is always
off so reasoning is not split into ``reasoning_content`` / ``<think>``.
"""

from __future__ import annotations

from typing import Any

from android_world.agents.PROMPT import (
    QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT,
    QWEN35_SESSION_USER_PROMPT,
)
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.utils import (
    extract_message_content,
    has_thought_prefix,
    normalize_thought_session_content,
    session_assistant_text_from_message,
)


class QwenThoughtSessionBackend(SessionBackend):
    """Fixed Thought: prompts; never auto-selected by model name."""

    key = "qwen_thought"

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        del model_name
        return False

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del dialect, enable_thinking, require_think_tags
        return SessionPrompts(
            system=QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT,
            user=QWEN35_SESSION_USER_PROMPT,
            runtime_name="qwen35_thought_session",
        )

    def create_kwargs(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        reasoning_effort: str = "max",
    ) -> dict[str, Any]:
        del enable_thinking, reasoning_effort
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "timeout": 300.0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        }

    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        del dialect
        response = session_assistant_text_from_message(message)
        response_norm = normalize_thought_session_content(response)
        return SessionTurnResult(
            response_text=response,
            response_raw=extract_message_content(message),
            response_norm=response_norm,
            history_message={"role": "assistant", "content": response_norm},
        )

    def requires_text_thinking(self) -> bool:
        return True

    def has_required_thinking(self, response_text: str) -> bool:
        return has_thought_prefix(response_text)

    def thinking_retry_user_message(self) -> dict[str, Any] | None:
        return {
            "role": "user",
            "content": (
                "Your previous reply was missing a Thought: paragraph. "
                "Output exactly: a Thought: paragraph, then one "
                "<tool_call>...</tool_call> block. Do not use <think> tags."
            ),
        }

    def io_flags(self) -> dict[str, bool]:
        return {self.key: True, "thought_in_content": True}
