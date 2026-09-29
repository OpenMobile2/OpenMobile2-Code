"""OSWorld-inspired Qwen session backend — Action + XML tool_call, no think retry."""

from __future__ import annotations

from typing import Any

from android_world.agents.PROMPT import (
    QWEN35_OSWORLD_SESSION_SYSTEM_PROMPT,
    QWEN35_OSWORLD_SESSION_USER_PROMPT,
)
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.osworld_history import normalize_osworld_assistant_content
from android_world.agents.session.utils import (
    extract_message_content,
    session_assistant_text_from_message,
)


class QwenOsworldSessionBackend(SessionBackend):
    """Fixed OSWorld-style prompts; never auto-selected by model name."""

    key = "qwen_osworld"

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
            system=QWEN35_OSWORLD_SESSION_SYSTEM_PROMPT,
            user=QWEN35_OSWORLD_SESSION_USER_PROMPT,
            runtime_name="qwen35_osworld_session",
        )

    def create_kwargs(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        reasoning_effort: str = "max",
    ) -> dict[str, Any]:
        del reasoning_effort
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "timeout": 60.0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            },
        }

    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        del dialect
        response = session_assistant_text_from_message(message)
        # Do NOT use normalize_session_assistant_content: it wraps Action in
        # <think>, which we then strip and accidentally drop Action.
        response_norm = normalize_osworld_assistant_content(response)
        return SessionTurnResult(
            response_text=response,
            response_raw=extract_message_content(message),
            response_norm=response_norm,
            history_message={"role": "assistant", "content": response_norm},
        )

    def requires_text_thinking(self) -> bool:
        return False
