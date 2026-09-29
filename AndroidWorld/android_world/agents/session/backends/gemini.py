"""Gemini session backend — native ``reasoning_content`` via Google thinking_config.

Tokenhub / OpenAI-compat nesting::

    extra_body={
        "extra_body": {
            "google": {
                "thinking_config": {
                    "include_thoughts": True,
                    "thinking_level": "high",
                }
            }
        }
    }

Follow-up turns are image-only user messages. Assistant history stores
``<think>`` (from ``reasoning_content``) + ``<tool_call>`` text.
"""

from __future__ import annotations

import re
from typing import Any

from android_world.agents.PROMPT import (
    GEMINI_SESSION_SYSTEM_PROMPT,
    GEMINI_SESSION_USER_PROMPT,
)
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.utils import (
    extract_reasoning_from_message,
    normalize_session_assistant_content,
    session_assistant_text_from_message,
)

_THINKING_RE = re.compile(r"<(thinking|think)\b", flags=re.I)

# Map CLI --reasoning_effort onto Google thinking_level.
_EFFORT_TO_LEVEL = {
    "low": "low",
    "high": "high",
    "max": "high",
}


class GeminiSessionBackend(SessionBackend):
    key = "gemini"

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        return "gemini" in (model_name or "").lower()

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del enable_thinking, require_think_tags
        runtime_name = (
            "qwen3vl_session" if dialect == "qwen3vl" else "qwen35_session"
        )
        return SessionPrompts(
            system=GEMINI_SESSION_SYSTEM_PROMPT,
            user=GEMINI_SESSION_USER_PROMPT,
            observation=None,
            runtime_name=runtime_name,
        )

    def observation_message(
        self,
        *,
        image_part: dict[str, Any],
        tool_call_counter: int,
        last_tool_call_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        del tool_call_counter, last_tool_call_id
        return (
            {
                "role": "user",
                "_session_observation": True,
                "content": [image_part],
            },
            None,
        )

    def create_kwargs(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        reasoning_effort: str = "max",
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "timeout": 60.0,
            "max_tokens": 8192,
            "tool_choice": "none",
        }
        if enable_thinking:
            level = _EFFORT_TO_LEVEL.get(
                (reasoning_effort or "max").strip().lower(), "high"
            )
            # Tokenhub Gemini: nested google.thinking_config (not top-level).
            kwargs["extra_body"] = {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "include_thoughts": True,
                            "thinking_level": level,
                        }
                    }
                }
            }
        return kwargs

    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        del dialect
        reasoning = extract_reasoning_from_message(message)
        try:
            content = (message.content or "").strip()
        except Exception:
            content = ""
        if not content:
            content = session_assistant_text_from_message(message).strip()

        if reasoning and not _THINKING_RE.search(content or ""):
            response = f"<thinking>\n{reasoning}\n</thinking>\n{content}".strip()
        else:
            response = content

        response_norm = normalize_session_assistant_content(response)
        try:
            response_raw = message.model_dump_json(indent=2, exclude_none=True)
        except Exception:
            response_raw = response
            if reasoning:
                response_raw = f"{response}\n[reasoning_content]\n{reasoning}"

        return SessionTurnResult(
            response_text=response,
            response_raw=response_raw,
            response_norm=response_norm,
            history_message={"role": "assistant", "content": response_norm},
        )

    def requires_text_thinking(self) -> bool:
        # Thinking comes from native reasoning_content; do not force text <thinking>.
        return False

    def has_required_thinking(self, response_text: str) -> bool:
        return True

    def thinking_retry_user_message(self) -> dict[str, Any] | None:
        return None

    def io_flags(self) -> dict[str, bool]:
        return {"gemini_text_session": True, "gemini_native_thinking": True}
