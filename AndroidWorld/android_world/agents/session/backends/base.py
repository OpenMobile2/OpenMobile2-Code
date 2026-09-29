"""Session backend protocol — one implementation per model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SessionPrompts:
    system: str
    user: str
    runtime_name: str
    observation: str | None = None


@dataclass
class SessionTurnResult:
    response_text: str
    response_raw: str
    response_norm: str
    history_message: dict[str, Any]
    last_tool_call_id: str | None = None


class SessionBackend(ABC):
    key: str

    @classmethod
    @abstractmethod
    def matches(cls, model_name: str | None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        raise NotImplementedError

    def observation_message(
        self,
        *,
        image_part: dict[str, Any],
        tool_call_counter: int,
        last_tool_call_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        tool_call_id = last_tool_call_id or f"mobile_use_{tool_call_counter}"
        return (
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": [image_part],
            },
            last_tool_call_id,
        )

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages

    @abstractmethod
    def create_kwargs(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        reasoning_effort: str = "max",
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        raise NotImplementedError

    def parse_tool_call(
        self,
        message: Any | None,
        response_norm: str,
        *,
        parse_text_tool_call: Callable[[str], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        return parse_text_tool_call(response_norm)

    def requires_text_thinking(self) -> bool:
        """If True, assistant text must include a thinking block before tool_call."""
        return False

    def has_required_thinking(self, response_text: str) -> bool:
        return True

    def thinking_retry_user_message(self) -> dict[str, Any] | None:
        """Ephemeral user message appended for API retry only (not stored in history)."""
        return None

    def io_flags(self) -> dict[str, bool]:
        return {self.key: True}
