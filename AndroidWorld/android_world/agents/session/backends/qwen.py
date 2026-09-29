"""Qwen session backend — tools= API with Qwen XML <function=><parameter=>."""

from __future__ import annotations

from typing import Any

from android_world.agents.PROMPT import (
    QWEN35_SESSION_SYSTEM_PROMPT,
    QWEN35_SESSION_SYSTEM_PROMPT_NATIVE,
    QWEN35_SESSION_SYSTEM_PROMPT_NO_THINK,
    QWEN35_SESSION_USER_PROMPT,
    QWEN35_TOOLS,
    QWEN3VL_SESSION_SYSTEM_PROMPT,
    QWEN3VL_SESSION_SYSTEM_PROMPT_NO_THINK,
    QWEN3VL_SESSION_USER_PROMPT,
)
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.backends.kimi_k3 import (
    _adapt_legacy_assistant_tool_pair,
    build_kimi_k3_assistant_record,
)
from android_world.agents.session.utils import (
    assistant_api_record_to_session_text,
    extract_message_content,
    normalize_session_assistant_content,
    parse_native_tool_call,
    session_assistant_text_from_message,
)


class QwenSessionBackend(SessionBackend):
    key = "qwen"

    def __init__(self, use_native_tools: bool = False):
        self.use_native_tools = bool(use_native_tools)

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        return False

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del enable_thinking
        if self.use_native_tools:
            return SessionPrompts(
                system=QWEN35_SESSION_SYSTEM_PROMPT_NATIVE,
                user=QWEN35_SESSION_USER_PROMPT,
                runtime_name=(
                    "qwen3vl_session" if dialect == "qwen3vl" else "qwen35_session"
                ),
            )
        if dialect == "qwen3vl":
            return SessionPrompts(
                system=(
                    QWEN3VL_SESSION_SYSTEM_PROMPT
                    if require_think_tags
                    else QWEN3VL_SESSION_SYSTEM_PROMPT_NO_THINK
                ),
                user=QWEN3VL_SESSION_USER_PROMPT,
                runtime_name="qwen3vl_session",
            )
        return SessionPrompts(
            system=(
                QWEN35_SESSION_SYSTEM_PROMPT
                if require_think_tags
                else QWEN35_SESSION_SYSTEM_PROMPT_NO_THINK
            ),
            user=QWEN35_SESSION_USER_PROMPT,
            runtime_name="qwen35_session",
        )

    def observation_message(
        self,
        *,
        image_part: dict[str, Any],
        tool_call_counter: int,
        last_tool_call_id: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        if not self.use_native_tools:
            return super().observation_message(
                image_part=image_part,
                tool_call_counter=tool_call_counter,
                last_tool_call_id=last_tool_call_id,
            )
        tool_call_id = last_tool_call_id or f"mobile_use_{tool_call_counter}"
        return (
            {
                "role": "tool",
                "name": "mobile_use",
                "tool_call_id": tool_call_id,
                "content": [image_part],
            },
            last_tool_call_id,
        )

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.use_native_tools:
            return super().prepare_messages(messages)
        out: list[dict[str, Any]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            clean = {k: v for k, v in msg.items() if not str(k).startswith("_")}
            role = clean.get("role")
            if role == "assistant" and (
                clean.get("reasoning_content") or clean.get("tool_calls")
            ):
                if "content" not in clean:
                    clean["content"] = None
                out.append(clean)
                i += 1
                continue
            if (
                role == "assistant"
                and i + 1 < len(messages)
                and messages[i + 1].get("role") == "tool"
            ):
                out.extend(
                    _adapt_legacy_assistant_tool_pair(msg, messages[i + 1], index=i)
                )
                i += 2
                continue
            if role == "tool":
                tool_msg = dict(clean)
                tool_msg.setdefault("name", "mobile_use")
                out.append(tool_msg)
                i += 1
                continue
            out.append(clean)
            i += 1
        return out

    def create_kwargs(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        enable_thinking: bool,
        reasoning_effort: str = "max",
    ) -> dict[str, Any]:
        del reasoning_effort
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "timeout": 60.0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            },
        }
        if self.use_native_tools:
            kwargs["tools"] = QWEN35_TOOLS
            kwargs["tool_choice"] = "auto"
        return kwargs

    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        if not self.use_native_tools:
            response = session_assistant_text_from_message(message)
            response_norm = normalize_session_assistant_content(response)
            return SessionTurnResult(
                response_text=response,
                response_raw=extract_message_content(message),
                response_norm=response_norm,
                history_message={"role": "assistant", "content": response_norm},
            )
        record = build_kimi_k3_assistant_record(message)
        response_norm = normalize_session_assistant_content(
            assistant_api_record_to_session_text(record, dialect=dialect)
        )
        last_tool_call_id = None
        tool_calls = record.get("tool_calls") or []
        if tool_calls:
            last_tool_call_id = str(tool_calls[0].get("id") or "")
        return SessionTurnResult(
            response_text=response_norm,
            response_raw=extract_message_content(message),
            response_norm=response_norm,
            history_message=record,
            last_tool_call_id=last_tool_call_id,
        )

    def parse_tool_call(
        self,
        message: Any | None,
        response_norm: str,
        *,
        parse_text_tool_call,
    ) -> dict[str, Any] | None:
        if self.use_native_tools and message is not None:
            native = parse_native_tool_call(message)
            if native:
                return native
        return parse_text_tool_call(response_norm)

    def io_flags(self) -> dict[str, bool]:
        flags = {self.key: True}
        if self.use_native_tools:
            flags["qwen_native_tools"] = True
        return flags
