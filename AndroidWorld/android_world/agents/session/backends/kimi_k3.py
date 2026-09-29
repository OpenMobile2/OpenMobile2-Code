"""Kimi K3 session backend — native tool_calls + reasoning_content (Preserved Thinking)."""

from __future__ import annotations

import json
from typing import Any

from android_world.agents.PROMPT import KIMI_K3_SESSION_SYSTEM_PROMPT, KIMI_K3_SESSION_USER_PROMPT, QWEN35_TOOLS
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.utils import (
    assistant_api_record_to_session_text,
    extract_reasoning_from_message,
    extract_think_text,
    extract_thinking_text,
    message_text_content,
    normalize_session_assistant_content,
    parse_native_tool_call,
    serialize_tool_calls,
    session_assistant_text_from_message,
)


def _parse_tool_call_any(block: str) -> dict[str, Any] | None:
    from android_world.agents.seeact_v import _parse_tool_call_json, _parse_tool_call_xml

    return _parse_tool_call_json(block) or _parse_tool_call_xml(block)


def _adapt_legacy_assistant_tool_pair(
    assistant_msg: dict[str, Any],
    tool_msg: dict[str, Any],
    *,
    index: int,
) -> list[dict[str, Any]]:
    """Convert legacy text tool_call + tool screenshot into OpenAI tool_calls shape."""
    text = message_text_content(assistant_msg.get("content"))
    tool_call = _parse_tool_call_any(text)
    tool_call_id = str(tool_msg.get("tool_call_id") or f"call_mobile_use_{index}")
    fn_name = str((tool_call or {}).get("name") or "mobile_use")
    if not tool_call:
        return [assistant_msg, tool_msg]
    thinking = extract_think_text(text) or extract_thinking_text(text)
    assistant_api: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(
                        tool_call.get("arguments") or {},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }
    if thinking:
        assistant_api["content"] = thinking
    else:
        assistant_api["content"] = None
    adapted_tool = dict(tool_msg)
    adapted_tool["name"] = fn_name
    adapted_tool["tool_call_id"] = tool_call_id
    return [assistant_api, adapted_tool]


def build_kimi_k3_assistant_record(message: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"role": "assistant"}
    reasoning = extract_reasoning_from_message(message)
    if reasoning:
        record["reasoning_content"] = reasoning
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        record["content"] = content.strip()
    elif content is not None:
        text = message_text_content(content)
        if text.strip():
            record["content"] = text.strip()
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        record["tool_calls"] = serialize_tool_calls(tool_calls)
    if "content" not in record:
        record["content"] = None
    return record


class KimiK3SessionBackend(SessionBackend):
    key = "kimi_k3"

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        name = (model_name or "").lower().replace("_", "-")
        return "kimi-k3" in name or name.startswith("kimi-k3")

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del dialect, enable_thinking, require_think_tags
        return SessionPrompts(
            system=KIMI_K3_SESSION_SYSTEM_PROMPT,
            user=KIMI_K3_SESSION_USER_PROMPT,
            runtime_name="qwen35_session",
        )

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
                "name": "mobile_use",
                "tool_call_id": tool_call_id,
                "content": [image_part],
            },
            last_tool_call_id,
        )

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        del enable_thinking  # K3 always thinks; strength is reasoning_effort only.
        effort = (reasoning_effort or "max").strip().lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError(
                f"reasoning_effort must be low|high|max, got {reasoning_effort!r}"
            )
        return {
            "model": model_name,
            "messages": messages,
            "timeout": 60.0,
            "tools": QWEN35_TOOLS,
            "tool_choice": "auto",
            "reasoning_effort": effort,
        }

    def parse_turn(self, message: Any, *, dialect: str) -> SessionTurnResult:
        record = build_kimi_k3_assistant_record(message)
        response = assistant_api_record_to_session_text(record, dialect=dialect)
        if not response.strip():
            try:
                response = message.model_dump_json(indent=2, exclude_none=True)
            except Exception:
                response = session_assistant_text_from_message(message)
        response_norm = normalize_session_assistant_content(
            assistant_api_record_to_session_text(record, dialect=dialect)
        )
        try:
            response_raw = message.model_dump_json(indent=2, exclude_none=True)
        except Exception:
            reasoning = extract_reasoning_from_message(message)
            response_raw = (
                f"{response}\n[reasoning_content]\n{reasoning}" if reasoning else response
            )
        last_tool_call_id = None
        tool_calls = record.get("tool_calls") or []
        if tool_calls:
            last_tool_call_id = str(tool_calls[0].get("id") or "")
        return SessionTurnResult(
            response_text=response,
            response_raw=response_raw,
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
        del parse_text_tool_call
        if message is not None:
            native = parse_native_tool_call(message)
            if native:
                return native
        return _parse_tool_call_any(response_norm)

    def io_flags(self) -> dict[str, bool]:
        return {"kimi_k3_session": True}
