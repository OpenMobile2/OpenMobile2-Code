"""DeepSeek-V4 vision session — native tool_calls + reasoning_content replay.

Official Chat Completions (``https://api.deepseek.com``):

- Vision: ``deepseek-v4-flash-vision-exp`` accepts ``image_url`` parts.
- Tools: OpenAI ``tools=`` / ``tool_calls``. With ``tools`` present, later
  turns **must** echo ``reasoning_content`` or the API returns 400.
- Thinking: top-level ``thinking: {type: enabled|disabled}`` plus
  ``reasoning_effort`` (``high`` / ``max``; we map CLI ``low`` → ``high``).
"""

from __future__ import annotations

from typing import Any

from android_world.agents.PROMPT import (
    DEEPSEEK_V4_SESSION_SYSTEM_PROMPT,
    DEEPSEEK_V4_SESSION_USER_PROMPT,
    QWEN35_TOOLS,
)
from android_world.agents.session.backends.base import (
    SessionBackend,
    SessionPrompts,
    SessionTurnResult,
)
from android_world.agents.session.backends.kimi_k3 import (
    _adapt_legacy_assistant_tool_pair,
    _parse_tool_call_any,
    build_kimi_k3_assistant_record,
)
from android_world.agents.session.utils import (
    assistant_api_record_to_session_text,
    extract_reasoning_from_message,
    normalize_session_assistant_content,
    parse_native_tool_call,
    session_assistant_text_from_message,
)

# DeepSeek V4 thinking mode documents high / max only.
_EFFORT_MAP = {
    "low": "high",
    "high": "high",
    "max": "max",
}


class DeepSeekV4SessionBackend(SessionBackend):
    key = "deepseek_v4"

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        name = (model_name or "").lower().replace("_", "-")
        return "deepseek" in name

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del dialect, enable_thinking, require_think_tags
        return SessionPrompts(
            system=DEEPSEEK_V4_SESSION_SYSTEM_PROMPT,
            user=DEEPSEEK_V4_SESSION_USER_PROMPT,
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
        thinking_type = "enabled" if enable_thinking else "disabled"
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "timeout": 180.0,
            "tools": QWEN35_TOOLS,
            "tool_choice": "auto",
            "extra_body": {"thinking": {"type": thinking_type}},
        }
        if enable_thinking:
            effort = _EFFORT_MAP.get(
                (reasoning_effort or "max").strip().lower(), "max"
            )
            kwargs["reasoning_effort"] = effort
        return kwargs

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
        return {"deepseek_v4_session": True}