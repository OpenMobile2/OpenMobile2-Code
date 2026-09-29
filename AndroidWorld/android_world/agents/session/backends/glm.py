"""GLM / Z.AI session backend — native tool_calls + thinking.type.

GLM-5.3 / GLM-5.3-Flash force thinking: ``thinking.type`` must stay ``enabled``.
Strength is ``reasoning_effort`` (``low`` / ``high`` / ``max``). Do not send
Qwen ``chat_template_kwargs.enable_thinking``.
"""

from __future__ import annotations

import copy
from typing import Any

from android_world.agents.PROMPT import (
    GLM_COORDINATE_DESCRIPTION,
    GLM_MOBILE_USE_DESCRIPTION,
    GLM_SESSION_SYSTEM_PROMPT,
    GLM_SESSION_USER_PROMPT,
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
    normalize_session_assistant_content,
    parse_native_tool_call,
    session_assistant_text_from_message,
)


def _glm_tools() -> list[dict[str, Any]]:
    tools = copy.deepcopy(QWEN35_TOOLS)
    fn = tools[0]["function"]
    fn["description"] = GLM_MOBILE_USE_DESCRIPTION
    props = fn["parameters"]["properties"]
    props["coordinate"]["description"] = GLM_COORDINATE_DESCRIPTION
    props["coordinate2"]["description"] = (
        "Normalized end [x, y] in the same 0–1000 space. Required only by swipe."
    )
    return tools


class GlmSessionBackend(SessionBackend):
    key = "glm"

    @classmethod
    def matches(cls, model_name: str | None) -> bool:
        name = (model_name or "").lower().replace("_", "-")
        return (
            name.startswith("glm")
            or "/glm" in name
            or "-glm-" in name
            or name.endswith("-glm")
            or name.startswith("z-ai/")
            or name.startswith("zai/")
            or "chatglm" in name
        )

    def prompts(
        self,
        dialect: str,
        *,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
    ) -> SessionPrompts:
        del dialect, enable_thinking, require_think_tags
        return SessionPrompts(
            system=GLM_SESSION_SYSTEM_PROMPT,
            user=GLM_SESSION_USER_PROMPT,
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
        # GLM-5.3-Flash rejects thinking.type=disabled. --enable-thinking false
        # maps to the lightest allowed effort instead of turning thinking off.
        effort = (reasoning_effort or "max").strip().lower()
        if effort not in {"low", "high", "max"}:
            effort = "max"
        if not enable_thinking:
            effort = "low"
        return {
            "model": model_name,
            "messages": messages,
            "temperature": 1.0,
            "top_p": 0.95,
            "timeout": 60.0,
            "tools": _glm_tools(),
            "tool_choice": "auto",
            "extra_body": {
                "thinking": {"type": "enabled", "clear_thinking": False},
                "reasoning_effort": effort,
            },
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
            response_raw = response
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
        return {"glm_session": True}
