"""Shared helpers for multi-turn session runtimes."""

from __future__ import annotations

import json
import re
from typing import Any


def message_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
        return "".join(parts)
    return ""


def parse_native_tool_call(message: Any) -> dict[str, Any] | None:
    """Convert OpenAI/vLLM native tool_calls into the mobile_use dict shape."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    call = tool_calls[0]
    function = getattr(call, "function", None)
    if function is None:
        return None
    raw_args = getattr(function, "arguments", "{}") or "{}"
    try:
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
    except Exception:
        return None
    return {"name": getattr(function, "name", "mobile_use"), "arguments": args}


def tool_call_dict_to_session_json(tool_call: dict[str, Any]) -> str:
    payload = {
        "name": str(tool_call.get("name") or "mobile_use"),
        "arguments": tool_call.get("arguments") or {},
    }
    return (
        "<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n</tool_call>"
    )


def tool_call_dict_to_session_xml(tool_call: dict[str, Any]) -> str:
    """Serialize {name, arguments} to Qwen3.5 XML ``<function=><parameter=>``."""
    name = str(tool_call.get("name") or "mobile_use")
    args = tool_call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.extend([f"<parameter={key}>", rendered, "</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)


def extract_message_content(message: Any) -> str:
    """Return API ``message.content`` as-is (no think wrapping)."""
    if message is None:
        return ""
    try:
        content = message.content
    except Exception:
        return ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return message_text_content(content)


def extract_reasoning_from_message(message: Any) -> str:
    for key in (
        "reasoning_content",
        "reasoning",
        "reasoning_text",
        "thinking",
        "thought",
        "thoughts",
    ):
        val = getattr(message, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    try:
        if hasattr(message, "model_dump"):
            data = message.model_dump()
            for key in (
                "reasoning_content",
                "reasoning",
                "reasoning_text",
                "thinking",
                "thought",
                "thoughts",
            ):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass
    return ""


def assistant_text_from_completion_message(message: Any) -> str:
    content = ""
    try:
        content = message.content or ""
    except Exception:
        content = ""
    reasoning = None
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        reasoning = getattr(message, key, None)
        if reasoning:
            break
    if isinstance(reasoning, str) and reasoning.strip():
        if not re.search(r"<think\b", content or "", flags=re.I):
            tool = (content or "").strip()
            # Drop dangling </think> left in content by enable_thinking split.
            tool = re.sub(r"^(?:\s*</think\s*>)+", "", tool, flags=re.I).strip()
            tool = re.sub(r"</think\s*>", "", tool, flags=re.I).strip()
            return (
                f"<think>\n{reasoning.strip()}\n</think>\n{tool}".strip()
            )
    return content or ""


def session_assistant_text_from_message(message: Any) -> str:
    reasoning = extract_reasoning_from_message(message)
    try:
        content = (message.content or "").strip()
    except Exception:
        content = ""

    if content and re.search(r"<tool_call\b", content, flags=re.I):
        merged = assistant_text_from_completion_message(message)
        if merged.strip():
            return merged
        if reasoning and not re.search(r"<think\b", content, flags=re.I):
            return f"<thinking>\n{reasoning}\n</thinking>\n{content}"
        return content

    native = parse_native_tool_call(message)
    if native:
        tool_part = tool_call_dict_to_session_xml(native)
        if reasoning:
            return f"<thinking>\n{reasoning}\n</thinking>\n{tool_part}"
        if content:
            if re.search(r"<tool_call\b", content, flags=re.I):
                return content
            return f"{content}\n{tool_part}"
        return tool_part

    merged = assistant_text_from_completion_message(message)
    if merged.strip():
        return merged
    if reasoning:
        return f"<thinking>\n{reasoning}\n</thinking>"
    return content


def extract_think_text(block: str) -> str:
    m = re.search(
        r"<think>\s*([\s\S]*?)\s*</think>", block or "", flags=re.I
    )
    return m.group(1).strip() if m else ""


def extract_thinking_text(block: str) -> str:
    m = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", block or "", flags=re.I)
    return m.group(1).strip() if m else ""


_THOUGHT_LINE_RE = re.compile(r"(?im)^\s*Thought\s*:")
_THOUGHT_BODY_RE = re.compile(
    r"(?is)^\s*Thought\s*:\s*(.*?)(?=\s*<tool_call\b|\Z)"
)


def extract_thought_prefix(text: str) -> str:
    """Return the ``Thought:`` body from assistant content, if present."""
    m = _THOUGHT_BODY_RE.search(text or "")
    return m.group(1).strip() if m else ""


def has_thought_prefix(text: str) -> bool:
    return bool(_THOUGHT_LINE_RE.search(text or ""))


def normalize_thought_session_content(response: str) -> str:
    """Keep ``Thought:`` + XML ``<tool_call>``; do not wrap the prefix in ``<think>``.

    ``normalize_session_assistant_content`` would otherwise turn a ``Thought:``
    paragraph into ``<think>``, which breaks this training format on later turns.
    """
    text = (response or "").strip()
    if not text:
        return text

    think = extract_think_text(text) or extract_thinking_text(text)
    m = re.search(r"<tool_call\b[\s\S]*?</tool_call>", text, flags=re.I)
    if not m:
        prefix = think or re.sub(
            r"</?think(?:ing)?\s*>", "", text, flags=re.I
        ).strip()
        if prefix and not _THOUGHT_LINE_RE.search(prefix):
            return f"Thought: {prefix}"
        return prefix or text

    tool = m.group(0).strip()
    prefix = text[: m.start()].strip()
    prefix = re.sub(r"</?think(?:ing)?\s*>", "", prefix, flags=re.I).strip()
    if think and not prefix:
        prefix = think
    if prefix and not _THOUGHT_LINE_RE.search(prefix):
        prefix = f"Thought: {prefix}"
    if prefix:
        return f"{prefix}\n{tool}"
    return tool


def _strip_orphan_think_closers(text: str) -> str:
    """Remove leftover ``</think>`` from Qwen enable_thinking content.

    With chat_template enable_thinking, vLLM often puts reasoning in a
    separate channel and leaves content like::

        {reasoning text}
        </think>
        <tool_call>...</tool_call>

    (open ``<think>`` already consumed). Re-wrapping that prefix would
    produce ``</think></think>``.
    """
    text = re.sub(r"(</think\s*>\s*){2,}", "</think>\n", text or "", flags=re.I)
    # Drop a lone closer that appears before the first <think> / <tool_call>.
    text = re.sub(
        r"^(?:\s*</think\s*>)+",
        "",
        text,
        flags=re.I,
    )
    return text.strip()


def normalize_session_assistant_content(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return text

    if re.search(r"<thinking\b", text, flags=re.I):
        text = re.sub(r"<thinking\s*>", "<think>", text, flags=re.I)
        text = re.sub(r"</thinking\s*>", "</think>", text, flags=re.I)

    text = _strip_orphan_think_closers(text)

    if re.search(r"<think\b", text, flags=re.I):
        # Collapse ``</think>\n</think>`` still present after merge.
        text = re.sub(r"(</think\s*>\s*){2,}", "</think>\n", text, flags=re.I)
        m_think = re.search(
            r"<think>\s*(.*?)\s*</think>\s*(<tool_call\b[\s\S]*?</tool_call>)",
            text,
            flags=re.I | re.S,
        )
        if m_think and not m_think.group(1).strip():
            return m_think.group(2).strip()
        return text.strip()

    m = re.search(r"<tool_call\b[\s\S]*?</tool_call>", text, flags=re.I)
    if not m:
        cleaned = re.sub(r"</?think\s*>", "", text, flags=re.I).strip()
        return f"<think>\n{cleaned}\n</think>" if cleaned else text

    prefix = text[: m.start()].strip()
    tool = m.group(0).strip()
    # Prefix may still contain a dangling closer from the thinking channel.
    prefix = re.sub(r"</?think\s*>", "", prefix, flags=re.I).strip()
    if not prefix:
        return tool
    return f"<think>\n{prefix}\n</think>\n{tool}"


def serialize_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get("function") or {}
        if isinstance(fn, dict):
            out.append(
                {
                    "id": str(
                        tc.get("id")
                        if isinstance(tc, dict)
                        else getattr(tc, "id", "")
                        or f"call_{len(out)}"
                    ),
                    "type": "function",
                    "function": {
                        "name": str(fn.get("name") or "mobile_use"),
                        "arguments": fn.get("arguments") or "{}",
                    },
                }
            )
            continue
        out.append(
            {
                "id": str(getattr(tc, "id", None) or f"call_{len(out)}"),
                "type": "function",
                "function": {
                    "name": str(getattr(fn, "name", None) or "mobile_use"),
                    "arguments": getattr(fn, "arguments", None) or "{}",
                },
            }
        )
    return out


def assistant_api_record_to_session_text(
    record: dict[str, Any],
    *,
    dialect: str = "qwen35",
) -> str:
    parts: list[str] = []
    reasoning = str(record.get("reasoning_content") or "").strip()
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    tool_calls = record.get("tool_calls") or []
    if tool_calls:
        tc0 = tool_calls[0]
        fn = tc0.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {}
        tool_call = {
            "name": fn.get("name") or "mobile_use",
            "arguments": args if isinstance(args, dict) else {},
        }
        if dialect == "qwen3vl":
            parts.append(tool_call_dict_to_session_json(tool_call))
        else:
            parts.append(tool_call_dict_to_session_xml(tool_call))
    else:
        content = record.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def redact_messages_for_io_dump(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _redact_part(part: Any) -> Any:
        if not isinstance(part, dict):
            return part
        if part.get("type") != "image_url":
            return part
        url = ""
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url") or "")
        elif isinstance(image_url, str):
            url = image_url
        return {
            "type": "image_url",
            "image_url": {"url": f"<omitted data-url, chars={len(url)}>"},
        }

    out: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = {k: v for k, v in msg.items() if k != "content"}
        content = msg.get("content")
        if isinstance(content, list):
            new_msg["content"] = [_redact_part(p) for p in content]
        else:
            new_msg["content"] = content
        out.append(new_msg)
    return out
