"""MemGUI state machine for AndroidWorld eval agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from android_world.agents import memgui_templates as T


_RE_THOUGHT = re.compile(
    r"^Thought:\s*(.+?)(?=\nAction:|\n<|\Z)", re.S | re.M | re.I
)
_RE_ACTION = re.compile(r"^Action:\s*(.+?)(?=\n<|\Z)", re.S | re.M | re.I)


def _apply_memory_action(
    memory: list[dict[str, Any]], args: dict[str, Any]
) -> list[dict[str, Any]]:
    action = str(args.get("action") or "")
    mid = str(args.get("memory_id") or args.get("id") or "").strip()
    desc = str(args.get("description") or "").strip()
    content = str(args.get("content") or "").strip()
    out = list(memory)

    if action == "memory_add":
        if not mid:
            mid = f"m{len(out) + 1}"
        out = [m for m in out if m.get("id") != mid]
        out.append({"id": mid, "description": desc, "content": content})
        return out
    if action == "memory_update":
        found = False
        for m in out:
            if m.get("id") == mid:
                if desc:
                    m["description"] = desc
                if content:
                    m["content"] = content
                found = True
                break
        if not found and mid:
            out.append({"id": mid, "description": desc, "content": content})
        return out
    if action == "memory_delete":
        return [m for m in out if m.get("id") != mid]
    return out


@dataclass
class MemGUIRuntime:
    """Per-episode MemGUI state for ``seeact_v`` eval agents."""

    prompt_format: str = "auto"

    ui_memory: list[dict[str, Any]] = field(default_factory=list)
    step_summaries: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.ui_memory = []
        self.step_summaries = []

    def resolve_format(self, *, agent_kind: str, model_name: str) -> str:
        if self.prompt_format not in ("auto", ""):
            return self.prompt_format
        # Agent family owns the tool-call dialect:
        # - qwen3vl → JSON <tool_call>{"name":...}</tool_call>
        # - qwen35vl → XML <function=...><parameter=...>
        if agent_kind == "qwen35vl":
            return "qwen35"
        if agent_kind == "qwen3vl":
            return "qwen3vl"
        model_lower = (model_name or "").lower()
        if "gemini" in model_lower or "kimi" in model_lower:
            return "qwen35"
        return "qwen3vl"

    def build_prompts(
        self,
        instruction: str,
        *,
        agent_kind: str,
        model_name: str,
    ) -> tuple[str, str]:
        fmt = self.resolve_format(agent_kind=agent_kind, model_name=model_name)
        user_kwargs = {
            "instruction": instruction,
            "task_progress": T.format_task_progress(self.step_summaries),
            "memory": T.format_ui_memory(self.ui_memory),
        }
        if fmt == "qwen35":
            return (
                T.MEMGUI_QWEN35_SYSTEM_PROMPT,
                T.MEMGUI_QWEN35_USER_PROMPT.format(**user_kwargs),
            )
        if fmt == "qwen3vl":
            return (
                T.MEMGUI_QWEN3VL_SYSTEM_PROMPT,
                T.MEMGUI_QWEN3VL_USER_PROMPT.format(**user_kwargs),
            )
        return (
            T.MEMGUI_SYSTEM_PROMPT,
            T.MEMGUI_USER_PROMPT.format(**user_kwargs),
        )

    def _extract_text_fields(self, response: str) -> tuple[str, str]:
        thinking = T.tag(response, "thinking") or ""
        intent = T.tag(response, "action_intent") or ""
        m_th = _RE_THOUGHT.search(response or "")
        if m_th:
            thinking = m_th.group(1).strip()
        m_ac = _RE_ACTION.search(response or "")
        if m_ac:
            intent = m_ac.group(1).strip()
        return thinking, intent

    def apply_model_output(
        self,
        response: str,
        tool_call: dict[str, Any] | None,
        *,
        turn_number: int,
        agent_kind: str,
        model_name: str,
    ) -> str:
        del turn_number, agent_kind, model_name
        thinking, intent = self._extract_text_fields(response)
        step_text = intent or thinking
        if step_text:
            self.step_summaries.append(step_text)

        args: dict[str, Any] = {}
        if isinstance(tool_call, dict):
            raw_args = tool_call.get("arguments", {})
            if isinstance(raw_args, dict):
                args = raw_args

        action = str(args.get("action") or "")
        if action in ("memory_add", "memory_update", "memory_delete"):
            self.ui_memory = _apply_memory_action(list(self.ui_memory), args)

        return step_text

    def after_action(
        self,
        tool_call: dict[str, Any] | None,
        result: str,
    ) -> None:
        del tool_call, result

    @property
    def step_history_text(self) -> str:
        return T.format_task_progress(self.step_summaries)

    def log_fields(self) -> dict[str, Any]:
        return {
            "prompt_pack": "memgui",
            "memgui_prompt_format": self.prompt_format,
            "ui_memory": list(self.ui_memory),
            "step_summaries": list(self.step_summaries),
            "step_history": self.step_history_text,
        }
