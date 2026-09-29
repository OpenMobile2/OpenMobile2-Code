"""Flat Qwen3VL ReAct agent for official MobileGym scoring.

Same request shape as AndroidWorld ``Qwen3VL.step`` in ``seeact_v.py``:

* one system message (``QWEN3VL_SYSTEM_PROMPT``)
* one user message: instruction + past Action lines, then the last N screenshots
* model output is Thought / Action plus a JSON ``<tool_call>``
* history is text inside the next user prompt, not a multi-turn session
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Any

from android_world.agents.PROMPT import QWEN3VL_SYSTEM_PROMPT, QWEN3VL_USER_PROMPT
from android_world.agents.utils import qwen3vl_action_transform
from bench_env.agent.base import AgentConfig, AgentStepRecord, BaseAgent
from bench_env.env.base import Action, Observation
from bench_env.llm import LLMClient

from openmobile_session_agent import SESSION_CFG, _parsed_to_action

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>")
_ACTION_LINE_RE = re.compile(
    r"Action:\s*(.+?)(?=\n(?:Memory|Thought)\s*:|\n\s*<tool_call>|$)",
    flags=re.S | re.I,
)


def _parse_tool_call_json(block: str) -> dict[str, Any] | None:
    match = _TOOL_CALL_RE.search(block or "")
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1).strip())
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_action_text(block: str) -> str:
    match = _ACTION_LINE_RE.search(block or "")
    if not match:
        return ""
    text = match.group(1).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.replace("\n", " ")


class OpenMobileQwen3VLBenchAgent(BaseAgent):
    """bench_env agent that mirrors ``seeact_v.Qwen3VL`` message packing."""

    SYSTEM_PROMPT = QWEN3VL_SYSTEM_PROMPT
    ACTION_MAP: dict[str, Any] = {}

    def __init__(self, llm: LLMClient, config: AgentConfig | None = None):
        super().__init__(config)
        self.llm = llm
        self.step_his = ""
        self.turn_number = 0
        self.last_n = max(1, int(SESSION_CFG.get("last_n") or 1))
        self._recent_images: deque[str] = deque(maxlen=self.last_n)
        merged = dict(self.config.model_args)
        merged["temperature"] = 0.0
        self.config.model_args = merged

    @property
    def name(self) -> str:
        return "qwen3vl"

    def reset(self, task: str) -> None:
        self._task = task
        self._history = []
        self.step_his = ""
        self.turn_number = 0
        self._recent_images.clear()

    def parse_response(self, response_text: str) -> Action:
        tool_call = _parse_tool_call_json(response_text)
        if not tool_call:
            return Action.abort("No <tool_call> JSON found", raw_response=response_text)
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        action_name = str(args.get("action") or "")
        try:
            # Coordinates are already 0-1000. Identity scale keeps them in bench_env space.
            parsed = qwen3vl_action_transform(action_name, args, 1000, 1000)
        except Exception as exc:
            return Action.abort(f"Failed to transform tool-call: {exc}", raw_response=response_text)
        return _parsed_to_action(parsed, 1000, 1000, response_text)

    def build_messages(self, obs: Observation) -> list[dict]:
        del obs
        user_prompt = QWEN3VL_USER_PROMPT.format(
            instruction=self._task,
            history=self.step_his,
        )
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for url in self._recent_images:
            user_content.append({"type": "image_url", "image_url": {"url": url}})
        return [
            {"role": "system", "content": [{"type": "text", "text": QWEN3VL_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    def act(self, obs: Observation) -> Action:
        self.turn_number += 1
        image_url = obs.image_data_url
        if image_url:
            self._recent_images.append(image_url)

        messages = self.build_messages(obs)
        response = self.llm.chat(
            messages=messages,
            args={
                **self.config.model_args,
                "stream": self.config.stream,
                "stream_print": self.config.stream,
            },
        )
        raw = response.content or ""
        action = self.parse_response(raw)
        op_text = _extract_action_text(raw)
        self.step_his += f"Step {self.turn_number}: {op_text}; "
        self._history.append(
            AgentStepRecord(
                step_idx=len(self._history),
                observation=obs,
                action=action,
                llm_response=raw,
                llm_prompt=messages,
            )
        )
        return action
