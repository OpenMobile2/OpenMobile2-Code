"""OpenMobile session agent for official MobileGym bench_env eval.

Predicts actions with Qwen35Session / Qwen35ThoughtSession; does not drive
Playwright itself. bench_env owns instantiate / step / state-diff judging.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image

from android_world.agents import infer
from android_world.agents.session.agent import Qwen35Session, Qwen35ThoughtSession
from android_world.env import interface
from android_world.env import json_action
from bench_env.agent.base import AgentConfig, BaseAgent, AgentStepRecord
from bench_env.env.base import Action, ActionType, Observation

SESSION_CFG: dict[str, Any] = {
    "runtime": "qwen35_session",
    "last_n": 3,
    "enable_thinking": True,
    "require_think_tags": False,
    "qwen35_tool_call_mode": "native",
    "model_base_url": "http://<openai-compatible-host>/v1",
    "model_api_key": "EMPTY",
    "model_name": "YOUR_MODEL",
}


class _ScreenshotPumpEnv(interface.AsyncEnv):
    """Minimal AsyncEnv so Qwen35Session.step() can read pixels without executing."""

    interaction_cache = ""

    def __init__(self) -> None:
        self._pixels = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self._controller = None
        self.last_action: json_action.JSONAction | None = None

    def feed(self, pixels: np.ndarray) -> None:
        self._pixels = np.asarray(pixels)
        self.last_action = None

    @property
    def controller(self) -> Any:
        return self._controller

    def reset(self, go_home: bool = False) -> interface.State:
        del go_home
        return self.get_state()

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        del wait_to_stabilize
        return interface.State(
            pixels=self._pixels,
            forest=None,
            ui_elements=[],
            auxiliaries=None,
        )

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> str | None:
        del question, timeout_seconds
        return None

    def execute_action(self, action: json_action.JSONAction) -> None:
        self.last_action = action

    @property
    def foreground_activity_name(self) -> str:
        return ""

    @property
    def device_screen_size(self) -> tuple[int, int]:
        h, w = self._pixels.shape[:2]
        return (int(w), int(h))

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        return self.device_screen_size

    def close(self) -> None:
        return None

    def hide_automation_ui(self) -> None:
        return None

    @property
    def orientation(self) -> int:
        return 0

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        w, h = self.device_screen_size
        return (0, 0, w, h)


def _obs_to_pixels(obs: Observation) -> np.ndarray:
    if obs.screenshot is not None:
        return np.asarray(obs.screenshot)
    raw = obs.get_screenshot_bytes()
    if not raw:
        raise ValueError("Observation has no screenshot")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _norm_point(x: float, y: float, width: int, height: int) -> list[int]:
    return [
        int(round(float(x) / max(int(width), 1) * 1000)),
        int(round(float(y) / max(int(height), 1) * 1000)),
    ]


def _parsed_to_action(parsed: dict[str, Any], width: int, height: int, raw: str = "") -> Action:
    at = str(parsed.get("action_type") or "")
    kwargs = {"raw_response": raw}
    if at == "click":
        return Action.click(_norm_point(parsed["x"], parsed["y"], width, height), **kwargs)
    if at == "long_press":
        return Action(
            ActionType.LONG_PRESS,
            {"point": _norm_point(parsed["x"], parsed["y"], width, height)},
            **kwargs,
        )
    if at == "double_tap":
        return Action(
            ActionType.DOUBLE_TAP,
            {"point": _norm_point(parsed["x"], parsed["y"], width, height)},
            **kwargs,
        )
    if at == "swipe":
        p1 = _norm_point(parsed["x"], parsed["y"], width, height)
        p2 = _norm_point(parsed.get("x2", parsed["x"]), parsed.get("y2", parsed["y"]), width, height)
        return Action.swipe(p1, p2, **kwargs)
    if at == "input_text":
        return Action.type_text(str(parsed.get("text") or ""), **kwargs)
    if at == "navigate_home":
        return Action.home(**kwargs)
    if at == "navigate_back":
        return Action.back(**kwargs)
    if at == "keyboard_enter":
        return Action(ActionType.ENTER, {}, **kwargs)
    if at == "open_app":
        return Action.awake(str(parsed.get("app_name") or ""), **kwargs)
    if at == "wait":
        return Action.wait(1.0, **kwargs)
    if at == "answer":
        return Action.answer(str(parsed.get("text") or ""), **kwargs)
    if at == "status":
        if str(parsed.get("goal_status") or "") == "complete":
            return Action.complete(str(parsed.get("text") or ""), **kwargs)
        return Action.abort(str(parsed.get("goal_status") or "infeasible"), **kwargs)
    return Action.wait(0.5, **kwargs)


def _build_session(pump: _ScreenshotPumpEnv, cfg: dict[str, Any]) -> Qwen35Session:
    common: dict[str, Any] = {
        "wait_after_action_seconds": 0.0,
        "model_base_url": str(cfg["model_base_url"]),
        "model_api_key": str(cfg["model_api_key"]),
        "model_name": str(cfg["model_name"]),
        "last_n": int(cfg["last_n"]),
    }
    runtime = str(cfg.get("runtime") or "qwen35_session")
    if runtime == "qwen35_thought_session":
        return Qwen35ThoughtSession(
            pump,
            infer.Gpt4Wrapper("gpt-4o"),
            **common,
        )
    return Qwen35Session(
        pump,
        infer.Gpt4Wrapper("gpt-4o"),
        enable_thinking=bool(cfg["enable_thinking"]),
        require_think_tags=bool(cfg["require_think_tags"]),
        qwen35_tool_call_mode=str(cfg["qwen35_tool_call_mode"]),
        **common,
    )


class OpenMobileSessionBenchAgent(BaseAgent):
    """bench_env agent backed by OpenMobile Qwen35Session / ThoughtSession."""

    SYSTEM_PROMPT = ""
    ACTION_MAP = {}

    def __init__(self, llm: Any = None, config: AgentConfig | None = None):
        del llm
        super().__init__(config)
        self._pump = _ScreenshotPumpEnv()
        self._session = _build_session(self._pump, SESSION_CFG)
        self._session.transition_pause = 0.0
        self._session.save_dir = None

    @property
    def name(self) -> str:
        return str(SESSION_CFG.get("runtime") or "qwen35_session")

    def parse_response(self, response_text: str) -> Action:
        del response_text
        return Action.wait(0)

    def build_messages(self, obs: Observation) -> list[dict]:
        del obs
        return []

    def reset(self, task: str) -> None:
        self._task = task
        self._history = []
        self._session.reset(go_home_on_reset=False)

    def act(self, obs: Observation) -> Action:
        pixels = _obs_to_pixels(obs)
        height, width = pixels.shape[:2]
        self._pump.feed(pixels)
        result = self._session.step(self._task)
        parsed = (result.data or {}).get("parsed") or {}
        raw = str((result.data or {}).get("response") or "")
        if parsed:
            action = _parsed_to_action(parsed, width, height, raw)
        elif result.done:
            action = Action.abort(str((result.data or {}).get("summary") or "no tool_call"), raw_response=raw)
        else:
            action = Action.wait(0.5, raw_response=raw)
        self._history.append(
            AgentStepRecord(step_idx=len(self._history), observation=obs, action=action, llm_response=raw)
        )
        return action
