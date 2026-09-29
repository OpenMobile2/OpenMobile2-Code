"""UI-Venus-1.5 agent for AndroidWorld / MobileWorld.

Official prompt and parse mirror MobileGym ``bench_env.agent.venus.VenusAgent``:
``<think>/<action>/<conclusion>``, current screenshot only, text previous_actions.
Coordinates are normalized 0–999 (same scale as Qwen3VL's /1000 mapping).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from io import BytesIO
from typing import Any, Optional

import numpy as np
from openai import OpenAI
from PIL import Image as PILImage

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents.utils import _coord_swipe_action
from android_world.env import interface
from android_world.env import json_action


PROMPT_TEMPLATE = """\n**你是一个手机图形界面智能体代理**
你的任务是根据历史操作和当前设备状态去执行一系列操作来完成用户的任务。

###你可以用的操作以及对应功能如下
- Click(box=(x1,y1))
>>点击操作，点击屏幕上的指定位置。坐标区间从左上角(0,0)到右下角(999,999)。
- Drag(start=(x1,y1), end=(x2,y2))
>>拖动操作，从起始坐标长按数秒之后拖动到结束坐标。用于调整app布局，滑动滑块验证码等。
- Scroll(start=(x1,y1), end=(x2,y2))
>>滑动操作，从起始坐标拖动到结束坐标。用于滚动查找内容，切换选项卡，下拉通知栏等。坐标区间从左上角(0,0)到右下角(999,999)。
- Type(content='')
>>输入操作，在当前激活的输入框输入指定内容。
- Launch(app='')
>>启动目标app。当目标app在当前界面不可见时，可以使用该动作打开app。
- Wait()
>>等待页面加载。
- Finished(content='')
>>任务结束，退出设备接管。
- CallUser(content='')
>>回答用户的问题或者当前界面有多个符合要求的选项时需要用户接管。
- LongPress(box=(x1,y1))
>>长按操作，在指定位置长按一定的时间。该操作可以触发更多功能选项，例如复制、转发消息，删除等。坐标区间从左上角(0,0)到右下角(999,999)。
- PressBack()
>>返回上一个界面，一般用于错误回退或继续执行剩余任务。
- PressHome()
>>返回系统桌面，一般用于跨app任务中快速打开下一个app或遇到严重错误时回退到系统桌面。
- PressEnter()
>>回车操作，用于换行或者在搜索框中输入内容之后执行搜索操作。
- PressRecent()
>>打开系统后台界面。

###用户任务
{user_task}

###先前的动作和推理过程
{previous_actions}

###输出格式
<think>你的思考过程</think>
<action>执行的操作</action>
<conclusion>总结当前操作</conclusion>

###额外的提示
-输入内容之前，确保输入框已经被激活（出现键盘或者'ADB Keyboard {{ON}}'字样代表输入框已经激活）。
-在app内找不到任务要求的入口时，尝试使用搜索功能，或者如果当前页面上方存在选多个项卡，尝试使用Scroll操作查看。
-如果在执行任务的过程中进入到和任务无关的界面，使用PressBack进行回退。
-任务结束之前，确保已经完整准确地完成用户的任务，如果存在漏做、错做的内容，需要返回重新执行。
"""

# Official Venus keeps only the current screenshot in the user turn.
HISTORY_N = 1


def _extract_tag(tag_name: str, data: str) -> Optional[str]:
    match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", data, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_venus_llm_output(response_text: str) -> dict[str, Any]:
    """Parse Venus ``<think>/<action>/<conclusion>`` into action + params."""
    raw = str(response_text or "").strip()
    if not raw:
        return {
            "action": "CallUser",
            "params": {"content": "empty_response"},
            "think": "",
            "conclusion": "",
            "raw_action": "",
        }

    think = _extract_tag("think", raw) or ""
    conclusion = _extract_tag("conclusion", raw) or ""
    action_str = _extract_tag("action", raw) or raw.strip()

    action_type = "CallUser"
    params: dict[str, Any] = {"content": "无法解析模型输出"}

    try:
        func_call_match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
        if func_call_match:
            func_name = func_call_match.group(1)
            params_str = func_call_match.group(2)
            parsed_params: dict[str, Any] = {}
            param_pattern = r"(\w+)\s*=\s*(\([^)]+\)|'[^']*'|\"[^\"]*\"|[^,]+)"
            for key, value in re.findall(param_pattern, params_str):
                key = key.strip()
                value = value.strip()
                if value.startswith("'") and value.endswith("'"):
                    parsed_params[key] = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
                    parsed_params[key] = value[1:-1]
                elif value.startswith("(") and value.endswith(")"):
                    try:
                        parts = [p.strip() for p in value[1:-1].split(",")]
                        parsed_params[key] = [int(float(p)) for p in parts]
                    except ValueError:
                        parsed_params[key] = value
                elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
                    parsed_params[key] = (
                        float(value) if "." in value else int(value)
                    )
                else:
                    parsed_params[key] = value
            action_type = func_name
            params = parsed_params
    except (ValueError, KeyError) as exc:
        action_type = "CallUser"
        params = {"content": f"解析失败: {exc}"}

    return {
        "action": action_type,
        "params": params,
        "think": think,
        "conclusion": conclusion,
        "raw_action": action_str,
    }


def _norm_point(point: Any, width: int, height: int) -> tuple[float, float]:
    """Map Venus 0–999 (or 0–1000) coords to screen pixels."""
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return 0.0, 0.0
    # Official prompt says 0–999; use /1000 like Qwen3VL for edge safety.
    x = float(point[0]) / 1000.0 * width
    y = float(point[1]) / 1000.0 * height
    return x, y


def venus_action_to_json(
    action_name: str,
    params: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Map Venus action name + params to AndroidWorld JSONAction kwargs."""
    name = (action_name or "").strip()
    params = params or {}

    if name == "Click":
        x, y = _norm_point(params.get("box"), width, height)
        return {"action_type": "click", "x": x, "y": y}
    if name == "LongPress":
        x, y = _norm_point(params.get("box"), width, height)
        return {"action_type": "long_press", "x": x, "y": y}
    if name in {"Scroll", "Drag"}:
        x0, y0 = _norm_point(params.get("start"), width, height)
        x1, y1 = _norm_point(params.get("end"), width, height)
        return _coord_swipe_action(x0, y0, x1, y1)
    if name == "Type":
        return {
            "action_type": "input_text",
            "text": str(params.get("content", "")),
        }
    if name == "Launch":
        return {
            "action_type": "open_app",
            "app_name": str(params.get("app", "")),
        }
    if name == "Wait":
        return {"action_type": "wait"}
    if name == "Finished":
        content = str(params.get("content", "") or "").strip()
        if content:
            return {"action_type": "answer", "text": content}
        return {"action_type": "status", "goal_status": "complete"}
    if name == "CallUser":
        return {
            "action_type": "answer",
            "text": str(params.get("content", "")),
        }
    if name == "PressBack":
        return {"action_type": "navigate_back"}
    if name == "PressHome":
        return {"action_type": "navigate_home"}
    if name == "PressEnter":
        return {"action_type": "keyboard_enter"}
    if name == "PressRecent":
        # JSONAction has no app-switch / recents; treat as no-op wait.
        return {"action_type": "wait"}
    return {"action_type": "wait"}


class VenusAgent(base_agent.EnvironmentInteractingAgent):
    """UI-Venus-1.5: current image + text previous_actions; HISTORY_N=1."""

    HISTORY_N = HISTORY_N

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "venus",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://<openai-compatible-host>/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        last_n: int = 1,
    ):
        del llm  # OpenAI client below; keep signature aligned with factory.
        super().__init__(env, name)
        self.wait_after_action_seconds = wait_after_action_seconds
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.client = OpenAI(
            api_key=model_api_key,
            base_url=model_base_url,
            default_headers=extra_headers,
        )
        # Venus official prompt always uses the current screenshot only.
        self.last_N = HISTORY_N
        if int(last_n) != HISTORY_N:
            print(
                f"[VenusAgent] ignoring last_n={last_n}; "
                f"forced to HISTORY_N={HISTORY_N}"
            )

        self._action_history: list[str] = []
        self.turn_number = 0
        self.last_action: str | None = None
        self.repeat_time = 0

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self._action_history = []
        self.turn_number = 0
        self.last_action = None
        self.repeat_time = 0

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format="PNG")
        return (
            "data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode()
        )

    def _build_messages(self, instruction: str, screenshot: np.ndarray) -> list[dict]:
        formatted = [
            f"Step {i}: {item}" for i, item in enumerate(self._action_history)
        ]
        prompt_text = PROMPT_TEMPLATE.format(
            user_task=instruction,
            previous_actions="\n".join(formatted),
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_base64_png(screenshot)},
                    },
                ],
            }
        ]

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1
        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        height, width = screenshot.shape[:2]

        if self.save_dir is not None:
            try:
                path = os.path.join(
                    self.save_dir, f"screenshot_step{self.turn_number - 1}.png"
                )
                PILImage.fromarray(screenshot).save(path)
            except Exception as exc:
                print(f"Failed to save screenshot: {exc}")

        messages = self._build_messages(instruction, screenshot)
        response = ""
        for _ in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                    timeout=60.0,
                )
                response = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"VenusAgent request failed: {exc}")
                response = ""
            if response.strip():
                break
            print("sleep 10 seconds and retry")
            time.sleep(10)

        print(response)
        print("=" * 50)

        parsed = parse_venus_llm_output(response)
        think = parsed.get("think", "")
        raw_action = parsed.get("raw_action", "")
        if think:
            hist_line = f"<think>{think}</think><action>{raw_action}</action>"
        else:
            hist_line = raw_action or str(parsed.get("action", ""))
        self._action_history.append(hist_line)

        try:
            action_dict = venus_action_to_json(
                str(parsed.get("action", "")),
                parsed.get("params") or {},
                width,
                height,
            )
        except Exception as exc:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to transform Venus action: {exc}",
                    "response": response,
                    "parsed": parsed,
                },
            )

        print(action_dict)

        if action_dict.get("action_type") in {"answer", "status"}:
            try:
                act = json_action.JSONAction(**action_dict)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute terminal action:", action_dict)
            return base_agent.AgentInteractionResult(
                True,
                {
                    "response": response,
                    "parsed": parsed,
                    "action": action_dict,
                    "step_history": "\n".join(self._action_history),
                },
            )

        try:
            action_sig = json.dumps(
                {"a": parsed.get("action"), "p": parsed.get("params")},
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            action_sig = str(parsed.get("raw_action"))
        if self.last_action == action_sig:
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = action_sig

        try:
            act = json_action.JSONAction(**action_dict)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
        except Exception:
            print("Failed to execute action:", action_dict)

        if self.repeat_time >= 10:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Terminated due to repeated identical actions.",
                    "response": response,
                    "parsed": parsed,
                    "action": action_dict,
                    "repeat_time": self.repeat_time,
                    "step_history": "\n".join(self._action_history),
                },
            )

        return base_agent.AgentInteractionResult(
            False,
            {
                "response": response,
                "parsed": parsed,
                "action": action_dict,
                "step_history": "\n".join(self._action_history),
            },
        )
