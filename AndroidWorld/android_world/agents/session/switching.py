"""Session-based policy switching: weak multi-turn actor + strong monitor/intervene.

Runtime name: ``qwen35_switching_session``.

Unlike flat ``qwen3vl_switching`` (text history only), both actors keep their own
multi-turn session message lists and dump ``session_io/`` each step.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents.seeact_v import (
    MONITOR_SYSTEM_PROMPT,
    _chat_with_retry,
    _extract_history_summary,
    _parse_monitor_output,
    _ui_element_to_metadata_dict,
)
from android_world.agents.session.agent import Qwen35Session
from android_world.agents.session.backends import resolve_session_backend
from android_world.agents.session.backends.base import SessionTurnResult
from android_world.agents.utils import qwen3vl_action_transform
from android_world.env import interface
from android_world.env import json_action


class Qwen35SwitchingSession(Qwen35Session):
    """Weak-first session rollout with monitor-triggered strong intervention.

    - Weak / strong each own a multi-turn ``_session_messages`` buffer + backend.
    - Monitor uses the strong client (same as flat switching).
    - On intervene: bootstrap a fresh strong session from instruction + history
      text + intervention feedback + current screenshot.
    - On hand-back to weak: append a continuity note + current screenshot.
    - ``session_io/`` dumps include ``policy_source`` / monitor fields.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen35SwitchingSession",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://<openai-compatible-host>/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        last_n: int = 1,
        session_dialect: str = "qwen35",
        enable_thinking: bool = True,
        require_think_tags: bool = True,
        reasoning_effort: str = "max",
        qwen35_tool_call_mode: str = "native",
        weak_model_base_url: str = "http://<weak-openai-compatible-host>/v1",
        weak_model_api_key: str = "EMPTY",
        weak_model_name: str = "",
        max_interventions: int = 2,
        strong_min_steps_after_intervention: int = 3,
    ):
        super().__init__(
            env=env,
            llm=llm,
            name=name,
            wait_after_action_seconds=wait_after_action_seconds,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=model_name,
            extra_headers=extra_headers,
            last_n=last_n,
            session_dialect=session_dialect,
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
            qwen35_tool_call_mode=qwen35_tool_call_mode,
        )
        self._strong_client = self.client
        self._strong_backend = self._backend
        self._strong_system_prompt = self._session_system_prompt
        self._strong_user_prompt = self._session_user_prompt
        self._strong_model_name = self.model_name
        self._strong_messages: list[dict[str, Any]] = []

        self.weak_model_name = weak_model_name or ""
        self.weak_client = OpenAI(
            api_key=weak_model_api_key,
            base_url=weak_model_base_url,
            default_headers=extra_headers,
        )
        self._weak_backend = resolve_session_backend(self.weak_model_name)
        if hasattr(self._weak_backend, "use_native_tools"):
            self._weak_backend.use_native_tools = bool(
                getattr(self._backend, "use_native_tools", False)
            )
        weak_prompts = self._weak_backend.prompts(
            session_dialect,
            enable_thinking=self.enable_thinking,
            require_think_tags=self.require_think_tags,
        )
        self._weak_system_prompt = weak_prompts.system
        self._weak_user_prompt = weak_prompts.user
        self._weak_messages: list[dict[str, Any]] = []

        self._session_runtime_name = "qwen35_switching_session"
        self.current_policy = "weak"
        self.has_intervened = False
        self.pending_intervention_feedback: str | None = None
        self.max_interventions = max(1, int(max_interventions))
        self.strong_min_steps_after_intervention = max(
            1, int(strong_min_steps_after_intervention)
        )
        self.intervention_count = 0
        self.strong_steps_since_intervention = 0
        self.should_monitor_next_step = False
        self._recent_screenshots: deque[np.ndarray] = deque(
            maxlen=max(1, int(last_n))
        )
        self._weak_needs_bootstrap = False
        # Chronological (response, policy) for result.json — survives strong bootstrap.
        self._export_steps: list[dict[str, Any]] = []
        self._activate_policy("weak")

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self._strong_messages = []
        self._weak_messages = []
        self._session_messages = []
        self.current_policy = "weak"
        self.has_intervened = False
        self.pending_intervention_feedback = None
        self.intervention_count = 0
        self.strong_steps_since_intervention = 0
        self.should_monitor_next_step = False
        self._recent_screenshots.clear()
        self._weak_needs_bootstrap = False
        self._export_steps = []
        self._activate_policy("weak")

    def export_session_record(
        self,
        *,
        episode_id: str,
        goal: str,
        save_dir: str | None = None,
    ) -> dict[str, Any]:
        """Full-episode export with every step screenshot (ignores last_n / policy buffers)."""
        del save_dir
        steps = self._export_steps
        images = [f"screenshot_step{i}.png" for i in range(len(steps))]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._weak_system_prompt},
            {
                "role": "user",
                "content": self._weak_user_prompt.format(instruction=goal)
                + ("\n<image>" if steps else ""),
            },
        ]
        for i, step in enumerate(steps):
            messages.append(
                {
                    "role": "assistant",
                    "content": step.get("response") or "",
                    "policy_source": step.get("policy_source"),
                }
            )
            if i + 1 < len(steps):
                messages.append({"role": "tool", "content": "<image>"})
        return {
            "id": episode_id,
            "runtime": self._session_runtime_name,
            "backend": "switching",
            "goal": goal,
            "messages": messages,
            "images": images,
        }

    def _activate_policy(self, policy: str) -> None:
        self.current_policy = policy
        if policy == "strong":
            self.client = self._strong_client
            self._backend = self._strong_backend
            self._session_system_prompt = self._strong_system_prompt
            self._session_user_prompt = self._strong_user_prompt
            self.model_name = self._strong_model_name
            self._session_messages = self._strong_messages
        else:
            self.client = self.weak_client
            self._backend = self._weak_backend
            self._session_system_prompt = self._weak_system_prompt
            self._session_user_prompt = self._weak_user_prompt
            self.model_name = self.weak_model_name
            self._session_messages = self._weak_messages

    def _persist_active_messages(self) -> None:
        if self.current_policy == "strong":
            self._strong_messages = self._session_messages
        else:
            self._weak_messages = self._session_messages

    def _to_base64_png(self, image: np.ndarray) -> str:
        import base64
        from io import BytesIO

        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _run_monitor(
        self, instruction: str, current_screenshot: np.ndarray
    ) -> dict[str, Any]:
        recent = list(self._recent_screenshots)[-self.last_N :]
        if not recent:
            recent = [current_screenshot]
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Instruction: {instruction}\n"
                    f"Action history so far: {self.step_his or 'No action has been executed yet.'}\n"
                    "The following screenshots are ordered from older to newer. "
                    "The last screenshot is the current screen.\n"
                    "Decide whether the weak agent should continue acting on this step."
                ),
            }
        ]
        for img in recent:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._to_base64_png(img)},
                }
            )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": MONITOR_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
        ]
        raw = _chat_with_retry(
            self._strong_client,
            self._strong_model_name,
            messages,
            retry_times=3,
            retry_sleep_s=5.0,
            retry_name="monitor_model",
            request_timeout_s=60.0,
        )
        print("[monitor judge]")
        print(raw)
        print("=" * 50)
        decision = _parse_monitor_output(raw)
        decision["raw_response"] = raw
        decision["checked"] = True
        decision["screens_considered"] = len(recent)
        return decision

    def _bootstrap_strong_session(
        self, instruction: str, screenshot: np.ndarray, feedback: str | None
    ) -> None:
        user_text = self._strong_user_prompt.format(instruction=instruction)
        user_text += (
            "\n\nPrevious actions so far:\n"
            f"{self.step_his or 'No action has been executed yet.'}\n"
        )
        if feedback:
            user_text += (
                "\nIntervention context:\n"
                "The weak agent's recent execution was judged to be off track. "
                "You are now taking over from the current screen.\n"
                f"{feedback}\n"
                "Use this assessment as additional context, but ground your next "
                "action in the current screen and full history.\n"
            )
        self._strong_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._strong_system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    self._image_content(screenshot),
                ],
            },
        ]
        self._activate_policy("strong")

    def _hand_back_to_weak(self, instruction: str, screenshot: np.ndarray) -> None:
        note = (
            "A stronger agent handled intervening steps. Continue the task from "
            "the current screen. Do not repeat completed actions.\n"
            f"History: {self.step_his or 'No action has been executed yet.'}"
        )
        if not self._weak_messages:
            self._weak_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._weak_system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._weak_user_prompt.format(
                                instruction=instruction
                            )
                            + "\n"
                            + note,
                        },
                        self._image_content(screenshot),
                    ],
                },
            ]
        else:
            self._activate_policy("weak")
            self._tool_call_counter += 1
            obs_msg, _ = self._backend.observation_message(
                image_part=self._image_content(screenshot),
                tool_call_counter=self._tool_call_counter,
                last_tool_call_id=self._last_tool_call_id,
            )
            # Attach continuity note as text alongside the observation when possible.
            content = obs_msg.get("content")
            if isinstance(content, list):
                content = [{"type": "text", "text": note}, *content]
                obs_msg = {**obs_msg, "content": content}
            self._session_messages.append(obs_msg)
            self._persist_active_messages()
            return
        self._activate_policy("weak")

    def _dump_step_io(
        self,
        *,
        instruction: str,
        messages_for_api: list[dict[str, Any]],
        response_raw: str,
        response_norm: str,
        create_kwargs: dict[str, Any] | None = None,
        response_message: Any | None = None,
        policy_source: str = "weak",
        monitor_output: dict[str, Any] | None = None,
        intervention_triggered: bool = False,
    ) -> None:
        if self.save_dir is None:
            return
        try:
            from android_world.agents.session.utils import (
                extract_message_content,
                extract_reasoning_from_message,
                redact_messages_for_io_dump,
                serialize_tool_calls,
            )

            io_dir = os.path.join(self.save_dir, "session_io")
            os.makedirs(io_dir, exist_ok=True)
            step_idx = self.turn_number - 1
            api_request: dict[str, Any] = {
                "messages": redact_messages_for_io_dump(messages_for_api)
            }
            if create_kwargs:
                api_request.update(
                    {k: v for k, v in create_kwargs.items() if k != "messages"}
                )
            if response_message is not None:
                raw_content = extract_message_content(response_message)
                reasoning = extract_reasoning_from_message(response_message)
                tool_calls = serialize_tool_calls(
                    getattr(response_message, "tool_calls", None)
                )
            else:
                raw_content = response_raw or ""
                reasoning = ""
                tool_calls = []
            payload = {
                "runtime": self._session_runtime_name,
                "backend": self._backend.key,
                "step": step_idx,
                "turn_number": self.turn_number,
                "last_n": self.last_N,
                "enable_thinking": self.enable_thinking,
                "require_think_tags": self.require_think_tags,
                "reasoning_effort": self.reasoning_effort,
                "model_name": self.model_name,
                "policy_source": policy_source,
                "monitor_output": monitor_output or {},
                "intervention_triggered": bool(intervention_triggered),
                "instruction": instruction,
                "api_request": api_request,
                "model_output_raw": raw_content,
                "model_reasoning_content": reasoning,
                "model_tool_calls": tool_calls,
                "model_output_stored": response_norm,
            }
            with open(
                os.path.join(io_dir, f"step{step_idx:02d}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with open(
                os.path.join(io_dir, f"step{step_idx:02d}_response_raw.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(raw_content or "")
        except Exception as e:
            print(f"Failed to dump session step IO: {e}")

    def _infer_one_turn(
        self, instruction: str
    ) -> tuple[Any, SessionTurnResult, list[dict[str, Any]], dict[str, Any]]:
        messages_for_api = self._backend.prepare_messages(
            self._build_messages_for_api(instruction)
        )
        create_kwargs = self._backend.create_kwargs(
            model_name=self.model_name,
            messages=messages_for_api,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )
        response_message = None
        turn = None
        api_messages = list(messages_for_api)
        for attempt in range(5):
            create_kwargs["messages"] = api_messages
            try:
                completion = self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                print(
                    f"{self.__class__.__name__} ({self.current_policy}) request "
                    f"failed or timed out after 60s: {e}"
                )
                response_message = None
                turn = None
                continue
            try:
                response_message = completion.choices[0].message
                turn = self._backend.parse_turn(
                    response_message, dialect=self.session_dialect
                )
            except Exception:
                response_message = None
                turn = None
            has_tool_calls = bool(
                response_message is not None
                and getattr(response_message, "tool_calls", None)
            )
            if not turn or not (turn.response_text.strip() or has_tool_calls):
                print("sleep 10 seconds and retry")
                time.sleep(10)
                continue
            if (
                self._backend.requires_text_thinking()
                and not self._backend.has_required_thinking(turn.response_text)
            ):
                retry_msg = self._backend.thinking_retry_user_message()
                if retry_msg is not None and attempt < 4:
                    print(
                        f"[{self._backend.key}] missing <thinking>; "
                        f"retrying ({attempt + 1}/4)"
                    )
                    api_messages = list(messages_for_api) + [retry_msg]
                    continue
            break

        if turn is None:
            turn = SessionTurnResult(
                response_text="",
                response_raw="",
                response_norm="",
                history_message={"role": "assistant", "content": ""},
            )
        return response_message, turn, api_messages, create_kwargs

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        height, width = screenshot.shape[:2]
        self._recent_screenshots.append(screenshot)

        if self.save_dir is not None:
            try:
                step_idx = self.turn_number - 1
                self._ui_elements_history.append(
                    {
                        "step": step_idx,
                        "logical_screen_size": list(self.env.logical_screen_size),
                        "ui_elements": [
                            _ui_element_to_metadata_dict(e) for e in state.ui_elements
                        ],
                    }
                )
                meta = {"goal": instruction, "steps": self._ui_elements_history}
                with open(
                    os.path.join(self.save_dir, "metadata.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save ui_elements metadata: {e}")
            try:
                screenshot_path = os.path.join(
                    self.save_dir, f"screenshot_step{self.turn_number - 1}.png"
                )
                Image.fromarray(screenshot).save(screenshot_path)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

        monitor_output: dict[str, Any] = {
            "checked": False,
            "should_intervene": False,
            "reason": "",
            "summary": "",
        }
        intervention_triggered = False

        if (
            self.current_policy == "weak"
            and self.should_monitor_next_step
            and self.intervention_count < self.max_interventions
        ):
            print("calling monitor...")
            monitor_output = self._run_monitor(instruction, screenshot)
            if bool(monitor_output.get("should_intervene")):
                summary = str(monitor_output.get("summary", "") or "").strip()
                reason = str(monitor_output.get("reason", "") or "").strip()
                parts = []
                if summary:
                    parts.append(f"Monitor summary: {summary}")
                if reason:
                    parts.append(f"Monitor reason: {reason}")
                feedback = "\n".join(parts).strip()
                self.pending_intervention_feedback = feedback
                self.has_intervened = True
                self.intervention_count += 1
                self.strong_steps_since_intervention = 0
                self.should_monitor_next_step = False
                intervention_triggered = True
                print(
                    "[switching] monitor triggered intervention, "
                    "switching to strong actor"
                )
                self._bootstrap_strong_session(instruction, screenshot, feedback)
            else:
                self._activate_policy("weak")
        else:
            self._activate_policy(self.current_policy)

        # Append observation into the active session (unless just bootstrapped).
        if intervention_triggered:
            pass  # strong session already has current screenshot
        elif self.current_policy == "weak" and self._weak_needs_bootstrap:
            self._hand_back_to_weak(instruction, screenshot)
            self._weak_needs_bootstrap = False
        elif not self._session_messages:
            self._session_messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self._session_system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._session_user_prompt.format(
                                instruction=instruction
                            ),
                        },
                        self._image_content(screenshot),
                    ],
                },
            ]
        else:
            self._tool_call_counter += 1
            obs_msg, _ = self._backend.observation_message(
                image_part=self._image_content(screenshot),
                tool_call_counter=self._tool_call_counter,
                last_tool_call_id=self._last_tool_call_id,
            )
            self._session_messages.append(obs_msg)

        step_policy_source = self.current_policy
        if self.turn_number == 1:
            print(self._session_user_prompt.format(instruction=instruction))
        print(f"[policy] {step_policy_source}")

        response_message, turn, api_messages, create_kwargs = self._infer_one_turn(
            instruction
        )
        print(turn.response_text)
        print("=" * 50)

        self._session_messages.append(turn.history_message)
        if turn.last_tool_call_id:
            self._last_tool_call_id = turn.last_tool_call_id
        self._persist_active_messages()
        self._export_steps.append(
            {
                "step": self.turn_number - 1,
                "policy_source": step_policy_source,
                "response": turn.response_norm,
                "intervention_triggered": intervention_triggered,
            }
        )

        self._dump_step_io(
            instruction=instruction,
            messages_for_api=api_messages,
            response_raw=turn.response_raw,
            response_norm=turn.response_norm,
            create_kwargs=create_kwargs,
            response_message=response_message,
            policy_source=step_policy_source,
            monitor_output=monitor_output,
            intervention_triggered=intervention_triggered,
        )

        tool_call = self._backend.parse_tool_call(
            response_message,
            turn.response_norm,
            parse_text_tool_call=self._parse_session_tool_call,
        )
        if not tool_call:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "No valid <tool_call> found in model output.",
                    "response": turn.response_norm,
                    "runtime": self._session_runtime_name,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        action_name = str(args.get("action", "") or "")
        op_text = _extract_history_summary(turn.response_norm) or action_name
        self.step_his += f"Step {self.turn_number}: {op_text}; "

        try:
            parsed = qwen3vl_action_transform(action_name, args, width, height)
            print(parsed)
        except Exception as e:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to transform tool-call into action: {e}",
                    "response": turn.response_norm,
                    "tool_call": tool_call,
                    "runtime": self._session_runtime_name,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        def _done_payload(**extra: Any) -> dict[str, Any]:
            return {
                "response": turn.response_norm,
                "parsed": parsed,
                "runtime": self._session_runtime_name,
                "policy_source": step_policy_source,
                "monitor_output": monitor_output,
                "intervention_triggered": intervention_triggered,
                "step_history": self.step_his,
                **extra,
            }

        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(True, _done_payload())

        action_executed = False
        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
            action_executed = True
        except Exception:
            print("Failed to execute action:", parsed)

        if action_executed and step_policy_source == "strong":
            self.strong_steps_since_intervention += 1
            self.should_monitor_next_step = False
            if (
                self.max_interventions >= 2
                and self.intervention_count == 1
                and self.strong_steps_since_intervention
                >= self.strong_min_steps_after_intervention
            ):
                self.strong_steps_since_intervention = 0
                print(
                    "[switching] first strong intervention finished, "
                    "switching back to weak actor"
                )
                self._persist_active_messages()
                self.current_policy = "weak"
                self._weak_needs_bootstrap = not bool(self._weak_messages)
                self._activate_policy("weak")
        elif action_executed and step_policy_source == "weak":
            self.should_monitor_next_step = True

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(True, _done_payload())

        return base_agent.AgentInteractionResult(False, _done_payload())
