"""Multi-turn session agents — shared loop, model-specific backends."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents.seeact_v import (
    Qwen35VL,
    _extract_history_summary,
    _parse_tool_call_json,
    _parse_tool_call_xml,
    _ui_element_to_metadata_dict,
)
from android_world.agents.session.backends import resolve_session_backend
from android_world.agents.session.backends.base import SessionBackend, SessionTurnResult
from android_world.agents.session.backends.qwen_osworld import QwenOsworldSessionBackend
from android_world.agents.session.backends.qwen_thought import QwenThoughtSessionBackend
from android_world.agents.session.osworld_history import (
    COLLAPSED_SCREENSHOT_TEXT,
    build_instruction_prompt,
    build_osworld_messages,
    extract_action_line,
    previous_actions_text,
    strip_think_tags,
    update_folding_state,
)
from android_world.agents.session.utils import (
    assistant_api_record_to_session_text,
    extract_message_content,
    extract_reasoning_from_message,
    extract_thought_prefix,
    message_text_content,
    normalize_session_assistant_content,
    normalize_thought_session_content,
    redact_messages_for_io_dump,
    serialize_tool_calls,
)
from android_world.agents.utils import qwen3vl_action_transform
from android_world.env import interface
from android_world.env import json_action


class Qwen35Session(Qwen35VL):
    """Multi-turn session runtime (``qwen35_session.md`` shape).

    Model-family behaviour (prompts, API kwargs, message shape) lives in
    ``session.backends.*``; this class owns the shared step / export loop.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen35Session",
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
    ):
        if session_dialect not in {"qwen35", "qwen3vl"}:
            raise ValueError(
                "session_dialect must be 'qwen35' or 'qwen3vl', "
                f"got {session_dialect!r}"
            )
        effort = (reasoning_effort or "max").strip().lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError(
                f"reasoning_effort must be low|high|max, got {reasoning_effort!r}"
            )
        mode = (qwen35_tool_call_mode or "native").strip().lower()
        if mode not in {"xml", "native"}:
            raise ValueError(
                "qwen35_tool_call_mode must be xml|native, "
                f"got {qwen35_tool_call_mode!r}"
            )
        super().__init__(
            env=env,
            llm=llm,
            name=name,
            wait_after_action_seconds=wait_after_action_seconds,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=model_name,
            extra_headers=extra_headers,
            qwen35_tool_call_mode=mode,
            use_memory_prompt=False,
            use_memgui_prompt=False,
            last_n=last_n,
        )
        self.session_dialect = session_dialect
        self.enable_thinking = bool(enable_thinking)
        self.require_think_tags = bool(require_think_tags)
        self.reasoning_effort = effort
        self._backend: SessionBackend = resolve_session_backend(model_name)
        if hasattr(self._backend, "use_native_tools"):
            self._backend.use_native_tools = mode == "native"
        prompts = self._backend.prompts(
            session_dialect,
            enable_thinking=self.enable_thinking,
            require_think_tags=self.require_think_tags,
        )
        self._session_system_prompt = prompts.system
        self._session_user_prompt = prompts.user
        self._session_runtime_name = prompts.runtime_name
        self._session_messages: list[dict[str, Any]] = []
        self._tool_call_counter: int = 0
        self._last_tool_call_id: str | None = None
        # Optional per-turn OpenAI tools= override (GUI-only stays QWEN35_TOOLS).
        self._turn_tools: list[dict[str, Any]] | None = None
        self._next_observation_text: str | None = None
        self._next_user_text: str | None = None
        self._last_tool_name: str = "mobile_use"
        self._request_timeout: float | None = None

    def set_turn_tools(self, tools: list[dict[str, Any]] | None) -> None:
        """Replace this turn's ``tools=`` list. ``None`` keeps the backend default."""
        self._turn_tools = tools

    def set_next_observation_text(self, text: str | None) -> None:
        """Optional JSON/text attached to the next tool-role screenshot message."""
        self._next_observation_text = text

    def set_next_user_text(self, text: str | None) -> None:
        """Optional extra user message after this turn's screenshot observation."""
        self._next_user_text = text

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self._session_messages = []
        self._tool_call_counter = 0
        self._last_tool_call_id = None
        self._turn_tools = None
        self._next_observation_text = None
        self._next_user_text = None
        self._last_tool_name = "mobile_use"

    def _parse_session_tool_call(self, response: str) -> dict[str, Any] | None:
        if self.session_dialect == "qwen3vl":
            return _parse_tool_call_json(response) or _parse_tool_call_xml(response)
        return _parse_tool_call_xml(response) or _parse_tool_call_json(response)

    def _normalize_exported_assistant(self, text: str) -> str:
        if getattr(self._backend, "key", "") == "qwen_thought":
            return normalize_thought_session_content(text)
        return normalize_session_assistant_content(text)

    @staticmethod
    def _message_has_image(msg: dict[str, Any]) -> bool:
        content = msg.get("content")
        return isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        )

    def _image_bearing_message_indices(
        self, messages: list[dict[str, Any]]
    ) -> list[int]:
        return [
            i
            for i, msg in enumerate(messages)
            if self._message_has_image(msg) or msg.get("_session_observation")
        ]

    def _kept_image_message_indices(
        self, messages: list[dict[str, Any]]
    ) -> set[int]:
        img_indices = self._image_bearing_message_indices(messages)
        keep_n = max(1, int(self.last_N))
        return set(img_indices[-keep_n:]) if img_indices else set()

    @staticmethod
    def _collapse_image_parts(content: list) -> list:
        text_parts = [
            p
            for p in content
            if not (isinstance(p, dict) and p.get("type") == "image_url")
        ]
        n_images = sum(
            1
            for p in content
            if isinstance(p, dict) and p.get("type") == "image_url"
        )
        if n_images == 0:
            n_images = 1
        for _ in range(n_images):
            text_parts.append({"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT})
        return text_parts

    def _strip_old_images(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keep = self._kept_image_message_indices(messages)
        out: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            has_image_slot = self._message_has_image(msg) or msg.get(
                "_session_observation"
            )
            if i in keep or not has_image_slot:
                out.append(msg)
                continue
            new_msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                new_msg["content"] = self._collapse_image_parts(content)
            else:
                new_msg["content"] = [
                    {"type": "text", "text": COLLAPSED_SCREENSHOT_TEXT}
                ]
            out.append(new_msg)
        return out

    def _build_messages_for_api(self, instruction: str) -> list[dict[str, Any]]:
        """Messages sent to the model this turn (default: last_n image strip)."""
        del instruction
        if self.turn_number > 1:
            print(
                f"[{self._session_runtime_name}/{self._backend.key} "
                f"turn {self.turn_number}] screenshot appended "
                f"(keep last_n={self.last_N})"
            )
        return self._strip_old_images(self._session_messages)

    def _image_content(self, screenshot: np.ndarray) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self._to_base64_png(screenshot)},
        }

    def _dump_step_io(
        self,
        *,
        instruction: str,
        messages_for_api: list[dict[str, Any]],
        response_raw: str,
        response_norm: str,
        create_kwargs: dict[str, Any] | None = None,
        response_message: Any | None = None,
    ) -> None:
        if self.save_dir is None:
            return
        try:
            io_dir = os.path.join(self.save_dir, "session_io")
            os.makedirs(io_dir, exist_ok=True)
            step_idx = self.turn_number - 1
            api_request = {
                "messages": redact_messages_for_io_dump(messages_for_api),
            }
            if create_kwargs:
                api_request.update(
                    {k: v for k, v in create_kwargs.items() if k != "messages"}
                )
            # Dump API fields as returned. Do not wrap reasoning in <think>.
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
            if not reasoning:
                reasoning = extract_thought_prefix(
                    response_norm or raw_content or ""
                )
            payload = {
                "runtime": self._session_runtime_name,
                "backend": self._backend.key,
                "step": step_idx,
                "turn_number": self.turn_number,
                "last_n": self.last_N,
                "enable_thinking": self.enable_thinking,
                "require_think_tags": self.require_think_tags,
                "qwen35_tool_call_mode": getattr(
                    self, "qwen35_tool_call_mode", "native"
                ),
                "reasoning_effort": self.reasoning_effort,
                "model_name": self.model_name,
                "instruction": instruction,
                "api_request": api_request,
                "model_output_raw": raw_content,
                "model_reasoning_content": reasoning,
                "model_tool_calls": tool_calls,
                "model_output_stored": response_norm,
                **self._backend.io_flags(),
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

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        height, width = screenshot.shape[:2]

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
                    os.path.join(self.save_dir, "metadata.json"), "w", encoding="utf-8"
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

        if not self._session_messages:
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
            obs_msg = dict(obs_msg)
            if self._last_tool_name:
                obs_msg["name"] = self._last_tool_name
            extra = self._next_observation_text
            self._next_observation_text = None
            if extra:
                content = obs_msg.get("content")
                extra_part = {"type": "text", "text": extra}
                if isinstance(content, list):
                    obs_msg["content"] = [extra_part, *content]
                elif content:
                    obs_msg["content"] = [
                        extra_part,
                        {"type": "text", "text": str(content)},
                    ]
                else:
                    obs_msg["content"] = [extra_part]
            self._session_messages.append(obs_msg)

        extra_user = self._next_user_text
        self._next_user_text = None
        if extra_user:
            self._session_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": extra_user}],
                }
            )

        messages_for_api = self._backend.prepare_messages(
            self._build_messages_for_api(instruction)
        )

        if self.turn_number == 1:
            print(self._session_user_prompt.format(instruction=instruction))

        create_kwargs = self._backend.create_kwargs(
            model_name=self.model_name,
            messages=messages_for_api,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )
        if self._turn_tools is not None:
            create_kwargs["tools"] = self._turn_tools
            create_kwargs["tool_choice"] = "auto"
        if self._request_timeout is not None:
            create_kwargs["timeout"] = float(self._request_timeout)

        response_message = None
        turn = None
        api_messages = list(messages_for_api)
        for attempt in range(5):
            create_kwargs["messages"] = api_messages
            try:
                completion = self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                timeout_s = create_kwargs.get("timeout", 60)
                print(
                    f"{self.__class__.__name__} request failed or timed out "
                    f"after {timeout_s}s: {e}"
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
                        f"[{self._backend.key}] missing required thought; "
                        f"retrying ({attempt + 1}/4)"
                    )
                    api_messages = list(messages_for_api) + [retry_msg]
                    continue
            break

        # Dump uses the messages actually sent on the last attempt.
        messages_for_api = api_messages

        if turn is None:
            turn = SessionTurnResult(
                response_text="",
                response_raw="",
                response_norm="",
                history_message={"role": "assistant", "content": ""},
            )

        print(turn.response_text)
        print("=" * 50)

        self._session_messages.append(turn.history_message)
        if turn.last_tool_call_id:
            self._last_tool_call_id = turn.last_tool_call_id

        self._dump_step_io(
            instruction=instruction,
            messages_for_api=messages_for_api,
            response_raw=turn.response_raw,
            response_norm=turn.response_norm,
            create_kwargs=create_kwargs,
            response_message=response_message,
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
                },
            )

        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        if not isinstance(args, dict):
            args = {}
        tool_name = str(tool_call.get("name") or "mobile_use")
        self._last_tool_name = tool_name
        action_name = str(args.get("action", "") or "")
        op_text = _extract_history_summary(turn.response_norm) or action_name or tool_name
        self.step_his += f"Step {self.turn_number}: {op_text}; "

        if tool_name != "mobile_use":
            parsed = {
                "action_type": "mcp_tool",
                "tool_name": tool_name,
                "arguments": args,
            }
            print(parsed)
            return base_agent.AgentInteractionResult(
                False,
                {
                    "response": turn.response_norm,
                    "parsed": parsed,
                    "tool_call": tool_call,
                    "runtime": self._session_runtime_name,
                },
            )

        try:
            if action_name == "open_app":
                parsed = {
                    "action_type": "open_app",
                    "app_name": str(args.get("text") or args.get("app_name") or ""),
                }
            else:
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
                },
            )

        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(
                True,
                {
                    "response": turn.response_norm,
                    "parsed": parsed,
                    "runtime": self._session_runtime_name,
                },
            )

        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
        except Exception:
            print("Failed to execute action:", parsed)

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(
                True,
                {
                    "response": turn.response_norm,
                    "parsed": parsed,
                    "runtime": self._session_runtime_name,
                },
            )

        return base_agent.AgentInteractionResult(
            False,
            {
                "response": turn.response_norm,
                "parsed": parsed,
                "runtime": self._session_runtime_name,
            },
        )

    def export_session_record(
        self,
        *,
        episode_id: str,
        goal: str,
        save_dir: str | None = None,
    ) -> dict[str, Any]:
        del save_dir
        images: list[str] = []
        messages: list[dict[str, str]] = []
        img_i = 0

        def _image_count(content: Any) -> int:
            if not isinstance(content, list):
                return 0
            return sum(
                1
                for p in content
                if isinstance(p, dict) and p.get("type") == "image_url"
            )

        # Export keeps every screenshot. last_n only applies to live API calls
        # via _strip_old_images / _build_messages_for_api.
        for msg in self._session_messages:
            role = str(msg.get("role") or "")
            content = msg.get("content")
            n_img = _image_count(content)

            if role == "system":
                messages.append(
                    {
                        "role": "system",
                        "content": message_text_content(content).strip(),
                    }
                )
            elif role == "user":
                text = message_text_content(content).rstrip()
                if msg.get("_session_observation"):
                    count = n_img if n_img > 0 else 1
                    messages.append({"role": "tool", "content": "<image>" * count})
                    for _ in range(count):
                        images.append(f"screenshot_step{img_i}.png")
                        img_i += 1
                    continue
                if n_img:
                    if "<image>" not in text:
                        text = (text + "\n" if text else "") + ("<image>" * n_img)
                    for _ in range(n_img):
                        images.append(f"screenshot_step{img_i}.png")
                        img_i += 1
                messages.append({"role": "user", "content": text})
            elif role == "assistant":
                if msg.get("reasoning_content") or msg.get("tool_calls"):
                    stored = self._normalize_exported_assistant(
                        assistant_api_record_to_session_text(
                            msg,
                            dialect=self.session_dialect,
                        )
                    )
                    messages.append({"role": "assistant", "content": stored})
                else:
                    text = (
                        message_text_content(content)
                        if not isinstance(content, str)
                        else content
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": self._normalize_exported_assistant(text),
                        }
                    )
            elif role == "tool":
                count = n_img if n_img > 0 else 1
                messages.append({"role": "tool", "content": "<image>" * count})
                for _ in range(count):
                    images.append(f"screenshot_step{img_i}.png")
                    img_i += 1

        return {
            "id": episode_id,
            "runtime": self._session_runtime_name,
            "backend": self._backend.key,
            "goal": goal,
            "messages": messages,
            "images": images,
        }


class Qwen35ThoughtSession(Qwen35Session):
    """Session with ``Thought:`` in assistant text (no native thinking).

    Forces :class:`QwenThoughtSessionBackend`. XML tools live in the system
    prompt (no ``tools=``). ``enable_thinking`` is always false so the model
    writes ``Thought:`` in content instead of ``reasoning_content`` / ``<think>``.
    ``--enable_thinking`` and ``--qwen35_tool_call_mode`` are ignored.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen35ThoughtSession",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://<openai-compatible-host>/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        last_n: int = 1,
        enable_thinking: bool = False,
        require_think_tags: bool = False,
        reasoning_effort: str = "max",
        qwen35_tool_call_mode: str = "xml",
    ):
        del enable_thinking, require_think_tags, qwen35_tool_call_mode
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
            session_dialect="qwen35",
            enable_thinking=False,
            require_think_tags=False,
            reasoning_effort=reasoning_effort,
            qwen35_tool_call_mode="xml",
        )
        self._backend = QwenThoughtSessionBackend()
        prompts = self._backend.prompts(
            self.session_dialect,
            enable_thinking=False,
            require_think_tags=False,
        )
        self._session_system_prompt = prompts.system
        self._session_user_prompt = prompts.user
        self._session_runtime_name = prompts.runtime_name


class Qwen3VLSession(Qwen35Session):
    """Qwen3-VL multi-turn session: ``<think>`` + JSON ``<tool_call>``."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen3VLSession",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://<openai-compatible-host>/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        last_n: int = 1,
        enable_thinking: bool = True,
        require_think_tags: bool = True,
        reasoning_effort: str = "max",
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
            session_dialect="qwen3vl",
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
        )


class Qwen35OsworldSession(Qwen35Session):
    """OSWorld-style Qwen3.5 session: ``Action:`` + XML ``mobile_use`` tool_call.

    Forces :class:`QwenOsworldSessionBackend` prompts. API messages are rebuilt
    each turn like OSWorld ``mm_agents.qwen``:

    - ``history_n``: dialogue window of the last N steps
    - outside the window: ``Previous actions`` text
    - inside the window, when visible images exceed ``image_max``: fold oldest
      screenshots into collapse text inside ``<tool_response>`` (slot kept)

    Disk / ``_session_messages`` still keep full history; only the API payload is
    rebuilt. ``last_n`` is ignored for API image retention on this runtime.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen35OsworldSession",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://<openai-compatible-host>/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        last_n: int = 1,
        enable_thinking: bool = False,
        require_think_tags: bool = False,
        reasoning_effort: str = "max",
        history_n: int = 100,
        image_max: int = 20,
        fold_size: int = 10,
        collapse_text: str | None = None,
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
            session_dialect="qwen35",
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
        )
        self._backend = QwenOsworldSessionBackend()
        prompts = self._backend.prompts(
            self.session_dialect,
            enable_thinking=self.enable_thinking,
            require_think_tags=self.require_think_tags,
        )
        self._session_system_prompt = prompts.system
        self._session_user_prompt = prompts.user
        self._session_runtime_name = prompts.runtime_name
        self.history_n = max(1, int(history_n))
        self.image_max = max(1, int(image_max))
        self.fold_size = max(1, int(fold_size))
        self.collapse_text = collapse_text or COLLAPSED_SCREENSHOT_TEXT
        self.folded_prefix_k = 0

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.folded_prefix_k = 0

    @staticmethod
    def _iter_image_urls(msg: dict[str, Any]) -> list[str]:
        content = msg.get("content")
        if not isinstance(content, list):
            return []
        urls: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = ((part.get("image_url") or {}).get("url")) or ""
            if url:
                urls.append(url)
        return urls

    def _collect_screenshots_and_responses(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        screenshots: list[str] = []
        responses: list[str] = []
        for msg in self._session_messages:
            role = msg.get("role")
            if role == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    responses.append(content)
                elif isinstance(content, list):
                    responses.append(message_text_content(content))
                else:
                    responses.append("")
                continue
            screenshots.extend(self._iter_image_urls(msg))
        actions = [extract_action_line(r) for r in responses]
        return screenshots, responses, actions

    def _build_messages_for_api(self, instruction: str) -> list[dict[str, Any]]:
        screenshots, responses, actions = self._collect_screenshots_and_responses()
        total_steps = len(screenshots)
        if total_steps < 1:
            return [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self._session_system_prompt}
                    ],
                }
            ]

        self.folded_prefix_k = update_folding_state(
            total_steps,
            self.folded_prefix_k,
            self.image_max,
            self.fold_size,
        )
        start_step = max(1, total_steps - self.history_n)
        previous = previous_actions_text(actions, start_step)
        instruction_prompt = build_instruction_prompt(instruction, previous)
        messages = build_osworld_messages(
            system_prompt=self._session_system_prompt,
            instruction_prompt=instruction_prompt,
            screenshots=screenshots,
            responses=responses,
            start_step=start_step,
            total_steps=total_steps,
            folded_prefix_k=self.folded_prefix_k,
            collapse_text=self.collapse_text,
            response_transform=strip_think_tags,
        )
        print(
            f"[{self._session_runtime_name}/{self._backend.key} "
            f"turn {self.turn_number}] osworld rebuild "
            f"history_n={self.history_n} window=[{start_step},{total_steps}] "
            f"folded_prefix_k={self.folded_prefix_k} image_max={self.image_max}"
        )
        return messages
