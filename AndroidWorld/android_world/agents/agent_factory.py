"""Factory for shared GUI agents across MobileGym / MobileWorld / AndroidWorld."""

from __future__ import annotations

from typing import Any

from android_world.agents import base_agent
from android_world.agents import human_agent
from android_world.agents import infer
from android_world.agents import random_agent
from android_world.agents import seeact_v
from android_world.agents import venus as venus_agent
from android_world.agents.session import (
    Qwen35OsworldSession,
    Qwen35Session,
    Qwen35SwitchingSession,
    Qwen35ThoughtSession,
    Qwen3VLSession,
)
from android_world.agents.session_runtimes import RuntimeSpec
from android_world.agents.session_runtimes import resolve_runtime
from android_world.env import interface


def create_gui_agent(
    env: interface.AsyncEnv,
    *,
    runtime: str = "",
    agent_name: str = "qwen3vl",
    use_memgui_prompt: bool = False,
    use_memory_prompt: bool = False,
    memgui_prompt_format: str = "auto",
    last_n: int = 1,
    enable_thinking: bool = True,
    require_think_tags: bool = True,
    reasoning_effort: str = "max",
    history_n: int = 100,
    image_max: int = 20,
    fold_size: int = 10,
        qwen35_tool_call_mode: str = "native",
    model_base_url: str = "http://<openai-compatible-host>/v1",
    model_api_key: str = "EMPTY",
    model_name: str = "",
    weak_model_base_url: str = "",
    weak_model_api_key: str = "EMPTY",
    weak_model_name: str = "",
    wait_after_action_seconds: float = 2.0,
    extra_headers: dict[str, str] | None = None,
) -> tuple[base_agent.EnvironmentInteractingAgent, RuntimeSpec]:
    """Build an agent from ``--runtime`` or legacy agent/prompt flags.

    Returns ``(agent, resolved_runtime_spec)``.

    ``last_n`` controls how many recent screenshots are visible to the model:
    - ``qwen3vl`` / ``qwen35vl`` / ``qwen25vl``: pack last N images into the current user turn
    - ``qwen3vl_session`` / ``qwen35_session``: keep images only in the last N
      image-bearing messages
    - ``venus``: ignored (official HISTORY_N=1, current image only)

    ``qwen35_osworld_session`` ignores ``last_n`` for API retention and instead
    uses OSWorld-style ``history_n`` / ``image_max`` / ``fold_size``.

    ``enable_thinking`` applies to Qwen session (vLLM ``chat_template_kwargs``).
    ``require_think_tags`` (qwen35/qwen3vl_session): if False, use the tool_call-only
    system prompt (no mandatory ``<think>`` in content). Independent of
    ``enable_thinking``.
    ``qwen35_thought_session`` ignores both flags: it always uses ``Thought:`` in
    assistant text with ``enable_thinking=false`` and XML tools in the system prompt.
    ``reasoning_effort``:
    - Kimi K3: API ``reasoning_effort`` (``low`` / ``high`` / ``max``)
    - DeepSeek-V4: API ``reasoning_effort`` (``low`` maps to ``high``)
    - Gemini: ``thinking_level`` via ``extra_body.extra_body.google.thinking_config``
    """
    spec = resolve_runtime(
        runtime,
        agent_name=agent_name,
        use_memgui_prompt=use_memgui_prompt,
        use_memory_prompt=use_memory_prompt,
        memgui_prompt_format=memgui_prompt_format,
    )
    name = spec.agent_name
    llm = infer.Gpt4Wrapper("gpt-4o")
    keep_n = max(1, int(last_n))
    common: dict[str, Any] = {
        "model_base_url": model_base_url,
        "model_api_key": model_api_key,
        "model_name": model_name,
        "extra_headers": extra_headers,
        "wait_after_action_seconds": wait_after_action_seconds,
        "last_n": keep_n,
    }

    if name == "human_agent":
        agent: base_agent.EnvironmentInteractingAgent = human_agent.HumanAgent(env)
    elif name == "random_agent":
        agent = random_agent.RandomAgent(env)
    elif name == "qwen3vl":
        agent = seeact_v.Qwen3VL(
            env,
            llm,
            use_memory_prompt=spec.use_memory_prompt,
            use_memgui_prompt=spec.use_memgui_prompt,
            memgui_prompt_format=spec.memgui_prompt_format,
            **common,
        )
    elif name == "qwen35vl":
        agent = seeact_v.Qwen35VL(
            env,
            llm,
            use_memory_prompt=spec.use_memory_prompt,
            use_memgui_prompt=spec.use_memgui_prompt,
            memgui_prompt_format=spec.memgui_prompt_format,
            qwen35_tool_call_mode=qwen35_tool_call_mode,
            **common,
        )
    elif name == "qwen35_session":
        agent = Qwen35Session(
            env,
            llm,
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
            qwen35_tool_call_mode=qwen35_tool_call_mode,
            **common,
        )
    elif name == "qwen35_switching_session":
        agent = Qwen35SwitchingSession(
            env,
            llm,
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
            qwen35_tool_call_mode=qwen35_tool_call_mode,
            weak_model_base_url=weak_model_base_url or "http://<weak-openai-compatible-host>/v1",
            weak_model_api_key=weak_model_api_key,
            weak_model_name=weak_model_name,
            **common,
        )
    elif name == "qwen35_thought_session":
        agent = Qwen35ThoughtSession(
            env,
            llm,
            reasoning_effort=reasoning_effort,
            **common,
        )
    elif name == "qwen35_osworld_session":
        agent = Qwen35OsworldSession(
            env,
            llm,
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
            history_n=history_n,
            image_max=image_max,
            fold_size=fold_size,
            **common,
        )
    elif name == "qwen3vl_session":
        agent = Qwen3VLSession(
            env,
            llm,
            enable_thinking=enable_thinking,
            require_think_tags=require_think_tags,
            reasoning_effort=reasoning_effort,
            **common,
        )
    elif name == "qwen25vl":
        agent = seeact_v.Qwen25VL(env, llm, **common)
    elif name == "qwen3vl_switching":
        agent = seeact_v.Qwen3VL_Switching(
            env,
            llm,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=model_name,
            weak_model_base_url=weak_model_base_url or "http://<weak-openai-compatible-host>/v1",
            weak_model_api_key=weak_model_api_key,
            weak_model_name=weak_model_name,
            wait_after_action_seconds=wait_after_action_seconds,
            extra_headers=extra_headers,
            monitor_num_recent_screens=keep_n,
        )
    elif name == "venus":
        agent = venus_agent.VenusAgent(env, llm, **common)
    else:
        raise ValueError(f"Unknown agent for runtime {spec.name!r}: {name!r}")

    agent.name = name
    return agent, spec
