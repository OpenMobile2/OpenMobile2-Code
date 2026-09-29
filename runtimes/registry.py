"""Shared GUI-agent runtime presets for MobileGym / MobileWorld / AndroidWorld.

Use ``--runtime <name>`` to select a preset. Legacy flags
(``--agent_name``, ``--use_memgui_prompt``, ``--use_memory_prompt``) still work
when ``--runtime`` is empty.

``--last_n`` (default 1) is orthogonal to the preset: how many recent screenshots
the model can see.

Runtime overview
----------------
| name               | Agent class      | Prompt / dialogue style                          |
|--------------------|------------------|--------------------------------------------------|
| qwen3vl            | Qwen3VL          | Thought/Action + JSON ``<tool_call>``; history in user text |
| qwen3vl_memgui     | Qwen3VL          | MemGUI + JSON tool_call                          |
| qwen3vl_memory     | Qwen3VL          | Thought/Action/Memory + JSON tool_call           |
| qwen3vl_session    | Qwen3VLSession   | Multi-turn session; model backend auto (see ``session/``) |
| qwen35vl           | Qwen35VL         | Thought/Action + XML tool_call; history in user text |
| qwen35vl_memgui    | Qwen35VL         | MemGUI + XML tool_call                           |
| qwen35vl_memory    | Qwen35VL         | Thought/Action/Memory + XML tool_call            |
| qwen35_session     | Qwen35Session    | Multi-turn session; model backend auto (see ``session/``) |
| qwen35_thought_session | Qwen35ThoughtSession | ``Thought:`` + XML; no native thinking (training sample) |
| qwen35_osworld_session | Qwen35OsworldSession | OSWorld-style Action + XML; fixed qwen_osworld backend |
| qwen35_switching_session | Qwen35SwitchingSession | Weak session + strong monitor/intervene; ``session_io`` |
| qwen25vl           | Qwen25VL         | Qwen2.5-VL action tags                           |
| qwen3vl_switching  | Qwen3VL_Switching| Weak-first + monitor intervention                |
| venus              | VenusAgent       | UI-Venus-1.5: ``<think>/<action>/<conclusion>``; current image only |

``last_n`` semantics
--------------------
- Flat runtimes (``qwen3vl`` / ``qwen35vl`` / ``qwen25vl``): pack the last N
  screenshots into the current user message.
- ``venus``: ignores ``last_n``; always current screenshot + text previous_actions
  (official HISTORY_N=1).
- Session runtimes (``qwen3vl_session`` / ``qwen35_session`` /
  ``qwen35_thought_session``): keep images only
  in the last N image-bearing messages (assistant text history always kept;
  older screenshot slots are replaced with collapse placeholder text).
- ``qwen35_osworld_session``: ignores ``last_n`` for API retention; uses
  ``history_n`` window + ``Previous actions``, and ``image_max``/``fold_size``
  collapse placeholders inside ``<tool_response>``.

Session model backends (``android_world/agents/session/backends/``)
-------------------------------------------------------------------
| backend | model name match   | API shape                                        |
|---------|--------------------|--------------------------------------------------|
| qwen    | default            | ``enable_thinking`` extra_body; tool screenshots |
| qwen_thought | ``qwen35_thought_session`` | ``Thought:`` in content; ``enable_thinking=false`` |
| gemini      | ``gemini``         | ``google.thinking_config`` → ``reasoning_content``; image obs |
| kimi_k3     | ``kimi-k3``        | native ``tool_calls`` + ``reasoning_content``    |
| glm         | ``glm`` / ``z-ai`` | ``thinking.type=enabled`` + ``reasoning_effort`` |
| deepseek_v4 | ``deepseek``       | vision + native ``tool_calls``; must replay ``reasoning_content`` |

``--reasoning_effort`` (``low`` / ``high`` / ``max``, default ``max``):
- Kimi K3: API ``reasoning_effort``
- GLM / Z.AI: API ``reasoning_effort`` (thinking cannot be disabled on GLM-5.3)
- DeepSeek-V4: API ``reasoning_effort`` (``low`` maps to ``high``)
- Gemini: maps to ``thinking_level`` (``low`` / ``high``) under
  ``extra_body.extra_body.google.thinking_config``
Qwen / DeepSeek session also use ``--enable_thinking``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSpec:
    """Resolved runtime used to construct an eval/rollout agent."""

    name: str
    agent_name: str
    use_memgui_prompt: bool = False
    use_memory_prompt: bool = False
    memgui_prompt_format: str = "auto"
    description: str = ""


RUNTIME_PRESETS: dict[str, RuntimeSpec] = {
    "qwen3vl": RuntimeSpec(
        name="qwen3vl",
        agent_name="qwen3vl",
        description="Qwen3-VL: Thought/Action + JSON tool_call; history text in user prompt.",
    ),
    "qwen3vl_memgui": RuntimeSpec(
        name="qwen3vl_memgui",
        agent_name="qwen3vl",
        use_memgui_prompt=True,
        memgui_prompt_format="qwen3vl",
        description="Qwen3-VL + MemGUI (task progress + UI memory actions).",
    ),
    "qwen3vl_memory": RuntimeSpec(
        name="qwen3vl_memory",
        agent_name="qwen3vl",
        use_memory_prompt=True,
        description="Qwen3-VL + simple Memory line in the response.",
    ),
    "qwen3vl_session": RuntimeSpec(
        name="qwen3vl_session",
        agent_name="qwen3vl_session",
        description=(
            "Qwen3-VL multi-turn session: "
            "<think> + JSON tool_call; query once; screenshots as tool messages."
        ),
    ),
    "qwen35vl": RuntimeSpec(
        name="qwen35vl",
        agent_name="qwen35vl",
        description="Qwen3.5-VL: Thought/Action + XML tool_call; history text in user prompt.",
    ),
    "qwen35vl_memgui": RuntimeSpec(
        name="qwen35vl_memgui",
        agent_name="qwen35vl",
        use_memgui_prompt=True,
        memgui_prompt_format="qwen35",
        description="Qwen3.5-VL + MemGUI (XML tool_call dialect).",
    ),
    "qwen35vl_memory": RuntimeSpec(
        name="qwen35vl_memory",
        agent_name="qwen35vl",
        use_memory_prompt=True,
        description="Qwen3.5-VL + simple Memory line in the response.",
    ),
    "qwen35_session": RuntimeSpec(
        name="qwen35_session",
        agent_name="qwen35_session",
        description=(
            "Qwen3.5 multi-turn session (qwen35_session.md): "
            "<think> + XML tool_call; query once; screenshots as tool messages. "
            "Model backend auto-selects (qwen / gemini / kimi_k3 / deepseek_v4)."
        ),
    ),
    "deepseek_v4_session": RuntimeSpec(
        name="deepseek_v4_session",
        agent_name="qwen35_session",
        description=(
            "DeepSeek-V4 vision session (same loop as qwen35_session): "
            "native tools= + reasoning_content replay. "
            "Set --model_name deepseek-v4-flash-vision-exp."
        ),
    ),
    "qwen35_thought_session": RuntimeSpec(
        name="qwen35_thought_session",
        agent_name="qwen35_thought_session",
        description=(
            "Qwen3.5 session with Thought: in assistant text (no native "
            "thinking / no <think>): XML tools in the system prompt, then "
            "one <tool_call>. Matches the Thought: training sample."
        ),
    ),
    "qwen35_osworld_session": RuntimeSpec(
        name="qwen35_osworld_session",
        agent_name="qwen35_osworld_session",
        description=(
            "Qwen3.5 OSWorld-style session: Action + XML mobile_use; "
            "history_n window + Previous actions; image_max/fold collapse "
            "inside <tool_response>."
        ),
    ),
    "qwen35_switching_session": RuntimeSpec(
        name="qwen35_switching_session",
        agent_name="qwen35_switching_session",
        description=(
            "Session policy-switching: weak multi-turn actor + strong monitor/"
            "intervene; dumps session_io with policy_source."
        ),
    ),
    "qwen25vl": RuntimeSpec(
        name="qwen25vl",
        agent_name="qwen25vl",
        description="Qwen2.5-VL native action-tag format.",
    ),
    "qwen3vl_switching": RuntimeSpec(
        name="qwen3vl_switching",
        agent_name="qwen3vl_switching",
        description="Weak-first Qwen3-VL with monitor-triggered strong intervention.",
    ),
    "venus": RuntimeSpec(
        name="venus",
        agent_name="venus",
        description=(
            "UI-Venus-1.5: <think>/<action>/<conclusion>; "
            "current screenshot only + text previous_actions (HISTORY_N=1)."
        ),
    ),
}

RUNTIME_CHOICES: tuple[str, ...] = tuple(RUNTIME_PRESETS.keys())


def list_runtimes() -> str:
    """Human-readable runtime table for CLI help / errors."""
    lines = []
    for name, spec in RUNTIME_PRESETS.items():
        lines.append(f"  {name:20s}  {spec.description}")
    return "\n".join(lines)


def resolve_runtime(
    runtime: str | None = None,
    *,
    agent_name: str = "qwen3vl",
    use_memgui_prompt: bool = False,
    use_memory_prompt: bool = False,
    memgui_prompt_format: str = "auto",
) -> RuntimeSpec:
    """Resolve ``--runtime`` or fall back to legacy agent/prompt flags."""
    key = (runtime or "").strip()
    if key:
        if key not in RUNTIME_PRESETS:
            raise ValueError(
                f"Unknown runtime {key!r}. Choose one of:\n{list_runtimes()}"
            )
        return RUNTIME_PRESETS[key]

    # Legacy path: derive a synthetic name from flags for logging.
    if use_memgui_prompt and use_memory_prompt:
        raise ValueError("use_memgui_prompt and use_memory_prompt are mutually exclusive")

    name = agent_name or "qwen3vl"
    if use_memgui_prompt:
        name = f"{agent_name}_memgui"
    elif use_memory_prompt:
        name = f"{agent_name}_memory"

    return RuntimeSpec(
        name=name,
        agent_name=agent_name or "qwen3vl",
        use_memgui_prompt=bool(use_memgui_prompt),
        use_memory_prompt=bool(use_memory_prompt),
        memgui_prompt_format=memgui_prompt_format or "auto",
        description="Resolved from legacy --agent_name / prompt flags.",
    )
