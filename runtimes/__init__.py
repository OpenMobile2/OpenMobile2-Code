"""Model runtimes shared by AndroidWorld, MobileGym, MobileWorld, and MobileGym++.

Prompt text and runtime names live here. This package does not drive an
emulator, a browser, or an HTTP device. Each benchmark turns a parsed action
into its own environment step.

Eval runtimes
-------------
qwen3vl
    Flat ReAct. One system prompt and one user prompt. History is past
    Action lines inside the user prompt, plus the last N screenshots.
qwen35_thought_session
    Multi-turn session used by the evaluations. ``Thought:`` in the assistant
    text, then one XML ``<tool_call>``. Native thinking is off.
qwen35_session
    Older multi-turn session. Native thinking and API ``tools=`` stay available
    for rollout code. Evaluations use ``qwen35_thought_session``.
venus
    Current screenshot only. ``<think>`` / ``<action>`` / ``<conclusion>``.
"""

from runtimes.registry import (
    RUNTIME_CHOICES,
    RUNTIME_PRESETS,
    RuntimeSpec,
    list_runtimes,
    resolve_runtime,
)

__all__ = [
    "RUNTIME_CHOICES",
    "RUNTIME_PRESETS",
    "RuntimeSpec",
    "list_runtimes",
    "resolve_runtime",
]
