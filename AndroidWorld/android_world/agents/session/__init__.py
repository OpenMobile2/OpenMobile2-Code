"""Multi-turn session runtimes (``qwen35_session`` / ``qwen3vl_session`` /
``qwen35_osworld_session`` / ``qwen35_thought_session``).

Model-specific API behaviour lives under ``session.backends``:

| backend      | models              | dialogue style                                      |
|--------------|---------------------|-----------------------------------------------------|
| qwen         | Qwen / vLLM default | tool screenshots + ``enable_thinking`` extra_body   |
| qwen_thought | forced by runtime   | ``Thought:`` + XML; no native thinking              |
| qwen_osworld | forced by runtime   | OSWorld-style Action + XML; no think retry          |
| gemini       | ``gemini-*``        | text ``<thinking>`` (no native thought API)         |
| kimi_k3      | ``kimi-k3``         | native tool_calls + ``reasoning_content`` replay    |
| glm          | ``glm`` / ``z-ai``  | native tool_calls + ``thinking.type`` / ``reasoning_effort`` |
| deepseek_v4  | ``deepseek*``       | vision + native tool_calls + ``reasoning_content``  |
"""

from android_world.agents.session.agent import (
    Qwen35OsworldSession,
    Qwen35Session,
    Qwen35ThoughtSession,
    Qwen3VLSession,
)
from android_world.agents.session.backends import resolve_session_backend
from android_world.agents.session.switching import Qwen35SwitchingSession

__all__ = [
    "Qwen35Session",
    "Qwen3VLSession",
    "Qwen35OsworldSession",
    "Qwen35ThoughtSession",
    "Qwen35SwitchingSession",
    "resolve_session_backend",
]
