"""OSWorld-style history window + screenshot folding for session rebuild."""

from __future__ import annotations

import re
from typing import Any, Callable, List


COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."


def update_folding_state(
    total_screenshots: int,
    folded_prefix_k: int,
    image_max: int,
    fold_size: int,
) -> int:
    while (total_screenshots - folded_prefix_k) > image_max:
        folded_prefix_k += fold_size
    if folded_prefix_k > total_screenshots:
        folded_prefix_k = total_screenshots
    return folded_prefix_k


def should_collapse_step(step_num_1based: int, folded_prefix_k: int) -> bool:
    return step_num_1based <= folded_prefix_k


def previous_actions_text(actions: List[str], start_step: int) -> str:
    previous_actions = [
        f"Step {i + 1}: {actions[i]}"
        for i in range(0, min(start_step - 1, len(actions)))
    ]
    return "\n".join(previous_actions) if previous_actions else "None"


def wrap_tool_response(parts: List[dict[str, Any]]) -> List[dict[str, Any]]:
    return (
        [{"type": "text", "text": "<tool_response>\n"}]
        + parts
        + [{"type": "text", "text": "\n</tool_response>"}]
    )


def build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous_actions_str}"
    )


def extract_action_line(response: str) -> str:
    """Pull the ``Action:`` imperative from an assistant turn."""
    text = response or ""
    match = re.search(r"(?im)^\s*Action:\s*(.+?)\s*$", text)
    if match:
        return match.group(1).strip()
    # Fallback: first non-empty line before tool_call.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("<"):
            continue
        if stripped.lower().startswith("action:"):
            return stripped.split(":", 1)[1].strip()
        return stripped[:200]
    return ""


def ensure_empty_think_prefix(response: str) -> str:
    text = response or ""
    if re.match(r"^\s*<think>.*?</think>\s*", text, re.DOTALL):
        return text
    return "<think>\n\n</think>\n\n" + text.lstrip("\n")


def strip_think_tags(response: str) -> str:
    """Remove think wrappers but keep their inner text (e.g. ``Action:``).

    Qwen3.5 / ``normalize_session_assistant_content`` may wrap ``Action`` inside
    ``<think>...</think>``. Deleting the whole block would leave only
    ``<tool_call>`` and teach the model to drop Action on later turns.
    """
    text = response or ""
    # Unwrap: keep inner content.
    text = re.sub(
        r"<think\b[^>]*>(.*?)</think\s*>",
        r"\1",
        text,
        flags=re.I | re.DOTALL,
    )
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.I)
    # Collapse excess blank lines left after unwrap.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_osworld_assistant_content(response: str) -> str:
    """Keep ``Action`` + ``<tool_call>``; never re-wrap Action in ``<think>``."""
    return strip_think_tags(response or "")


def build_osworld_messages(
    *,
    system_prompt: str,
    instruction_prompt: str,
    screenshots: List[str],
    responses: List[str],
    start_step: int,
    total_steps: int,
    folded_prefix_k: int,
    collapse_text: str,
    response_transform: Callable[[str], str] | None = None,
) -> List[dict[str, Any]]:
    """Rebuild the chat window like OSWorld ``mm_agents.qwen.history.build_messages``.

    Observations after the first window turn are user messages wrapping content in
    ``<tool_response>...</tool_response>`` (image or collapse placeholder).
    """
    transform = response_transform or (lambda text: text)
    messages: List[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = should_collapse_step(step_num, folded_prefix_k)

        if is_collapsed:
            if is_first_turn:
                user_content: List[dict[str, Any]] = [
                    {"type": "text", "text": instruction_prompt}
                ]
            else:
                user_content = wrap_tool_response(
                    [{"type": "text", "text": collapse_text}]
                )
            messages.append({"role": "user", "content": user_content})
        else:
            raw = screenshots[step_num - 1]
            if raw.startswith("data:image/"):
                img_url = raw
            else:
                img_url = f"data:image/png;base64,{raw}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}]
                )
            messages.append({"role": "user", "content": user_content})

        if step_num <= total_steps - 1 and (step_num - 1) < len(responses):
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": transform(responses[step_num - 1]),
                        }
                    ],
                }
            )

    return messages
