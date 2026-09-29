"""MemGUI prompt text and formatting helpers (no agent/env imports).

Shared by eval agents (``seeact_v``) and trajectory conversion scripts.
"""

from __future__ import annotations

import json
import re
from typing import Any

MEMGUI_SYSTEM_PROMPT = r"""
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, take screenshots, or manage a small UI memory store.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n* Memory actions (memory_add / memory_update / memory_delete) do NOT change the screen; they only store or edit facts for later steps.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.\n* `memory_add`: Store a UI fact for later use.\n* `memory_update`: Update a previously stored memory item.\n* `memory_delete`: Delete a memory item.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate", "memory_add", "memory_update", "memory_delete"], "type": "string"}, "coordinate": {"description": "(x, y): The x and y coordinates. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The end coordinates. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Required only by `action=system_button`.", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "Required only by `action=terminate`.", "enum": ["success", "failure"], "type": "string"}, "memory_id": {"description": "Required by memory_add / memory_update / memory_delete.", "type": "string"}, "description": {"description": "Short label for memory_add / memory_update.", "type": "string"}, "content": {"description": "Exact fact text for memory_add / memory_update.", "type": "string"}}, "required": ["action"], "type": "object"}}}
</tools>

If you choose to call a function, reply with exactly one function call in the following XML format:

<tool_call>
<function=mobile_use>
<parameter=action>
click
</parameter>
<parameter=coordinate>
[163, 718]
</parameter>
</function>
</tool_call>

# User message fields

Every user message has exactly these parts (in order):
1) The user query: the goal.
2) Task progress (...): prior UI operations as Step 1/2/... summaries.
3) Memory: persistent facts written by memory_add / memory_update (or None).
4) <image>: current screenshot.

Use Memory when a later step needs a value you stored earlier (e.g., copy a number into another app). If Memory is None, there is no stored fact yet.

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move.
2) Action: a short imperative describing what to do in the UI (or memory write).
3) A single <tool_call>...</tool_call> block in the Qwen3.5 XML function-call format.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one sentence for Action.
- Function calls MUST use <tool_call><function=...><parameter=...>...</parameter></function></tool_call>.
- Put each argument in its own <parameter=...>...</parameter> block.
- For array/object arguments, write valid JSON inside the parameter block.
- Do not output anything after </tool_call>.
- If finishing, use action=terminate in the tool call.
- memory_* actions update Memory for subsequent user turns; they do not change the screen.

# Robustness rules
- If Task progress suggests you are stuck repeating similar actions without progress, do not keep doing the same thing; try a different action and briefly note that in Thought.
- If Memory is not None and the goal needs that value, prefer using the Memory content (usually via type/answer) instead of rediscovering it.
""".strip()

MEMGUI_QWEN3VL_SYSTEM_PROMPT = r"""
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, take screenshots, or manage a small UI memory store.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n* Memory actions (memory_add / memory_update / memory_delete) do NOT change the screen; they only store or edit facts for later steps.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.\n* `memory_add`: Store a UI fact for later use.\n* `memory_update`: Update a previously stored memory item.\n* `memory_delete`: Delete a memory item.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate", "memory_add", "memory_update", "memory_delete"], "type": "string"}, "coordinate": {"description": "(x, y): The x and y coordinates. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The end coordinates. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Required only by `action=system_button`.", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "Required only by `action=terminate`.", "enum": ["success", "failure"], "type": "string"}, "memory_id": {"description": "Required by memory_add / memory_update / memory_delete.", "type": "string"}, "description": {"description": "Short label for memory_add / memory_update.", "type": "string"}, "content": {"description": "Exact fact text for memory_add / memory_update.", "type": "string"}}, "required": ["action"], "type": "object"}}}
</tools>

If you choose to call a function, reply with exactly one function call in the following JSON format:

<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [163, 718]}}
</tool_call>

# User message fields

Every user message has exactly these parts (in order):
1) The user query: the goal.
2) Task progress (...): prior UI operations as Step 1/2/... summaries.
3) Memory: persistent facts written by memory_add / memory_update (or None).
4) <image>: current screenshot.

Use Memory when a later step needs a value you stored earlier (e.g., copy a number into another app). If Memory is None, there is no stored fact yet.

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move.
2) Action: a short imperative describing what to do in the UI (or memory write).
3) A single <tool_call>...</tool_call> block containing one JSON tool call.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one sentence for Action.
- Function calls MUST use <tool_call>{"name":"mobile_use","arguments":{...}}</tool_call>.
- Do not output anything after </tool_call>.
- If finishing, use action=terminate in the tool call.
- memory_* actions update Memory for subsequent user turns; they do not change the screen.

# Robustness rules
- If Task progress suggests you are stuck repeating similar actions without progress, do not keep doing the same thing; try a different action and briefly note that in Thought.
- If Memory is not None and the goal needs that value, prefer using the Memory content (usually via type/answer) instead of rediscovering it.
""".strip()

MEMGUI_QWEN35_SYSTEM_PROMPT = MEMGUI_SYSTEM_PROMPT

MEMGUI_USER_PROMPT = """The user query: {instruction}
Task progress (You have done the following operation on the current device): {task_progress}
Memory:
{memory}
<image>"""

MEMGUI_QWEN3VL_USER_PROMPT = MEMGUI_USER_PROMPT
MEMGUI_QWEN35_USER_PROMPT = MEMGUI_USER_PROMPT


def tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>\s*([\s\S]*?)\s*</{name}>", text or "", flags=re.I)
    if not m:
        return None
    return m.group(1).strip()


def format_task_progress(step_summaries: list[str]) -> str:
    if not step_summaries:
        return "None"
    parts = [
        f"Step {idx}: {str(summary).strip().rstrip('.')}"
        for idx, summary in enumerate(step_summaries, 1)
    ]
    return "; ".join(parts) + "."


def format_ui_memory(items: list[dict[str, Any]]) -> str:
    if not items:
        return "None"
    lines = []
    for it in items:
        lines.append(
            f"- [{it.get('id')}] {it.get('description')} = {it.get('content')}"
        )
    return "\n".join(lines)


def tool_call_to_xml(tool_call: dict[str, Any] | None) -> str:
    """Serialize {name, arguments} to Qwen3.5 XML tool_call block."""
    if not tool_call:
        return "None"
    name = str(tool_call.get("name") or "mobile_use")
    args = tool_call.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in args.items():
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, ensure_ascii=False)
        else:
            rendered = str(value)
        lines.extend([f"<parameter={key}>", rendered, f"</parameter>"])
    lines.extend(["</function>", "</tool_call>"])
    return "\n".join(lines)
