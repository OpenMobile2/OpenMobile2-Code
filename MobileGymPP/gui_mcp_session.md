# GUI+MCP session (qwen35_session hybrid)

Source: converted 0825 training `result.json`
(`successful_mcp_episodes/*/result.json`).

- Runtime name: `qwen35_session`
- API `tools=`: always `[mobile_use]` (App Tools are **not** in this list)
- `last_n`: 3 (live eval); export/training keeps every attached `<image>`)
- Full system prompt text: [`gui_mcp_system_prompt.txt`](gui_mcp_system_prompt.txt)

## System prompt

```text
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `answer`: Output the answer.\n* `system_button`: Press the system button.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. The coordinates should be values from 0 to 999. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Session interaction

- The task query is provided only once in the first user message, together with the initial screenshot(s).
- After each tool call, the next user message contains only the resulting screenshot(s).
- All previous assistant responses remain available as the response history.
- On each turn, inspect the currently provided screenshot(s) and the complete response history, then choose exactly one next action.
- Do not invent screenshot placeholders; they are inserted by the caller.
- Continue until the task is complete or cannot be completed, then call terminate with the appropriate status.

# Response format

Response format for every assistant turn:
- A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly one <tool_call> per turn.
- Do not output anything after </tool_call>.
- If finishing, use action=terminate in the tool call.

# Visual context

- Only the most recent screenshot(s) are kept as images (controlled by last_n); older screenshot slots are replaced with "This screenshot has been collapsed.", but the complete textual response history is still retained.
- Always base the next action on the currently available screenshots and the response history.
- Do not assume access to screenshots that are not currently provided.

# Robustness rules

- If the response history suggests you are stuck repeating similar actions without progress, try a different action.
- Use the complete response history to avoid repeating actions that have already been performed.
- Base each action on the currently available screenshots; never assume that an earlier action succeeded without checking its result.

# Extra app tools

The API `tools=` list is only `mobile_use` for the entire session; it does not change
when you open another app.
When the foreground app changes, that turn's tool observation lists extra tools for
the new app inside `<tools></tools>`, together with the new screenshot. Those extra
tools stay valid only while that app remains in the foreground. A later observation
with a new app's tools replaces them.
Never invent a tool from another app. On the launcher / recents / control center,
only `mobile_use` is available — open the app first (`mobile_use` action `open_app`
with its name or id, or click the icon).
Call extra app tools with the same `<tool_call>` XML as `mobile_use`.
After a non-`mobile_use` call: if the tool does not change the UI, the next tool
message is JSON only (no new screenshot); if it changes the UI, the next tool
message contains the JSON result and a new screenshot.
```

GUI-only is the same prompt **without** `# Extra app tools`.

## Runtime record

```json
{
  "id": "hybrid_tools_0825_explore.HTV1Ht0825CalendarDental",
  "runtime": "qwen35_session",
  "backend": "gemini",
  "goal": "{instruction}",
  "messages": [
    { "role": "system", "content": "<system prompt above>" },
    { "role": "user", "content": "The user query: {instruction}\n<image>" },
    {
      "role": "assistant",
      "content": "<think>...</think>\n<tool_call>\n<function=mobile_use>\n<parameter=action>\nclick\n</parameter>\n...</function>\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "Foreground app changed to calendar. Extra app tools ...\n<tools>\n{...app tool schemas...}\n</tools>\n<image>"
    },
    {
      "role": "assistant",
      "content": "<think>...</think>\n<tool_call>\n<function=calendar__prepare_event_draft>\n...\n</function>\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"call_id\": \"tool_call_1\", \"name\": \"calendar__prepare_event_draft\", \"ok\": true, \"output\": {...}}\n<image>"
    },
    {
      "role": "assistant",
      "content": "<think>...</think>\n<tool_call>\n<function=mobile_use>\n<parameter=action>\nclick\n</parameter>\n...</function>\n</tool_call>"
    },
    { "role": "tool", "content": "<image>" },
    {
      "role": "assistant",
      "content": "<think>...</think>\n<tool_call>\n<function=mobile_use>\n<parameter=action>\nterminate\n</parameter>\n<parameter=status>\nsuccess\n</parameter>\n</function>\n</tool_call>"
    }
  ],
  "images": [
    "screenshot_step0.png",
    "screenshot_step1.png",
    "screenshot_step2.png",
    "screenshot_step3.png"
  ],
  "done": true
}
```

Tool observation packing:

| Previous call | `uiEffect` | Next `role=tool` content |
|---|---|---|
| `mobile_use` (GUI) | — | `<image>` ; if foreground app changed, prepend schema `<tools>` |
| App Tool | `none` | JSON only, no `<image>` |
| App Tool | `draft` / `navigation` | JSON + `\n<image>` |

`images` lists only filenames that correspond to `<image>` tokens, in order.
