"""Stable session ``tools=[mobile_use]`` plus App Tools in the switch-turn user text.

Official qwen35_session only sends ``QWEN35_TOOLS`` (``mobile_use``). Hybrid keeps
that ``tools=`` list unchanged for the whole episode. When the foreground app
changes, the new App Tool schemas are written into that turn's user/observation
text instead of mutating ``tools=`` (which would rewrite the system prefix).
"""

from __future__ import annotations

import copy
import json
from typing import Any

from android_world.agents.PROMPT import QWEN35_TOOLS

HYBRID_SESSION_ADDENDUM = """
# Extra app tools

The API `tools=` list is only `mobile_use` for the entire session; it does not change
when you open another app.
When the foreground app changes, that turn's user message lists extra tools for the
new app inside `<tools></tools>`. Those extra tools stay valid only while that app
remains in the foreground. A later user message with a new app's tools replaces them.
Never invent a tool from another app. On the launcher / recents / control center,
only `mobile_use` is available — open the app first (`mobile_use` action `open_app`
with its name or id, or click the icon).
Call extra app tools with the same `<tool_call>` XML as `mobile_use`.
After a non-`mobile_use` call, the next user/tool message contains a JSON result and
a new screenshot.
"""

THOUGHT_HYBRID_ADDENDUM = """
# Extra app tools

Do not use an API `tools=` list. `mobile_use` is described in the system prompt.
When the foreground app changes, that turn's user message lists extra tools for the
new app inside `<tools></tools>`. Those extra tools stay valid only while that app
remains in the foreground. A later user message with a new app's tools replaces them.
Never invent a tool from another app. On the launcher / recents / control center,
only `mobile_use` is available — open the app first (`mobile_use` action `open_app`
with its name or id, or click the icon).
Call extra app tools with the same `<tool_call>` XML as `mobile_use`.
After a non-`mobile_use` call, the next user/tool message contains a JSON result and
a new screenshot.
"""

SECRETARY_EXTRA_POLICY = """
- Prefer mobile_use action open_app with the launcher label exactly: 腾讯会议, 携程旅行, 微信, 卡皮记账. Do not swipe launcher pages to hunt icons.
- Cross-app to WeChat must use the in-app 分享 button (system share), then pick 张伟. Do not use wechat__send_text_message or wechat__open_chat to skip the share sheet.
- After a share, the WeChat picker or sent result must be on the next screenshot.
- If the task names 卡皮记账: switch to 公司账本, prepare the housing and transport drafts, tap 完成, then open 分享 and 发到微信, pick 张伟. Do not stop after saving the poster to the album. Write the meeting materials in the share caption.
- Booking pages: fill the form and stop before pay.
- If the task is a Ctrip hotel: search 杭州 / 西湖国际会议中心, pick a hotel that stays within the 住房 budget, enter that booking page. Also open a Shanghai-Hangzhou train booking page. Do not pay.
"""

GLM_SESSION_ADDENDUM = """
# Extra app tools

The API `tools=` list is `mobile_use` plus extra tools for the current foreground app.
When the foreground app changes, extra tools in `tools=` are replaced.
Call extra tools as native function calls (name like `ctrip__search_hotels`).
Never pass a tool name to `mobile_use` `open_app`. `open_app` only takes an app
display name or id (微信, 腾讯会议, ctrip).
On the launcher / recents / control center, only `mobile_use` is available.
If a GUI click misses twice, use an extra app tool instead of re-deriving coordinates.
"""


def _fill_array_items(node: Any, key: str | None = None) -> Any:
    """Gemini function_declarations reject ``type: array`` without ``items``."""
    if isinstance(node, dict):
        if node.get("type") == "array" and "items" not in node:
            node["items"] = (
                {"type": "number"}
                if key in {"coordinate", "coordinate2"}
                else {"type": "string"}
            )
        for child_key, value in node.items():
            _fill_array_items(value, child_key if isinstance(child_key, str) else key)
    elif isinstance(node, list):
        for value in node:
            _fill_array_items(value, key)
    return node


def mobile_use_tool(*, include_open_app: bool = True, glm: bool = False) -> dict[str, Any]:
    from android_world.agents.PROMPT import (
        GLM_COORDINATE_DESCRIPTION,
        GLM_MOBILE_USE_DESCRIPTION,
    )

    tool = _fill_array_items(copy.deepcopy(QWEN35_TOOLS[0]))
    if glm:
        tool["function"]["description"] = GLM_MOBILE_USE_DESCRIPTION
        props = tool["function"]["parameters"]["properties"]
        props["coordinate"]["description"] = GLM_COORDINATE_DESCRIPTION
        props["coordinate2"]["description"] = (
            "Normalized end [x, y] in the same 0–1000 space. Required only by swipe."
        )
    if not include_open_app:
        return tool
    action = tool["function"]["parameters"]["properties"]["action"]
    enum = list(action.get("enum") or [])
    if "open_app" not in enum:
        enum.append("open_app")
        action["enum"] = enum
        action["description"] = (
            str(action.get("description") or "")
            + "\n* `open_app`: Open an installed app. Put its display name or id in `text` (e.g. 美团 / meituan)."
        )
        text = tool["function"]["parameters"]["properties"]["text"]
        text["description"] = (
            "Required by `type`, `answer`, and `open_app` (app name or id)."
        )
    return tool


def app_tool_to_openai(definition: dict[str, Any]) -> dict[str, Any]:
    description = str(definition.get("description") or "")
    annotations = definition.get("annotations") or {}
    ui_effect = str(annotations.get("uiEffect") or "none")
    if ui_effect == "none":
        description += " This tool does not promise to change the current UI."
    else:
        description += (
            " When this call returns, its target UI is rendered. "
            "Inspect the next screenshot before continuing with GUI actions."
        )
    return {
        "type": "function",
        "function": {
            "name": str(definition["name"]),
            "description": description.strip(),
            "parameters": definition.get("inputSchema")
            or {"type": "object", "properties": {}},
        },
    }


def stable_session_tools(*, glm: bool = False) -> list[dict[str, Any]]:
    """Episode ``tools=`` payload: ``mobile_use`` (GLM uses 0–1000 coords, not 999x999)."""
    return [mobile_use_tool(glm=glm)]


def build_session_tools(app_definitions: list[dict[str, Any]], *, glm: bool = False) -> list[dict[str, Any]]:
    tools = stable_session_tools(glm=glm)
    seen = {"mobile_use"}
    for item in app_definitions:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        if name in seen:
            continue
        seen.add(name)
        tools.append(_fill_array_items(app_tool_to_openai(item)))
    return tools


def format_app_switch_user_text(
    app: str | None,
    definitions: list[dict[str, Any]],
) -> str:
    """User-message dump used only on the turn the foreground app changes."""
    label = str(app or "").strip() or "launcher"
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in definitions:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        if name in seen or name == "mobile_use":
            continue
        seen.add(name)
        rows.append(_fill_array_items(app_tool_to_openai(item)))
    if not rows:
        return (
            f"Foreground app changed to {label}. "
            "No extra app tools are available. Only call `mobile_use` until a later "
            "user message lists tools for a new app."
        )
    dumped = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    return (
        f"Foreground app changed to {label}. Extra app tools for this app are listed "
        "below. They are valid only while this app stays in the foreground. "
        "Do not call tools from a previous app. The API `tools=` list remains only "
        "`mobile_use`; call these extra tools with the same `<tool_call>` XML format.\n"
        "<tools>\n"
        f"{dumped}\n"
        "</tools>"
    )


def format_app_switch_user_text_native(
    app: str | None,
    definitions: list[dict[str, Any]],
) -> str:
    """Short reminder when App Tools live in ``tools=`` (GLM / native FC)."""
    label = str(app or "").strip() or "launcher"
    names: list[str] = []
    seen: set[str] = set()
    for item in definitions:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        if name in seen or name == "mobile_use":
            continue
        seen.add(name)
        names.append(name)
    if not names:
        return (
            f"Foreground app changed to {label}. "
            "No extra app tools are available. Only call `mobile_use`."
        )
    return (
        f"Foreground app changed to {label}. Extra tools now in the API tools= list: "
        f"{', '.join(names)}. Call them as native function calls. "
        "Do not pass these names to mobile_use open_app."
    )
