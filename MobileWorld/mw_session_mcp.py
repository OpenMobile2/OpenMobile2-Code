"""Attach MobileWorld MCP servers to qwen35_session ``tools=``.

Official MW eval uses MCPAgent + AndroidMCPEnvClient. OpenMobile session
eval only sent ``mobile_use`` even with ``--enable_mcp``. This bridge:

1. loads DashScope / ModelScope MCP tools on the eval host
2. injects them into ``session.set_turn_tools`` (with ``mobile_use``)
3. executes non-``mobile_use`` calls and feeds JSON back as the next
   tool-role observation
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

from android_world.agents.PROMPT import QWEN35_TOOLS

MCP_SESSION_ADDENDUM = """
# Extra MCP tools

This turn's tools list is `mobile_use` plus MCP functions for this task
(maps, GitHub, arXiv, web extract, stocks). Only call a function that appears
in this turn's tools list. After a non-`mobile_use` call, the next tool message
contains the JSON result and a new screenshot. Then continue with GUI actions
or another MCP call as needed.

For tool arguments: use JSON types that match the schema (numbers as bare
numbers like 42, not strings like "42"; booleans as true/false).
"""


def _schema_type(schema: dict[str, Any] | None) -> str | None:
    if not isinstance(schema, dict):
        return None
    t = schema.get("type")
    if isinstance(t, list):
        for item in t:
            if item != "null":
                return str(item)
        return None
    return str(t) if t is not None else None


def _coerce_value(value: Any, schema: dict[str, Any] | None) -> Any:
    """Best-effort cast of LLM tool args to JSON-schema types (e.g. \"12\" → 12)."""
    if schema is None or value is None:
        return value
    t = _schema_type(schema)
    if t == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value.strip()))
            except ValueError:
                return value
        return value
    if t == "number":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str) and value.strip():
            try:
                num = float(value.strip())
                return int(num) if num.is_integer() else num
            except ValueError:
                return value
        return value
    if t == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes"}:
                return True
            if low in {"false", "0", "no"}:
                return False
        return value
    if t == "string":
        if isinstance(value, (dict, list)):
            return value
        return str(value) if not isinstance(value, str) else value
    if t == "array" and isinstance(value, list):
        item_schema = schema.get("items") if isinstance(schema, dict) else None
        if isinstance(item_schema, dict):
            return [_coerce_value(v, item_schema) for v in value]
        return value
    if t == "object" and isinstance(value, dict):
        return _coerce_args(value, schema)
    return value


def _coerce_args(arguments: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}
    if not isinstance(schema, dict):
        return dict(arguments)
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return dict(arguments)
    out = dict(arguments)
    for key, raw in list(out.items()):
        prop_schema = props.get(key)
        if isinstance(prop_schema, dict):
            out[key] = _coerce_value(raw, prop_schema)
    return out


_DEFAULT_MW_SRC = Path(__file__).resolve().parent.parent.parent / "MobileWorld" / "src"


def _fill_array_items(node: Any, key: str | None = None) -> Any:
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


def _mobile_use_tool() -> dict[str, Any]:
    return _fill_array_items(copy.deepcopy(QWEN35_TOOLS[0]))


def _mcp_to_openai(tool: dict[str, Any]) -> dict[str, Any] | None:
    name = str(tool.get("name") or "").strip()
    if not name:
        return None
    params = tool.get("inputSchema") or tool.get("parameters") or {
        "type": "object",
        "properties": {},
    }
    return _fill_array_items(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or name),
                "parameters": params,
            },
        }
    )


def _ensure_mobileworld_on_path(mw_src: Path | None = None) -> Path:
    src = (mw_src or _DEFAULT_MW_SRC).expanduser().resolve()
    if not src.is_dir():
        raise RuntimeError(
            f"MobileWorld src not found at {src}. "
            "Install/clone MobileWorld so eval can import mobile_world.runtime.mcp_server."
        )
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)
    return src


def _preload_dotenv(mw_src: Path) -> None:
    """Load MobileWorld ``.env`` before ``mcp_server`` bakes API keys into URLs."""
    candidates = [
        mw_src.parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            print(f"[mcp] loaded env from {path}")
            return


# Official https://mcp.jina.ai/v1 tool names do not contain the "jina_" prefix
# used by the ModelScope-hosted PsychArch server.
JINA_OFFICIAL_TOOLS = frozenset(
    {
        "primer",
        "read_url",
        "capture_screenshot_url",
        "guess_datetime_url",
        "search_web",
        "search_web_deep",
        "search_arxiv",
        "search_ssrn",
        "search_images",
        "search_jina_blog",
        "search_bibtex",
        "expand_query",
        "parallel_read_url",
        "parallel_search_web",
        "parallel_search_arxiv",
        "parallel_search_ssrn",
        "sort_by_relevance",
        "classify_text",
        "deduplicate_strings",
        "deduplicate_images",
        "extract_pdf",
        "show_api_key",
    }
)


def tool_belongs_to_amap(name: str) -> bool:
    low = str(name or "").lower()
    return low.startswith("amap_") or low.startswith("maps_")


def tool_belongs_to_jina(name: str) -> bool:
    low = str(name or "").lower()
    return low.startswith("jina_") or low in {t.lower() for t in JINA_OFFICIAL_TOOLS}


def _refresh_mcp_auth(mcp_mod: Any) -> None:
    dash = os.getenv("DASHSCOPE_API_KEY") or ""
    ms = os.getenv("MODELSCOPE_API_KEY") or ""
    jina = (os.getenv("JINA_API_KEY") or "").strip().strip('"').strip("'")
    amap_key = (os.getenv("AMAP_MAPS_API_KEY") or "").strip().strip('"').strip("'")
    mcp_mod.DASHSCOPE_API_KEY = dash
    mcp_mod.MODELSCOPE_API_KEY = ms
    mcp_mod.JINA_API_KEY = jina or None
    mcp_mod.AMAP_MAPS_API_KEY = amap_key or None
    servers = (mcp_mod.MCP_CONFIG or {}).get("mcpServers") or {}
    url_env = {
        "gitHub": "MCP_GITHUB_URL",
        "jina": "MCP_JINA_URL",
        "arXiv": "MCP_ARXIV_URL",
        "amap": "MCP_AMAP_URL",
    }
    for name, server in servers.items():
        headers = server.setdefault("headers", {})
        if name == "amap":
            if amap_key or os.getenv("MCP_AMAP_URL"):
                headers.pop("Authorization", None)
                server["transport"] = "http"
                server["url"] = os.getenv("MCP_AMAP_URL") or (
                    f"https://mcp.amap.com/mcp?key={amap_key}"
                )
            else:
                headers["Authorization"] = f"Bearer {dash}"
        elif name == "stockstar":
            headers["Authorization"] = f"Bearer {dash}"
        elif name == "jina":
            if jina:
                headers["Authorization"] = f"Bearer {jina}"
            else:
                headers.pop("Authorization", None)
        else:
            headers["Authorization"] = f"Bearer {ms}"
        env_key = url_env.get(name)
        if env_key and os.getenv(env_key) and name != "amap":
            server["url"] = os.getenv(env_key)
        elif name == "amap" and os.getenv("MCP_AMAP_URL"):
            server["url"] = os.getenv("MCP_AMAP_URL")
    missing = []
    if not dash:
        missing.append("DASHSCOPE_API_KEY")
    if not ms:
        missing.append("MODELSCOPE_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing " + ", ".join(missing) + ". "
            "Put them in <MOBILEWORLD_ROOT>\\.env "
            "(see MobileWorld/docs/mcp_setup.md)."
        )


class MobileWorldMcpBridge:
    def __init__(self, mw_src: Path | None = None):
        src = _ensure_mobileworld_on_path(mw_src)
        _preload_dotenv(src)
        try:
            from mobile_world.runtime import mcp_server as mcp_mod
            from mobile_world.runtime.mcp_server import init_mcp_clients
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import MobileWorld MCP client "
                f"({type(exc).__name__}: {exc}). "
                "In the android_world conda env, install MobileWorld MCP deps: "
                "pip install fastmcp loguru python-dotenv"
            ) from exc
        _refresh_mcp_auth(mcp_mod)
        self._client = init_mcp_clients()
        self._all_tools = list(self._client.list_tools_sync() or [])
        self._schemas: dict[str, dict[str, Any]] = {}
        for tool in self._all_tools:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            schema = tool.get("inputSchema") or tool.get("parameters") or {}
            if isinstance(schema, dict):
                self._schemas[name] = schema
        self._active = list(self._all_tools)
        if not self._all_tools:
            raise RuntimeError(
                "MCP list_tools returned empty. Check DASHSCOPE_API_KEY / "
                "MODELSCOPE_API_KEY and network access to DashScope / ModelScope."
            )
        print(
            f"[mcp] loaded {len(self._all_tools)} tools: "
            + ", ".join(self.tool_names)
        )

    @property
    def tool_names(self) -> list[str]:
        return [str(t.get("name")) for t in self._active if t.get("name")]

    def select_for_task(self, metadata: dict[str, Any] | None) -> list[str]:
        meta = metadata or {}
        tags = meta.get("tags") or []
        if "agent-mcp" not in tags:
            self._active = []
            return []
        filters: list[str] = []
        for app in meta.get("apps") or []:
            text = str(app)
            if "MCP" in text:
                filters.append(text.split("-")[-1])
        if not filters:
            self._active = list(self._all_tools)
        else:
            filtered = [
                tool
                for tool in self._all_tools
                if any(
                    (
                        f.lower() in str(tool.get("name") or "").lower()
                        or (f.lower() == "jina" and tool_belongs_to_jina(str(tool.get("name") or "")))
                        or (f.lower() == "amap" and tool_belongs_to_amap(str(tool.get("name") or "")))
                    )
                    for f in filters
                )
            ]
            if filtered:
                self._active = filtered
            else:
                # Official filter is a substring on tool name; some servers
                # (GitHub / Amap) do not put that token in every name.
                print(
                    f"[mcp] filter {filters} matched 0 tools; "
                    "falling back to the full MCP tool list"
                )
                self._active = list(self._all_tools)
        names = self.tool_names
        print(f"[mcp] task tools ({len(names)}): {names}")
        return names

    def openai_tools(self) -> list[dict[str, Any]]:
        tools = [_mobile_use_tool()]
        seen = {"mobile_use"}
        for raw in self._active:
            converted = _mcp_to_openai(raw)
            if not converted:
                continue
            name = converted["function"]["name"]
            if name in seen:
                continue
            seen.add(name)
            tools.append(converted)
        return tools

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.tool_names:
            return {
                "ok": False,
                "error": f"{name} is not in this task's MCP tool list",
                "available": self.tool_names,
            }
        raw_args = arguments if isinstance(arguments, dict) else {}
        coerced = _coerce_args(raw_args, self._schemas.get(name))
        if coerced != raw_args:
            print(f"[mcp] coerced args for {name}: {raw_args} -> {coerced}")
        try:
            result = self._client.call_tool_sync(name, coerced)
            return {"ok": True, "name": name, "result": result}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"ok": False, "name": name, "error": repr(exc)}


class SessionMcpWrapper:
    """Keep session ``tools=`` after reset, and execute MCP tool calls."""

    def __init__(self, agent: Any, bridge: MobileWorldMcpBridge):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_bridge", bridge)
        prompt = str(getattr(agent, "_session_system_prompt", "") or "")
        if "# Extra MCP tools" not in prompt:
            agent._session_system_prompt = prompt.rstrip() + "\n" + MCP_SESSION_ADDENDUM

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_agent", "_bridge"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._agent, name, value)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        result = self._agent.reset(*args, **kwargs)
        self._agent.set_turn_tools(self._bridge.openai_tools())
        return result

    def step(self, instruction: str) -> Any:
        result = self._agent.step(instruction)
        data = result.data or {}
        parsed = data.get("parsed") or {}
        if parsed.get("action_type") != "mcp_tool":
            return result
        name = str(parsed.get("tool_name") or "")
        arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
        payload = self._bridge.call(name, arguments)
        text = json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > 8000:
            text = text[:8000] + "...[truncated]"
        self._agent.set_next_observation_text(text)
        print(f"[mcp] {name} ok={payload.get('ok')}")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)
