"""Smoke-test MCP servers/tools used by MobileWorld ``--mcp_only`` eval.

Uses the same keys as eval (``<MOBILEWORLD_ROOT>\\.env`` via
``mw_session_mcp`` loading rules) and the same ``mobile_world.runtime.mcp_server``
client.

Examples:

  # Only check env + that expected tool names are listed (no API quota burn)
  python test_mcp_tools.py --list-only

  # One cheap live call per MCP server family (default)
  python test_mcp_tools.py

  # Every probe used by MobileWorld task helpers (burns more amap quota)
  python test_mcp_tools.py --full

  # Call EVERY tool returned by list_tools (auto args from schema; heavy)
  python test_mcp_tools.py --all-listed

  # Only amap / only github+jina
  python test_mcp_tools.py --servers amap
  python test_mcp_tools.py --servers github,jina,arxiv,stockstar
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mw_session_mcp import _DEFAULT_MW_SRC, tool_belongs_to_amap, tool_belongs_to_jina

_HERE = Path(__file__).resolve().parent

# Tools that MobileWorld task helpers actually call (see
# mobile_world.runtime.app_helpers.mcp).
# Names MobileWorld task helpers still call (app_helpers/mcp.py).
HELPER_EXPECTED_BY_SERVER: dict[str, list[str]] = {
    "amap": [
        "amap_maps_weather",
        "amap_maps_distance",
        "amap_maps_around_search",
        "amap_maps_direction_driving",
        "amap_maps_direction_bicycling",
        "amap_maps_direction_walking",
    ],
    "github": [
        "gitHub_list_issues",
        "gitHub_search_issues",
        "gitHub_list_commits",
        "gitHub_search_users",
        "gitHub_search_repositories",
    ],
    "jina": [],  # resolved from list_tools (prefix jina_)
    "stockstar": [
        "stockstar_stk_eval_filter_by_roe_3y",
        "stockstar_stk_eval_filter_by_div_rate",
        "stockstar_miotech_esg_rating",
    ],
    "arxiv": [
        "arXiv_search_papers",
    ],
}

# Live / currently advertised names (used for --list-only presence of "working" API).
LIVE_EXPECTED_BY_SERVER: dict[str, list[str]] = {
    "amap": HELPER_EXPECTED_BY_SERVER["amap"],
    "github": HELPER_EXPECTED_BY_SERVER["github"],
    "jina": [],
    "stockstar": HELPER_EXPECTED_BY_SERVER["stockstar"],
    "arxiv": [
        "arXiv_search_papers",
        "arXiv_read_paper",
        "arXiv_get_abstract",
    ],
}

# Old helper names → live replacements (kept for migration checks).
HELPER_ALIASES: dict[str, str] = {
    "arXiv_search_arxiv": "arXiv_search_papers",
    "arXiv_get_recent_ai_papers": "arXiv_search_papers",
}

# Hangzhou CBD-ish point used by several MW map tasks.
_HZ = "120.155070,30.274084"


@dataclass
class Probe:
    server: str
    name: str
    arguments: dict[str, Any]
    smoke: bool = False  # included in default (non --full) run
    note: str = ""


def _default_probes() -> list[Probe]:
    return [
        # amap — weather is cheapest; around_search is what hit OVER_LIMIT
        Probe("amap", "amap_maps_weather", {"city": "杭州"}, smoke=True),
        Probe(
            "amap",
            "amap_maps_around_search",
            {"location": _HZ, "radius": "500", "keywords": "药店"},
            smoke=False,
            note="burns daily quota",
        ),
        Probe(
            "amap",
            "amap_maps_distance",
            {"origins": _HZ, "destination": "120.219375,30.259244"},
            smoke=False,
        ),
        Probe(
            "amap",
            "amap_maps_direction_walking",
            {"origin": _HZ, "destination": "120.219375,30.259244"},
            smoke=False,
        ),
        Probe(
            "amap",
            "amap_maps_direction_driving",
            {"origin": _HZ, "destination": "120.219375,30.259244"},
            smoke=False,
        ),
        Probe(
            "amap",
            "amap_maps_direction_bicycling",
            {"origin": _HZ, "destination": "120.219375,30.259244"},
            smoke=False,
        ),
        # github
        Probe(
            "github",
            "gitHub_search_repositories",
            {"query": "android_world", "perPage": 3},
            smoke=True,
        ),
        Probe(
            "github",
            "gitHub_list_issues",
            {"owner": "google-research", "repo": "android_world", "state": "open"},
            smoke=False,
        ),
        Probe(
            "github",
            "gitHub_list_commits",
            {"owner": "google-research", "repo": "android_world", "page": 1},
            smoke=False,
        ),
        Probe(
            "github",
            "gitHub_search_users",
            {"q": "language:Python", "per_page": 3, "sort": "followers", "order": "desc"},
            smoke=False,
        ),
        Probe(
            "github",
            "gitHub_search_issues",
            {"q": "repo:google-research/android_world is:issue"},
            smoke=False,
        ),
        # arxiv (current ModelScope tool names)
        Probe(
            "arxiv",
            "arXiv_list_papers",
            {"compact": True},
            smoke=True,
            note="local list; closest to helper arXiv_get_recent_ai_papers",
        ),
        Probe(
            "arxiv",
            "arXiv_get_abstract",
            {"paper_id": "1706.03762"},
            smoke=False,
        ),
        Probe(
            "arxiv",
            "arXiv_search_papers",
            {"query": 'ti:"attention is all you need"', "max_results": 3},
            smoke=False,
            note="helper still calls arXiv_search_arxiv",
        ),
        # stockstar
        Probe(
            "stockstar",
            "stockstar_stk_eval_filter_by_roe_3y",
            {"filter_value": 15.0, "filter_type": 1},
            smoke=True,
        ),
        Probe(
            "stockstar",
            "stockstar_miotech_esg_rating",
            {"security_code": "600519"},
            smoke=False,
        ),
        Probe(
            "stockstar",
            "stockstar_stk_eval_filter_by_div_rate",
            {"filter_value": 3.0, "filter_type": 1},
            smoke=False,
        ),
    ]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    server: str = ""
    kind: str = "call"  # list | call | env


@dataclass
class Report:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> None:
        self.results.append(r)
        mark = "OK " if r.ok else "FAIL"
        prefix = f"[{mark}] {r.kind}"
        server = f" ({r.server})" if r.server else ""
        print(f"{prefix}{server}: {r.name} — {r.detail}")

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if not r.ok]


def _payload_text(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=False)[:800]
    except Exception:
        return repr(result)[:800]


def _looks_failed(result: Any) -> str | None:
    """Return error reason if the MCP payload looks like a failure."""
    text = _payload_text(result).lower()
    needles = (
        "user_daily_query_over_limit",
        "over_limit",
        "quota",
        "api 调用失败",
        "failed to call tool",
        "fetch failed",
        "unauthorized",
        "invalid api",
        "permission denied",
        "rate limit",
        "401",
        "403",
    )
    for n in needles:
        if n in text:
            return n
    if isinstance(result, dict):
        if result.get("is_error"):
            return "is_error=true"
        if isinstance(result.get("result"), str) and "failed" in result["result"].lower():
            return result["result"][:200]
    if result is None or result == [] or result == {}:
        return "empty result"
    return None


def _bootstrap_mcp(mw_src: Path):
    """Import MobileWorld MCP client the same way eval does."""
    from mw_session_mcp import _ensure_mobileworld_on_path, _preload_dotenv, _refresh_mcp_auth

    src = _ensure_mobileworld_on_path(mw_src)
    _preload_dotenv(src)
    from mobile_world.runtime import mcp_server as mcp_mod
    from mobile_world.runtime.mcp_server import SyncMCPClient

    _refresh_mcp_auth(mcp_mod)
    # Fail fast: eval client retries 5x with long backoff; probes should not.
    client = SyncMCPClient(config=mcp_mod.MCP_CONFIG, max_retries=1, retry_delay=1)
    return mcp_mod, client


def _server_for_tool(name: str) -> str:
    low = name.lower()
    if low.startswith("amap_") or tool_belongs_to_amap(name):
        return "amap"
    if low.startswith("github_"):
        return "github"
    if tool_belongs_to_jina(name):
        return "jina"
    if low.startswith("stockstar_"):
        return "stockstar"
    if low.startswith("arxiv_"):
        return "arxiv"
    return "other"


def _dummy_arg(prop_name: str, schema: dict[str, Any]) -> Any:
    """Build a safe-ish dummy value for schema-driven --all-listed probes."""
    t = schema.get("type")
    enum = schema.get("enum")
    if enum:
        return enum[0]
    name = prop_name.lower()
    if name in {"city", "city_name"}:
        return "杭州"
    if name in {"location", "origins", "origin", "destination"}:
        return _HZ
    if name in {"radius"}:
        return "500"
    if name in {"keywords", "query", "q"}:
        return "android_world"
    if name in {"owner"}:
        return "google-research"
    if name in {"repo"}:
        return "android_world"
    if name in {"url", "urls"}:
        return "https://arxiv.org/abs/1706.03762" if name == "url" else ["https://example.com"]
    if name in {"paper_id", "id"}:
        return "1706.03762"
    if name in {"security_code"}:
        return "600519"
    if name in {"state"}:
        return "open"
    if name in {"filter_type"}:
        return 1
    if name in {"filter_value"}:
        return 10.0
    if name in {"max_results", "count", "per_page", "perpage", "page", "limit"}:
        return 1
    if name in {"compact"}:
        return True
    if t == "boolean":
        return False
    if t == "integer" or t == "number":
        return 1
    if t == "array":
        return []
    if t == "object":
        return {}
    return "test"


def _args_from_schema(tool: dict[str, Any]) -> dict[str, Any]:
    schema = tool.get("inputSchema") or tool.get("parameters") or {}
    props = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    # Always fill required; also fill a few common optional keys used by MW.
    keys = list(dict.fromkeys(required + [k for k in props if k in {"query", "q", "url", "city"}]))
    args: dict[str, Any] = {}
    for key in keys:
        args[key] = _dummy_arg(key, props.get(key) or {})
    return args


def _probes_for_all_listed(tools: list[dict[str, Any]], wanted: set[str]) -> list[Probe]:
    probes: list[Probe] = []
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        server = _server_for_tool(name)
        if server not in wanted and "other" not in wanted:
            # Allow --servers other to include unknowns; default wanted skips other.
            if server == "other":
                continue
            continue
        probes.append(
            Probe(
                server=server,
                name=name,
                arguments=_args_from_schema(tool),
                smoke=True,
                note="auto from list_tools schema",
            )
        )
    return probes


def _pick_jina_probe(tool_names: list[str]) -> Probe | None:
    """Prefer a cheap reader call; skip show_api_key (prints the secret)."""
    by_lower = {n.lower(): n for n in tool_names if tool_belongs_to_jina(n)}
    for candidate, args, note in (
        ("jina_read_url", {"url": "https://example.com"}, "official mcp.jina.ai read_url"),
        ("read_url", {"url": "https://example.com"}, "official mcp.jina.ai read_url"),
        ("jina_search_web", {"query": "MobileWorld Android"}, "official mcp.jina.ai search_web"),
        ("search_web", {"query": "MobileWorld Android"}, "official mcp.jina.ai search_web"),
        ("jina_jina_search", {"query": "MobileWorld Android", "count": 3}, "modelscope jina search"),
        ("jina_jina_reader", {"url": "https://arxiv.org/abs/1706.03762"}, "modelscope jina reader"),
    ):
        if candidate in by_lower:
            return Probe("jina", by_lower[candidate], args, smoke=True, note=note)
    names = [
        n
        for n in tool_names
        if tool_belongs_to_jina(n) and "show_api_key" not in n.lower()
    ]
    if not names:
        return None
    first = names[0]
    args = {"url": "https://example.com"} if "url" in first.lower() or "read" in first.lower() else {"query": "test"}
    return Probe("jina", first, args, smoke=True, note="auto-picked first jina tool")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mw-src",
        type=Path,
        default=_DEFAULT_MW_SRC,
        help="Path to MobileWorld/src (for importing mobile_world).",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only verify env + tool listing (no live call_tool).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Call every known probe (more amap quota).",
    )
    parser.add_argument(
        "--all-listed",
        action="store_true",
        help="Call every tool from list_tools with schema-derived dummy args.",
    )
    parser.add_argument(
        "--servers",
        type=str,
        default="amap,github,jina,stockstar,arxiv",
        help="Comma-separated server families to check.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write machine-readable report.",
    )
    parser.add_argument(
        "--ignore-helper-compat",
        action="store_true",
        help="Do not fail when helper hard-coded names are missing from list_tools.",
    )
    args = parser.parse_args()

    wanted = {
        s.strip().lower()
        for s in args.servers.replace("，", ",").split(",")
        if s.strip()
    }
    report = Report()

    try:
        mcp_mod, client = _bootstrap_mcp(args.mw_src)
    except Exception as exc:
        print(f"[FAIL] env: bootstrap — {exc!r}", file=sys.stderr)
        return 2

    dash = bool(getattr(mcp_mod, "DASHSCOPE_API_KEY", None))
    ms = bool(getattr(mcp_mod, "MODELSCOPE_API_KEY", None))
    jina_key = bool(getattr(mcp_mod, "JINA_API_KEY", None))
    report.add(
        CheckResult(
            "DASHSCOPE_API_KEY",
            dash,
            "set" if dash else "MISSING (amap/stockstar)",
            kind="env",
        )
    )
    report.add(
        CheckResult(
            "MODELSCOPE_API_KEY",
            ms,
            "set" if ms else "MISSING (github/arxiv)",
            kind="env",
        )
    )
    report.add(
        CheckResult(
            "JINA_API_KEY",
            True,
            "set (search_web)" if jina_key else "unset (read_url still works; search_web needs it)",
            kind="env",
        )
    )
    if not dash or not ms:
        return 2

    print("\n=== list_tools ===")
    tools = client.list_tools_sync() or []
    names = [str(t.get("name") or "") for t in tools if t.get("name")]
    report.add(
        CheckResult(
            "list_tools",
            bool(names),
            f"{len(names)} tools" if names else "empty (check keys/network)",
            kind="list",
        )
    )
    if names:
        print("  " + ", ".join(names))

    # Presence: live API surface
    print("\n=== live tool names (API today) ===")
    for server, expected in LIVE_EXPECTED_BY_SERVER.items():
        if server not in wanted:
            continue
        if server == "jina":
            jina_names = [n for n in names if tool_belongs_to_jina(n)]
            ok = bool(jina_names)
            report.add(
                CheckResult(
                    "jina_tools",
                    ok,
                    (", ".join(jina_names) if ok else "no jina tools listed"),
                    server=server,
                    kind="list",
                )
            )
            continue
        for tool in expected:
            ok = tool in names
            report.add(
                CheckResult(
                    tool,
                    ok,
                    "listed" if ok else "NOT in list_tools",
                    server=server,
                    kind="list",
                )
            )

    # Presence: names helpers still hard-code (compat warning for eval)
    print("\n=== helper tool names (app_helpers/mcp.py) ===")
    for server, expected in HELPER_EXPECTED_BY_SERVER.items():
        if server not in wanted or not expected:
            continue
        for tool in expected:
            ok = tool in names
            alias = HELPER_ALIASES.get(tool)
            if ok:
                detail = "listed (helper OK)"
            elif alias and alias in names:
                detail = f"MISSING; helper will break — live alias present: {alias}"
                ok = False
            else:
                detail = "MISSING; helper will break"
            if not ok and args.ignore_helper_compat:
                print(
                    f"[WARN] list ({server}): helper:{tool} — {detail} "
                    "(ignored via --ignore-helper-compat)"
                )
                continue
            report.add(
                CheckResult(
                    f"helper:{tool}",
                    ok,
                    detail,
                    server=server,
                    kind="list",
                )
            )

    if args.list_only:
        return _finish(report, args.json_out)

    print("\n=== live call_tool probes ===")
    if args.all_listed:
        probes = _probes_for_all_listed(tools, wanted)
        print(f"  --all-listed: {len(probes)} tools")
    else:
        probes = [p for p in _default_probes() if p.server in wanted]
        if not args.full:
            probes = [p for p in probes if p.smoke]

        if "jina" in wanted:
            jina = _pick_jina_probe(names)
            if jina is None:
                report.add(
                    CheckResult(
                        "jina_probe",
                        False,
                        "no jina tool to call",
                        server="jina",
                        kind="call",
                    )
                )
            else:
                probes.append(jina)

    for probe in probes:
        if probe.name not in names and not tool_belongs_to_jina(probe.name):
            report.add(
                CheckResult(
                    probe.name,
                    False,
                    "skipped: not listed",
                    server=probe.server,
                    kind="call",
                )
            )
            continue
        try:
            result = client.call_tool_sync(probe.name, probe.arguments)
            err = _looks_failed(result)
            if err:
                report.add(
                    CheckResult(
                        probe.name,
                        False,
                        f"{err}; snippet={_payload_text(result)[:240]}",
                        server=probe.server,
                        kind="call",
                    )
                )
            else:
                extra = f" ({probe.note})" if probe.note else ""
                report.add(
                    CheckResult(
                        probe.name,
                        True,
                        f"ok args={probe.arguments}{extra}; "
                        f"snippet={_payload_text(result)[:160]}",
                        server=probe.server,
                        kind="call",
                    )
                )
        except Exception as exc:
            report.add(
                CheckResult(
                    probe.name,
                    False,
                    repr(exc),
                    server=probe.server,
                    kind="call",
                )
            )

    return _finish(report, args.json_out)


def _finish(report: Report, json_out: Path | None) -> int:
    failed = report.failed
    print("\n" + "=" * 64)
    print(
        f"SUMMARY: {len(report.results) - len(failed)}/{len(report.results)} checks passed, "
        f"{len(failed)} failed"
    )
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  - [{r.kind}] {r.server + ':' if r.server else ''}{r.name}: {r.detail}")
        quota = [r for r in failed if "over_limit" in r.detail.lower() or "quota" in r.detail.lower()]
        if quota:
            print(
                "\nHint: amap USER_DAILY_QUERY_OVER_LIMIT → rotate DASHSCOPE_API_KEY "
                "in <MOBILEWORLD_ROOT>\\.env or wait for daily reset, then re-run."
            )
    if json_out is not None:
        payload = [
            {
                "name": r.name,
                "ok": r.ok,
                "detail": r.detail,
                "server": r.server,
                "kind": r.kind,
            }
            for r in report.results
        ]
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {json_out}")
    print("=" * 64)
    return 0 if not failed else 1


if __name__ == "__main__":
    # Allow `python test_mcp_tools.py` from this folder to import mw_session_mcp.
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    raise SystemExit(main())
