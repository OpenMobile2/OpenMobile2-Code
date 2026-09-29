"""Roll hybrid GUI+App-Tools trajectories via mobilegym-mock hybrid_benchmark.

This is data collection, not official GUI-only eval (see ../MobileGym/).

Default runtime is qwen35_session: same message / tools= format as GUI-only
eval. ``tools=`` stays ``[mobile_use]``; new App Tools are written into the
user/observation text on the turn the foreground app changes.

``--runtime hybrid_xml`` is the older HybridXmlAgent path (rebuilds system
every turn, not session-shaped).

Examples:
  python run_rollout.py --list
  python run_rollout.py --list --app meituan
  python run_rollout.py --env-url http://127.0.0.1:4173 --limit 2 \\
      --model-base-url http://<openai-compatible-host>/v1 --model-name YOUR_MODEL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OPENMOBILE = _HERE.parent


def _default_mock_root() -> Path:
    env = (os.environ.get("MOBILEGYM_MOCK") or "").strip()
    candidates = [
        Path(env) if env else None,
        _OPENMOBILE.parent / "mobilegym++" / "mobilegym-mock" / "trial_apps" / "mobilegym",
        _OPENMOBILE.parent / "mobilegym-mock" / "trial_apps" / "mobilegym",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "bench_env" / "hybrid_benchmark").is_dir():
            return candidate
    return candidates[1]


_DEFAULT_MOCK = _default_mock_root()


def _ensure_session_imports() -> None:
    for path in (
        _HERE,
        _OPENMOBILE / "AndroidWorld",
        _OPENMOBILE / "MobileGym",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _ensure_mock_on_path(mock_root: Path) -> Path:
    root = mock_root.expanduser().resolve()
    if not (root / "bench_env" / "hybrid_benchmark").is_dir():
        raise SystemExit(
            f"mobilegym-mock bench_env not found under {root}. "
            "Set --mock-root or MOBILEGYM_MOCK to trial_apps/mobilegym "
            "(the mock repo that contains hybrid_tools_custom_v5)."
        )
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    os.chdir(root)
    return root


def _thinking_extra_body(enable_thinking: bool) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}


def _is_local_or_private_model_url(base_url: str) -> bool:
    """Loopback and RFC1918 hosts (internal vLLM) accept EMPTY; public gateways do not."""

    from urllib.parse import urlparse
    import ipaddress

    text = str(base_url or "").strip().lower()
    if any(token in text for token in ("127.0.0.1", "localhost", "0.0.0.0")):
        return True
    host = urlparse(text if "://" in text else f"http://{text}").hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private)


def _resolve_api_key(flag_value: str, *, base_url: str) -> str:
    key = str(flag_value or "").strip()
    if not key or key.upper() == "EMPTY":
        key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("MOBILEGYM_API_KEY")
            or ""
        ).strip()
    if not key or key.upper() == "EMPTY":
        if _is_local_or_private_model_url(base_url):
            return "EMPTY"
        raise SystemExit(
            "missing API key: pass --model-api-key or set OPENAI_API_KEY. "
            "Remote gateways reject EMPTY (401 auth_api_key_missing)."
        )
    return key


def _filter_task_ids(suite: str, app: str, explicit: list[str]) -> list[str]:
    from bench_env.task.registry import TaskRegistry

    registry = TaskRegistry()
    rows = []
    for name in registry.list_tasks(suite):
        task = registry.create_task(f"{suite}.{name}")
        spec = getattr(task, "spec", None)
        dataset_id = str(getattr(spec, "task_id", "") or task.id)
        if app and app.lower() not in dataset_id.lower() and app.lower() not in task.id.lower():
            continue
        rows.append(task.id)
    if explicit:
        wanted = set(explicit)
        rows = [tid for tid in rows if tid in wanted or any(tid.endswith(x) or x in tid for x in wanted)]
        if not rows:
            raise SystemExit(f"No tasks in {suite} matched --task-ids {explicit!r}")
    if app and not rows:
        raise SystemExit(f"No tasks in {suite} matched --app {app!r}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roll MobileGym custom hybrid tasks with foreground App Tools."
    )
    parser.add_argument("--mock-root", type=Path, default=_DEFAULT_MOCK)
    parser.add_argument("--suite", default="hybrid_tools_custom_v5")
    parser.add_argument("--app", default="", help="Keep tasks whose id contains this substring, e.g. meituan")
    parser.add_argument("--task-ids", default="", help="Comma-separated full task IDs or class names")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--env-url", default="")
    parser.add_argument("--interaction-mode", choices=["hybrid", "gui_only"], default="hybrid")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--live-console",
        action="store_true",
        help="Headed demo: open a second Chromium window with live thought/action. Implies not --headless.",
    )
    parser.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after the phone window opens before the agent starts (recording).",
    )
    parser.add_argument(
        "--runtime",
        choices=["qwen35_session", "hybrid_xml"],
        default="qwen35_session",
        help="qwen35_session = official session format; extra app tools in switch-turn user text; hybrid_xml = mock HybridXmlAgent",
    )
    parser.add_argument("--last-n", type=int, default=3)
    parser.add_argument("--agent-protocol", choices=["xml", "gpt_responses", "kimi_k2_6"], default="xml")
    parser.add_argument("--coord-space", default="norm_0_1000")
    parser.add_argument("--model-base-url", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument(
        "--model-api-key",
        default="",
        help="Gateway API key. Falls back to OPENAI_API_KEY / MOBILEGYM_API_KEY. "
        "Required for remote --model-base-url (default EMPTY is only for local vLLM).",
    )
    parser.add_argument("--enable-thinking", type=_parse_bool, default=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high", "max"],
        default="low",
        help=(
            "Thinking strength: Gemini → thinking_level; "
            "GLM/Z.AI → reasoning_effort (thinking cannot be turned off on GLM-5.3); "
            "Kimi/DeepSeek → reasoning_effort."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run every selected task. Default skips episodes that already have trace.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=_HERE / "runs" / "c5-hybrid-roll")
    args = parser.parse_args(argv)
    if not args.output_dir.is_absolute():
        args.output_dir = (_HERE / args.output_dir).resolve()

    _ensure_session_imports()
    mock_root = _ensure_mock_on_path(args.mock_root)
    explicit_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]

    if args.list or args.app or explicit_ids:
        selected = _filter_task_ids(args.suite, args.app, explicit_ids)
    else:
        selected = []

    if args.list:
        from bench_env.task.registry import TaskRegistry

        registry = TaskRegistry()
        out = []
        names = registry.list_tasks(args.suite)
        for name in names:
            task = registry.create_task(f"{args.suite}.{name}")
            spec = getattr(task, "spec", None)
            dataset_id = str(getattr(spec, "task_id", "") or "")
            if selected and task.id not in selected:
                continue
            out.append({
                "task_id": task.id,
                "dataset_id": dataset_id,
                "instruction": task.description,
                "apps": list(getattr(spec, "apps", ()) or getattr(task, "apps", [])),
            })
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"# {len(out)} tasks  suite={args.suite}  mock={mock_root}", file=sys.stderr)
        return 0

    if not args.env_url or not args.model_base_url or not args.model_name:
        raise SystemExit("required for rollout: --env-url --model-base-url --model-name")

    extra = _thinking_extra_body(args.enable_thinking)
    args.model_api_key = _resolve_api_key(args.model_api_key, base_url=args.model_base_url)
    name = (args.model_name or "").lower().replace("_", "-")
    glm_mode = (
        name.startswith("glm")
        or "/glm" in name
        or "-glm-" in name
        or name.endswith("-glm")
        or name.startswith("z-ai/")
        or "chatglm" in name
    )
    print(
        f"MobileGymPP rollout  runtime={args.runtime} suite={args.suite} "
        f"mode={args.interaction_mode} parallel={args.parallel} "
        f"thinking={args.enable_thinking} effort={args.reasoning_effort} "
        f"{'backend=glm ' if glm_mode else ''}"
        f"tasks={len(selected) or 'all'} "
        f"mock={mock_root}",
        flush=True,
    )

    if args.runtime == "qwen35_session":
        import asyncio

        from bench_env.hybrid_benchmark.runner import HybridBenchmarkConfig
        from session_runtime import run_session_benchmark

        headed = bool(args.live_console) or not args.headless
        if args.live_console and args.headless:
            print("note: --live-console implies headed; ignoring --headless", flush=True)
        config = HybridBenchmarkConfig(
            env_url=args.env_url,
            model_base_url=args.model_base_url,
            model_name=args.model_name,
            model_api_key=args.model_api_key or "EMPTY",
            model_extra_body=extra,
            interaction_mode=args.interaction_mode,
            suite=args.suite,
            task_ids=tuple(selected),
            limit=args.limit,
            parallel=args.parallel,
            output_dir=Path(args.output_dir).expanduser().resolve(),
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            coord_space=args.coord_space,
            headless=not headed,
            start_delay=float(args.start_delay or 0),
        )
        object.__setattr__(config, "live_console", bool(args.live_console))
        results = asyncio.run(
            run_session_benchmark(
                config,
                last_n=args.last_n,
                resume=not args.no_resume,
                reasoning_effort=args.reasoning_effort,
            )
        )
        return 0 if not any(row.get("error") for row in results) else 2

    forwarded = [
        "--suite", args.suite,
        "--interaction-mode", args.interaction_mode,
        "--env-url", args.env_url,
        "--model-base-url", args.model_base_url,
        "--model-name", args.model_name,
        "--model-api-key", args.model_api_key or "EMPTY",
        "--model-extra-body-json", json.dumps(extra, ensure_ascii=False),
        "--parallel", str(args.parallel),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--agent-protocol", args.agent_protocol,
        "--coord-space", args.coord_space,
        "--output-dir", str(Path(args.output_dir).expanduser().resolve()),
    ]
    if args.headless:
        forwarded.append("--headless")
    if args.max_steps is not None:
        forwarded.extend(["--max-steps", str(args.max_steps)])
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    if selected:
        forwarded.extend(["--task-ids", ",".join(selected)])

    from bench_env.hybrid_benchmark.cli import async_main, create_parser
    import asyncio

    hb_args = create_parser().parse_args(forwarded)
    return asyncio.run(async_main(hb_args))


if __name__ == "__main__":
    raise SystemExit(main())
