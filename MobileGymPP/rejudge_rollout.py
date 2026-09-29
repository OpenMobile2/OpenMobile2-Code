"""Replay saved traces onto a fresh env and run the semantic verifier.

Does not call the model. Use this when ``results.jsonl`` has resumed
placeholders (``stop_reason=resumed``, ``success`` defaulted True).

  python rejudge_rollout.py `
    --run-dir <OPENMOBILE_ROOT>\\MobileGymPP\\runs_rollout\\gemini\\hybrid_tools_0831_explore `
    --env-url http://127.0.0.1:4172 `
    --parallel 8 --headless
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent.parent


def _default_mock_root() -> Path:
    env = (os.environ.get("MOBILEGYM_MOCK") or "").strip()
    candidates = [
        Path(env) if env else None,
        _CODE / "mobilegym++" / "mobilegym-mock" / "trial_apps" / "mobilegym",
        _CODE / "mobilegym-mock" / "trial_apps" / "mobilegym",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "bench_env" / "hybrid_benchmark").is_dir():
            return candidate
    return candidates[1]


_DEFAULT_MOCK = _default_mock_root()

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*\"?\s*>(.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*\"?\s*>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _ensure_mock_on_path(mock_root: Path) -> Path:
    root = mock_root.expanduser().resolve()
    if not (root / "bench_env" / "hybrid_benchmark").is_dir():
        raise SystemExit(f"mobilegym-mock bench_env not found under {root}")
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    os.chdir(root)
    return root


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = row.get("task_id")
        if task_id:
            rows[str(task_id)] = row
    return rows


def _decode_param(raw: str) -> Any:
    text = str(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _call_from_xml(response: str) -> dict[str, Any] | None:
    blocks = _TOOL_CALL_RE.findall(response or "")
    if not blocks:
        return None
    match = _FUNCTION_RE.search(blocks[-1].strip())
    if not match:
        return None
    arguments: dict[str, Any] = {}
    for item in _PARAMETER_RE.finditer(match.group(2)):
        arguments[item.group(1).strip()] = _decode_param(item.group(2))
    return {"name": match.group(1).strip(), "arguments": arguments}


def _normalize_mobile_use(arguments: dict[str, Any]) -> dict[str, Any]:
    args = dict(arguments or {})
    action = str(args.get("action") or "").strip().lower()
    if action == "open_app" and not str(args.get("app") or "").strip():
        args["app"] = str(args.get("app_name") or args.get("text") or "").strip()
    if action in {"status", "terminate"} or args.get("goal_status"):
        status = str(args.get("status") or args.get("goal_status") or "").strip().lower()
        if status in {"success", "complete"}:
            args["action"] = "terminate"
            args["status"] = "success"
        elif status:
            args["action"] = "terminate"
            args["status"] = "failure"
            args.setdefault("text", status)
    return args


def _step_call(step: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _call_from_xml(str(step.get("response") or ""))
    stored = step.get("call") if isinstance(step.get("call"), dict) else None
    if parsed and parsed.get("name"):
        call = parsed
    elif stored and stored.get("name"):
        call = {"name": stored["name"], "arguments": dict(stored.get("arguments") or {})}
    else:
        return None
    if call["name"] == "mobile_use":
        call["arguments"] = _normalize_mobile_use(call.get("arguments") or {})
    return call


def _simulated_time(suite: str) -> str | None:
    try:
        module = importlib.import_module(f"bench_env.task.{suite}")
    except Exception:
        return None
    value = getattr(module, "SIMULATED_TIME", None)
    return str(value) if value else None


def _write_outputs(out_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("task_id") or ""))
    (out_dir / "rejudge_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in ordered),
        encoding="utf-8",
    )
    successes = [row for row in ordered if row.get("success") and not row.get("error")]
    summary = {
        "total": len(ordered),
        "successful": len(successes),
        "sr": (len(successes) / len(ordered)) if ordered else 0.0,
        "errors": sum(bool(row.get("error")) for row in ordered),
        "clean": sum(bool(row.get("clean")) for row in ordered),
        "replay_errors": sum(bool(row.get("replay_error")) for row in ordered),
        "stop_reasons": dict(Counter(str(row.get("stop_reason")) for row in ordered)),
        "source": "rejudge_rollout.py",
    }
    (out_dir / "rejudge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rejudge_success_ids.json").write_text(
        json.dumps(
            [row["task_id"] for row in successes],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


async def _replay_one(env: Any, task: Any, trace_path: Path) -> dict[str, Any]:
    from dataclasses import replace as dc_replace

    from bench_env.env.base import ActionType
    from bench_env.hybrid_agent.browser_tools import BrowserToolClient
    from bench_env.hybrid_agent.mobile_use import (
        MobileUseArgumentError,
        to_mobilegym_action,
    )
    from bench_env.hybrid_benchmark.runner import _task_payload
    from bench_env.task.judge import JudgeInput

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    steps = list(payload.get("trace") or [])
    browser = BrowserToolClient(env.page)
    started = time.time()
    initial_obs = await task.setup(env)
    replay_error = None
    executed = 0
    calls: list[str] = []
    try:
        for step in steps:
            call = _step_call(step)
            if not call:
                continue
            name = str(call["name"])
            arguments = dict(call.get("arguments") or {})
            calls.append(name)
            executed += 1
            if name == "mobile_use":
                try:
                    action = to_mobilegym_action(
                        arguments,
                        coord_space=str(getattr(env, "coord_space", "norm_0_1000")),
                        physical_size=(
                            int(getattr(env, "physical_width", 1080)),
                            int(getattr(env, "physical_height", 2400)),
                        ),
                    )
                except MobileUseArgumentError as error:
                    replay_error = f"INVALID_ARGUMENT:{error}"
                    continue
                result = await env.step(action)
                if action.action_type in {ActionType.COMPLETE, ActionType.ABORT} or result.done:
                    break
            else:
                await browser.call_tool(name, arguments)
                await env.get_observation()
    except Exception as error:
        replay_error = f"{type(error).__name__}: {error}"

    final_obs = await env.get_observation()
    final_state = await env.get_state(required_apps=list(task.apps))
    final_obs = dc_replace(final_obs, state=final_state)
    judge = task.evaluate(
        JudgeInput(
            init_obs=initial_obs,
            last_obs=final_obs,
            answer=getattr(env, "agent_answer", None),
        )
    )
    try:
        task.teardown(env)
    except Exception:
        pass
    return {
        **_task_payload(task),
        "success": bool(judge.success),
        "clean": bool(judge.clean),
        "progress": judge.progress,
        "issues": judge.issues,
        "warnings": judge.warnings,
        "judge_error": judge.judge_error,
        "stop_reason": "replayed",
        "original_stop_reason": payload.get("stop_reason"),
        "steps": executed,
        "tool_calls": calls,
        "app_tool_calls": [name for name in calls if name != "mobile_use"],
        "runtime_s": time.time() - started,
        "artifact_dir": str(trace_path.parent),
        "replay_error": replay_error,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved MobileGym traces and run semantic verifiers."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-url", default="")
    parser.add_argument("--mock-root", type=Path, default=_DEFAULT_MOCK)
    parser.add_argument("--suite", default="")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rejudge every episode even if rejudge_results.jsonl already has it.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")
    mock_root = _ensure_mock_on_path(args.mock_root)

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    suite = args.suite or str(meta.get("suite") or "")
    if not suite:
        raise SystemExit("pass --suite or put suite in meta.json")
    env_url = args.env_url or str(meta.get("env_url") or "")
    if not env_url:
        raise SystemExit("pass --env-url (simulator preview, e.g. http://127.0.0.1:4172)")

    from bench_env.env.mobile_gym import MobileGymEnv
    from bench_env.task.hybrid_tools_v4.time_control import simulated_time_bootstrap_url
    from bench_env.task.registry import TaskRegistry

    registry = TaskRegistry()
    tasks = [registry.create_task(f"{suite}.{name}") for name in registry.list_tasks(suite)]
    wanted = {item.strip() for item in args.task_ids.split(",") if item.strip()}
    if wanted:
        tasks = [
            task
            for task in tasks
            if task.id in wanted or task.name in wanted or any(token in task.id for token in wanted)
        ]
    jobs: list[tuple[Any, Path]] = []
    for task in tasks:
        trace_path = run_dir / "episodes" / task.id / "trace.json"
        if trace_path.is_file():
            jobs.append((task, trace_path))
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]
    if not jobs:
        raise SystemExit(f"no trace.json episodes found under {run_dir / 'episodes'}")

    existing = {} if args.no_resume else _load_jsonl(run_dir / "rejudge_results.jsonl")
    done_rows = [existing[task.id] for task, _ in jobs if task.id in existing]
    pending = [(task, path) for task, path in jobs if task.id not in existing]
    print(
        f"rejudge suite={suite} traces={len(jobs)} "
        f"already={len(done_rows)} pending={len(pending)} "
        f"mock={mock_root} env={env_url}",
        flush=True,
    )
    if not pending:
        summary = _write_outputs(run_dir, done_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    clock = _simulated_time(suite)
    url = simulated_time_bootstrap_url(env_url, clock) if clock else env_url
    if clock:
        print(f"simulated time: {clock}", flush=True)

    queue: asyncio.Queue[tuple[Any, Path]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    results: list[dict[str, Any]] = list(done_rows)
    lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        env = MobileGymEnv(
            url=url,
            headless=args.headless,
            coord_space=str(meta.get("coord_space") or "norm_0_1000"),
            delay_after_action=float(meta.get("delay_after_action") or 0.8),
            verbose=False,
            viewport_size=(360, 800),
            physical_size=(1080, 2400),
            device_scale_factor=3,
        )
        env._log_prefix = f"[rejudge-{worker_id}]"
        await env.start()
        try:
            while True:
                try:
                    task, trace_path = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    row = await _replay_one(env, task, trace_path)
                except Exception as error:
                    from bench_env.hybrid_benchmark.runner import _task_payload

                    row = {
                        **_task_payload(task),
                        "success": False,
                        "clean": False,
                        "progress": 0.0,
                        "issues": [],
                        "warnings": [],
                        "judge_error": None,
                        "stop_reason": "replay_failed",
                        "steps": 0,
                        "tool_calls": [],
                        "app_tool_calls": [],
                        "runtime_s": 0,
                        "artifact_dir": str(trace_path.parent),
                        "replay_error": f"{type(error).__name__}: {error}",
                        "error": f"{type(error).__name__}: {error}",
                    }
                async with lock:
                    results.append(row)
                    _write_outputs(run_dir, results)
                    status = "PASS" if row.get("success") else "FAIL"
                    print(
                        f"[{len(results)}/{len(jobs)}] {status} {task.id} "
                        f"steps={row.get('steps')} issues={len(row.get('issues') or [])}",
                        flush=True,
                    )
                queue.task_done()
        finally:
            await env.close()

    async def run() -> None:
        workers = min(max(1, args.parallel), len(pending))
        await asyncio.gather(*(worker(index + 1) for index in range(workers)))

    asyncio.run(run())
    summary = _write_outputs(run_dir, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary.get("errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
