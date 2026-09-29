"""Rebuild results.jsonl from episode traces + terminal_state (no model).

Use this when resume overwrote results.jsonl with a later slice but traces
are still on disk. Needs the mock preview only for task.setup() init state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from eval_bench215 import _load_jsonl, _load_official, _merge_summary, _write_comparison
from run_rollout import _DEFAULT_MOCK, _ensure_mock_on_path


def _backup(path: Path) -> Path | None:
    if not path.is_file():
        return None
    bak = path.with_name(path.name + ".pre_rebuild")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def _row_from_episode(
    *,
    task: Any,
    episode_dir: Path,
    init_obs: Any,
    meta: dict[str, Any],
) -> dict[str, Any]:
    from bench_env.env.base import Observation
    from bench_env.hybrid_benchmark.runner import _task_payload
    from bench_env.task.judge import JudgeInput

    trace = json.loads((episode_dir / "trace.json").read_text(encoding="utf-8"))
    steps = trace.get("trace") if isinstance(trace.get("trace"), list) else []
    calls = [
        str(step.get("call", {}).get("name") or "")
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("call"), dict)
    ]
    calls = [name for name in calls if name]
    route = {}
    if steps and isinstance(steps[-1], dict):
        route = dict(steps[-1].get("route_after") or {})
    state_path = episode_dir / "terminal_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    last_obs = Observation(state=state, route=route)
    judge = task.evaluate(
        JudgeInput(
            init_obs=init_obs,
            last_obs=last_obs,
            answer=trace.get("agent_answer"),
        )
    )
    return {
        **_task_payload(task),
        "worker_id": None,
        "interaction_mode": meta.get("interaction_mode") or trace.get("interaction_mode"),
        "runtime": meta.get("runtime") or "qwen35_thought_session",
        "success": judge.success,
        "clean": judge.clean,
        "progress": judge.progress,
        "issues": judge.issues,
        "warnings": judge.warnings,
        "judge_error": judge.judge_error,
        "stop_reason": trace.get("stop_reason"),
        "steps": trace.get("steps") if isinstance(trace.get("steps"), int) else len(steps),
        "agent_answer": trace.get("agent_answer"),
        "agent_message": trace.get("agent_message"),
        "tool_calls": calls,
        "app_tool_calls": [name for name in calls if name != "mobile_use"],
        "runtime_s": None,
        "artifact_dir": str(episode_dir),
        "terminal_state_artifact": str(state_path) if state_path.is_file() else None,
        "error": None,
        "rebuilt_from_episode": True,
    }


async def _rebuild_pack(pack_dir: Path, env_url: str) -> dict[str, Any]:
    from bench_env.env.mobile_gym import MobileGymEnv
    from bench_env.hybrid_benchmark.runner import (
        HybridBenchmarkConfig,
        _attach_runtime,
        _build_summary,
        _configured_env_url,
        _write_results,
    )
    from bench_env.task.registry import TaskRegistry

    meta = {}
    meta_path = pack_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    existing = {
        str(row["task_id"]): row
        for row in _load_jsonl(pack_dir / "results.jsonl")
        if row.get("task_id")
    }
    episodes = pack_dir / "episodes"
    missing = sorted(
        path.name
        for path in episodes.iterdir()
        if path.is_dir() and (path / "trace.json").is_file() and path.name not in existing
    )
    print(
        f"[rebuild] {pack_dir} existing={len(existing)} missing={len(missing)}",
        flush=True,
    )
    if not missing:
        return {
            "pack_dir": str(pack_dir),
            "existing": len(existing),
            "rebuilt": 0,
            "failed": 0,
        }

    config = HybridBenchmarkConfig(
        env_url=env_url or str(meta.get("env_url") or ""),
        model_base_url=str(meta.get("model_base_url") or ""),
        model_name=str(meta.get("model_name") or ""),
        interaction_mode=str(meta.get("interaction_mode") or "hybrid"),
        suite=str(meta.get("suite") or ""),
        output_dir=pack_dir,
        headless=True,
        runtime=str(meta.get("runtime") or "qwen35_thought_session"),
        last_n=int(meta.get("last_n") or 3),
        enable_thinking=False,
        qwen35_tool_call_mode="xml",
        coord_space=str(meta.get("coord_space") or "norm_0_1000"),
    )
    url = _configured_env_url(config)
    registry = TaskRegistry()
    failed = 0
    rebuilt: list[dict[str, Any]] = []

    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    browser = None
    last_error: Exception | None = None
    for channel in ("chrome", "msedge"):
        try:
            browser = await pw.chromium.launch(
                headless=True,
                channel=channel,
                args=["--ignore-certificate-errors"],
            )
            print(f"[rebuild] launched chromium channel={channel}", flush=True)
            break
        except Exception as error:
            last_error = error
    if browser is None:
        try:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--ignore-certificate-errors"],
            )
            print("[rebuild] launched bundled chromium", flush=True)
        except Exception as error:
            last_error = error
            raise RuntimeError(
                "Playwright could not launch a browser. Run this script in your "
                f"eval terminal (not the sandbox). Last error: {last_error}"
            ) from error

    env = MobileGymEnv(
        url=url,
        browser=browser,
        headless=True,
        coord_space=config.coord_space,
        delay_after_action=0.2,
        verbose=False,
        viewport_size=(360, 800),
        physical_size=(1080, 2400),
        device_scale_factor=3,
    )
    env._log_prefix = "[rebuild]"
    await env.start()
    try:
        for index, tid in enumerate(missing, start=1):
            episode_dir = episodes / tid
            task = None
            try:
                task = registry.create_task(tid)
                init_obs = await task.setup(env)
                row = _row_from_episode(
                    task=task,
                    episode_dir=episode_dir,
                    init_obs=init_obs,
                    meta=meta,
                )
                rebuilt.append(row)
                existing[tid] = row
                status = "PASS" if row["success"] else "FAIL"
                print(
                    f"[rebuild {index}/{len(missing)}] {status} {tid} "
                    f"stop={row.get('stop_reason')} progress={row.get('progress')}",
                    flush=True,
                )
            except Exception as error:
                failed += 1
                print(f"[rebuild {index}/{len(missing)}] ERROR {tid}: {type(error).__name__}: {error}", flush=True)
            finally:
                if task is not None:
                    try:
                        task.teardown(env)
                    except Exception:
                        pass
    finally:
        await env.close()
        try:
            await browser.close()
        except Exception:
            pass
        try:
            await pw.stop()
        except Exception:
            pass

    _backup(pack_dir / "results.jsonl")
    _write_results(config, [], rebuilt)
    summary = _attach_runtime(
        _build_summary(list(existing.values()), config.interaction_mode),
        config,
    )
    print(
        f"[rebuild] wrote {len(existing)} rows sr={summary.get('sr'):.3f} "
        f"ok={summary.get('successful')} failed={failed} -> {pack_dir}",
        flush=True,
    )
    return {
        "pack_dir": str(pack_dir),
        "existing": len(existing) - len(rebuilt),
        "rebuilt": len(rebuilt),
        "failed": failed,
        "n": len(existing),
        "successful": summary.get("successful"),
        "sr": summary.get("sr"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild overwritten bench215 results.jsonl from traces.")
    parser.add_argument("--run-dir", type=Path, required=True, help=".../bench215-mock/hybrid or .../hybrid/main")
    parser.add_argument("--env-url", default="http://127.0.0.1:4172")
    parser.add_argument("--mock-root", type=Path, default=_DEFAULT_MOCK)
    parser.add_argument("--model-name", default="")
    args = parser.parse_args(argv)

    mock_root = _ensure_mock_on_path(args.mock_root)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    packs = []
    if (run_dir / "episodes").is_dir():
        packs = [run_dir]
        parent = run_dir.parent
    else:
        packs = [run_dir / "main", run_dir / "secretary"]
        packs = [path for path in packs if (path / "episodes").is_dir()]
        parent = run_dir
    if not packs:
        raise SystemExit(f"no episodes/ under {run_dir}")

    reports = [asyncio.run(_rebuild_pack(pack, args.env_url)) for pack in packs]
    print(json.dumps({"packs": reports}, ensure_ascii=False, indent=2))

    if (parent / "hybrid").is_dir() or parent.name in {"hybrid", "gui_only"}:
        mode_dir = parent if parent.name in {"hybrid", "gui_only"} else None
        if (run_dir / "main").is_dir():
            mode_dir = run_dir
        if mode_dir is not None and (mode_dir / "main").is_dir():
            official = _load_official(mock_root, None)
            model_name = args.model_name or json.loads(
                (mode_dir / "summary.json").read_text(encoding="utf-8")
            ).get("model_name") or ""
            summary = _merge_summary(
                mode_dir,
                mode=mode_dir.name,
                model_name=str(model_name),
                official=official,
            )
            print(
                f"[rebuild] merged {summary['successful']}/{summary['n']} "
                f"SR={summary['sr']:.3f} expected={summary['expected_n']} -> {mode_dir}",
                flush=True,
            )
            bench_parent = mode_dir.parent
            hybrid = bench_parent / "hybrid" / "summary.json"
            gui = bench_parent / "gui_only" / "summary.json"
            if hybrid.is_file() and gui.is_file():
                summaries = {
                    "hybrid": json.loads(hybrid.read_text(encoding="utf-8")),
                    "gui_only": json.loads(gui.read_text(encoding="utf-8")),
                }
                comparison = _write_comparison(bench_parent, summaries)
                print(
                    f"[rebuild] comparison hybrid SR={comparison['hybrid']['sr']} "
                    f"gui_only SR={comparison['gui_only']['sr']} "
                    f"delta={comparison['delta_sr_hybrid_minus_gui']}",
                    flush=True,
                )
    return 0 if all(item.get("failed", 0) == 0 for item in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
