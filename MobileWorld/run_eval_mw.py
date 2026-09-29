"""
MobileWorld benchmark evaluation with OpenMobile GUI runtimes.

Runs official MobileWorld tasks (from /task/list), executes an OpenMobile agent
(--runtime qwen35_thought_session / qwen3vl / ...), then scores via /task/eval.

Example (single backend):
  conda activate android_world
  cd <OPENMOBILE_ROOT>\\MobileWorld

  python run_eval_mw.py \\
    --aw_host http://127.0.0.1:6800 \\
    --runtime qwen35_thought_session \\
    --last_n 3 \\
    --tasks ALL \\
    --qwen3vl_model_base_url https://<openai-compatible-host>/v1 \\
    --qwen3vl_model_name gemini-3.1-pro-preview \\
    --qwen3vl_model_api_key YOUR_KEY

MCP tasks only (session + DashScope/ModelScope tools):
  python parallel_eval_mw.py \\
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 \\
    --runtime qwen35_thought_session --mcp_only --tasks ALL ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AW_ROOT = _HERE.parent / "AndroidWorld"
if str(_AW_ROOT) not in sys.path:
    sys.path.insert(0, str(_AW_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from android_world.agents import agent_factory  # noqa: E402
from android_world.agents import base_agent  # noqa: E402
from android_world.agents.session_runtimes import list_runtimes  # noqa: E402
from android_world.episode_runner import run_episode  # noqa: E402
from android_world.env import interface  # noqa: E402

from mw_env import MobileWorldEnv  # noqa: E402
from mw_session_mcp import MobileWorldMcpBridge, SessionMcpWrapper  # noqa: E402

SUCCESS_THRESHOLD = 0.99


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _split_csv(raw: str) -> list[str]:
    text = (raw or "").replace("，", ",").replace(";", ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def _task_shard(task_name: str, num_shards: int) -> int:
    # hashlib, not builtin hash(): PYTHONHASHSEED differs per process, so two
    # workers would otherwise pick overlapping / missing task sets.
    digest = hashlib.md5(str(task_name).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_shards)


def _write_result_txt(path: Path, score: float, reason: str) -> None:
    path.write_text(f"score: {score}\n{reason}\n", encoding="utf-8")


def _build_agent(args: argparse.Namespace, env: interface.AsyncEnv) -> base_agent.EnvironmentInteractingAgent:
    agent, spec = agent_factory.create_gui_agent(
        env,
        runtime=args.runtime,
        agent_name=args.agent_name,
        use_memgui_prompt=args.use_memgui_prompt,
        use_memory_prompt=args.use_memory_prompt,
        memgui_prompt_format=args.memgui_prompt_format,
        last_n=args.last_n,
        enable_thinking=args.enable_thinking,
        require_think_tags=args.require_think_tags,
        qwen35_tool_call_mode=args.qwen35_tool_call_mode,
        reasoning_effort=args.reasoning_effort,
        model_base_url=args.qwen3vl_model_base_url,
        model_api_key=args.qwen3vl_model_api_key,
        model_name=args.qwen3vl_model_name,
        weak_model_base_url=args.qwen3vl_switching_weak_model_base_url,
        weak_model_api_key=args.qwen3vl_switching_weak_model_api_key,
        weak_model_name=args.qwen3vl_switching_weak_model_name,
    )
    print(f"Runtime: {spec.name} (last_n={args.last_n}; {spec.description})")
    return agent


def _select_tasks(env: MobileWorldEnv, args: argparse.Namespace) -> list[str]:
    tasks = env.list_tasks(
        enable_mcp=args.enable_mcp,
        enable_user_interaction=args.enable_user_interaction,
    )
    if args.mcp_only:
        before = len(tasks)
        tasks = [t for t in tasks if "agent-mcp" in (t.get("tags") or [])]
        print(f"MCP-only: {len(tasks)}/{before} tasks tagged agent-mcp")
    names = sorted({t["name"] for t in tasks})
    if not names:
        raise RuntimeError("No tasks returned from /task/list")

    raw = (args.tasks or "ALL").strip()
    if raw.upper() == "ALL":
        selected = names
    else:
        wanted = set(_split_csv(raw))
        selected = [n for n in names if n in wanted]
        missing = sorted(wanted - set(selected))
        if missing:
            print(f"WARNING: unknown task(s) ignored: {', '.join(missing)}")

    if args.num_shards > 1:
        before = len(selected)
        selected = [
            n for n in selected if _task_shard(n, args.num_shards) == args.shard_index
        ]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(selected)}/{before} tasks (by task_name hash)"
        )
    return selected


def _load_finished_scores(output_dir: Path, task_names: list[str]) -> dict[str, float]:
    finished: dict[str, float] = {}
    for name in task_names:
        task_dir = output_dir / name
        # Env crashes write error.txt; never treat those as completed.
        if (task_dir / "error.txt").is_file():
            continue
        result_txt = task_dir / "result.txt"
        if not result_txt.is_file():
            continue
        first = result_txt.read_text(encoding="utf-8").splitlines()
        if not first or "score:" not in first[0]:
            continue
        try:
            finished[name] = float(first[0].split("score:", 1)[1].strip())
        except ValueError:
            continue
    return finished


_ENV_ERROR_HINTS = (
    "not healthy",
    "device is not healthy",
    "connection refused",
    "connection aborted",
    "connection reset",
    "timed out",
    "timeout",
    "max retries exceeded",
    "failed to establish",
    "task/eval",
    "verifier crashed",
    "eval exception",
    "502",
    "503",
    "504",
)


def _is_env_error(err: BaseException | str) -> bool:
    text = str(err).lower()
    return any(h in text for h in _ENV_ERROR_HINTS)


def _is_eval_error(err: BaseException | str) -> bool:
    text = str(err).lower()
    return any(
        h in text
        for h in (
            "task/eval",
            "verifier crashed",
            "eval exception",
        )
    )


def _should_retry_env(err: BaseException, env: MobileWorldEnv) -> bool:
    return _is_env_error(err) or not env.health()


def _reset_save_dir(save_dir: Path) -> None:
    if save_dir.is_dir():
        shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    results: list[dict],
) -> None:
    scored = [r for r in results if r.get("score") is not None]
    success = [r for r in scored if float(r["score"]) > SUCCESS_THRESHOLD]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "runtime": args.runtime or args.agent_name,
        "last_n": args.last_n,
        "hosts": [args.aw_host],
        "success_threshold": SUCCESS_THRESHOLD,
        "total_tasks": len(results),
        "scored_tasks": len(scored),
        "success_count": len(success),
        "success_rate": (len(success) / len(scored)) if scored else 0.0,
        "avg_score": (
            sum(float(r["score"]) for r in scored) / len(scored) if scored else 0.0
        ),
        "tasks": results,
    }
    summary_name = (
        f"eval_summary_shard{args.shard_index}.json"
        if args.num_shards > 1
        else "eval_summary.json"
    )
    (output_dir / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("EVAL SUMMARY")
    print(f"  output   : {output_dir}")
    print(f"  runtime  : {summary['runtime']}")
    print(f"  scored   : {summary['scored_tasks']}/{summary['total_tasks']}")
    print(f"  success  : {summary['success_count']} ({summary['success_rate']:.1%})")
    print(f"  avg_score: {summary['avg_score']:.4f}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MobileWorld benchmark eval with OpenMobile runtimes."
    )
    parser.add_argument("--aw_host", type=str, default="http://127.0.0.1:6800")
    parser.add_argument("--device", type=str, default="emulator-5554")
    parser.add_argument(
        "--runtime",
        type=str,
        default="",
        help="OpenMobile runtime preset (preferred).\n" + list_runtimes(),
    )
    parser.add_argument("--agent_name", type=str, default="qwen3vl")
    parser.add_argument("--tasks", type=str, default="ALL", help='Task names or "ALL".')
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Log root. Default: eval_runs/<runtime>_<timestamp>/",
    )
    parser.add_argument("--max_n_steps", type=int, default=50)
    parser.add_argument("--last_n", type=int, default=3)
    parser.add_argument(
        "--enable_thinking",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Session: chat_template_kwargs.enable_thinking (true/false).",
    )
    parser.add_argument(
        "--require_think_tags",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="qwen35/qwen3vl_session: if false, use tool_call-only prompt (no mandatory <think>).",
    )
    parser.add_argument(
        "--qwen35_tool_call_mode",
        type=str,
        default="native",
        choices=["xml", "native"],
        help="native = OpenAI tools= schema; xml = dump tools into the system prompt.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="max",
        choices=["low", "high", "max"],
    )
    parser.add_argument("--enable_mcp", action="store_true")
    parser.add_argument(
        "--mcp_only",
        action="store_true",
        help="Evaluate only agent-mcp tasks and attach MCP tools to the session runtime.",
    )
    parser.add_argument("--enable_user_interaction", action="store_true")
    parser.add_argument("--use_memgui_prompt", action="store_true")
    parser.add_argument("--use_memory_prompt", action="store_true")
    parser.add_argument(
        "--memgui_prompt_format",
        type=str,
        default="auto",
        choices=["auto", "qwen3vl", "qwen35"],
    )
    parser.add_argument("--qwen3vl_model_base_url", type=str, default="http://<openai-compatible-host>/v1")
    parser.add_argument("--qwen3vl_model_name", type=str, default="")
    parser.add_argument("--qwen3vl_model_api_key", type=str, default="EMPTY")
    parser.add_argument("--qwen3vl_switching_weak_model_base_url", type=str, default="")
    parser.add_argument("--qwen3vl_switching_weak_model_name", type=str, default="")
    parser.add_argument("--qwen3vl_switching_weak_model_api_key", type=str, default="EMPTY")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--task_retries",
        type=int,
        default=3,
        help="Retries per task when the emulator is unhealthy / init 500.",
    )
    parser.add_argument(
        "--health_retries",
        type=int,
        default=20,
        help="Health-check attempts before/after a task (emulator reboot can take ~1 min).",
    )
    parser.add_argument(
        "--health_sleep",
        type=float,
        default=5.0,
        help="Seconds between health-check attempts.",
    )
    args = parser.parse_args()
    if args.mcp_only:
        args.enable_mcp = True

    if args.num_shards < 1:
        print("--num_shards must be >= 1", file=sys.stderr)
        return 2
    if not (0 <= args.shard_index < args.num_shards):
        print("--shard_index must be in [0, num_shards)", file=sys.stderr)
        return 2

    runtime_tag = (args.runtime or args.agent_name).replace("/", "_")
    output_dir = Path(
        args.output_dir.strip()
        or str(_HERE / "eval_runs" / f"{runtime_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": {k: v for k, v in vars(args).items() if "api_key" not in k},
        "output_dir": str(output_dir),
    }
    (output_dir / "eval_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    env = MobileWorldEnv(base_url=args.aw_host, device=args.device)
    if not env.wait_until_healthy(
        retries=args.health_retries, sleep_s=args.health_sleep
    ):
        print(f"Backend not healthy: {args.aw_host}", file=sys.stderr)
        return 2

    if args.enable_mcp:
        runtime = (args.runtime or "").strip().lower()
        if "session" not in runtime:
            print(
                "--mcp_only / --enable_mcp requires a session runtime "
                "(qwen35_thought_session or qwen35_session)",
                file=sys.stderr,
            )
            return 2
        if args.qwen35_tool_call_mode != "native":
            print(
                "WARNING: MCP tools are injected via tools=; "
                "prefer --qwen35_tool_call_mode=native"
            )

    task_names = _select_tasks(env, args)
    if not task_names:
        print("No tasks selected.", file=sys.stderr)
        return 2
    if args.mcp_only:
        print("MCP tasks:")
        for name in task_names:
            print(f"  {name}")

    finished = _load_finished_scores(output_dir, task_names)
    if finished:
        print(f"Resume: {len(finished)} task(s) already have result.txt")

    agent = _build_agent(args, env)
    mcp_bridge = None
    if args.enable_mcp:
        mcp_bridge = MobileWorldMcpBridge()
        agent = SessionMcpWrapper(agent, mcp_bridge)
    results: list[dict] = []
    consecutive_env_fails = 0

    for idx, task_name in enumerate(task_names):
        save_dir = output_dir / task_name
        entry: dict = {"task_name": task_name, "score": None, "reason": "", "skipped": False}

        if task_name in finished:
            entry["score"] = finished[task_name]
            entry["reason"] = "resumed"
            entry["skipped"] = True
            entry["success"] = float(finished[task_name]) > SUCCESS_THRESHOLD
            results.append(entry)
            continue

        goal = env.get_task_goal(task_name)
        print(f"\n[{idx + 1}/{len(task_names)}] task={task_name}")
        print(f"  goal: {goal[:120]}{'...' if len(goal) > 120 else ''}")

        if not env.health():
            print("  device unhealthy; waiting to recover before this task")
            if not env.wait_until_healthy(
                retries=args.health_retries, sleep_s=args.health_sleep
            ):
                consecutive_env_fails += 1
                print(
                    f"  SKIP {task_name}: backend still unhealthy at {args.aw_host} "
                    "(no result.txt; resume will retry)"
                )
                entry["reason"] = f"skipped: device unhealthy at {args.aw_host}"
                entry["skipped"] = True
                entry["success"] = False
                results.append(entry)
                if consecutive_env_fails >= 2:
                    print(
                        "  device stayed down; leaving remaining tasks unfinished "
                        "so a later resume can pick them up"
                    )
                    break
                continue

        started = time.time()
        score: float | None = None
        reason = ""
        env_failed = False
        last_err: BaseException | None = None
        attempts = max(1, int(args.task_retries))

        for attempt in range(1, attempts + 1):
            _reset_save_dir(save_dir)
            try:
                if mcp_bridge is not None:
                    try:
                        meta = env.get_task_metadata(task_name)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        print(f"  metadata warning: {exc!r}")
                        meta = {"tags": ["agent-mcp"], "apps": []}
                    mcp_bridge.select_for_task(meta)
                env.initialize_task(task_name)
                run_episode(
                    goal=goal,
                    agent=agent,
                    max_n_steps=args.max_n_steps,
                    start_on_home_screen=True,
                    termination_fn=None,
                    save_dir=str(save_dir),
                )
                try:
                    score, reason = env.get_task_score(task_name)
                except Exception as eval_err:  # pylint: disable=broad-exception-caught
                    # Episode is done. Do not wipe traces or re-run the model.
                    last_err = eval_err
                    reason = repr(eval_err)
                    print(f"  EVAL ERROR (in-place retries exhausted): {eval_err!r}")
                    env_failed = True
                    try:
                        env.tear_down_task(task_name)
                    except Exception as te:  # pylint: disable=broad-exception-caught
                        print(f"  tear_down ERROR: {te!r}")
                    break
                env_failed = False
                last_err = None
                try:
                    env.tear_down_task(task_name)
                except Exception as te:  # pylint: disable=broad-exception-caught
                    print(f"  tear_down ERROR: {te!r}")
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_err = e
                env_failed = _should_retry_env(e, env)
                print(f"  ERROR (attempt {attempt}/{attempts}): {e!r}")
                try:
                    env.tear_down_task(task_name)
                except Exception as te:  # pylint: disable=broad-exception-caught
                    print(f"  tear_down ERROR: {te!r}")
                if env_failed and attempt < attempts:
                    print("  emulator glitch; waiting for health then retrying this task")
                    env.wait_until_healthy(
                        retries=args.health_retries, sleep_s=args.health_sleep
                    )
                    continue
                score, reason = 0.0, repr(e)
                break

        if not env.health():
            env.wait_until_healthy(
                retries=args.health_retries, sleep_s=args.health_sleep
            )

        duration = time.time() - started

        if env_failed:
            consecutive_env_fails += 1
            err_path = save_dir / "error.txt"
            err_path.write_text(reason or repr(last_err), encoding="utf-8")
            print(
                f"  ENV FAIL {task_name}: not writing result.txt "
                f"(resume will retry). {duration:.1f}s"
            )
            entry.update(
                {
                    "task_name": task_name,
                    "goal": goal,
                    "runtime": args.runtime or args.agent_name,
                    "score": None,
                    "reason": reason or repr(last_err),
                    "success": False,
                    "skipped": True,
                    "duration_seconds": round(duration, 2),
                    "save_dir": str(save_dir),
                    "env_error": True,
                }
            )
            results.append(entry)
            if consecutive_env_fails >= 2 and not env.health():
                print(
                    "  device still unhealthy after retries; "
                    "stop this shard so remaining tasks are not marked 0"
                )
                break
            continue

        consecutive_env_fails = 0
        assert score is not None
        _write_result_txt(save_dir / "result.txt", score, reason)
        eval_record = {
            "task_name": task_name,
            "goal": goal,
            "runtime": args.runtime or args.agent_name,
            "score": score,
            "reason": reason,
            "success": float(score) > SUCCESS_THRESHOLD,
            "duration_seconds": round(duration, 2),
            "save_dir": str(save_dir),
        }
        (save_dir / "eval.json").write_text(
            json.dumps(eval_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        entry.update(eval_record)
        results.append(entry)
        print(f"  score={score:.4f} success={entry['success']} ({duration:.1f}s)")

    _write_summary(output_dir, args=args, results=results)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
