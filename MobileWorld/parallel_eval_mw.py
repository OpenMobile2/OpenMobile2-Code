"""
Launch parallel MobileWorld benchmark eval workers (one backend per worker).

Before launch, task folders containing error.txt under --output_dir are deleted
so failed tasks are retried (they also have result.txt score=0 and would otherwise
be skipped). Pass --keep_errors to disable that cleanup.

Device-not-healthy / init-500 is retried in-process (wait for emulator, retry the
same task). If the device stays down, remaining tasks are left unfinished (no
result.txt) instead of being scored 0.

Example:

python parallel_eval_mw.py `
  --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
  --runtime qwen35_thought_session `
  --qwen35_tool_call_mode=native `
  --last_n 3 `
  --tasks ALL `
  --n_runs 3 `
  --output_dir eval_runs/YOUR_MODEL/gui-only `
  --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
  --qwen3vl_model_name "YOUR_MODEL" `
  --qwen3vl_model_api_key empty


  python parallel_eval_mw.py `
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
    --runtime qwen35_thought_session `
    --enable_thinking=true `
    --require_think_tags=false `
    --qwen35_tool_call_mode=native `
    --last_n 3 `
    --tasks ALL `
    --output_dir eval_runs/YOUR_MODEL/gui-only `
    --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
    --qwen3vl_model_name YOUR_MODEL `
    --qwen3vl_model_api_key EMPTY

  python parallel_eval_mw.py `
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
    --runtime qwen35_thought_session `
    --qwen35_tool_call_mode=native `
    --last_n 3 `
    --tasks ALL `
    --output_dir eval_runs/YOUR_MODEL/gui-only `
    --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
    --qwen3vl_model_name "YOUR_MODEL" `
    --qwen3vl_model_api_key EMPTY

MCP tasks only:
  python parallel_eval_mw.py `
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
    --runtime qwen35_thought_session `
    --enable_thinking=true `
    --require_think_tags=false `
    --qwen35_tool_call_mode=native `
    --mcp_only `
    --last_n 3 `
    --tasks ALL `
    --output_dir eval_runs/YOUR_MODEL/mcp-only `
    --qwen3vl_model_base_url http://<openai-compatible-host>/v1 `
    --qwen3vl_model_name YOUR_MODEL `
    --qwen3vl_model_api_key EMPTY

Repeat 3 independent evals (sequential; same emulators):
  python parallel_eval_mw.py `
    ... `
    --output_dir eval_runs/YOUR_MODEL `
    --n_runs 3
  # writes eval_runs/YOUR_MODEL/1, /2, /3

  python parallel_eval_mw.py `
    ... `
    --output_dir eval_runs/YOUR_MODEL/mcp-only `
    --n_runs 3
  # writes eval_runs/YOUR_MODEL/1/mcp-only, /2/mcp-only, /3/mcp-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "run_eval_mw.py"
_MODE_LEAFS = frozenset({"gui-only", "mcp-only", "hybrid", "gui_only", "mcp_only"})


def resolve_run_dirs(output_dir: Path, n_runs: int) -> list[Path]:
    """Map --output_dir + --n_runs to one directory per independent eval.

    n_runs=1 keeps ``output_dir`` unchanged. n_runs>1 writes
    ``{model}/{1..N}`` or ``{model}/{1..N}/{mode}`` when the last component
    is gui-only / mcp-only / hybrid. A trailing run index in output_dir is
    stripped so an existing ``model/1/mcp-only`` command can add ``--n_runs 3``.
    """
    if n_runs <= 1:
        return [output_dir]
    root = output_dir
    mode = ""
    if root.name.isdigit():
        root = root.parent
    if root.name in _MODE_LEAFS:
        mode = root.name
        root = root.parent
        if root.name.isdigit():
            root = root.parent
    if mode:
        return [root / str(i) / mode for i in range(1, n_runs + 1)]
    return [root / str(i) for i in range(1, n_runs + 1)]


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def delete_error_task_folders(output_dir: Path) -> list[str]:
    """Delete task dirs that contain error.txt so resume will re-run them.

    Failed runs still write result.txt (score=0), so without deleting the folder
    ``run_eval_mw`` would skip them forever.
    """
    deleted: list[str] = []
    if not output_dir.is_dir():
        return deleted
    for task_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        if not (task_dir / "error.txt").is_file():
            continue
        deleted.append(task_dir.name)
        shutil.rmtree(task_dir)
        print(f"deleted error folder: {task_dir.name}")
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="ALL")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--runtime", type=str, default="")
    parser.add_argument("--agent_name", type=str, default="qwen3vl")
    parser.add_argument("--max_n_steps", type=int, default=50)
    parser.add_argument("--last_n", type=int, default=3)
    parser.add_argument(
        "--enable_thinking",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--require_think_tags",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--qwen35_tool_call_mode",
        type=str,
        default="native",
        choices=["xml", "native"],
    )
    parser.add_argument("--enable_mcp", action="store_true")
    parser.add_argument(
        "--mcp_only",
        action="store_true",
        help="Only agent-mcp tasks; implies --enable_mcp and session MCP tools.",
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
    parser.add_argument(
        "--keep_errors",
        action="store_true",
        help="Do not delete task folders that contain error.txt before launch.",
    )
    parser.add_argument(
        "--task_retries",
        type=int,
        default=3,
        help="Per-task retries when the emulator is unhealthy (forwarded to run_eval_mw).",
    )
    parser.add_argument(
        "--health_retries",
        type=int,
        default=20,
        help="Health-check attempts between tasks (forwarded to run_eval_mw).",
    )
    parser.add_argument(
        "--health_sleep",
        type=float,
        default=5.0,
        help="Seconds between health-check attempts.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help=(
            "Independent eval repeats, sequential. n_runs>1 writes "
            "{output_dir}/1 .. /N, or {model}/1/{mode} when output_dir ends "
            "with gui-only/mcp-only/hybrid."
        ),
    )
    args = parser.parse_args()
    if int(args.n_runs) < 1:
        print("--n_runs must be >= 1", file=sys.stderr)
        return 2

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (_HERE / out).resolve()
    args.output_dir = str(out)

    hosts = [h.strip() for h in args.hosts.replace("，", ",").split(",") if h.strip()]
    if not hosts:
        print("No hosts provided", file=sys.stderr)
        return 2

    run_dirs = resolve_run_dirs(out, int(args.n_runs))
    if len(run_dirs) > 1:
        print("Repeat evals:")
        for i, run_dir in enumerate(run_dirs, start=1):
            print(f"  run {i}/{len(run_dirs)} -> {run_dir}")

    for i, run_dir in enumerate(run_dirs, start=1):
        if len(run_dirs) > 1:
            print(f"\n===== eval run {i}/{len(run_dirs)}: {run_dir} =====")
        code = run_once(args, run_dir, hosts)
        if code != 0:
            print(
                f"Run {i}/{len(run_dirs)} failed (exit {code}); remaining runs skipped.",
                file=sys.stderr,
            )
            return code

    print("All eval workers finished.")
    if len(run_dirs) == 1:
        print(f"Results under: {run_dirs[0]}")
    else:
        print("Results under:")
        for run_dir in run_dirs:
            print(f"  {run_dir}")
    print("See eval_summary.json written by each worker shard (same output_dir).")
    return 0


def run_once(args: argparse.Namespace, out: Path, hosts: list[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    output_dir = str(out)

    if not args.keep_errors:
        deleted = delete_error_task_folders(out)
        print(f"error folders removed: {len(deleted)}")

    procs = []
    for i, host in enumerate(hosts):
        cmd = [
            sys.executable,
            str(_SCRIPT),
            f"--aw_host={host}",
            f"--tasks={args.tasks}",
            f"--output_dir={output_dir}",
            f"--agent_name={args.agent_name}",
            f"--max_n_steps={args.max_n_steps}",
            f"--last_n={int(args.last_n)}",
            f"--qwen35_tool_call_mode={args.qwen35_tool_call_mode}",
            f"--qwen3vl_model_base_url={args.qwen3vl_model_base_url}",
            f"--qwen3vl_model_name={args.qwen3vl_model_name}",
            f"--qwen3vl_model_api_key={args.qwen3vl_model_api_key}",
            f"--memgui_prompt_format={args.memgui_prompt_format}",
            f"--shard_index={i}",
            f"--num_shards={len(hosts)}",
            f"--task_retries={int(args.task_retries)}",
            f"--health_retries={int(args.health_retries)}",
            f"--health_sleep={float(args.health_sleep)}",
        ]
        if args.runtime:
            cmd.append(f"--runtime={args.runtime}")
        cmd.append(f"--enable_thinking={str(bool(args.enable_thinking)).lower()}")
        cmd.append(f"--require_think_tags={str(bool(args.require_think_tags)).lower()}")
        if args.use_memgui_prompt:
            cmd.append("--use_memgui_prompt")
        if args.use_memory_prompt:
            cmd.append("--use_memory_prompt")
        if args.enable_mcp:
            cmd.append("--enable_mcp")
        if args.mcp_only:
            cmd.append("--mcp_only")
        if args.enable_user_interaction:
            cmd.append("--enable_user_interaction")
        if args.qwen3vl_switching_weak_model_base_url:
            cmd.append(
                f"--qwen3vl_switching_weak_model_base_url={args.qwen3vl_switching_weak_model_base_url}"
            )
        if args.qwen3vl_switching_weak_model_name:
            cmd.append(
                f"--qwen3vl_switching_weak_model_name={args.qwen3vl_switching_weak_model_name}"
            )
        if args.qwen3vl_switching_weak_model_api_key:
            cmd.append(
                f"--qwen3vl_switching_weak_model_api_key={args.qwen3vl_switching_weak_model_api_key}"
            )
        print("Launch:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    codes = [p.wait() for p in procs]
    bad = [c for c in codes if c != 0]
    if bad:
        print(f"Some workers failed: exit codes={codes}", file=sys.stderr)
        return 1

    shard_files = sorted(out.glob("eval_summary_shard*.json"))
    if shard_files:
        merged_tasks: list[dict] = []
        for sf in shard_files:
            data = json.loads(sf.read_text(encoding="utf-8"))
            merged_tasks.extend(data.get("tasks") or [])
        scored = [t for t in merged_tasks if t.get("score") is not None]
        success = [t for t in scored if float(t["score"]) > 0.99]
        merged = {
            "output_dir": str(out),
            "runtime": args.runtime or args.agent_name,
            "total_tasks": len(merged_tasks),
            "scored_tasks": len(scored),
            "success_count": len(success),
            "success_rate": (len(success) / len(scored)) if scored else 0.0,
            "avg_score": (
                sum(float(t["score"]) for t in scored) / len(scored) if scored else 0.0
            ),
            "tasks": merged_tasks,
        }
        (out / "eval_summary.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Merged summary: {merged['success_count']}/{merged['scored_tasks']} "
            f"({merged['success_rate']:.1%}), avg_score={merged['avg_score']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
