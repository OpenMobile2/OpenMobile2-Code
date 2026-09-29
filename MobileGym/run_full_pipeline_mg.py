"""
End-to-end MobileGym data pipeline in one script.

Unlike MobileWorld, this pipeline does NOT explore / synthesize tasks.
It uses the prebuilt grounded task bank under MobileGym/tasks/.

Stages (in order):
  1. prepare_tasks        tasks_*.jsonl -> prepared OpenMobile samples JSON
  2. rollout              parallel run_diy_mg against --env_urls
  3. merge                process_trajs.py -> data_merge_success.json
  4. refine_conclusion    process_refine.py --mode conclusion
  5. refine_thinking      process_refine.py --mode thinking
                          -> data_merge_success_conclusion_thinking.json
  6. upload (optional)    if thinking json exists, scp it + success traj folders
                          to /mnt/afs/<user>/data_gym (use --no_upload to skip)

Each invocation writes under:

  MobileGym/pipeline_runs/<run_id>/
    synthesis/prepared_tasks.json
    rollout/
      <sample_id>_MobileGymFreeform/
      data_merge_success.json
      data_merge_success_conclusion.json
      data_merge_success_conclusion_thinking.json
    run_config.json

Examples:

  # Full pipeline (MobileGym must already be serving, e.g. npm run dev)
  python run_full_pipeline_mg.py \\
    --env_urls http://127.0.0.1:3000 \\
    --use_memgui_prompt \\
    --qwen3vl_model_base_url https://<openai-compatible-host>/v1 \\
    --qwen3vl_model_name gemini-3.1-pro-preview \\
    --qwen3vl_model_api_key YOUR_KEY

  # Only first 50 easy tasks
  python run_full_pipeline_mg.py \\
    --env_urls http://127.0.0.1:3000 \\
    --tasks_input tasks/tasks_easy.jsonl \\
    --limit 50 \\
    --stop_after prepare_tasks

  # Resume from rollout of an existing run
  python run_full_pipeline_mg.py --run_id 20260803_150000 --start_from rollout ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MW = _ROOT / "MobileWorld"

STAGES: list[str] = [
    "prepare_tasks",
    "rollout",
    "merge",
    "refine_conclusion",
    "refine_thinking",
]


def _split_urls(raw: str) -> list[str]:
    text = (raw or "").replace("，", ",").replace(";", ",")
    return [h.strip() for h in text.split(",") if h.strip()]


def _run(cmd: Sequence[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("\n" + "=" * 72)
    print("RUN:", " ".join(str(c) for c in cmd))
    if cwd:
        print("CWD:", cwd)
    print("=" * 72, flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=merged)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}")


def _select_stages(start_from: str, stop_after: str) -> list[str]:
    if start_from not in STAGES:
        raise ValueError(f"--start_from must be one of {STAGES}")
    if stop_after not in STAGES:
        raise ValueError(f"--stop_after must be one of {STAGES}")
    i0 = STAGES.index(start_from)
    i1 = STAGES.index(stop_after)
    if i1 < i0:
        raise ValueError("--stop_after is before --start_from")
    return STAGES[i0 : i1 + 1]


def _paths(run_root: Path) -> dict[str, Path]:
    synthesis = run_root / "synthesis"
    rollout = run_root / "rollout"
    return {
        "run_root": run_root,
        "synthesis": synthesis,
        "prepared_tasks": synthesis / "prepared_tasks.json",
        "rollout": rollout,
        "merge_json": rollout / "data_merge_success.json",
        "conclusion_json": rollout / "data_merge_success_conclusion.json",
        "thinking_json": rollout / "data_merge_success_conclusion_thinking.json",
        "config": run_root / "run_config.json",
    }


def stage_prepare_tasks(args: argparse.Namespace, p: dict[str, Path]) -> None:
    p["synthesis"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_HERE / "prepare_tasks.py"),
        f"--input={args.tasks_input}",
        f"--output={p['prepared_tasks']}",
    ]
    if args.limit and args.limit > 0:
        cmd.append(f"--limit={args.limit}")
    if args.difficulty:
        cmd.append(f"--difficulty={args.difficulty}")
    if args.apps_filter:
        cmd.append(f"--apps={args.apps_filter}")
    _run(cmd, cwd=_HERE)
    if not p["prepared_tasks"].exists():
        raise FileNotFoundError(f"Expected prepared tasks at {p['prepared_tasks']}")


def stage_rollout(args: argparse.Namespace, p: dict[str, Path], env_urls: list[str]) -> None:
    p["rollout"].mkdir(parents=True, exist_ok=True)
    if not p["prepared_tasks"].exists():
        raise FileNotFoundError(f"Missing prepared tasks: {p['prepared_tasks']}")
    cmd = [
        sys.executable,
        str(_HERE / "parallel_rollout_mg.py"),
        f"--env_urls={','.join(env_urls)}",
        f"--parallel={int(args.parallel)}",
        f"--input_json={p['prepared_tasks']}",
        f"--output_dir={p['rollout']}",
        f"--agent_name={args.agent_name}",
        f"--max_n_steps={args.max_n_steps}",
        f"--step_wait_time={args.step_wait_time}",
        f"--qwen3vl_model_base_url={args.qwen3vl_model_base_url}",
        f"--qwen3vl_model_name={args.qwen3vl_model_name}",
        f"--qwen3vl_model_api_key={args.qwen3vl_model_api_key}",
        f"--memgui_prompt_format={args.memgui_prompt_format}",
    ]
    if getattr(args, "runtime", ""):
        cmd.append(f"--runtime={args.runtime}")
    if getattr(args, "last_n", None) is not None:
        cmd.append(f"--last_n={int(args.last_n)}")
    if getattr(args, "enable_thinking", None) is not None:
        cmd.append(f"--enable_thinking={str(args.enable_thinking).lower()}")
    if getattr(args, "require_think_tags", None) is not None:
        cmd.append(f"--require_think_tags={str(args.require_think_tags).lower()}")
    if getattr(args, "qwen35_tool_call_mode", None):
        cmd.append(f"--qwen35_tool_call_mode={args.qwen35_tool_call_mode}")
    if args.use_memgui_prompt:
        cmd.append("--use_memgui_prompt")
    if args.use_memory_prompt:
        cmd.append("--use_memory_prompt")
    if args.no_headless:
        cmd.append("--no_headless")
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
    _run(cmd, cwd=_HERE)


def stage_merge(p: dict[str, Path]) -> None:
    script = _HERE / "process_trajs.py"
    if not script.exists():
        script = _MW / "process_trajs.py"
    _run(
        [
            sys.executable,
            str(script),
            f"--runs_dir={p['rollout']}",
            "--output-name=data_merge_success.json",
        ],
        cwd=script.parent,
    )
    if not p["merge_json"].exists():
        raise FileNotFoundError(f"Missing merge output: {p['merge_json']}")


def stage_refine(
    args: argparse.Namespace,
    *,
    input_path: Path,
    output_path: Path,
    mode: str,
) -> None:
    script = _HERE / "process_refine.py"
    if not script.exists():
        script = _MW / "process_refine.py"
    cmd = [
        sys.executable,
        str(script),
        f"--input={input_path}",
        f"--output={output_path}",
        f"--mode={mode}",
        f"--workers={args.refine_workers}",
    ]
    if args.openai_base_url or args.qwen3vl_model_base_url:
        cmd.append(f"--base-url={args.openai_base_url or args.qwen3vl_model_base_url}")
    if args.openai_api_key or args.qwen3vl_model_api_key:
        cmd.append(f"--api-key={args.openai_api_key or args.qwen3vl_model_api_key}")
    if args.openai_model or args.qwen3vl_model_name:
        cmd.append(f"--model={args.openai_model or args.qwen3vl_model_name}")
    _run(cmd, cwd=script.parent)
    if not output_path.exists():
        raise FileNotFoundError(f"Missing refine output: {output_path}")


def stage_upload(args: argparse.Namespace, *, run_id: str, thinking_json: Path) -> None:
    """Upload thinking json + success traj folders when the final artifact exists."""
    if not thinking_json.is_file():
        print(
            f"Skip upload: {thinking_json.name} not found at {thinking_json}",
            flush=True,
        )
        return
    cmd = [
        sys.executable,
        str(_HERE / "upload_success_to_remote.py"),
        f"--run_id={run_id}",
        f"--pipeline_root={args.pipeline_root}",
        f"--remote_user={args.remote_user}",
        f"--remote_host={args.remote_host}",
        f"--remote_port={args.remote_port}",
        f"--remote_base={args.remote_base}",
    ]
    if args.upload_dry_run:
        cmd.append("--dry_run")
    _run(cmd, cwd=_HERE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MobileGym full pipeline: prepare_tasks -> rollout -> merge -> refine."
    )
    parser.add_argument("--run_id", type=str, default="", help="Default: YYYYMMDD_HHMMSS.")
    parser.add_argument(
        "--pipeline_root",
        type=str,
        default=str(_HERE / "pipeline_runs"),
        help="Root directory for all runs.",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default="",
        help="Single MobileGym frontend URL (preferred). Combine with --parallel N.",
    )
    parser.add_argument(
        "--env_urls",
        type=str,
        default="",
        help="Comma-separated MobileGym frontend URLs (legacy). Default falls back to --env_url or :4173.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Playwright worker count. With one URL, expands to N identical shards (state is per-browser).",
    )
    parser.add_argument("--start_from", type=str, default="prepare_tasks", choices=STAGES)
    parser.add_argument("--stop_after", type=str, default="refine_thinking", choices=STAGES)

    # prepare_tasks
    parser.add_argument(
        "--tasks_input",
        type=str,
        default=str(_HERE / "tasks" / "tasks_5000.jsonl"),
        help="Prebuilt MobileGym tasks json/jsonl.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max tasks (0=all).")
    parser.add_argument("--difficulty", type=str, default="", help="easy|medium|hard filter.")
    parser.add_argument("--apps_filter", type=str, default="", help="Comma-separated app filter.")

    # rollout
    parser.add_argument("--agent_name", type=str, default="qwen3vl")
    parser.add_argument(
        "--runtime",
        type=str,
        default="",
        help="Shared runtime preset (overrides agent_name/memgui/memory when set).",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=1,
        help="Recent screenshots kept for the model (default 1).",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str,
        default="true",
        help="true/false. Session chat_template_kwargs.enable_thinking.",
    )
    parser.add_argument(
        "--require_think_tags",
        type=str,
        default="true",
        help="true/false. If false, use tool_call-only prompt.",
    )
    parser.add_argument(
        "--qwen35_tool_call_mode",
        type=str,
        default="native",
        choices=["xml", "native"],
    )
    parser.add_argument("--max_n_steps", type=int, default=30)
    parser.add_argument("--step_wait_time", type=float, default=1.0)
    parser.add_argument("--no_headless", action="store_true")
    parser.add_argument("--qwen3vl_model_base_url", type=str, default="http://<openai-compatible-host>/v1")
    parser.add_argument("--qwen3vl_model_name", type=str, default="")
    parser.add_argument("--qwen3vl_model_api_key", type=str, default="EMPTY")
    parser.add_argument("--use_memgui_prompt", action="store_true")
    parser.add_argument("--use_memory_prompt", action="store_true")
    parser.add_argument(
        "--memgui_prompt_format",
        type=str,
        default="auto",
        choices=["auto", "qwen3vl", "qwen35"],
    )
    parser.add_argument("--qwen3vl_switching_weak_model_base_url", type=str, default="")
    parser.add_argument("--qwen3vl_switching_weak_model_name", type=str, default="")
    parser.add_argument("--qwen3vl_switching_weak_model_api_key", type=str, default="EMPTY")

    # refine
    parser.add_argument("--refine_workers", type=int, default=16)
    parser.add_argument("--openai_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai_model", type=str, default=os.getenv("OPENAI_MODEL", ""))

    # upload (after all stages; only runs if thinking json exists)
    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Do not upload success data to the remote archive after the pipeline.",
    )
    parser.add_argument(
        "--upload_dry_run",
        action="store_true",
        help="Plan remote upload without transferring files.",
    )
    parser.add_argument("--remote_user", type=str, default="<user>")
    parser.add_argument("--remote_host", type=str, default="<gateway-host>")
    parser.add_argument("--remote_port", type=int, default=22)
    parser.add_argument(
        "--remote_base",
        type=str,
        default="/mnt/afs/<user>/data_gym",
    )

    args = parser.parse_args()
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    env_urls = _split_urls(args.env_urls)
    if not env_urls and args.env_url.strip():
        env_urls = [args.env_url.strip().rstrip("/")]
    if not env_urls:
        # Match common local preview default (vite preview --port 4173).
        env_urls = ["http://127.0.0.1:4173"]
    if int(args.parallel) < 1:
        print("--parallel must be >= 1", file=sys.stderr)
        return 2

    stages = _select_stages(args.start_from, args.stop_after)
    if "rollout" in stages and not env_urls:
        print("--env_url or --env_urls required for rollout", file=sys.stderr)
        return 2

    run_root = Path(args.pipeline_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    p = _paths(run_root)

    config = {
        "run_id": run_id,
        "env_urls": env_urls,
        "parallel": int(args.parallel),
        "stages": stages,
        "args": {k: v for k, v in vars(args).items() if "api_key" not in k},
        "paths": {k: str(v) for k, v in p.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "No explore/synthesize — uses prebuilt MobileGym/tasks",
    }
    p["config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run directory: {run_root}")
    print(f"Stages: {' -> '.join(stages)}")

    try:
        for stage in stages:
            print(f"\n######## STAGE: {stage} ########")
            if stage == "prepare_tasks":
                stage_prepare_tasks(args, p)
            elif stage == "rollout":
                stage_rollout(args, p, env_urls)
            elif stage == "merge":
                stage_merge(p)
            elif stage == "refine_conclusion":
                stage_refine(
                    args,
                    input_path=p["merge_json"],
                    output_path=p["conclusion_json"],
                    mode="conclusion",
                )
            elif stage == "refine_thinking":
                stage_refine(
                    args,
                    input_path=p["conclusion_json"],
                    output_path=p["thinking_json"],
                    mode="thinking",
                )
    except Exception as e:
        print(f"\nPIPELINE FAILED at run_id={run_id}: {e!r}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print("DONE")
    print(f"run_id : {run_id}")
    print(f"root   : {run_root}")
    print(f"final  : {p['thinking_json']}")
    print("=" * 72)

    if not args.no_upload:
        print("\n######## STAGE: upload ########")
        try:
            stage_upload(args, run_id=run_id, thinking_json=p["thinking_json"])
        except Exception as e:
            print(f"\nUPLOAD FAILED at run_id={run_id}: {e!r}", file=sys.stderr)
            return 1
    else:
        print("\nSkip upload (--no_upload).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
