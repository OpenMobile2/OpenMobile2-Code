"""
End-to-end MobileWorld data pipeline in one script.

Stages (in order):
  1. explore              parallel random_walk_mw against --hosts
  2. process_explore      trajectories -> state_transfer_explore.json
  3. synthesize           task_synthesis/pipeline.py -> *_final.json
  4. rollout              parallel run_diy_mw against --hosts
  5. merge                process_trajs.py -> data_merge_success.json
  6. refine_conclusion    process_refine.py --mode conclusion
  7. refine_thinking      process_refine.py --mode thinking
                          -> data_merge_success_conclusion_thinking.json
  8. upload (optional)    if thinking json exists, scp it + success traj folders
                          to <user>@<gateway-host>:/mnt/afs/<user>/data_mobile/<NNNN>_<run_id>

Each invocation writes under an isolated run directory so re-runs do not clash:

  MobileWorld/pipeline_runs/<run_id>/
    explore/
      screenshots/ trajectories/ params/ ...
      state_transfer_explore.json
    synthesis/
      synthesized_tasks_<run_id>_final.json
    rollout/
      <sample_id>_<task>/
      data_merge_success.json
      data_merge_success_conclusion.json
      data_merge_success_conclusion_thinking.json
    run_config.json
    run.log

Examples (PowerShell):

  # Full pipeline (session runtime), auto run_id = timestamp
  python run_full_pipeline_mw.py `
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802 `
    --runtime qwen35_session `
    --last_n 3 `
    --stop_after rollout `
    --model_base_url https://<openai-compatible-host>/v1 `
    --model_name gemini-3.1-pro-preview `
    --model_api_key YOUR_KEY

  # Resume from rollout of an existing run
  python run_full_pipeline_mw.py --run_id 20260803_143000 --start_from rollout ...

  # Only explore + process_explore
  python run_full_pipeline_mw.py --hosts ... --stop_after process_explore
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
_AW = _ROOT / "AndroidWorld"
_SYN = _ROOT / "task_synthesis"

STAGES: list[str] = [
    "explore",
    "process_explore",
    "synthesize",
    "rollout",
    "merge",
    "refine_conclusion",
    "refine_thinking",
]


def _split_hosts(raw: str) -> list[str]:
    # Accept English/Chinese commas and whitespace.
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


def _paths(run_root: Path, run_id: str) -> dict[str, Path]:
    explore = run_root / "explore"
    # pipeline.py writes to outputs_root/dataset_id/...
    synthesis_root = run_root / "synthesis"
    synthesis = synthesis_root / run_id
    rollout = run_root / "rollout"
    return {
        "run_root": run_root,
        "explore": explore,
        "traj_dir": explore / "trajectories",
        "screenshots_dir": explore / "screenshots",
        "params_dir": explore / "params",
        "state_transfer": explore / "state_transfer_explore.json",
        "synthesis_root": synthesis_root,
        "synthesis": synthesis,
        "final_tasks": synthesis / f"synthesized_tasks_{run_id}_final.json",
        "rollout": rollout,
        "merge_json": rollout / "data_merge_success.json",
        "conclusion_json": rollout / "data_merge_success_conclusion.json",
        "thinking_json": rollout / "data_merge_success_conclusion_thinking.json",
        "config": run_root / "run_config.json",
    }


def stage_explore(args: argparse.Namespace, p: dict[str, Path], hosts: list[str]) -> None:
    p["explore"].mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_HERE / "parallel_explore_mw.py"),
        f"--hosts={','.join(hosts)}",
        f"--output_dir={p['explore']}",
        f"--num_step={args.num_step}",
        f"--max_tasks={args.max_tasks}",
        f"--step_wait_time={args.step_wait_time}",
        f"--post_action_sleep={args.post_action_sleep}",
    ]
    if args.enable_mcp:
        cmd.append("--enable_mcp")
    if args.enable_user_interaction:
        cmd.append("--enable_user_interaction")
    env = {}
    if args.openai_base_url:
        env["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_model:
        env["OPENAI_MODEL"] = args.openai_model
        env["OPENAI_TEXT_INPUT_MODEL"] = args.openai_model
    _run(cmd, cwd=_HERE, env=env or None)


def stage_process_explore(p: dict[str, Path]) -> None:
    traj_dir = p["traj_dir"]
    if not traj_dir.exists():
        raise FileNotFoundError(f"Missing trajectories dir: {traj_dir}")
    _run(
        [
            sys.executable,
            str(_AW / "process_explore.py"),
            f"--traj_dir={traj_dir}",
            f"--out={p['state_transfer']}",
            "--indent=2",
        ],
        cwd=_AW,
    )


def stage_synthesize(args: argparse.Namespace, p: dict[str, Path], run_id: str) -> None:
    p["synthesis"].mkdir(parents=True, exist_ok=True)
    if not p["state_transfer"].exists():
        raise FileNotFoundError(f"Missing state_transfer: {p['state_transfer']}")
    cmd = [
        sys.executable,
        str(_SYN / "pipeline.py"),
        f"--dataset_id={run_id}",
        f"--outputs_root={p['synthesis_root']}",
        f"--state_transfer={p['state_transfer']}",
        f"--screenshots_dir={p['screenshots_dir']}",
        f"--max_num_syn_screen={args.max_num_syn_screen}",
        f"--max_workers={args.syn_max_workers}",
        f"--context_embedding_model={args.context_embedding_model}",
        f"--prompt_suite={args.prompt_suite}",
    ]
    env = {}
    if args.openai_base_url:
        env["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_model:
        env["OPENAI_MODEL"] = args.openai_model
        # Explore uses this for editable-field typing; keep it aligned with synthesis model.
        env.setdefault("OPENAI_TEXT_INPUT_MODEL", args.openai_model)
    _run(cmd, cwd=_SYN, env=env or None)
    if not p["final_tasks"].exists():
        raise FileNotFoundError(f"Expected final tasks at {p['final_tasks']}")


def stage_rollout(args: argparse.Namespace, p: dict[str, Path], hosts: list[str]) -> None:
    p["rollout"].mkdir(parents=True, exist_ok=True)
    if not p["final_tasks"].exists():
        raise FileNotFoundError(f"Missing final tasks: {p['final_tasks']}")
    cmd = [
        sys.executable,
        str(_HERE / "parallel_rollout_mw.py"),
        f"--hosts={','.join(hosts)}",
        f"--input_json={p['final_tasks']}",
        f"--output_dir={p['rollout']}",
        f"--params_dir={p['params_dir']}",
        f"--agent_name={args.agent_name}",
        f"--max_n_steps={args.max_n_steps}",
        f"--model_base_url={args.model_base_url}",
        f"--model_name={args.model_name}",
        f"--model_api_key={args.model_api_key}",
        f"--memgui_prompt_format={args.memgui_prompt_format}",
    ]
    if getattr(args, "runtime", ""):
        cmd.append(f"--runtime={args.runtime}")
    if getattr(args, "last_n", None) is not None:
        cmd.append(f"--last_n={int(args.last_n)}")
    if args.use_memgui_prompt:
        cmd.append("--use_memgui_prompt")
    if args.use_memory_prompt:
        cmd.append("--use_memory_prompt")
    if args.switching_weak_model_base_url:
        cmd.append(
            f"--switching_weak_model_base_url={args.switching_weak_model_base_url}"
        )
    if args.switching_weak_model_name:
        cmd.append(
            f"--switching_weak_model_name={args.switching_weak_model_name}"
        )
    if args.switching_weak_model_api_key:
        cmd.append(
            f"--switching_weak_model_api_key={args.switching_weak_model_api_key}"
        )
    _run(cmd, cwd=_HERE)


def stage_merge(p: dict[str, Path]) -> None:
    _run(
        [
            sys.executable,
            str(_HERE / "process_trajs.py"),
            f"--runs_dir={p['rollout']}",
            "--output-name=data_merge_success.json",
        ],
        cwd=_HERE,
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
    cmd = [
        sys.executable,
        str(_HERE / "process_refine.py"),
        f"--input={input_path}",
        f"--output={output_path}",
        f"--mode={mode}",
        f"--workers={args.refine_workers}",
    ]
    if args.openai_base_url or args.model_base_url:
        cmd.append(f"--base-url={args.openai_base_url or args.model_base_url}")
    if args.openai_api_key or args.model_api_key:
        cmd.append(f"--api-key={args.openai_api_key or args.model_api_key}")
    if args.openai_model or args.model_name:
        cmd.append(f"--model={args.openai_model or args.model_name}")
    _run(cmd, cwd=_HERE)
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
        description="MobileWorld full pipeline: explore -> synthesize -> rollout -> refine."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Unique id for this run. Default: YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--pipeline_root",
        type=str,
        default=str(_HERE / "pipeline_runs"),
        help="Root directory for all runs.",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default="http://127.0.0.1:6800",
        help="Comma-separated MobileWorld backends (English commas).",
    )
    parser.add_argument("--start_from", type=str, default="explore", choices=STAGES)
    parser.add_argument("--stop_after", type=str, default="refine_thinking", choices=STAGES)

    # explore
    parser.add_argument("--num_step", type=int, default=10)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--step_wait_time", type=float, default=0.6)
    parser.add_argument("--post_action_sleep", type=float, default=0.8)
    parser.add_argument("--enable_mcp", action="store_true")
    parser.add_argument("--enable_user_interaction", action="store_true")

    # synthesize
    parser.add_argument("--max_num_syn_screen", type=int, default=1000)
    parser.add_argument("--syn_max_workers", type=int, default=64)
    parser.add_argument("--context_embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--prompt_suite",
        type=str,
        default="mobileworld",
        choices=["androidworld", "mobileworld", "aw", "mw"],
        help="Task synthesis/judge prompt suite (default mobileworld).",
    )
    parser.add_argument("--openai_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai_model", type=str, default=os.getenv("OPENAI_MODEL", ""))

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
    parser.add_argument("--max_n_steps", type=int, default=30)
    parser.add_argument("--model_base_url", type=str, default="http://<openai-compatible-host>/v1")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_api_key", type=str, default="EMPTY")
    parser.add_argument("--use_memgui_prompt", action="store_true")
    parser.add_argument("--use_memory_prompt", action="store_true")
    parser.add_argument(
        "--memgui_prompt_format",
        type=str,
        default="qwen3vl",
        choices=["auto", "qwen3vl", "qwen35"],
    )
    parser.add_argument("--switching_weak_model_base_url", type=str, default="")
    parser.add_argument("--switching_weak_model_name", type=str, default="")
    parser.add_argument("--switching_weak_model_api_key", type=str, default="EMPTY")

    # refine
    parser.add_argument("--refine_workers", type=int, default=16)

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
        default="/mnt/afs/<user>/data_mobile",
    )

    args = parser.parse_args()
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    hosts = _split_hosts(args.hosts)
    if not hosts:
        print("No hosts provided", file=sys.stderr)
        return 2

    need_env = any(
        s in _select_stages(args.start_from, args.stop_after)
        for s in ("explore", "rollout")
    )
    if need_env and len(hosts) < 1:
        print("--hosts required for explore/rollout", file=sys.stderr)
        return 2

    run_root = Path(args.pipeline_root) / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    p = _paths(run_root, run_id)
    stages = _select_stages(args.start_from, args.stop_after)

    config = {
        "run_id": run_id,
        "hosts": hosts,
        "stages": stages,
        "args": {k: v for k, v in vars(args).items() if "api_key" not in k},
        "paths": {k: str(v) for k, v in p.items()},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    p["config"].write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run directory: {run_root}")
    print(f"Stages: {' -> '.join(stages)}")

    try:
        for stage in stages:
            print(f"\n######## STAGE: {stage} ########")
            if stage == "explore":
                stage_explore(args, p, hosts)
            elif stage == "process_explore":
                stage_process_explore(p)
            elif stage == "synthesize":
                stage_synthesize(args, p, run_id)
            elif stage == "rollout":
                stage_rollout(args, p, hosts)
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
