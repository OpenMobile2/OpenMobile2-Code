"""
Launch N parallel MobileWorld rollouts against N backends.

Example (after `uv run mw env run --count 4` in MobileWorld repo):

  python parallel_rollout_mw.py \
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803 \
    --input_json ../task_synthesis/output/mobileworld_explore/synthesized_tasks_mobileworld_explore_final.json \
    --output_dir runs/mobileworld_explore_rollout \
    --runtime qwen35_session \
    --last_n 3 \
    --model_base_url http://YOUR_VLLM/v1 \
    --model_name YOUR_MODEL

Each worker owns a disjoint sample shard and writes into the same output_dir
(existing dirs are skipped, so restarts are safe).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "run_diy_mw.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hosts",
        type=str,
        required=True,
        help="Comma-separated MobileWorld backend URLs.",
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_HERE / "runs" / "mobileworld_explore_rollout"),
    )
    parser.add_argument(
        "--params_dir",
        type=str,
        default=str(_HERE / "explore_results" / "params"),
    )
    parser.add_argument("--agent_name", type=str, default="qwen3vl")
    parser.add_argument(
        "--runtime",
        type=str,
        default="",
        help="Shared runtime preset (overrides agent_name/memgui/memory when set).",
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
        default="auto",
        choices=["auto", "qwen3vl", "qwen35"],
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=1,
        help="Recent screenshots kept for the model (default 1).",
    )
    parser.add_argument("--switching_weak_model_base_url", type=str, default="")
    parser.add_argument("--switching_weak_model_name", type=str, default="")
    parser.add_argument("--switching_weak_model_api_key", type=str, default="EMPTY")
    args = parser.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        print("No hosts provided", file=sys.stderr)
        return 2

    procs = []
    for i, host in enumerate(hosts):
        cmd = [
            sys.executable,
            str(_SCRIPT),
            f"--aw_host={host}",
            f"--input_json={args.input_json}",
            f"--output_dir={args.output_dir}",
            f"--params_dir={args.params_dir}",
            f"--agent_name={args.agent_name}",
            f"--max_n_steps={args.max_n_steps}",
            f"--model_base_url={args.model_base_url}",
            f"--model_name={args.model_name}",
            f"--model_api_key={args.model_api_key}",
            f"--memgui_prompt_format={args.memgui_prompt_format}",
            f"--last_n={int(args.last_n)}",
            f"--shard_index={i}",
            f"--num_shards={len(hosts)}",
        ]
        if args.runtime:
            cmd.append(f"--runtime={args.runtime}")
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
        print("Launch:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    codes = [p.wait() for p in procs]
    bad = [c for c in codes if c != 0]
    if bad:
        print(f"Some workers failed: exit codes={codes}", file=sys.stderr)
        return 1
    print("All rollout workers finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
