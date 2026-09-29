"""
Launch N parallel MobileWorld explorers against N backends.

Example (after `uv run mw env run --count 4` in MobileWorld repo):

  python parallel_explore_mw.py \
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803 \
    --output_dir explore_results \
    --num_step 30

Each worker writes into the same explore_results/ tree (unique UUIDs avoid clashes).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "random_walk_mw.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hosts",
        type=str,
        required=True,
        help="Comma-separated MobileWorld backend URLs.",
    )
    parser.add_argument("--output_dir", type=str, default=str(_HERE / "explore_results"))
    parser.add_argument("--num_step", type=int, default=10)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--step_wait_time", type=float, default=0.6)
    parser.add_argument("--post_action_sleep", type=float, default=0.8)
    parser.add_argument("--enable_mcp", action="store_true")
    parser.add_argument("--enable_user_interaction", action="store_true")
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
            f"--output_dir={args.output_dir}",
            f"--num_step={args.num_step}",
            f"--max_tasks={args.max_tasks}",
            f"--step_wait_time={args.step_wait_time}",
            f"--post_action_sleep={args.post_action_sleep}",
            f"--shard_index={i}",
            f"--num_shards={len(hosts)}",
        ]
        if args.enable_mcp:
            cmd.append("--enable_mcp")
        if args.enable_user_interaction:
            cmd.append("--enable_user_interaction")
        print("Launch:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    codes = [p.wait() for p in procs]
    bad = [c for c in codes if c != 0]
    if bad:
        print(f"Some workers failed: exit codes={codes}", file=sys.stderr)
        return 1
    print("All explore workers finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
