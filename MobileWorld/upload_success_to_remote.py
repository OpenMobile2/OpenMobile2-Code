"""
Upload refined success data to the remote archive.

After a pipeline run finishes, this script:
  1. Checks for data_merge_success_conclusion_thinking.json
  2. Collects every successful trajectory folder referenced by that JSON
  3. Picks the next numeric remote directory (0001, 0002, ...) to avoid clashes
  4. Uploads the JSON + trajectory folders over SSH/SCP

Defaults:
  host  <gateway-host>
  port  <ssh-port>
  user  <user>
  path  /mnt/afs/<user>/data_mobile

Examples (PowerShell):

  # Upload one finished run
  python upload_success_to_remote.py --run_id 20260803_171524

  # Explicit rollout directory
  python upload_success_to_remote.py `
    --rollout_dir pipeline_runs/20260803_171524/rollout

  # Dry-run (list what would be uploaded)
  python upload_success_to_remote.py --run_id 20260803_171524 --dry_run
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEFAULT_REMOTE_USER = "<user>"
DEFAULT_REMOTE_HOST = "<gateway-host>"
DEFAULT_REMOTE_PORT = 22  # set your SSH port
DEFAULT_REMOTE_BASE = "/mnt/afs/<user>/data_mobile"
THINKING_NAME = "data_merge_success_conclusion_thinking.json"
NUM_DIR_RE = re.compile(r"^(\d+)$")


def _ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-p",
        str(args.remote_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{args.remote_user}@{args.remote_host}",
    ]


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print(proc.stderr.rstrip(), file=sys.stderr)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _resolve_rollout_dir(args: argparse.Namespace) -> Path:
    if args.rollout_dir:
        return Path(args.rollout_dir).expanduser().resolve()
    if args.run_id:
        return (Path(args.pipeline_root) / args.run_id / "rollout").resolve()
    raise ValueError("Provide --run_id or --rollout_dir")


def _load_success_dirs(thinking_json: Path) -> list[str]:
    data = json.loads(thinking_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {thinking_json}")
    names: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("save_dir") or "").strip()
        if not name or name in seen:
            continue
        # Accept absolute paths written by older merges; keep basename only.
        name = Path(name).name
        seen.add(name)
        names.append(name)
    return names


def _next_remote_index(args: argparse.Namespace) -> int:
    """List remote base dir and return max existing numeric prefix + 1."""
    remote = shlex.quote(args.remote_base.rstrip("/"))
    proc = _run(
        _ssh_base(args)
        + [
            f"mkdir -p {remote} && "
            f"ls -1 {remote} 2>/dev/null || true"
        ],
        check=True,
    )
    max_idx = 0
    for line in (proc.stdout or "").splitlines():
        name = line.strip().rstrip("/")
        if not name:
            continue
        # Match "0001", "0001_xxx", "12_runid", etc.
        m = re.match(r"^(\d+)(?:_|$)", name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _remote_dest_name(idx: int, run_id: str) -> str:
    prefix = f"{idx:04d}"
    rid = (run_id or "").strip()
    return f"{prefix}_{rid}" if rid else prefix


def upload(args: argparse.Namespace) -> str:
    rollout = _resolve_rollout_dir(args)
    thinking = rollout / THINKING_NAME
    if not thinking.is_file():
        raise FileNotFoundError(
            f"Missing {THINKING_NAME} under {rollout}. "
            "Finish refine_thinking before uploading."
        )

    success_dirs = _load_success_dirs(thinking)
    missing = [d for d in success_dirs if not (rollout / d).is_dir()]
    if missing:
        preview = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"{len(missing)} success traj folder(s) missing under {rollout}: "
            f"{preview}{more}"
        )

    run_id = args.run_id.strip() if args.run_id else rollout.parent.name
    idx = _next_remote_index(args)
    dest_name = _remote_dest_name(idx, run_id)
    remote_dir = f"{args.remote_base.rstrip('/')}/{dest_name}"

    print(f"rollout        : {rollout}")
    print(f"thinking json  : {thinking}")
    print(f"success trajs  : {len(success_dirs)}")
    print(f"remote dest    : {args.remote_user}@{args.remote_host}:{remote_dir}")

    if args.dry_run:
        print("[dry_run] would upload:")
        print(f"  {THINKING_NAME}")
        for d in success_dirs[:20]:
            print(f"  {d}/")
        if len(success_dirs) > 20:
            print(f"  ... and {len(success_dirs) - 20} more folders")
        return remote_dir

    _run(_ssh_base(args) + [f"mkdir -p {shlex.quote(remote_dir)}"])

    # Upload JSON first, then each traj folder. scp -r is widely available on Win/Linux.
    scp_base = [
        "scp",
        "-P",
        str(args.remote_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    remote_target = f"{args.remote_user}@{args.remote_host}:{remote_dir}/"

    _run(scp_base + [str(thinking), remote_target])

    # Batch folders in chunks to keep command lines reasonable on Windows.
    chunk_size = max(1, int(args.scp_chunk_size))
    for i in range(0, len(success_dirs), chunk_size):
        chunk = success_dirs[i : i + chunk_size]
        local_paths = [str(rollout / d) for d in chunk]
        print(
            f"Uploading traj folders {i + 1}-{i + len(chunk)} / {len(success_dirs)} ...",
            flush=True,
        )
        _run(scp_base + ["-r", *local_paths, remote_target])

    # Small marker for bookkeeping on the remote side.
    marker = {
        "local_run_id": run_id,
        "local_rollout": str(rollout),
        "remote_dir": remote_dir,
        "thinking_json": THINKING_NAME,
        "num_success_trajs": len(success_dirs),
        "success_dirs": success_dirs,
    }
    marker_path = rollout / f"_upload_manifest_{dest_name}.json"
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
    _run(scp_base + [str(marker_path), remote_target])

    print(f"UPLOAD OK -> {remote_dir}")
    return remote_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload data_merge_success_conclusion_thinking.json "
        "and success trajectory folders to the remote archive."
    )
    parser.add_argument("--run_id", type=str, default="", help="pipeline_runs/<run_id>")
    parser.add_argument(
        "--pipeline_root",
        type=str,
        default=str(_HERE / "pipeline_runs"),
        help="Root directory for pipeline runs.",
    )
    parser.add_argument(
        "--rollout_dir",
        type=str,
        default="",
        help="Direct path to a rollout/ directory (overrides --run_id).",
    )
    parser.add_argument("--remote_user", type=str, default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote_host", type=str, default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote_port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote_base", type=str, default=DEFAULT_REMOTE_BASE)
    parser.add_argument(
        "--scp_chunk_size",
        type=int,
        default=20,
        help="How many traj folders to scp per batch.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the planned remote destination and file list.",
    )
    args = parser.parse_args()

    try:
        upload(args)
    except Exception as e:
        print(f"UPLOAD FAILED: {e!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
