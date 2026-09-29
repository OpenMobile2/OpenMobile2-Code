"""Replay a saved MobileGym hybrid episode in the headed live-console window."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a saved hybrid episode (screenshots + thought/action console).",
    )
    parser.add_argument("episode", nargs="?", type=Path, help="Episode directory or trace.json")
    parser.add_argument("--mock-root", type=Path, default=_DEFAULT_MOCK)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--step-delay", type=float, default=1.1)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    root = args.mock_root.expanduser().resolve()
    if not (root / "bench_env" / "hybrid_benchmark").is_dir():
        raise SystemExit(f"mobilegym-mock not found under {root}")
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    os.chdir(root)

    from bench_env.hybrid_benchmark.replay import main as replay_main

    forwarded: list[str] = []
    if args.episode is not None:
        forwarded.append(str(args.episode))
    if args.run_dir is not None:
        forwarded.extend(["--run-dir", str(args.run_dir)])
    if args.task_id:
        forwarded.extend(["--task-id", args.task_id])
    forwarded.extend([
        "--step-delay", str(args.step_delay),
        "--start-delay", str(args.start_delay),
        "--speed", str(args.speed),
    ])
    if args.loop:
        forwarded.append("--loop")
    if args.no_browser:
        forwarded.append("--no-browser")
    return replay_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
