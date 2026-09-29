"""MobileGym official benchmark eval with OpenMobile session runtimes.

This is scoring eval (SR/PR/FC via bench_env judges), not data rollout.

Example:
  conda activate android_world
  cd <OPENMOBILE_ROOT>\\MobileGym
  python run_eval_mg.py --env-url http://127.0.0.1:4173 --split test --parallel 2

  python run_eval_mg.py --agent qwen35_thought_session --last_n 3 --model-name YOUR_MODEL ...
  python run_eval_mg.py --agent qwen3vl --model-name OpenMobile-8B ...
  python run_eval_mg.py --agent venus --model-name UI-Venus-1.5-8B ...

  # Fixed output folder (resume with --resume <same path>):
  python run_eval_mg.py ... --run-dir runs/YOUR_MODEL
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows consoles are often GBK; force UTF-8 so model print()/logs are visible.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_AW_ROOT = _HERE.parent / "AndroidWorld"
_MG_FRONTEND = Path(
    os.environ.get(
        "MOBILEGYM_FRONTEND",
        str(_HERE.parent.parent / "mobilegym" / "mobilegym"),
    )
)

for p in (_AW_ROOT, _MG_FRONTEND, _HERE):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

from bench_env.agent import register_agent  # noqa: E402
from bench_env.config import RunnerConfig  # noqa: E402
from bench_env.logger import configure_logging  # noqa: E402
from bench_env.run import async_main, create_parser  # noqa: E402
from openmobile_qwen3vl_agent import OpenMobileQwen3VLBenchAgent  # noqa: E402
from openmobile_session_agent import OpenMobileSessionBenchAgent, SESSION_CFG  # noqa: E402

register_agent("qwen35_session", OpenMobileSessionBenchAgent)
register_agent("qwen35_thought_session", OpenMobileSessionBenchAgent)
register_agent("qwen3vl", OpenMobileQwen3VLBenchAgent)

_ORIG_FROM_ARGS = RunnerConfig.from_args


def _patch_recorder_append_existing() -> None:
    """If --run-dir already has results.jsonl, append instead of wiping it.

    bench_env RunRecorder.start_run always opens results.jsonl with \"w\".
    OpenMobile resume writes pending tasks into the same folder, so we keep
    existing rows and meta.json.
    """
    from bench_env.env.recorder import RunRecorder

    orig = RunRecorder.start_run

    def start_run(self, agent="", model_name="", extra_meta=None, repeat_n=1):
        dest = self.fixed_run_dir
        results_path = dest / "results.jsonl" if dest else None
        append = bool(
            results_path is not None
            and results_path.is_file()
            and results_path.stat().st_size > 0
        )
        if not append:
            return orig(
                self,
                agent=agent,
                model_name=model_name,
                extra_meta=extra_meta,
                repeat_n=repeat_n,
            )

        self._run_start_time = datetime.now()
        self._repeat_n = repeat_n
        self._run_dir = dest
        self._run_dir.mkdir(parents=True, exist_ok=True)
        if self.save_trajectory:
            self._trajectory_dir = self.trajectory_dir_override or (
                self._run_dir / "trajectory"
            )
            self._trajectory_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = (self._run_dir / "results.jsonl").open("a", encoding="utf-8")
        self._errors_file = (self._run_dir / "errors.jsonl").open("a", encoding="utf-8")
        meta_path = self._run_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st = meta.get("start_time")
                if st:
                    self._run_start_time = datetime.fromisoformat(str(st))
            except Exception:
                pass
        print(f"[resume] appending into existing {self._run_dir}", flush=True)
        return self._run_dir

    RunRecorder.start_run = start_run  # type: ignore[method-assign]


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"expected true/false, got {value!r}")


@classmethod  # type: ignore[misc]
def _from_args_with_run_dir(cls, args):  # noqa: N805
    cfg = _ORIG_FROM_ARGS(args)
    run_dir = getattr(args, "run_dir", None)
    if run_dir:
        cfg.run_dir = Path(str(run_dir)).expanduser().resolve()
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    parser.set_defaults(agent="qwen35_thought_session", split="test")
    parser.add_argument("--last_n", type=int, default=3)
    parser.add_argument("--enable_thinking", type=_parse_bool, default=True)
    parser.add_argument("--require_think_tags", type=_parse_bool, default=False)
    parser.add_argument(
        "--qwen35_tool_call_mode",
        type=str,
        default="native",
        choices=["xml", "native"],
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Fixed run directory (otherwise bench_env uses runs/<timestamp>).",
    )
    args = parser.parse_args(argv)
    if args.agent in {"venus", "qwen3vl"} and int(args.last_n) != 1:
        print(
            f"[bench] {args.agent} locks last_n=1 (got {args.last_n})",
            flush=True,
        )
        args.last_n = 1

    SESSION_CFG.update(
        {
            "runtime": str(args.agent or "qwen35_session"),
            "last_n": int(args.last_n),
            "enable_thinking": bool(args.enable_thinking),
            "require_think_tags": bool(args.require_think_tags),
            "qwen35_tool_call_mode": str(args.qwen35_tool_call_mode),
            "model_base_url": args.model_base_url or SESSION_CFG["model_base_url"],
            "model_api_key": args.model_api_key or "EMPTY",
            "model_name": args.model_name or SESSION_CFG["model_name"],
        }
    )
    if not args.model_base_url:
        args.model_base_url = SESSION_CFG["model_base_url"]
    if not args.model_name:
        args.model_name = SESSION_CFG["model_name"]
    if not args.model_api_key:
        args.model_api_key = "EMPTY"

    if args.run_dir:
        RunnerConfig.from_args = _from_args_with_run_dir  # type: ignore[method-assign]
        _patch_recorder_append_existing()
        print(f"Fixed run dir: {Path(args.run_dir).expanduser().resolve()}")

    configure_logging(quiet=args.quiet)
    print(
        "MobileGym EVAL (not rollout)  "
        f"split={args.split} agent={args.agent} last_n={args.last_n} "
        f"thinking={args.enable_thinking} require_think_tags={args.require_think_tags} "
        f"mode={args.qwen35_tool_call_mode} model={args.model_name}",
        flush=True,
    )
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[Interrupted]", flush=True)
        return 130
    finally:
        RunnerConfig.from_args = _ORIG_FROM_ARGS  # type: ignore[method-assign]


if __name__ == "__main__":
    raise SystemExit(main())
