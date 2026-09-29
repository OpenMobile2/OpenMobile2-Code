"""In-place resume for MobileGym — OpenMobile-Code only.

Does NOT use bench_env --resume (no .resume_tmp inside the run folder).
Does NOT modify <MOBILEGYM_FRONTEND>.

Flow:
  1) Read target run's meta.json + results.jsonl
  2) Find pending / partial / error task ids (errors are retried)
  3) Drop error/partial rows that will be retried
  4) Run those task ids with --run-dir pointed at the original folder
     (run_eval_mg.py appends results.jsonl; does not wipe it)
  5) Rewrite summary.json / errors.jsonl from the full jsonl

Example:
  python resume_eval_mg.py --run-dir runs/YOUR_MODEL --env-url http://127.0.0.1:4173 ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
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

from bench_env import factory  # noqa: E402
from bench_env.config import RunnerConfig  # noqa: E402
from bench_env.metrics import (  # noqa: E402
    load_jsonl,
    result_is_error,
    result_is_success,
    result_key,
    write_errors_jsonl,
    write_summary_json,
)  # noqa: E402
from bench_env.rerun import merge_results  # noqa: E402

_RUN_EVAL = _HERE / "run_eval_mg.py"
_WORK_ROOT = _HERE / "runs" / "_resume_work"


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _pending_task_ids(run_dir: Path, meta: dict) -> tuple[list[str], list[str], int]:
    """Return (pending_ids, partial_ids, repeat_n)."""
    repeat_n = int(meta.get("repeat_n") or 1)
    existing = load_jsonl(run_dir / "results.jsonl")
    recorded = {result_key(r) for r in existing}

    config = RunnerConfig.from_meta(meta)
    tasks = factory.load_tasks(config)

    pending: list[str] = []
    partial: list[str] = []
    for task in tasks:
        keys = [f"{task.id}__t{t}" for t in range(repeat_n)]
        n = sum(1 for k in keys if k in recorded)
        if n == 0:
            pending.append(task.id)
        elif n < repeat_n:
            partial.append(task.id)
    return pending, partial, repeat_n


def _error_task_ids(run_dir: Path, meta: dict) -> list[str]:
    """Task ids that recorded an exec/judge error and should be retried."""
    config = RunnerConfig.from_meta(meta)
    valid = {task.id for task in factory.load_tasks(config)}
    ids: list[str] = []
    seen: set[str] = set()
    for row in load_jsonl(run_dir / "results.jsonl"):
        if not result_is_error(row):
            continue
        tid = str(row.get("id") or "").strip()
        if tid and tid in valid and tid not in seen:
            seen.add(tid)
            ids.append(tid)
    return ids


def _dedupe_results(rows: list[dict]) -> list[dict]:
    by_key: dict[str, dict] = {}
    for row in rows:
        by_key[result_key(row)] = row
    return list(by_key.values())


def _overall_progress(run_dir: Path, meta: dict) -> dict[str, int]:
    """Counts for the full split, not just the resume batch."""
    repeat_n = int(meta.get("repeat_n") or 1)
    config = RunnerConfig.from_meta(meta)
    tasks = factory.load_tasks(config)
    rows = _dedupe_results(load_jsonl(run_dir / "results.jsonl"))
    success = sum(1 for r in rows if result_is_success(r))
    error = sum(1 for r in rows if result_is_error(r))
    return {
        "total": len(tasks) * repeat_n,
        "offset": len(rows),
        "success": success,
        "fail": len(rows) - success,
        "error": error,
        "failed": len(rows) - success - error,
    }


def _apply_progress_env(progress: dict[str, int]) -> None:
    os.environ["MG_EVAL_PROGRESS_TOTAL"] = str(progress["total"])
    os.environ["MG_EVAL_PROGRESS_OFFSET"] = str(progress["offset"])
    os.environ["MG_EVAL_PROGRESS_SUCCESS"] = str(progress["success"])
    os.environ["MG_EVAL_PROGRESS_FAIL"] = str(progress["fail"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows),
        encoding="utf-8",
    )


def _drop_result_keys(run_dir: Path, keys: set[str]) -> int:
    """Remove rows that are about to be retried so append does not duplicate them."""
    path = run_dir / "results.jsonl"
    if not keys or not path.is_file():
        return 0
    rows = load_jsonl(path)
    kept = [r for r in rows if result_key(r) not in keys]
    dropped = len(rows) - len(kept)
    if dropped:
        _write_jsonl(path, kept)
    return dropped


def _salvage_resume_work(run_dir: Path, meta: dict) -> int:
    """Merge leftover runs/_resume_work/<name>_* dirs from the old temp-dir flow."""
    if not _WORK_ROOT.is_dir():
        return 0
    repeat_n = int(meta.get("repeat_n") or 1)
    pass_k = meta.get("pass_k")
    merged = 0
    prefix = f"{run_dir.name}_"
    for child in sorted(p for p in _WORK_ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        src_results = child / "results.jsonl"
        if not src_results.is_file() or src_results.stat().st_size == 0:
            continue
        rows = load_jsonl(src_results)
        if not rows:
            continue
        keys = {result_key(r) for r in rows}
        existing_keys = {result_key(r) for r in load_jsonl(run_dir / "results.jsonl")}
        if keys <= existing_keys:
            shutil.rmtree(child, ignore_errors=True)
            print(f"[resume] leftover {child.name} already in original dir, removed")
            continue
        merge_results(run_dir, child, keys, repeat_n, pass_k)
        shutil.rmtree(child, ignore_errors=True)
        merged += len(rows)
        print(f"[resume] salvaged {len(rows)} episodes from {child.name}")
    return merged


def _merge_legacy_resume_tmp(run_dir: Path, meta: dict) -> int:
    """One-shot salvage of leftover .resume_tmp from old bench_env --resume runs."""
    tmp_root = run_dir / ".resume_tmp"
    if not tmp_root.is_dir():
        return 0
    repeat_n = int(meta.get("repeat_n") or 1)
    pass_k = meta.get("pass_k")
    merged = 0
    for child in sorted(p for p in tmp_root.iterdir() if p.is_dir()):
        src_results = child / "results.jsonl"
        if not src_results.is_file() or src_results.stat().st_size == 0:
            continue
        rows = load_jsonl(src_results)
        if not rows:
            continue
        keys = {result_key(r) for r in rows}
        merge_results(run_dir, child, keys, repeat_n, pass_k)
        shutil.rmtree(child, ignore_errors=True)
        merged += len(rows)
        print(f"[resume] salvaged {len(rows)} episodes from legacy {child.name}")
    # remove empty .resume_tmp
    try:
        if tmp_root.is_dir() and not any(tmp_root.iterdir()):
            tmp_root.rmdir()
    except OSError:
        pass
    return merged


def _forward_eval_argv(argv: list[str], *, task_ids: list[str], run_dir: Path) -> list[str]:
    """Build run_eval_mg.py argv: drop resume/run-dir/task filters, inject ours."""
    out: list[str] = []
    skip_next = False
    drop_flags = {
        "--resume",
        "--rerun",
        "--run-dir",
        "--run-name",
        "--task-id",
        "--task-ids",
        "--suite",
        "--split",
        "--filter-difficulty",
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        if a in drop_flags:
            # value may be next token or --flag=value
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                skip_next = True
            i += 1
            continue
        if any(a.startswith(f"{f}=") for f in drop_flags):
            i += 1
            continue
        out.append(a)
        i += 1

    out.extend(["--task-ids", ",".join(task_ids)])
    out.extend(["--run-dir", str(run_dir)])
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="In-place MobileGym resume (OpenMobile wrapper, no .resume_tmp)."
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Original run directory")
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Unused. Old temp-dir resume leftover; kept for CLI compatibility.",
    )
    args, forward = parser.parse_known_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not (run_dir / "meta.json").is_file():
        print(f"[ERROR] not a run dir (missing meta.json): {run_dir}")
        return 2

    meta = _load_meta(run_dir)
    salvaged = _merge_legacy_resume_tmp(run_dir, meta)
    salvaged += _salvage_resume_work(run_dir, meta)
    if salvaged:
        print(f"[resume] salvaged total {salvaged} leftover episodes")
        meta = _load_meta(run_dir)

    pending, partial, repeat_n = _pending_task_ids(run_dir, meta)
    error_ids = _error_task_ids(run_dir, meta)
    seen_ids: set[str] = set()
    resume_ids: list[str] = []
    for tid in pending + partial + error_ids:
        if tid not in seen_ids:
            seen_ids.add(tid)
            resume_ids.append(tid)
    existing = _dedupe_results(load_jsonl(run_dir / "results.jsonl"))
    progress = _overall_progress(run_dir, meta)
    valid = max(1, progress["offset"] - progress["error"])
    sr = progress["success"] / valid

    if existing:
        write_summary_json(
            run_dir, existing, repeat_n=repeat_n, pass_k=meta.get("pass_k")
        )
        write_errors_jsonl(run_dir, existing)

    print("=" * 60)
    print("  MOBILEGYM RESUME (OpenMobile, append into original dir)")
    print("=" * 60)
    print(f"  run_dir:  {run_dir}")
    print(
        f"  overall:  {progress['offset']}/{progress['total']}  "
        f"✓{progress['success']} ✗{progress['failed']}  "
        f"error={progress['error']}  SR={sr:.1%}"
    )
    print(f"  pending:  {len(pending)} tasks")
    print(f"  partial:  {len(partial)} tasks")
    print(f"  retry errors: {len(error_ids)} tasks")
    print("  progress bar below counts ALL tasks, not only this batch")
    print("=" * 60)

    if not resume_ids:
        print("[INFO] nothing to resume — run looks complete")
        return 0

    rerun_keys = {f"{tid}__t{t}" for tid in error_ids + partial for t in range(repeat_n)}
    dropped = _drop_result_keys(run_dir, rerun_keys)
    if dropped:
        print(f"[resume] dropped {dropped} old error/partial rows before append", flush=True)

    before = {result_key(r) for r in load_jsonl(run_dir / "results.jsonl")}
    eval_argv = _forward_eval_argv(forward, task_ids=resume_ids, run_dir=run_dir)
    cmd = [sys.executable, str(_RUN_EVAL), *eval_argv]
    # Errors are being rerun, so do not count them as already finished on the bar.
    _apply_progress_env(
        {
            "total": progress["total"],
            "offset": progress["success"] + progress["failed"],
            "success": progress["success"],
            "fail": progress["failed"],
        }
    )
    print("\nLaunch:", " ".join(cmd), flush=True)

    interrupted = False
    code = 1
    try:
        code = subprocess.call(cmd)
    except KeyboardInterrupt:
        interrupted = True
        code = 130
        print("\n[WARN] interrupted — keeping rows already appended to original run", flush=True)

    raw_rows = load_jsonl(run_dir / "results.jsonl")
    rows = _dedupe_results(raw_rows)
    if len(rows) != len(raw_rows):
        _write_jsonl(run_dir / "results.jsonl", rows)
    if rows:
        write_summary_json(run_dir, rows, repeat_n=repeat_n, pass_k=meta.get("pass_k"))
        write_errors_jsonl(run_dir, rows)
    added = [r for r in rows if result_key(r) not in before]
    print(
        f"[resume] appended {len(added)} episodes into {run_dir} "
        f"(now {len(rows)} rows)",
        flush=True,
    )
    try:
        history = meta.setdefault("resume_history", [])
        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "count": len(added),
                "tasks": resume_ids,
            }
        )
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[WARN] resume_history update failed: {e}")

    if interrupted:
        return 130
    return 0 if added or code == 0 else code


if __name__ == "__main__":
    raise SystemExit(main())
