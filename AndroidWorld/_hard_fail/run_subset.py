#!/usr/bin/env python3
"""Self-contained hard-fail AndroidWorld eval. Delete this folder to undo.

Does not modify AndroidWorld source. Injects --tasks from groups.json, then
writes 全做错 / 易做错 averages next to the result CSV.

Usage (from AndroidWorld/):
  python _hard_fail/run_subset.py hard --runtime qwen35_session --checkpoint_dir runs/foo ...
  python _hard_fail/run_subset.py summarize runs/foo/result_*.csv
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
AW_ROOT = HERE.parent
GROUPS_PATH = HERE / "groups.json"
LABEL_NEVER = "========= 全做错 ========="
LABEL_LT50 = "========= 易做错 ========="
LABEL_AVERAGE = "========= Average ========="
GROUP_CHOICES = ("never", "lt50", "hard")


def load_groups() -> dict:
    data = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    never = list(data["groups"]["never"]["tasks"])
    lt50 = list(data["groups"]["lt50"]["tasks"])
    return {
        "never": never,
        "lt50": lt50,
        "hard": never + lt50,
        "labels": {
            "never": data["groups"]["never"]["label"],
            "lt50": data["groups"]["lt50"]["label"],
        },
    }


def is_summary_row(name: str) -> bool:
    name = (name or "").strip()
    return (not name) or name.startswith("=========")


def parse_group_and_args(argv: list[str]) -> tuple[str, list[str]]:
    group = "hard"
    rest = list(argv)
    if rest and rest[0] in GROUP_CHOICES:
        group = rest.pop(0)
    filtered: list[str] = []
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg in ("--group", "--eval_group"):
            if i + 1 >= len(rest):
                raise SystemExit(f"{arg} needs a value: {', '.join(GROUP_CHOICES)}")
            group = rest[i + 1]
            i += 2
            continue
        if arg.startswith("--group=") or arg.startswith("--eval_group="):
            group = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--tasks" or arg.startswith("--tasks="):
            raise SystemExit("Do not pass --tasks; this folder injects it from groups.json.")
        filtered.append(arg)
        i += 1
    if group not in GROUP_CHOICES:
        raise SystemExit(f"Unknown group={group!r}. Use: {', '.join(GROUP_CHOICES)}")
    return group, filtered


def flag_value(args: list[str], name: str) -> str | None:
    prefix = f"--{name}="
    for i, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
        if arg == f"--{name}" and i + 1 < len(args):
            return args[i + 1]
    return None


def latest_result_csv(*dirs: Path) -> Path | None:
    files: list[Path] = []
    for d in dirs:
        if d and d.is_dir():
            files.extend(d.glob("result_*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if not is_summary_row(r.get("task") or "")]
    return fieldnames, rows


def mean_of(rows: list[dict[str, str]], field: str) -> str:
    vals: list[float] = []
    for row in rows:
        raw = (row.get(field) or "").strip()
        if raw == "":
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    if not vals:
        return ""
    return f"{sum(vals) / len(vals):.6g}"


def group_rows(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    names: list[str],
    label: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    name_set = set(names)
    sub = [r for r in rows if (r.get("task") or "") in name_set]
    avg = {k: mean_of(sub, k) for k in fieldnames if k != "task"}
    avg["task"] = label
    if "task_num" in avg:
        avg["task_num"] = "0"
    return sub, avg


def write_grouped_csv(
    src: Path,
    dest: Path,
    groups: dict,
) -> dict[str, dict]:
    fieldnames, rows = load_rows(src)
    if "task" not in fieldnames:
        fieldnames = ["task"] + fieldnames

    never_rows, never_avg = group_rows(fieldnames, rows, groups["never"], LABEL_NEVER)
    lt50_rows, lt50_avg = group_rows(fieldnames, rows, groups["lt50"], LABEL_LT50)
    _, overall_avg = group_rows(fieldnames, rows, [r["task"] for r in rows], LABEL_AVERAGE)

    extras = []
    stats = {}
    if never_rows:
        extras.append(never_avg)
        stats["never"] = {
            "n": len(never_rows),
            "mean_success_rate": never_avg.get("mean_success_rate", ""),
        }
    if lt50_rows:
        extras.append(lt50_avg)
        stats["lt50"] = {
            "n": len(lt50_rows),
            "mean_success_rate": lt50_avg.get("mean_success_rate", ""),
        }
    extras.append(overall_avg)
    stats["overall"] = {
        "n": len(rows),
        "mean_success_rate": overall_avg.get("mean_success_rate", ""),
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(extras)
    return stats


def print_stats(csv_path: Path, stats: dict) -> None:
    print(csv_path)
    if "never" in stats:
        print(
            f"  全做错  n={stats['never']['n']}  "
            f"mean_success_rate={stats['never']['mean_success_rate']}"
        )
    if "lt50" in stats:
        print(
            f"  易做错  n={stats['lt50']['n']}  "
            f"mean_success_rate={stats['lt50']['mean_success_rate']}"
        )
    print(
        f"  overall n={stats['overall']['n']}  "
        f"mean_success_rate={stats['overall']['mean_success_rate']}"
    )


def cmd_summarize(paths: list[str]) -> int:
    groups = load_groups()
    if not paths:
        raise SystemExit("summarize needs at least one result_*.csv")
    rc = 0
    for raw in paths:
        src = Path(raw).expanduser().resolve()
        if not src.is_file():
            print(f"Missing: {src}", file=sys.stderr)
            rc = 1
            continue
        dest = src.with_name(src.stem + "_grouped.csv")
        stats = write_grouped_csv(src, dest, groups)
        print_stats(dest, stats)
    return rc


def cmd_run(argv: list[str]) -> int:
    groups = load_groups()
    group, rest = parse_group_and_args(argv)
    tasks = groups[group]
    print(
        f"_hard_fail group={group}: {len(tasks)} tasks "
        f"(全做错={sum(1 for t in tasks if t in groups['never'])}, "
        f"易做错={sum(1 for t in tasks if t in groups['lt50'])})"
    )
    for name in groups["never"]:
        if name in tasks:
            print(f"  [全做错] {name}")
    for name in groups["lt50"]:
        if name in tasks:
            print(f"  [易做错] {name}")

    cmd = [
        sys.executable,
        str(AW_ROOT / "run.py"),
        f"--tasks={','.join(tasks)}",
        *rest,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(AW_ROOT))
    if proc.returncode != 0:
        return proc.returncode

    ckpt_raw = flag_value(rest, "checkpoint_dir")
    ckpt = Path(ckpt_raw).expanduser() if ckpt_raw else None
    if ckpt and not ckpt.is_absolute():
        ckpt = (AW_ROOT / ckpt).resolve()
    src = latest_result_csv(ckpt, AW_ROOT / "output") if ckpt else latest_result_csv(AW_ROOT / "output")
    if src is None:
        print("No result_*.csv found to summarize.", file=sys.stderr)
        return 0

    out_dir = ckpt if ckpt else src.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir / f"result_grouped_{ts}.csv"
    stats = write_grouped_csv(src, dest, groups)
    print_stats(dest, stats)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in ("summarize", "stats"):
        return cmd_summarize(argv[1:])
    os.chdir(AW_ROOT)
    return cmd_run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
