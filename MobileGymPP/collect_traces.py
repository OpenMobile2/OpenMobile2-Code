"""Flatten hybrid_benchmark episodes into one JSONL for SFT inspection.

custom_v5 has no semantic verifier: do not treat results.jsonl `success` as
task completion. Use --require-app-tool to keep only traces that actually
called a foreground App Tool.

  python collect_traces.py --run-dir runs/c5-hybrid-roll --output runs/c5-hybrid-roll/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect MobileGymPP episode traces.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--require-app-tool", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    episodes = run_dir / "episodes"
    if not episodes.is_dir():
        raise SystemExit(f"no episodes/ under {run_dir}")

    results = {}
    results_path = run_dir / "results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            results[str(row.get("task_id") or row.get("dataset_id") or "")] = row

    out_path = args.output or (run_dir / "traces.jsonl")
    kept = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for trace_path in sorted(episodes.glob("*/trace.json")):
            data = json.loads(trace_path.read_text(encoding="utf-8"))
            # HybridRunResult.to_dict: "steps" is a count; the list is "trace".
            raw_steps = data.get("trace")
            steps = raw_steps if isinstance(raw_steps, list) else []
            tool_calls = [
                step.get("call")
                for step in steps
                if isinstance(step, dict)
                and isinstance(step.get("call"), dict)
                and step["call"].get("name")
                and step["call"].get("name") != "mobile_use"
            ]
            if args.require_app_tool and not tool_calls:
                skipped += 1
                continue
            folder = trace_path.parent.name
            task_meta = {}
            task_json = trace_path.parent / "task.json"
            if task_json.is_file():
                task_meta = json.loads(task_json.read_text(encoding="utf-8"))
            meta = (
                results.get(folder)
                or results.get(str(task_meta.get("task_id") or ""))
                or results.get(str(task_meta.get("dataset_id") or ""))
                or {}
            )
            handle.write(json.dumps({
                "task_id": task_meta.get("task_id") or meta.get("task_id") or folder,
                "dataset_id": task_meta.get("dataset_id") or meta.get("dataset_id"),
                "instruction": task_meta.get("instruction") or meta.get("instruction"),
                "interaction_mode": data.get("interaction_mode") or meta.get("interaction_mode"),
                "stop_reason": data.get("stop_reason"),
                "agent_answer": data.get("agent_answer"),
                "n_steps": len(steps),
                "n_app_tools": len(tool_calls),
                "app_tools": [row.get("name") for row in tool_calls if isinstance(row, dict)],
                "steps": steps,
                "episode_dir": str(trace_path.parent),
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"wrote {kept} traces ({skipped} skipped) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
