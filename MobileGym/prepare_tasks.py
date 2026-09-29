"""Convert MobileGym grounded task files into OpenMobile rollout JSON.

Input (json / jsonl from MobileGym/tasks/):
  {"id": "...", "instruction": "...", "apps": ["Alipay"], "difficulty": "easy", ...}

Output (OpenMobile run_diy format):
  [
    {
      "sample_id": "...",
      "instruction": "...",
      "base_task_name": "MobileGymFreeform",
      "apps": [...],
      "difficulty": "...",
      ...
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FREEFORM = "MobileGymFreeform"


def _load_tasks(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        out = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{i+1} not an object")
            out.append(item)
        return out
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return list(data["tasks"])
    raise ValueError(f"Unsupported JSON shape in {path}")


def convert_item(item: dict, *, index: int) -> dict:
    instruction = (item.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"Item {index} missing instruction")
    sample_id = item.get("sample_id") or item.get("id") or f"mg_{index:04d}"
    sample_id = str(sample_id)
    apps = item.get("apps") or []
    if isinstance(apps, str):
        apps = [apps]
    base = item.get("base_task_name") or FREEFORM
    out = {
        "sample_id": sample_id,
        "instruction": instruction,
        "base_task_name": base,
        "apps": list(apps),
        "difficulty": item.get("difficulty"),
        "horizon": item.get("horizon"),
        "category": item.get("category"),
        "language": item.get("language", "en"),
        "grounded": item.get("grounded", True),
    }
    # Drop Nones for cleaner JSON
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MobileGym tasks for OpenMobile rollout.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "tasks" / "tasks_5000.jsonl"),
        help="tasks_*.json / tasks_*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output OpenMobile samples JSON path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max samples (0 = all).")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="",
        help="Optional filter: easy|medium|hard",
    )
    parser.add_argument(
        "--apps",
        type=str,
        default="",
        help="Optional comma-separated app filter (any match).",
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Missing input: {src}")

    items = _load_tasks(src)
    if args.difficulty:
        items = [t for t in items if t.get("difficulty") == args.difficulty]
    if args.apps:
        wanted = {a.strip() for a in args.apps.replace("，", ",").split(",") if a.strip()}
        items = [
            t
            for t in items
            if wanted.intersection(set(t.get("apps") or []))
        ]
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    samples = [convert_item(t, index=i) for i, t in enumerate(items, 1)]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
