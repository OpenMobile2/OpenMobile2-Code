"""bench215 eval on the MobileGym++ mock frontend.

One entry for every model. ``--runtime`` selects the dialogue format;
``--model-base-url`` and ``--model-name`` select the server. The agent is
mock ``bench_env.hybrid_benchmark`` (not a second OpenMobile session loop).

  python eval_bench215.py --list
  python eval_bench215.py --runtime qwen35_thought_session --env-url http://127.0.0.1:4173 ^
      --model-base-url http://<openai-compatible-host>/v1 --model-name YOUR_MODEL
  python eval_bench215.py --runtime qwen3vl --mode gui_only --model-name OpenMobile-8B ...
  python eval_bench215.py --runtime venus --model-name UI-Venus-1.5-8B ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from run_rollout import (
    _DEFAULT_MOCK,
    _ensure_mock_on_path,
    _resolve_api_key,
)

_DEFAULT_RUNTIME = "qwen35_thought_session"
_RUNTIMES = ("qwen35_thought_session", "gui_owl", "mai_ui", "venus", "qwen3vl")

_HERE = Path(__file__).resolve().parent


def _load_official(mock_root: Path, official_json: Path | None) -> dict[str, Any]:
    path = official_json or (mock_root / "bench215" / "official.json")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"bench215 official.json not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data.get("protocol") or {}
    suites = [str(s) for s in (data.get("suites") or []) if s]
    main_ids = [str(s) for s in (data.get("main_ids") or []) if s]
    secretary = str(data.get("secretary_task_id") or "")
    expected = int(data.get("n") or 0)
    if not suites or not main_ids or not secretary:
        raise SystemExit(f"official.json missing suites/main_ids/secretary_task_id: {path}")
    n = len(main_ids) + 1
    if expected and n != expected:
        raise SystemExit(
            f"official.json n={expected} but main_ids+secretary={n} ({path})"
        )
    return {
        "path": path,
        "n": n,
        "protocol": protocol,
        "suites": suites,
        "main_ids": main_ids,
        "secretary_task_id": secretary,
        "name": data.get("name") or "bench215-official",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _merge_summary(
    out: Path,
    *,
    mode: str,
    model_name: str,
    official: dict[str, Any],
    runtime: str | None = None,
    last_n: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    packs: dict[str, Any] = {}
    for pack in ("main", "secretary"):
        pack_dir = out / pack
        pack_rows = _load_jsonl(pack_dir / "results.jsonl")
        rows.extend(pack_rows)
        pack_sum_path = pack_dir / "summary.json"
        packs[pack] = (
            json.loads(pack_sum_path.read_text(encoding="utf-8"))
            if pack_sum_path.is_file()
            else {"n": len(pack_rows)}
        )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = str(row.get("task_id") or "")
        if tid:
            by_id[tid] = row
    ordered = list(by_id.values())
    (out / "results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in ordered),
        encoding="utf-8",
    )
    ok = sum(1 for r in ordered if r.get("success"))
    steps = [r["steps"] for r in ordered if isinstance(r.get("steps"), (int, float))]
    errors = sum(1 for r in ordered if r.get("error"))
    if runtime is None or last_n is None:
        for pack in ("main", "secretary"):
            meta_path = out / pack / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if runtime is None:
                runtime = meta.get("runtime")
            if last_n is None and meta.get("last_n") is not None:
                last_n = meta.get("last_n")
            if runtime is not None and last_n is not None:
                break
    summary = {
        "official": official["name"],
        "official_json": str(official["path"]),
        "n": len(ordered),
        "expected_n": official["n"],
        "successful": ok,
        "sr": (ok / len(ordered)) if ordered else 0.0,
        "errors": errors,
        "mean_steps": (sum(steps) / len(steps)) if steps else None,
        "interaction_mode": mode,
        "model_name": model_name,
        "runtime": runtime or "qwen35_thought_session",
        "last_n": 3 if last_n is None else last_n,
        "packs": packs,
        "output_dir": str(out),
    }
    if str(summary["runtime"]) not in {"gui_owl", "mai_ui", "venus"}:
        summary["enable_thinking"] = False
        summary["qwen35_tool_call_mode"] = "xml"
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _write_comparison(parent: Path, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hybrid = summaries.get("hybrid") or {}
    gui = summaries.get("gui_only") or {}
    runtime = (
        (hybrid.get("runtime") if hybrid else None)
        or (gui.get("runtime") if gui else None)
        or "qwen35_thought_session"
    )
    comparison = {
        "runtime": runtime,
        "order": ["hybrid", "gui_only"],
        "hybrid": {
            "n": hybrid.get("n"),
            "successful": hybrid.get("successful"),
            "sr": hybrid.get("sr"),
            "errors": hybrid.get("errors"),
            "output_dir": hybrid.get("output_dir"),
        },
        "gui_only": {
            "n": gui.get("n"),
            "successful": gui.get("successful"),
            "sr": gui.get("sr"),
            "errors": gui.get("errors"),
            "output_dir": gui.get("output_dir"),
        },
        "delta_sr_hybrid_minus_gui": None,
    }
    if hybrid.get("sr") is not None and gui.get("sr") is not None:
        comparison["delta_sr_hybrid_minus_gui"] = float(hybrid["sr"]) - float(gui["sr"])
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return comparison


def _pending_ids(dest: Path, task_ids: list[str], *, resume: bool) -> list[str]:
    if not resume:
        return list(task_ids)
    pending = [
        tid
        for tid in task_ids
        if not (dest / "episodes" / tid / "trace.json").is_file()
    ]
    skipped = len(task_ids) - len(pending)
    if skipped:
        print(
            f"[bench215] resume skip {skipped}, remaining {len(pending)} -> {dest}",
            flush=True,
        )
    return pending


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        tid = str(row.get("task_id") or "")
        if tid:
            out[tid] = row
    return out


def _write_pack_results(dest: Path, rows_by_id: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(rows_by_id.values(), key=lambda row: str(row.get("task_id") or ""))
    (dest / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in ordered),
        encoding="utf-8",
    )
    ok = sum(1 for row in ordered if row.get("success"))
    errors = sum(1 for row in ordered if row.get("error"))
    summary = {
        "total": len(ordered),
        "successful": ok,
        "sr": (ok / len(ordered)) if ordered else 0.0,
        "errors": errors,
        "merged_resume": True,
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _parse_task_ids(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _locked_last_n(runtime: str, requested: int | None) -> int:
    if runtime == "gui_owl":
        return 5
    if runtime == "mai_ui":
        return 3
    if runtime in {"venus", "qwen3vl"}:
        return 1
    return 3 if requested is None else requested


def _hybrid_benchmark_argv(
    *,
    suite: str,
    task_ids: list[str],
    dest: Path,
    mode: str,
    args: argparse.Namespace,
    parallel: int,
) -> list[str]:
    forwarded = [
        "--suite",
        suite,
        "--task-ids",
        ",".join(task_ids),
        "--interaction-mode",
        mode,
        "--runtime",
        args.runtime,
        "--last-n",
        str(args.last_n),
        "--enable-thinking",
        "false",
        "--qwen35-tool-call-mode",
        "xml",
        "--env-url",
        args.env_url,
        "--model-base-url",
        args.model_base_url,
        "--model-name",
        args.model_name,
        "--model-api-key",
        args.model_api_key or "EMPTY",
        "--parallel",
        str(parallel),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--infer-timeout",
        str(args.infer_timeout),
        "--coord-space",
        args.coord_space,
        "--start-delay",
        str(args.start_delay or 0),
        "--output-dir",
        str(dest),
    ]
    if args.quiet:
        forwarded.append("--quiet")
    if args.max_steps is not None:
        forwarded.extend(["--max-steps", str(args.max_steps)])
    if not args.no_headless:
        forwarded.append("--headless")
    return forwarded


async def _run_pack(
    *,
    pack: str,
    suite: str,
    task_ids: list[str],
    dest: Path,
    args: argparse.Namespace,
) -> None:
    from bench_env.hybrid_benchmark.cli import async_main, create_parser

    dest.mkdir(parents=True, exist_ok=True)
    kept_path = dest / "results.jsonl.kept"
    kept: dict[str, dict[str, Any]] = {}
    if not args.no_resume:
        if kept_path.is_file():
            kept.update(_rows_by_id(kept_path))
        kept.update(_rows_by_id(dest / "results.jsonl"))
        if kept:
            print(
                f"[bench215] resume rows kept={len(kept)} -> {dest}",
                flush=True,
            )
    if kept:
        kept_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in kept.values()
            ),
            encoding="utf-8",
        )
    pending = _pending_ids(dest, task_ids, resume=not args.no_resume)
    if not pending:
        n_traces = sum(
            1
            for tid in task_ids
            if (dest / "episodes" / tid / "trace.json").is_file()
        )
        if kept:
            _write_pack_results(dest, kept)
        print(f"[bench215] {args.mode} pack={pack} already complete -> {dest}", flush=True)
        if len(kept) < n_traces:
            print(
                f"[bench215] WARNING: {n_traces} traces but results.jsonl only "
                f"has {len(kept)}. Rebuild with rebuild_bench215_results.py, do not --no-resume.",
                flush=True,
            )
        return
    parallel = max(1, args.parallel if pack == "main" else min(args.parallel, 1))
    forwarded = _hybrid_benchmark_argv(
        suite=suite,
        task_ids=pending,
        dest=dest,
        mode=args.mode,
        args=args,
        parallel=parallel,
    )
    print(
        f"[bench215] {args.mode} pack={pack} n={len(pending)} "
        f"runtime={args.runtime} parallel={parallel} -> {dest}",
        flush=True,
    )
    hb_args = create_parser().parse_args(forwarded)
    await async_main(hb_args)
    merged = dict(kept)
    for tid in pending:
        merged.pop(tid, None)
    for row in _load_jsonl(dest / "results.jsonl"):
        tid = str(row.get("task_id") or "")
        if tid:
            merged[tid] = row
    _write_pack_results(dest, merged)
    print(
        f"[bench215] merged results kept={len(kept)} new={len(pending)} "
        f"total={len(merged)} -> {dest / 'results.jsonl'}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Eval bench215 on the MobileGym++ mock. "
            "Pass --runtime and the model server; do not use a per-model script."
        )
    )
    parser.add_argument("--mock-root", type=Path, default=_DEFAULT_MOCK)
    parser.add_argument("--official-json", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["both", "hybrid", "gui_only"],
        default="both",
        help="both = run hybrid first, then gui_only (default).",
    )
    parser.add_argument("--pack", choices=["all", "main", "secretary"], default="all")
    parser.add_argument(
        "--runtime",
        choices=list(_RUNTIMES),
        default=_DEFAULT_RUNTIME,
        help=(
            "qwen35_thought_session = Thought + XML; "
            "qwen3vl = flat ReAct; venus / gui_owl / mai_ui = that model's official prompt."
        ),
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Screenshot slots. Locked to 5/3/1/1 for gui_owl/mai_ui/venus/qwen3vl.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke: first N main tasks (skips secretary)")
    parser.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated official task ids. Resume still skips existing trace.json.",
    )
    parser.add_argument("--env-url", default="")
    parser.add_argument("--model-base-url", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--model-api-key", default="")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--infer-timeout", type=float, default=300.0)
    parser.add_argument("--coord-space", default="norm_0_1000")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--start-delay", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Single-mode dir, or parent dir when --mode both. "
        "Default: eval_runs/<model>/bench215-mock[/hybrid|/gui_only]",
    )
    args = parser.parse_args(argv)
    args.last_n = _locked_last_n(args.runtime, args.last_n)
    if args.output_dir is not None and not args.output_dir.is_absolute():
        args.output_dir = (_HERE / args.output_dir).resolve()

    mock_root = _ensure_mock_on_path(args.mock_root)
    official = _load_official(mock_root, args.official_json)

    if args.list:
        payload = {
            "backend": "mobilegym-mock hybrid_benchmark",
            "official": official["name"],
            "n": official["n"],
            "suites": official["suites"],
            "main": len(official["main_ids"]),
            "secretary": official["secretary_task_id"],
            "runtimes": list(_RUNTIMES),
            "protocol": {
                "runtime": args.runtime,
                "last_n": args.last_n,
            },
            "mock": str(mock_root),
            "official_json": str(official["path"]),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.env_url or not args.model_base_url or not args.model_name:
        raise SystemExit("required: --env-url --model-base-url --model-name")

    args.model_api_key = _resolve_api_key(args.model_api_key, base_url=args.model_base_url)

    main_ids = list(official["main_ids"])
    wanted = _parse_task_ids(args.task_ids)
    if wanted:
        wanted_set = set(wanted)
        known = set(official["main_ids"]) | {official["secretary_task_id"]}
        missing = [tid for tid in wanted if tid not in known]
        if missing:
            raise SystemExit(f"unknown --task-ids: {', '.join(missing)}")
        main_ids = [tid for tid in main_ids if tid in wanted_set]
        if official["secretary_task_id"] not in wanted_set and args.pack == "all":
            args.pack = "main"
        print(f"[bench215] --task-ids n={len(wanted)} main={len(main_ids)}", flush=True)
    if args.limit is not None:
        main_ids = main_ids[: max(0, args.limit)]
        if args.pack == "all":
            args.pack = "main"
        print(f"[bench215] --limit {args.limit}: main only, n={len(main_ids)}", flush=True)

    modes = ["hybrid", "gui_only"] if args.mode == "both" else [args.mode]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in args.model_name)
    default_parent = _HERE / "eval_runs" / safe / "bench215-mock"
    if args.output_dir is None:
        parent = default_parent
        mode_dirs = {mode: parent / mode for mode in modes}
    elif args.mode == "both":
        parent = args.output_dir.expanduser().resolve()
        mode_dirs = {mode: parent / mode for mode in modes}
    else:
        parent = args.output_dir.expanduser().resolve()
        mode_dirs = {args.mode: parent}

    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    print(
        f"bench215 eval  runtime={args.runtime} last_n={args.last_n} modes={modes} pack={args.pack} "
        f"model={args.model_name} mock={mock_root}",
        flush=True,
    )

    summaries: dict[str, dict[str, Any]] = {}
    rc = 0
    for mode in modes:
        out = mode_dirs[mode]
        out.mkdir(parents=True, exist_ok=True)
        args.mode = mode
        print(f"[bench215] start mode={mode} -> {out}", flush=True)

        async def _run(dest: Path) -> None:
            if args.pack in {"all", "main"}:
                await _run_pack(
                    pack="main",
                    suite=",".join(official["suites"]),
                    task_ids=main_ids,
                    dest=dest / "main",
                    args=args,
                )
            if args.pack in {"all", "secretary"}:
                await _run_pack(
                    pack="secretary",
                    suite="hybrid_demo_crossapp",
                    task_ids=[official["secretary_task_id"]],
                    dest=dest / "secretary",
                    args=args,
                )

        asyncio.run(_run(out))
        summary = _merge_summary(
            out,
            mode=mode,
            model_name=args.model_name,
            official=official,
            runtime=args.runtime,
            last_n=args.last_n,
        )
        summaries[mode] = summary
        print(
            f"[bench215] {mode} {summary['successful']}/{summary['n']} "
            f"SR={summary['sr']:.3f} output={out}",
            flush=True,
        )
        if args.pack == "all" and args.limit is None and summary["n"] != official["n"]:
            rc = 2

    if len(modes) > 1:
        compare_dir = parent.resolve()
        comparison = _write_comparison(compare_dir, summaries)
        print(
            f"[bench215] comparison hybrid SR={comparison['hybrid']['sr']} "
            f"gui_only SR={comparison['gui_only']['sr']} "
            f"delta={comparison['delta_sr_hybrid_minus_gui']} "
            f"-> {compare_dir / 'comparison.json'}",
            flush=True,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
