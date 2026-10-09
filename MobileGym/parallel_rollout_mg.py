#!/usr/bin/env python3
"""Launch N parallel MobileGym rollouts (OpenMobile freeform).

Parallelism model (important):
  - Each worker is a separate Playwright browser process with its OWN SPA state.
  - The frontend URL is only a static asset server (preview / nginx).
  - Therefore ONE URL can back N workers: pass --env_url once + --parallel N.
  - nginx gateway (https://localhost:4180) is optional: same single-URL model,
    but with sendfile/HTTP2/multi-worker static serving so high N is stabler
    than `npm run preview`.

Examples:
  # 8-way against your current preview
  python parallel_rollout_mg.py \\
    --env_url http://127.0.0.1:4173 \\
    --parallel 8 \\
    --input_json prepared_tasks.json \\
    --output_dir runs/mobilegym_rollout \\
    --agent_name qwen3vl \\
    --use_memgui_prompt \\
    --model_base_url https://<openai-compatible-host>/v1 \\
    --model_name gemini-3.1-pro-preview \\
    --model_api_key "$OPENAI_API_KEY"

  # Or list URLs explicitly (legacy)
  python parallel_rollout_mg.py \\
    --env_urls http://127.0.0.1:4173,http://127.0.0.1:4173 \\
    ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "run_diy_mg.py"


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _split_urls(raw: str) -> list[str]:
    text = (raw or "").replace("，", ",").replace(";", ",")
    return [h.strip() for h in text.split(",") if h.strip()]


def _resolve_worker_urls(args: argparse.Namespace) -> list[str]:
    """Build the list of frontend URLs, one per parallel worker."""
    if args.env_urls:
        urls = _split_urls(args.env_urls)
        if args.parallel and args.parallel > 1 and len(urls) == 1:
            # Convenience: --env_urls http://x --parallel 8
            return [urls[0]] * int(args.parallel)
        if args.parallel and args.parallel > 1 and len(urls) != args.parallel:
            print(
                f"[warn] --parallel={args.parallel} ignored; using {len(urls)} URLs from --env_urls",
                file=sys.stderr,
            )
        return urls

    if not args.env_url:
        raise SystemExit("Provide --env_url (with --parallel) or --env_urls")

    n = max(1, int(args.parallel or 1))
    return [args.env_url.rstrip("/")] * n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel MobileGym rollouts for OpenMobile (shard by sample_id)."
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default="",
        help="Single MobileGym frontend URL. Combine with --parallel N.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of Playwright workers (default 1). Uses the same --env_url N times.",
    )
    parser.add_argument(
        "--env_urls",
        type=str,
        default="",
        help="Comma-separated URLs (legacy). Length = worker count unless single URL + --parallel.",
    )
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_HERE / "runs" / "mobilegym_rollout"),
    )
    parser.add_argument("--agent_name", type=str, default="qwen3vl")
    parser.add_argument(
        "--runtime",
        type=str,
        default="",
        help="Shared runtime preset (overrides agent_name/memgui/memory when set).",
    )
    parser.add_argument("--max_n_steps", type=int, default=30)
    parser.add_argument("--step_wait_time", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_headless", action="store_true")
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
    parser.add_argument(
        "--enable_thinking",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--require_think_tags",
        type=_parse_bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--qwen35_tool_call_mode",
        type=str,
        default="native",
        choices=["xml", "native"],
    )
    parser.add_argument("--switching_weak_model_base_url", type=str, default="")
    parser.add_argument("--switching_weak_model_name", type=str, default="")
    parser.add_argument("--switching_weak_model_api_key", type=str, default="EMPTY")
    args = parser.parse_args()

    urls = _resolve_worker_urls(args)
    if not urls:
        print("No env URLs resolved", file=sys.stderr)
        return 2

    print(f"Parallel workers: {len(urls)}")
    print(f"Frontend URL(s): {sorted(set(urls))}")

    headless = not args.no_headless
    procs = []
    for i, url in enumerate(urls):
        cmd = [
            sys.executable,
            str(_SCRIPT),
            f"--env_url={url}",
            f"--input_json={args.input_json}",
            f"--output_dir={args.output_dir}",
            f"--agent_name={args.agent_name}",
            f"--max_n_steps={args.max_n_steps}",
            f"--step_wait_time={args.step_wait_time}",
            f"--headless={'true' if headless else 'false'}",
            f"--model_base_url={args.model_base_url}",
            f"--model_name={args.model_name}",
            f"--model_api_key={args.model_api_key}",
            f"--use_memgui_prompt={'true' if args.use_memgui_prompt else 'false'}",
            f"--use_memory_prompt={'true' if args.use_memory_prompt else 'false'}",
            f"--memgui_prompt_format={args.memgui_prompt_format}",
            f"--last_n={int(args.last_n)}",
            f"--enable_thinking={str(bool(args.enable_thinking)).lower()}",
            f"--require_think_tags={str(bool(args.require_think_tags)).lower()}",
            f"--qwen35_tool_call_mode={args.qwen35_tool_call_mode}",
            f"--shard_index={i}",
            f"--num_shards={len(urls)}",
        ]
        if args.runtime:
            cmd.append(f"--runtime={args.runtime}")
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
        print(f"[worker {i}/{len(urls)}] {url}")
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
