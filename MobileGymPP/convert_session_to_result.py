"""Convert MobileGymPP episodes into qwen35_session training ``result.json``.

Agreed packing (train-time, aligned with official MW MCP agents):

- GUI ``mobile_use``: next tool message is the new screenshot.
- App-switch schema is merged into that screenshot tool message (not a
  separate user turn).
- App Tool with ``uiEffect=none``: tool message is JSON only, no image.
- App Tool that changes UI (draft / navigation / ...): JSON + screenshot.

Schema: ``{id, runtime, backend, goal, messages, images, done}``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

UI_CHANGING_EFFECTS = {"draft", "navigation", "open", "mutate"}

SYSTEM_ADDENDUM_OLD = """# Extra app tools

The API `tools=` list is only `mobile_use` for the entire session; it does not change
when you open another app.
When the foreground app changes, that turn's user message lists extra tools for the
new app inside `<tools></tools>`. Those extra tools stay valid only while that app
remains in the foreground. A later user message with a new app's tools replaces them.
Never invent a tool from another app. On the launcher / recents / control center,
only `mobile_use` is available — open the app first (`mobile_use` action `open_app`
with its name or id, or click the icon).
Call extra app tools with the same `<tool_call>` XML as `mobile_use`.
After a non-`mobile_use` call, the next user/tool message contains a JSON result and
a new screenshot."""

SYSTEM_ADDENDUM_NEW = """# Extra app tools

The API `tools=` list is only `mobile_use` for the entire session; it does not change
when you open another app.
When the foreground app changes, that turn's tool observation lists extra tools for
the new app inside `<tools></tools>`, together with the new screenshot. Those extra
tools stay valid only while that app remains in the foreground. A later observation
with a new app's tools replaces them.
Never invent a tool from another app. On the launcher / recents / control center,
only `mobile_use` is available — open the app first (`mobile_use` action `open_app`
with its name or id, or click the icon).
Call extra app tools with the same `<tool_call>` XML as `mobile_use`.
After a non-`mobile_use` call: if the tool does not change the UI, the next tool
message is JSON only (no new screenshot); if it changes the UI, the next tool
message contains the JSON result and a new screenshot."""


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


def _image_count(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    )


def _latest_prompt(episode_dir: Path) -> list[dict[str, Any]]:
    prompts = sorted(episode_dir.glob("step_*_prompt.json"))
    if not prompts:
        raise FileNotFoundError(f"no step_*_prompt.json under {episode_dir}")
    return json.loads(prompts[-1].read_text(encoding="utf-8"))


def _load_trace_steps(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "trace.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("trace")
    return steps if isinstance(steps, list) else []


def _meta(episode_dir: Path) -> dict[str, Any]:
    session_path = episode_dir / "session.json"
    if session_path.is_file():
        return json.loads(session_path.read_text(encoding="utf-8"))
    task_path = episode_dir / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8")) if task_path.is_file() else {}
    return {
        "id": task.get("task_id") or episode_dir.name,
        "runtime": "qwen35_session",
        "backend": "gemini",
        "goal": task.get("instruction") or "",
    }


def _patch_system(text: str) -> str:
    if SYSTEM_ADDENDUM_OLD in text:
        return text.replace(SYSTEM_ADDENDUM_OLD, SYSTEM_ADDENDUM_NEW)
    if "# Extra app tools" in text and "JSON only" not in text:
        return re.sub(
            r"# Extra app tools\n(?:.*\n)*",
            SYSTEM_ADDENDUM_NEW.rstrip() + "\n",
            text,
            count=1,
        )
    return text


def _ui_changes(effect: str) -> bool:
    return str(effect or "").strip().lower() in UI_CHANGING_EFFECTS


def _merge_schema_into_tool(tool_content: str, schema: str) -> str:
    n_img = tool_content.count("<image>")
    body = tool_content.replace("<image>", "").strip()
    parts = [schema.strip()]
    if body:
        parts.append(body)
    if n_img:
        parts.append("<image>" * n_img)
    return "\n".join(parts)


def convert_episode(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.expanduser().resolve()
    raw = _latest_prompt(episode_dir)
    meta = _meta(episode_dir)
    trace = _load_trace_steps(episode_dir)
    messages: list[dict[str, str]] = []
    images: list[str] = []
    img_i = 0
    obs_i = 0
    stats = {
        "mcp_json_only": 0,
        "mcp_json_image": 0,
        "gui_image": 0,
        "schema_merged": 0,
        "schema_orphan": 0,
    }

    for msg in raw:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        n_img = _image_count(content)
        text = _text(content).rstrip()

        if role == "system":
            messages.append({"role": "system", "content": _patch_system(text.strip())})
            continue

        if role == "assistant":
            stored = content if isinstance(content, str) else text
            messages.append({"role": "assistant", "content": stored})
            continue

        if role == "user" and msg.get("_session_observation"):
            step = trace[obs_i] if obs_i < len(trace) else {}
            obs_i += 1
            call = step.get("call") if isinstance(step.get("call"), dict) else {}
            call_name = str(call.get("name") or msg.get("name") or "")
            effect = str(step.get("ui_effect") or "")
            is_mcp = bool(call_name) and call_name != "mobile_use"
            keep_image = (not is_mcp) or _ui_changes(effect)
            count = n_img if n_img > 0 else 1
            body = text.strip()
            if keep_image:
                tool_content = (body + "\n" if body else "") + ("<image>" * count)
                messages.append({"role": "tool", "content": tool_content})
                for _ in range(count):
                    images.append(f"screenshot_step{img_i}.png")
                    img_i += 1
                if is_mcp:
                    stats["mcp_json_image"] += 1
                else:
                    stats["gui_image"] += 1
            else:
                if not body:
                    body = json.dumps(
                        {"name": call_name, "ok": True, "note": "uiEffect=none"},
                        ensure_ascii=False,
                    )
                messages.append({"role": "tool", "content": body})
                img_i += count
                stats["mcp_json_only"] += 1
            continue

        if role in {"user", "tool"}:
            if text.startswith("Foreground app changed"):
                if messages and messages[-1]["role"] == "tool":
                    messages[-1]["content"] = _merge_schema_into_tool(
                        messages[-1]["content"], text
                    )
                    stats["schema_merged"] += 1
                else:
                    messages.append({"role": "user", "content": text})
                    stats["schema_orphan"] += 1
                continue
            if n_img:
                if "<image>" not in text:
                    text = (text + "\n" if text else "") + ("<image>" * n_img)
                for _ in range(n_img):
                    images.append(f"screenshot_step{img_i}.png")
                    img_i += 1
            messages.append({"role": "user" if role == "user" else "tool", "content": text})

    last = messages[-1] if messages else {}
    done = (
        last.get("role") == "assistant"
        and "terminate" in str(last.get("content") or "")
        and "success" in str(last.get("content") or "")
    )
    record = {
        "id": str(meta.get("id") or episode_dir.name),
        "runtime": str(meta.get("runtime") or "qwen35_session"),
        "backend": str(meta.get("backend") or "gemini"),
        "goal": str(meta.get("goal") or ""),
        "messages": messages,
        "images": images,
        "done": bool(done),
    }
    record["_convert_stats"] = stats
    return record


def _write_record(episode_dir: Path, record: dict[str, Any]) -> Path:
    stats = record.pop("_convert_stats", {})
    out = episode_dir / "result.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record["_convert_stats"] = stats
    return out


def _episode_dirs(root: Path) -> list[Path]:
    if (root / "step_001_prompt.json").is_file() or list(root.glob("step_*_prompt.json")):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("step_*_prompt.json")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write qwen35_session result.json")
    parser.add_argument("path", type=Path, help="Episode dir, or a folder of episodes")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    target = args.path.expanduser().resolve()
    dirs = _episode_dirs(target)
    if not dirs:
        raise SystemExit(f"no episodes under {target}")

    totals = {
        "episodes": 0,
        "errors": 0,
        "mcp_json_only": 0,
        "mcp_json_image": 0,
        "gui_image": 0,
        "schema_merged": 0,
        "schema_orphan": 0,
        "done": 0,
    }
    errors: list[str] = []
    for i, episode_dir in enumerate(dirs, 1):
        try:
            record = convert_episode(episode_dir)
            stats = record.get("_convert_stats") or {}
            if args.output and len(dirs) == 1:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                payload = {k: v for k, v in record.items() if k != "_convert_stats"}
                args.output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                out = args.output
            else:
                out = _write_record(episode_dir, record)
            totals["episodes"] += 1
            for key in (
                "mcp_json_only",
                "mcp_json_image",
                "gui_image",
                "schema_merged",
                "schema_orphan",
            ):
                totals[key] += int(stats.get(key) or 0)
            if record.get("done"):
                totals["done"] += 1
            if i == 1 or i % 50 == 0 or i == len(dirs):
                print(
                    f"{i}/{len(dirs)} {episode_dir.name} "
                    f"msgs={len(record['messages'])} imgs={len(record['images'])} "
                    f"json_only={stats.get('mcp_json_only')} "
                    f"json+img={stats.get('mcp_json_image')} "
                    f"-> {out.name}",
                    flush=True,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            totals["errors"] += 1
            errors.append(f"{episode_dir.name}: {exc}")
            print(f"ERROR {episode_dir.name}: {exc}", flush=True)

    summary_path = (target / "convert_summary.json") if len(dirs) > 1 else None
    if summary_path is not None:
        summary_path.write_text(
            json.dumps({"totals": totals, "errors": errors}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print("DONE", json.dumps(totals, ensure_ascii=False))
    if errors:
        print("errors", len(errors))
        print("\n".join(errors[:20]))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
