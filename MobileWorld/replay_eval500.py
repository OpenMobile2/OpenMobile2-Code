"""Replay saved MobileWorld MCP-eval trajectories, then call /task/eval again.

Use this for tasks that finished the episode but scored 0 because
``/task/eval`` returned HTTP 500. Does not call the model.

Default is GUI-only replay: skip MCP tool calls. MCP waits (quota retries of
10/20/40/80s) are what originally left the emulator idle; replaying those
gaps makes later clicks land on a dimmed / ANR / home screen ("empty tap").
Typed text from the original ``mobile_use type`` steps is already in the
trace, so maps/GitHub results do not need to be fetched again for scoring.

Before each click/long_press/swipe, compare the live UI tree to
``metadata.json`` from that step (package + widget under the tap). On drift,
wait for the UI to settle and retry; do not blindly tap.

  python replay_eval500.py `
    --eval-dir <OPENMOBILE_ROOT>\\MobileWorld\\eval_runs\\YOUR_MODEL `
    --hosts http://127.0.0.1:6800,http://127.0.0.1:6801 `
    --parallel 2

  python replay_eval500.py --eval-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_AW_ROOT = _HERE.parent / "AndroidWorld"
if str(_AW_ROOT) not in sys.path:
    sys.path.insert(0, str(_AW_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

SUCCESS_THRESHOLD = 0.99
_EVAL_500_RE = re.compile(r"500 Server Error.*task/eval|/task/eval.*500", re.I)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.I | re.DOTALL)
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*\"?\s*>(.*?)</function>",
    re.I | re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_.:-]*)\s*\"?\s*>(.*?)</parameter>",
    re.I | re.DOTALL,
)
_GUI_POINTER = {"click", "long_press", "swipe", "left_click"}


def _qwen3vl_action_transform(action: str, arguments: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Same 0-1000 coordinate mapping as android_world.agents.utils.qwen3vl_action_transform."""
    if action in {"click", "left_click"}:
        x, y = (arguments.get("coordinate") or [0, 0])[:2]
        return {"action_type": "click", "x": x / 1000 * width, "y": y / 1000 * height}
    if action == "long_press":
        x, y = (arguments.get("coordinate") or [0, 0])[:2]
        return {"action_type": "long_press", "x": x / 1000 * width, "y": y / 1000 * height}
    if action == "swipe":
        x0, y0 = (arguments.get("coordinate") or [0, 0])[:2]
        x1, y1 = (arguments.get("coordinate2") or [0, 0])[:2]
        return {
            "action_type": "swipe",
            "x": x0 / 1000 * width,
            "y": y0 / 1000 * height,
            "x2": x1 / 1000 * width,
            "y2": y1 / 1000 * height,
        }
    if action == "type":
        return {"action_type": "input_text", "text": arguments.get("text", "")}
    if action == "system_button":
        button = str(arguments.get("button") or "").lower()
        mapping = {
            "home": "navigate_home",
            "back": "navigate_back",
            "enter": "keyboard_enter",
        }
        return {"action_type": mapping.get(button, "wait")}
    if action in {"open", "open_app"}:
        text = arguments.get("text") or arguments.get("app_name") or arguments.get("app") or ""
        return {"action_type": "open_app", "app_name": str(text)}
    if action == "wait":
        return {"action_type": "wait"}
    if action == "answer":
        return {"action_type": "answer", "text": arguments.get("text", "")}
    if action == "terminate":
        status = str(arguments.get("status") or "").lower()
        if status == "success":
            return {"action_type": "status", "goal_status": "complete"}
        return {"action_type": "status", "goal_status": "infeasible"}
    return {"action_type": "wait"}


def _decode_param(raw: str) -> Any:
    text = str(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _call_from_xml(response: str) -> dict[str, Any] | None:
    blocks = _TOOL_CALL_RE.findall(response or "")
    if not blocks:
        return None
    match = _FUNCTION_RE.search(blocks[-1].strip())
    if not match:
        return None
    arguments: dict[str, Any] = {}
    for item in _PARAMETER_RE.finditer(match.group(2)):
        arguments[item.group(1).strip()] = _decode_param(item.group(2))
    return {"name": match.group(1).strip(), "arguments": arguments}


def _call_from_native(tool_calls: Any) -> dict[str, Any] | None:
    if not tool_calls:
        return None
    raw = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
    if not isinstance(raw, dict):
        return None
    fn = raw.get("function") or {}
    if not isinstance(fn, dict):
        return None
    name = str(fn.get("name") or "").strip()
    if not name:
        return None
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


def _step_call(step: dict[str, Any]) -> dict[str, Any] | None:
    native = _call_from_native(step.get("model_tool_calls"))
    xml = _call_from_xml(str(step.get("model_output_stored") or step.get("model_output_raw") or ""))
    if native and native.get("name"):
        return native
    if xml and xml.get("name"):
        return xml
    return None


def _is_eval_500(reason: str) -> bool:
    text = str(reason or "")
    return bool(_EVAL_500_RE.search(text)) or (
        "500" in text and "task/eval" in text.replace("\\", "/")
    )


def _load_summary_tasks(eval_dir: Path) -> list[dict[str, Any]]:
    path = eval_dir / "eval_summary.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data.get("tasks") or []
        if tasks:
            return tasks
    out: list[dict[str, Any]] = []
    for child in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        eval_json = child / "eval.json"
        if not eval_json.is_file():
            continue
        try:
            row = json.loads(eval_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        row.setdefault("task_name", child.name)
        row.setdefault("save_dir", str(child))
        out.append(row)
    return out


def _task_dir(eval_dir: Path, row: dict[str, Any]) -> Path:
    save = row.get("save_dir")
    if save:
        path = Path(str(save))
        if path.is_dir():
            return path
    return eval_dir / str(row.get("task_name") or "")


def _load_session_steps(task_dir: Path) -> list[dict[str, Any]]:
    io_dir = task_dir / "session_io"
    if not io_dir.is_dir():
        return []
    steps = []
    for path in sorted(io_dir.glob("step*.json")):
        if path.name.endswith("_response_raw.txt"):
            continue
        try:
            steps.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    steps.sort(key=lambda s: int(s.get("step") or 0))
    return steps


def _load_metadata_steps(task_dir: Path) -> dict[int, dict[str, Any]]:
    path = task_dir / "metadata.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in data.get("steps") or []:
        if isinstance(item, dict) and item.get("step") is not None:
            out[int(item["step"])] = item
    return out


def _el_dict(el: Any) -> dict[str, Any]:
    if isinstance(el, dict):
        return el
    bbox = getattr(el, "bbox_pixels", None)
    box = None
    if bbox is not None:
        box = {
            "x_min": getattr(bbox, "x_min", None),
            "x_max": getattr(bbox, "x_max", None),
            "y_min": getattr(bbox, "y_min", None),
            "y_max": getattr(bbox, "y_max", None),
        }
    return {
        "package_name": getattr(el, "package_name", "") or "",
        "resource_id": getattr(el, "resource_id", "") or "",
        "text": getattr(el, "text", "") or "",
        "content_description": getattr(el, "content_description", "") or "",
        "class_name": getattr(el, "class_name", "") or "",
        "bbox_pixels": box,
        "is_clickable": bool(getattr(el, "is_clickable", False)),
    }


def _bbox(el: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = el.get("bbox_pixels") or {}
    try:
        return (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _foreground_package(elements: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for el in elements:
        pkg = str(el.get("package_name") or "").strip()
        if pkg and pkg != "android":
            counts[pkg] += 1
    return counts.most_common(1)[0][0] if counts else ""


def _element_at(elements: list[dict[str, Any]], x: float, y: float) -> dict[str, Any] | None:
    best = None
    best_area = float("inf")
    for el in elements:
        box = _bbox(el)
        if not box:
            continue
        x0, y0, x1, y1 = box
        if x0 <= x <= x1 and y0 <= y <= y1:
            area = max(1.0, (x1 - x0) * (y1 - y0))
            if area < best_area:
                best = el
                best_area = area
    return best


def _widget_key(el: dict[str, Any] | None) -> tuple[str, str, str]:
    if not el:
        return ("", "", "")
    return (
        str(el.get("resource_id") or ""),
        str(el.get("text") or "")[:80],
        str(el.get("content_description") or "")[:80],
    )


def _load_replay_record(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / "replay_eval.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_actions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in steps:
        call = _step_call(step)
        if not call:
            continue
        name = str(call.get("name") or "")
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        actions.append(
            {
                "step": int(step.get("step") or 0),
                "name": name,
                "arguments": args,
                "kind": "gui" if name == "mobile_use" else "mcp",
            }
        )
    return actions


def _pointer_xy(parsed: dict[str, Any]) -> tuple[float, float] | None:
    if parsed.get("action_type") in {"click", "long_press"}:
        if parsed.get("x") is None or parsed.get("y") is None:
            return None
        return float(parsed["x"]), float(parsed["y"])
    if parsed.get("action_type") == "swipe":
        if parsed.get("x") is None or parsed.get("y") is None:
            return None
        return float(parsed["x"]), float(parsed["y"])
    return None


def _wake_screen(env: Any) -> None:
    from android_world.env import json_action

    try:
        env.execute_action(json_action.JSONAction(action_type=json_action.WAIT))
    except Exception:
        pass


def _ui_matches(
    expected_elements: list[dict[str, Any]],
    live_elements: list[dict[str, Any]],
    xy: tuple[float, float] | None,
) -> tuple[bool, str]:
    exp_pkg = _foreground_package(expected_elements)
    live_pkg = _foreground_package(live_elements)
    if exp_pkg and live_pkg and exp_pkg != live_pkg:
        return False, f"package {live_pkg!r} != expected {exp_pkg!r}"
    if xy is None:
        return True, "ok"
    exp_el = _element_at(expected_elements, *xy)
    live_el = _element_at(live_elements, *xy)
    exp_key = _widget_key(exp_el)
    live_key = _widget_key(live_el)
    if exp_key == ("", "", "") and live_key == ("", "", ""):
        return False, f"no widget under tap {xy}"
    if exp_key[0] and live_key[0] and exp_key[0] != live_key[0]:
        return False, f"resource_id {live_key[0]!r} != {exp_key[0]!r}"
    if exp_key[1] and live_key[1] and exp_key[1] != live_key[1]:
        return False, f"text {live_key[1]!r} != {exp_key[1]!r}"
    if exp_key[2] and live_key[2] and exp_key[2] != live_key[2]:
        return False, f"desc {live_key[2]!r} != {exp_key[2]!r}"
    return True, "ok"


def _wait_aligned(
    env: Any,
    expected_elements: list[dict[str, Any]],
    xy: tuple[float, float] | None,
    *,
    retries: int,
) -> tuple[bool, str]:
    last = "no ui"
    for i in range(max(1, retries)):
        state = env.get_state(wait_to_stabilize=True)
        live = [_el_dict(el) for el in (state.ui_elements or [])]
        ok, last = _ui_matches(expected_elements, live, xy)
        if ok:
            return True, last
        if i + 1 < retries:
            _wake_screen(env)
            time.sleep(1.0)
    return False, last


def _score_with_retry(env: Any, task_name: str, retries: int) -> tuple[float | None, str]:
    last = ""
    for i in range(max(1, retries)):
        try:
            score, reason = env.get_task_score(task_name)
            return score, str(reason or "")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last = repr(exc)
            if i + 1 < retries:
                time.sleep(min(8.0, 2.0 * (i + 1)))
    return None, last


def replay_task(
    *,
    env: Any,
    task_name: str,
    task_dir: Path,
    skip_mcp: bool,
    drift_retries: int,
    on_drift: str,
    eval_retries: int,
) -> dict[str, Any]:
    steps = _load_session_steps(task_dir)
    actions = _parse_actions(steps)
    meta = _load_metadata_steps(task_dir)
    gui_actions = [a for a in actions if a["kind"] == "gui"]
    record: dict[str, Any] = {
        "task_name": task_name,
        "save_dir": str(task_dir),
        "n_steps": len(steps),
        "n_actions": len(actions),
        "n_gui": len(gui_actions),
        "n_mcp": sum(1 for a in actions if a["kind"] == "mcp"),
        "skip_mcp": skip_mcp,
        "played": [],
        "drift": [],
    }
    if not gui_actions:
        record.update(
            {
                "score": None,
                "reason": "no replayable GUI actions in session_io (native dump empty?)",
                "success": False,
                "skipped": True,
            }
        )
        return record

    env.initialize_task(task_name)
    width, height = env.logical_screen_size
    aborted = False
    abort_reason = ""
    mcp_bridge = None
    if not skip_mcp:
        from mw_session_mcp import MobileWorldMcpBridge

        mcp_bridge = MobileWorldMcpBridge()
        try:
            mcp_bridge.select_for_task(env.get_task_metadata(task_name))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record["mcp_select_error"] = repr(exc)

    from android_world.env import json_action

    for action in actions:
        name = action["name"]
        args = dict(action["arguments"] or {})
        step_idx = int(action["step"])
        if name != "mobile_use":
            played = {"step": step_idx, "kind": "mcp", "name": name, "skipped": skip_mcp}
            if mcp_bridge is not None:
                payload = mcp_bridge.call(name, args)
                played["ok"] = bool(payload.get("ok"))
                played["skipped"] = False
            record["played"].append(played)
            continue

        action_name = str(args.get("action") or "").strip().lower()
        if action_name in {"terminate", "status", "answer"}:
            record["played"].append({"step": step_idx, "kind": "gui", "action": action_name, "done": True})
            break
        parsed = _qwen3vl_action_transform(action_name, args, width, height)

        xy = _pointer_xy(parsed) if action_name in _GUI_POINTER else None
        expected = [_el_dict(el) for el in (meta.get(step_idx) or {}).get("ui_elements") or []]
        if expected and action_name in _GUI_POINTER:
            aligned, why = _wait_aligned(env, expected, xy, retries=drift_retries)
            if not aligned:
                drift = {"step": step_idx, "action": action_name, "xy": xy, "reason": why}
                record["drift"].append(drift)
                if on_drift == "abort":
                    aborted = True
                    abort_reason = f"ui drift at step {step_idx}: {why}"
                    break
                if on_drift == "skip-click":
                    record["played"].append({**drift, "skipped": True})
                    continue
                # continue: tap anyway
        try:
            env.execute_action(json_action.JSONAction(**parsed))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            record["played"].append(
                {"step": step_idx, "kind": "gui", "action": action_name, "error": repr(exc)}
            )
            continue
        record["played"].append({"step": step_idx, "kind": "gui", "action": action_name, "ok": True})

    if aborted:
        score, reason = None, abort_reason
    else:
        score, reason = _score_with_retry(env, task_name, eval_retries)
    try:
        env.tear_down_task(task_name)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        record["tear_down_error"] = repr(exc)

    record["score"] = score
    record["reason"] = reason
    record["success"] = score is not None and float(score) > SUCCESS_THRESHOLD
    record["skipped"] = False
    record["aborted"] = aborted
    return record


def _plan_tasks(eval_dir: Path) -> list[tuple[dict[str, Any], Path, dict[str, Any]]]:
    planned = []
    for row in _load_summary_tasks(eval_dir):
        if not _is_eval_500(str(row.get("reason") or "")):
            continue
        task_dir = _task_dir(eval_dir, row)
        if not task_dir.is_dir():
            continue
        steps = _load_session_steps(task_dir)
        actions = _parse_actions(steps)
        planned.append((row, task_dir, {"n_steps": len(steps), "n_gui": sum(1 for a in actions if a["kind"] == "gui"), "n_mcp": sum(1 for a in actions if a["kind"] == "mcp")}))
    return planned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True, help="Folder with eval_summary.json and per-task dirs")
    parser.add_argument("--hosts", default="http://127.0.0.1:6800", help="Comma-separated MW backends")
    parser.add_argument("--device", default="emulator-5554")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replay-mcp",
        action="store_true",
        help="Also re-call MCP tools (slow, quota, idle). Default skips them.",
    )
    parser.add_argument(
        "--on-drift",
        choices=["abort", "skip-click", "continue"],
        default="abort",
        help="When live UI does not match metadata before a tap. Default abort.",
    )
    parser.add_argument("--drift-retries", type=int, default=4)
    parser.add_argument("--eval-retries", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Replay even if replay_eval.json exists")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).expanduser().resolve()
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    planned = _plan_tasks(eval_dir)
    if args.limit:
        planned = planned[: args.limit]

    print(f"eval-dir={eval_dir}")
    print(f"eval-500 tasks={len(planned)} hosts={hosts} skip_mcp={not args.replay_mcp} on_drift={args.on_drift}")
    for row, task_dir, info in planned:
        name = row.get("task_name")
        print(
            f"  {name}: steps={info['n_steps']} gui={info['n_gui']} mcp={info['n_mcp']} "
            f"dir={task_dir.name}"
        )
    if args.dry_run:
        return
    if not planned:
        print("nothing to replay")
        return

    pending = []
    for row, task_dir, info in planned:
        if info["n_gui"] == 0:
            rec = {
                "task_name": row.get("task_name"),
                "save_dir": str(task_dir),
                **info,
                "score": None,
                "reason": "no replayable GUI actions in session_io",
                "success": False,
                "skipped": True,
            }
            _write_json(task_dir / "replay_eval.json", rec)
            print(f"SKIP {row.get('task_name')}: empty trace")
            continue
        existing = _load_replay_record(task_dir)
        if existing and existing.get("score") is not None and not args.overwrite:
            print(f"RESUME skip {row.get('task_name')} score={existing.get('score')}")
            continue
        pending.append((str(row.get("task_name")), task_dir, info))

    def _worker(item: tuple[str, Path, dict[str, Any]], host: str) -> dict[str, Any]:
        task_name, task_dir, _info = item
        from mw_env import MobileWorldEnv

        env = MobileWorldEnv(base_url=host, device=args.device, step_wait_time=1.5)
        print(f"[{host}] replay {task_name}")
        started = time.time()
        rec = replay_task(
            env=env,
            task_name=task_name,
            task_dir=task_dir,
            skip_mcp=not args.replay_mcp,
            drift_retries=args.drift_retries,
            on_drift=args.on_drift,
            eval_retries=args.eval_retries,
        )
        rec["host"] = host
        rec["duration_seconds"] = round(time.time() - started, 2)
        _write_json(task_dir / "replay_eval.json", rec)
        flag = "PASS" if rec.get("success") else ("DRIFT" if rec.get("aborted") else "FAIL")
        print(
            f"[{host}] {flag} {task_name} score={rec.get('score')} "
            f"gui={rec.get('n_gui')} drift={len(rec.get('drift') or [])} "
            f"reason={(rec.get('reason') or '')[:120]}"
        )
        return rec

    results: list[dict[str, Any]] = []
    workers = max(1, min(int(args.parallel), len(hosts), len(pending) or 1))
    if workers <= 1 or len(pending) <= 1:
        host = hosts[0]
        for item in pending:
            results.append(_worker(item, host))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = []
            for i, item in enumerate(pending):
                host = hosts[i % len(hosts)]
                futs.append(pool.submit(_worker, item, host))
            for fut in as_completed(futs):
                results.append(fut.result())

    scored = [r for r in results if r.get("score") is not None]
    success = [r for r in scored if float(r["score"]) > SUCCESS_THRESHOLD]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_dir": str(eval_dir),
        "hosts": hosts,
        "skip_mcp": not args.replay_mcp,
        "on_drift": args.on_drift,
        "total_eval500": len(planned),
        "replayed": len(results),
        "scored": len(scored),
        "success_count": len(success),
        "success_rate": (len(success) / len(scored)) if scored else 0.0,
        "tasks": results,
    }
    out = eval_dir / "replay_eval_summary.json"
    _write_json(out, summary)
    print(f"wrote {out} scored={len(scored)} success={len(success)}")


if __name__ == "__main__":
    main()
