"""Session-format hybrid loop: qwen35_session messages + App Tools on app switch.

Same conversation shape as official GUI-only eval (system once, query once,
assistant + screenshot, last_n). Qwen keeps ``tools=[mobile_use]`` and dumps
App Tools into the switch-turn user text. GLM / Z.AI puts current App Tools
into ``tools=`` and sends ``thinking.type=enabled`` + ``reasoning_effort``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from android_world.agents import infer
from android_world.agents.session.agent import Qwen35Session, Qwen35ThoughtSession
from bench_env.hybrid_agent.artifacts import HybridArtifactRecorder
from bench_env.hybrid_agent.browser_tools import BrowserToolClient
from bench_env.hybrid_agent.modes import GUI_ONLY_MODE, validate_interaction_mode
from bench_env.hybrid_agent.types import (
    FunctionCall,
    HybridRunResult,
    HybridStepRecord,
    StepFeedback,
)
from bench_env.hybrid_benchmark.runner import (
    HybridBenchmarkConfig,
    _create_env,
    _jsonable_config,
    _load_tasks,
    _task_payload,
    _write_results,
)

try:
    from bench_env.hybrid_benchmark import live_console
except ImportError:
    class _NoLiveConsole:
        async def attach(self, env: Any) -> None:
            return None

        async def close(self, env: Any) -> None:
            return None

    live_console = _NoLiveConsole()

try:
    from bench_env.hybrid_benchmark.runner import _raise_headed_window
except ImportError:
    async def _raise_headed_window(env: Any) -> None:
        page = getattr(env, "page", None)
        if page is None:
            return
        try:
            await page.bring_to_front()
        except Exception:
            pass
from bench_env.task.judge import JudgeInput
from openmobile_session_agent import (
    _ScreenshotPumpEnv,
    _obs_to_pixels,
    _parsed_to_action,
)

from session_tools import (
    GLM_SESSION_ADDENDUM,
    HYBRID_SESSION_ADDENDUM,
    SECRETARY_EXTRA_POLICY,
    THOUGHT_HYBRID_ADDENDUM,
    build_session_tools,
    format_app_switch_user_text,
    format_app_switch_user_text_native,
    stable_session_tools,
)


def format_live_action(call: FunctionCall | None) -> str:
    if call is None:
        return ""
    if call.name != "mobile_use":
        return call.name
    args = call.arguments or {}
    return str(args.get("action") or args.get("action_type") or "mobile_use")


def format_live_cli(call: FunctionCall | None) -> str:
    if call is None:
        return ""
    args = dict(call.arguments or {})
    if call.name != "mobile_use":
        payload = json.dumps(args, ensure_ascii=False) if args else ""
        return f"{call.name} {payload}".strip()
    action = str(args.get("action") or args.get("action_type") or call.name)
    bits = [action]
    for key in ("coordinate", "coordinate2", "text", "button", "status", "time"):
        value = args.get(key)
        if value is not None and value != "":
            bits.append(repr(value) if key == "text" else str(value))
    return " ".join(bits)


def _extract_thought(raw: str) -> str:
    text = str(raw or "")
    for pattern in (
        r"<think>\s*([\s\S]*?)\s*</think>",
        r"<thinking>\s*([\s\S]*?)\s*</thinking>",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    from bench_env.hybrid_agent.xml_protocol import _extract_label

    return _extract_label(text, "Thought")


def _publish_live(
    recorder: HybridArtifactRecorder,
    *,
    step: int,
    phase: str,
    current_app: str | None,
    call: FunctionCall | None = None,
    raw: str = "",
    thought: str = "",
) -> None:
    if not hasattr(recorder, "publish_live"):
        return
    is_tool = bool(call is not None and call.name != "mobile_use")
    recorder.publish_live({
        "step": step,
        "phase": phase,
        "kind": "tool" if is_tool else "gui",
        "kind_label": "App Tool" if is_tool else "GUI",
        "app": current_app or "launcher",
        "thought": thought or _extract_thought(raw),
        "action": format_live_action(call) if call is not None else "",
        "cli": format_live_cli(call) if call is not None else "",
        "tool_return": "",
    })


def _load_results_jsonl(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "results.jsonl"
    by_id: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return by_id
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = row.get("task_id")
        if task_id:
            by_id[str(task_id)] = row
    return by_id


def _episode_complete(output_dir: Path, task_id: str) -> bool:
    """A finished episode always writes ``trace.json``; crashes usually do not."""
    return (output_dir / "episodes" / task_id / "trace.json").is_file()


def _resumed_row(task: Any, existing: dict[str, Any] | None, episode_dir: Path) -> dict[str, Any]:
    if existing and not existing.get("error"):
        row = dict(existing)
        row["skipped"] = True
        return row
    return {
        **_task_payload(task),
        "skipped": True,
        "worker_id": existing.get("worker_id") if existing else None,
        "interaction_mode": existing.get("interaction_mode") if existing else None,
        "runtime": "qwen35_session",
        "success": bool((existing or {}).get("success", True)),
        "clean": bool((existing or {}).get("clean", True)),
        "progress": (existing or {}).get("progress", 1.0),
        "issues": (existing or {}).get("issues") or [],
        "warnings": (existing or {}).get("warnings") or [],
        "judge_error": (existing or {}).get("judge_error"),
        "stop_reason": (existing or {}).get("stop_reason") or "resumed",
        "steps": (existing or {}).get("steps") or 0,
        "agent_answer": (existing or {}).get("agent_answer"),
        "agent_message": (existing or {}).get("agent_message"),
        "tool_calls": (existing or {}).get("tool_calls") or [],
        "app_tool_calls": (existing or {}).get("app_tool_calls") or [],
        "runtime_s": (existing or {}).get("runtime_s") or 0,
        "artifact_dir": str(episode_dir),
        "error": None,
    }


def _agent_runtime(config: HybridBenchmarkConfig, explicit: str = "") -> str:
    name = str(explicit or getattr(config, "agent_runtime", "") or "").strip()
    if name:
        return name
    return "qwen35_session"


def create_session(
    config: HybridBenchmarkConfig,
    *,
    last_n: int = 3,
    enable_thinking: bool = True,
    reasoning_effort: str = "low",
    agent_runtime: str = "",
) -> tuple[_ScreenshotPumpEnv, Qwen35Session]:
    pump = _ScreenshotPumpEnv()
    extra = config.model_extra_body or {}
    thinking = extra.get("chat_template_kwargs") or {}
    if "enable_thinking" in thinking:
        enable_thinking = bool(thinking["enable_thinking"])
    runtime = _agent_runtime(config, agent_runtime)
    common = dict(
        wait_after_action_seconds=0.0,
        model_base_url=config.model_base_url,
        model_api_key=config.model_api_key or "EMPTY",
        model_name=config.model_name,
        last_n=last_n,
        reasoning_effort=reasoning_effort,
    )
    if runtime == "qwen35_thought_session":
        session: Qwen35Session = Qwen35ThoughtSession(
            pump,
            infer.Gpt4Wrapper("gpt-4o"),
            **common,
        )
    else:
        session = Qwen35Session(
            pump,
            infer.Gpt4Wrapper("gpt-4o"),
            enable_thinking=enable_thinking,
            require_think_tags=False,
            qwen35_tool_call_mode="native",
            **common,
        )
    session.transition_pause = 0.0
    session._request_timeout = float(config.infer_timeout)
    if validate_interaction_mode(config.interaction_mode) != GUI_ONLY_MODE:
        if _is_thought_session(session):
            addendum = THOUGHT_HYBRID_ADDENDUM
        elif _uses_native_app_tools(session):
            addendum = GLM_SESSION_ADDENDUM
        else:
            addendum = HYBRID_SESSION_ADDENDUM
        session._session_system_prompt = (
            str(session._session_system_prompt).rstrip() + "\n" + addendum
        )
    if str(config.suite).startswith("hybrid_demo_crossapp"):
        session._session_system_prompt = (
            str(session._session_system_prompt).rstrip() + "\n" + SECRETARY_EXTRA_POLICY
        )
    return pump, session


def _uses_native_app_tools(session: Qwen35Session) -> bool:
    return str(getattr(getattr(session, "_backend", None), "key", "") or "") == "glm"


def _is_thought_session(session: Qwen35Session) -> bool:
    return str(getattr(getattr(session, "_backend", None), "key", "") or "") == "qwen_thought"


def _looks_like_app_tool_name(name: str) -> bool:
    text = str(name or "").strip()
    return "__" in text and " " not in text and not text.startswith("<")


async def _run_session_episode(
    env: Any,
    pump: _ScreenshotPumpEnv,
    session: Qwen35Session,
    task: Any,
    config: HybridBenchmarkConfig,
    recorder: HybridArtifactRecorder,
) -> HybridRunResult:
    mode = validate_interaction_mode(config.interaction_mode)
    browser = BrowserToolClient(env.page)
    session.reset(go_home_on_reset=False)
    session.save_dir = str(recorder.output_dir)
    obs = await env.get_observation()
    feedback: StepFeedback | None = None
    trace: list[HybridStepRecord] = []
    stop_reason = "MAX_STEPS"
    max_steps = config.max_steps or task.max_steps or 45
    local_seq = 0
    previous_app: str | None = None

    for step in range(1, max_steps + 1):
        if mode == GUI_ONLY_MODE:
            current_app = await browser.resolve_foreground_app(obs.current_app)
            definitions: list[dict[str, Any]] = []
        else:
            tool_context = await browser.discover_tools(obs.current_app)
            current_app = tool_context.app
            definitions = tool_context.tools
        definitions_by_name = {
            str(item.get("name")): item
            for item in definitions
            if isinstance(item, dict) and item.get("name")
        }
        available_names = ["mobile_use", *definitions_by_name]
        native_tools = _uses_native_app_tools(session)
        if _is_thought_session(session):
            session.set_turn_tools(None)
        elif native_tools and mode != GUI_ONLY_MODE:
            session.set_turn_tools(build_session_tools(definitions, glm=True))
        else:
            session.set_turn_tools(stable_session_tools(glm=native_tools))
        app_key = str(current_app or "")
        if (
            mode != GUI_ONLY_MODE
            and previous_app is not None
            and app_key != previous_app
        ):
            switch_text = (
                format_app_switch_user_text_native(current_app, definitions)
                if native_tools
                else format_app_switch_user_text(current_app, definitions)
            )
            session.set_next_user_text(switch_text)
        if feedback is not None and feedback.name != "mobile_use":
            session.set_next_observation_text(
                json.dumps(feedback.to_dict(), ensure_ascii=False)
            )
        previous_app = app_key

        before_obs = obs
        pixels = _obs_to_pixels(obs)
        height, width = pixels.shape[:2]
        pump.feed(pixels)
        route_before = dict(obs.route)
        started_at = time.time()
        _publish_live(
            recorder,
            step=step,
            phase="thinking",
            current_app=current_app,
        )
        result = await asyncio.to_thread(session.step, task.description)
        data = result.data or {}
        parsed = data.get("parsed") or {}
        raw = str(data.get("response") or "")
        if (
            parsed.get("action_type") == "open_app"
            and _looks_like_app_tool_name(str(parsed.get("app_name") or ""))
        ):
            parsed = {
                "action_type": "mcp_tool",
                "tool_name": str(parsed.get("app_name") or ""),
                "arguments": {},
            }

        if parsed.get("action_type") == "mcp_tool":
            name = str(parsed.get("tool_name") or "")
            arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
            call = FunctionCall(name=name, arguments=arguments)
            _publish_live(
                recorder, step=step, phase="acting", current_app=current_app, call=call, raw=raw
            )
            definition = definitions_by_name.get(name)
            if definition is None:
                local_seq += 1
                feedback = StepFeedback(
                    f"call_{local_seq}",
                    name,
                    False,
                    {
                        "code": "TOOL_NOT_AVAILABLE",
                        "message": (
                            f"{name} is not exposed by the foreground app "
                            f"{current_app or '<none>'}"
                        ),
                    },
                )
                ui_effect = "none"
            else:
                actual = await browser.discover_tools(obs.current_app)
                actual_names = {
                    str(item.get("name"))
                    for item in actual.tools
                    if isinstance(item, dict) and item.get("name")
                }
                if actual.app != current_app or name not in actual_names:
                    local_seq += 1
                    feedback = StepFeedback(
                        f"call_{local_seq}",
                        name,
                        False,
                        {
                            "code": "STALE_TOOL_CONTEXT",
                            "message": (
                                f"foreground tool context changed from "
                                f"{current_app or '<system>'} to {actual.app or '<system>'}"
                            ),
                        },
                    )
                else:
                    raw_result = await browser.call_tool(name, arguments)
                    call_id = str(raw_result.get("callId") or f"call_{step}")
                    if bool(raw_result.get("ok")):
                        feedback = StepFeedback(call_id, name, True, raw_result.get("output"))
                    else:
                        feedback = StepFeedback(
                            call_id,
                            name,
                            False,
                            raw_result.get("error")
                            or {"code": "UNKNOWN_TOOL_ERROR", "message": "Tool failed"},
                        )
                ui_effect = str(
                    ((definition or {}).get("annotations") or {}).get("uiEffect") or "none"
                )
            obs = await env.get_observation()
            done = False
        elif parsed:
            action = _parsed_to_action(parsed, width, height, raw)
            call = FunctionCall(
                name="mobile_use",
                arguments={
                    k: v
                    for k, v in parsed.items()
                    if k != "action_type"
                }
                | {"action": parsed.get("action_type")},
            )
            _publish_live(
                recorder, step=step, phase="acting", current_app=current_app, call=call, raw=raw
            )
            step_result = await env.step(action)
            local_seq += 1
            feedback = StepFeedback(
                f"call_{local_seq}",
                "mobile_use",
                True,
                {
                    "action_type": action.action_type.value,
                    "done": step_result.done,
                    **(step_result.info or {}),
                },
            )
            obs = step_result.observation
            done = bool(result.done or step_result.done)
            ui_effect = "navigation" if action.action_type.value not in {
                "WAIT",
                "ANSWER",
                "COMPLETE",
                "ABORT",
            } else "none"
        else:
            call = None
            local_seq += 1
            feedback = StepFeedback(
                f"call_{local_seq}",
                "protocol",
                False,
                {
                    "code": "INVALID_TOOL_CALL",
                    "message": str(data.get("summary") or "No valid tool_call"),
                },
            )
            done = True
            ui_effect = "none"

        record = HybridStepRecord(
            step=step,
            current_app=current_app,
            available_tools=available_names,
            response=raw,
            call=call,
            feedback=feedback,
            ui_effect=ui_effect,
            route_before=route_before,
            route_after=dict(obs.route),
            started_at=started_at,
            finished_at=time.time(),
        )
        trace.append(record)
        recorder.record_step(record, before_obs, obs, list(session._session_messages))
        if not config.quiet:
            label = call.name if call else "none"
            print(
                f"[session step {step}] {label} -> ok={feedback.ok} "
                f"app={obs.current_app or 'launcher'}"
            )
        if done:
            stop_reason = "TERMINATED"
            if isinstance(feedback.result, dict) and feedback.result.get("stop_reason"):
                stop_reason = str(feedback.result["stop_reason"])
            elif result.done:
                stop_reason = str(data.get("summary") or "TERMINATED")
            break

    run_result = HybridRunResult(
        trace=trace,
        stop_reason=stop_reason,
        task=task.description,
        interaction_mode=mode,
        agent_answer=getattr(env, "agent_answer", None),
        agent_message=getattr(env, "agent_message", None),
    )
    recorder.finish(run_result)
    try:
        payload = session.export_session_record(
            episode_id=str(task.id),
            goal=task.description,
            save_dir=str(recorder.output_dir),
        )
        (recorder.output_dir / "session.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"Failed to export session.json: {error}")
    return run_result


async def _run_one(
    env: Any,
    pump: _ScreenshotPumpEnv,
    session: Qwen35Session,
    task: Any,
    config: HybridBenchmarkConfig,
    worker_id: int,
) -> dict[str, Any]:
    started = time.time()
    episode_dir = config.output_dir / "episodes" / task.id
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "task.json").write_text(
        json.dumps(_task_payload(task), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    initial_obs = None
    try:
        initial_obs = await task.setup(env)
        if not config.headless:
            await _raise_headed_window(env)
        recorder = HybridArtifactRecorder(episode_dir)
        pump_live = getattr(env, "_live_pump", None)
        if pump_live is not None:
            recorder.live_sink = pump_live.submit
        start_delay = float(getattr(config, "start_delay", 0) or 0)
        if start_delay > 0:
            _publish_live(
                recorder,
                step=0,
                phase="thinking",
                current_app="launcher",
                thought=f"窗口已打开，先等 {int(start_delay)} 秒再开始，方便录屏。",
            )
            await asyncio.sleep(start_delay)
        run_result = await _run_session_episode(
            env, pump, session, task, config, recorder
        )
        final_obs = await env.get_observation()
        final_state = await env.get_state(required_apps=list(task.apps))
        final_obs = replace(final_obs, state=final_state)
        judge = task.evaluate(
            JudgeInput(
                init_obs=initial_obs,
                last_obs=final_obs,
                answer=run_result.agent_answer,
            )
        )
        calls = [
            record.call.name
            for record in run_result.trace
            if record.call is not None
        ]
        return {
            **_task_payload(task),
            "worker_id": worker_id,
            "interaction_mode": config.interaction_mode,
            "runtime": "qwen35_session",
            "success": judge.success,
            "clean": judge.clean,
            "progress": judge.progress,
            "issues": judge.issues,
            "warnings": judge.warnings,
            "judge_error": judge.judge_error,
            "stop_reason": run_result.stop_reason,
            "steps": len(run_result.trace),
            "agent_answer": run_result.agent_answer,
            "agent_message": run_result.agent_message,
            "tool_calls": calls,
            "app_tool_calls": [name for name in calls if name != "mobile_use"],
            "runtime_s": time.time() - started,
            "artifact_dir": str(episode_dir),
            "error": None,
        }
    except Exception as error:
        return {
            **_task_payload(task),
            "worker_id": worker_id,
            "interaction_mode": config.interaction_mode,
            "runtime": "qwen35_session",
            "success": False,
            "clean": True,
            "progress": 0.0,
            "issues": [],
            "warnings": [],
            "judge_error": None,
            "stop_reason": "ERROR",
            "steps": 0,
            "agent_answer": None,
            "agent_message": None,
            "tool_calls": [],
            "app_tool_calls": [],
            "runtime_s": time.time() - started,
            "artifact_dir": str(episode_dir),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        try:
            task.teardown(env)
        except Exception:
            pass


async def run_session_benchmark(
    config: HybridBenchmarkConfig,
    *,
    last_n: int = 3,
    resume: bool = True,
    reasoning_effort: str = "low",
    agent_runtime: str = "qwen35_session",
) -> list[dict[str, Any]]:
    config = replace(
        config,
        interaction_mode=validate_interaction_mode(config.interaction_mode),
    )
    tasks = _load_tasks(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _agent_runtime(config, agent_runtime)
    meta = _jsonable_config(config)
    meta["runtime"] = runtime
    meta["last_n"] = last_n
    meta["resume"] = bool(resume)
    meta["reasoning_effort"] = reasoning_effort
    meta["enable_thinking"] = runtime != "qwen35_thought_session"
    (config.output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    existing = _load_results_jsonl(config.output_dir) if resume else {}
    pending: list[Any] = []
    results: list[dict[str, Any]] = []
    for task in tasks:
        if resume and _episode_complete(config.output_dir, task.id):
            results.append(
                _resumed_row(
                    task,
                    existing.get(task.id),
                    config.output_dir / "episodes" / task.id,
                )
            )
            continue
        pending.append(task)
    if resume:
        print(
            f"Resume: {len(results)} already have trace.json, "
            f"{len(pending)} remaining (of {len(tasks)})",
            flush=True,
        )
    if not pending:
        _write_results(config, tasks, results)
        return sorted(results, key=lambda row: row["task_id"])

    queue: asyncio.Queue[Any] = asyncio.Queue()
    for task in pending:
        queue.put_nowait(task)
    lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        env = _create_env(config, worker_id)
        pump, session = create_session(
            config,
            last_n=last_n,
            reasoning_effort=reasoning_effort,
            agent_runtime=runtime,
        )
        await env.start()
        if getattr(config, "live_console", False) and not config.headless:
            env._live_pump = await live_console.attach(env)
        try:
            while True:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                row = await _run_one(env, pump, session, task, config, worker_id)
                async with lock:
                    results.append(row)
                    _write_results(config, tasks, results)
                    if not config.quiet:
                        status = "PASS" if row["success"] else "FAIL"
                        print(
                            f"[{len(results)}/{len(tasks)}] {status} {task.id} "
                            f"steps={row['steps']} stop={row['stop_reason']}",
                            flush=True,
                        )
                        if row.get("error"):
                            print(f"  error: {row['error']}", flush=True)
                queue.task_done()
        finally:
            await live_console.close(env)
            await env.close()

    workers = min(max(1, config.parallel), len(pending))
    await asyncio.gather(*(worker(index + 1) for index in range(workers)))
    _write_results(config, tasks, results)
    return sorted(results, key=lambda row: row["task_id"])
