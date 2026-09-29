"""
DIY rollout on MobileWorld for synthesized instructions.

Input JSON (same as AndroidWorld/run_diy.py):
[
  {"base_task_name": "...", "instruction": "...", "sample_id": "...", "task_id": "..."},
  ...
]

Uses OpenMobile agents (qwen3vl / qwen3vl_switching / ...) against MobileWorldEnv,
and writes the same save_dir layout expected by process_trajs.py / convert_traj.py.

Run from OpenMobile-Code/AndroidWorld so agent imports resolve:
  python ../MobileWorld/run_diy_mw.py \
    --input_json ../task_synthesis/output/mobileworld_explore/synthesized_tasks_mobileworld_explore_final.json \
    --output_dir runs/mobileworld_explore_rollout \
    --aw_host http://127.0.0.1:6800 \
    --agent_name qwen3vl \
    --qwen3vl_model_base_url http://.../v1 \
    --qwen3vl_model_name ...
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

from absl import app
from absl import flags
from absl import logging

_HERE = Path(__file__).resolve().parent
_AW_ROOT = _HERE.parent / "AndroidWorld"
if str(_AW_ROOT) not in sys.path:
    sys.path.insert(0, str(_AW_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from android_world.agents import agent_factory  # noqa: E402
from android_world.agents import base_agent  # noqa: E402
from android_world.agents.session_runtimes import list_runtimes  # noqa: E402
from android_world.episode_runner import run_episode  # noqa: E402
from android_world.env import interface  # noqa: E402

from mw_env import MobileWorldEnv  # noqa: E402

logging.set_verbosity(logging.WARNING)

_AW_HOST = flags.DEFINE_string("aw_host", "http://127.0.0.1:6800", "MobileWorld backend URL.")
_DEVICE = flags.DEFINE_string("device", "emulator-5554", "Device id.")
_RUNTIME = flags.DEFINE_string(
    "runtime",
    "",
    "Shared runtime preset (preferred). Empty = use --agent_name + prompt flags.\n"
    "Choices:\n" + list_runtimes(),
)
_AGENT_NAME = flags.DEFINE_string(
    "agent_name",
    "qwen3vl",
    "Agent name (legacy; ignored when --runtime is set).",
)
_INPUT_JSON = flags.DEFINE_string("input_json", None, "Synthesized tasks JSON.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "Output directory for trajectories.")
_PARAMS_DIR = flags.DEFINE_string(
    "params_dir",
    str(_HERE / "explore_results" / "params"),
    "Directory of exploration params pickles.",
)
_USE_PARAMS_INIT = flags.DEFINE_boolean(
    "use_params_init",
    True,
    "If True, require sample task_id and load matching params pickle (for bookkeeping).",
)
_MAX_N_STEPS = flags.DEFINE_integer("max_n_steps", 30, "Max steps per episode.")
_QWEN3VL_MODEL_BASE_URL = flags.DEFINE_string(
    "qwen3vl_model_base_url", "http://<openai-compatible-host>/v1", "OpenAI-compatible base URL."
)
_QWEN3VL_MODEL_API_KEY = flags.DEFINE_string("qwen3vl_model_api_key", "EMPTY", "API key.")
_QWEN3VL_MODEL_NAME = flags.DEFINE_string("qwen3vl_model_name", "", "Model name.")
_QWEN3VL_SWITCHING_WEAK_MODEL_BASE_URL = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_base_url",
    "http://127.0.0.1:32011/v1",
    "Weak model base URL for switching.",
)
_QWEN3VL_SWITCHING_WEAK_MODEL_API_KEY = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_api_key", "EMPTY", "Weak model API key."
)
_QWEN3VL_SWITCHING_WEAK_MODEL_NAME = flags.DEFINE_string(
    "qwen3vl_switching_weak_model_name",
    "Qwen2.5-VL-7B-Instruct-baseline",
    "Weak model name.",
)
_USE_MEMORY_PROMPT = flags.DEFINE_boolean(
    "use_memory_prompt",
    False,
    "If True, use memory-augmented baseline prompts (mutually exclusive with MemGUI).",
)
_USE_MEMGUI_PROMPT = flags.DEFINE_boolean(
    "use_memgui_prompt",
    False,
    "If True, use MemGUI prompts (task progress + Memory + memory_* actions).",
)
_MEMGUI_PROMPT_FORMAT = flags.DEFINE_enum(
    "memgui_prompt_format",
    "auto",
    ["auto", "qwen3vl", "qwen35"],
    "MemGUI format: auto / qwen3vl (JSON tool_call) / qwen35 (XML tool_call).",
)
_LAST_N = flags.DEFINE_integer(
    "last_n",
    1,
    "Number of recent screenshots visible to the model (session: last N image turns; "
    "flat prompts: last N images in the current user message).",
)
_ENABLE_THINKING = flags.DEFINE_boolean(
    "enable_thinking",
    True,
    "Session runtimes only: pass chat_template_kwargs.enable_thinking (default True).",
)
_REASONING_EFFORT = flags.DEFINE_enum(
    "reasoning_effort",
    "max",
    ["low", "high", "max"],
    "Session thinking strength: low | high | max (default max). "
    "Kimi: reasoning_effort; Gemini: google thinking_level.",
)
_SHARD_INDEX = flags.DEFINE_integer(
    "shard_index",
    0,
    "Worker shard index for parallel rollout (0-based).",
)
_NUM_SHARDS = flags.DEFINE_integer(
    "num_shards",
    1,
    "Total parallel rollout workers. Samples are split by sample_id hash.",
)


def _get_agent(env: interface.AsyncEnv) -> base_agent.EnvironmentInteractingAgent:
    agent, spec = agent_factory.create_gui_agent(
        env,
        runtime=_RUNTIME.value,
        agent_name=_AGENT_NAME.value,
        use_memgui_prompt=_USE_MEMGUI_PROMPT.value,
        use_memory_prompt=_USE_MEMORY_PROMPT.value,
        memgui_prompt_format=_MEMGUI_PROMPT_FORMAT.value,
        last_n=_LAST_N.value,
        enable_thinking=_ENABLE_THINKING.value,
        reasoning_effort=_REASONING_EFFORT.value,
        model_base_url=_QWEN3VL_MODEL_BASE_URL.value,
        model_api_key=_QWEN3VL_MODEL_API_KEY.value,
        model_name=_QWEN3VL_MODEL_NAME.value,
        weak_model_base_url=_QWEN3VL_SWITCHING_WEAK_MODEL_BASE_URL.value,
        weak_model_api_key=_QWEN3VL_SWITCHING_WEAK_MODEL_API_KEY.value,
        weak_model_name=_QWEN3VL_SWITCHING_WEAK_MODEL_NAME.value,
    )
    print(f"Runtime: {spec.name} (last_n={_LAST_N.value}; {spec.description})")
    return agent


def _read_samples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input_json must be a JSON list.")
    for i, item in enumerate(data):
        for k in ("base_task_name", "instruction"):
            if k not in item or not isinstance(item[k], str) or not item[k].strip():
                raise ValueError(f"Item {i} missing/invalid field: {k}")
        if "sample_id" not in item or item["sample_id"] is None:
            raise ValueError(f"Item {i} missing sample_id")
        if isinstance(item["sample_id"], int):
            item["sample_id"] = str(item["sample_id"])
    return data


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sample_shard(sample_id: str, num_shards: int) -> int:
    """Stable shard assignment from sample_id (process-independent)."""
    digest = hashlib.md5(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % int(num_shards)


def _main(argv: Sequence[str]) -> None:
    del argv
    if not _INPUT_JSON.value:
        raise ValueError("--input_json is required.")
    if not _OUTPUT_DIR.value:
        raise ValueError("--output_dir is required.")

    num_shards = int(_NUM_SHARDS.value)
    shard_index = int(_SHARD_INDEX.value)
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    samples = _read_samples(_INPUT_JSON.value)
    if num_shards > 1:
        before = len(samples)
        samples = [
            s for s in samples if _sample_shard(s["sample_id"], num_shards) == shard_index
        ]
        print(
            f"Shard {shard_index}/{num_shards}: {len(samples)}/{before} samples "
            f"(by sample_id hash)"
        )

    env = MobileWorldEnv(base_url=_AW_HOST.value, device=_DEVICE.value)
    if not env.health():
        raise RuntimeError(f"MobileWorld backend not healthy at {_AW_HOST.value}")

    agent = _get_agent(env)
    agent.transition_pause = None

    base_out = _OUTPUT_DIR.value
    if not os.path.isabs(base_out):
        base_out = str((_HERE / base_out).resolve())
    _ensure_dir(base_out)

    known_tasks = {t["name"] for t in env.list_tasks(enable_mcp=True, enable_user_interaction=True)}

    for idx, item in enumerate(samples):
        base_task_name = item["base_task_name"]
        instruction = item["instruction"]
        sample_id = item["sample_id"]

        if base_task_name not in known_tasks:
            print(f"[{idx+1}/{len(samples)}] SKIP unknown task: {base_task_name}")
            continue

        save_dir = os.path.join(base_out, sample_id + "_" + base_task_name)
        if os.path.exists(save_dir):
            print(f"[{idx+1}/{len(samples)}] SKIP (dir exists): {save_dir}")
            continue

        if _USE_PARAMS_INIT.value:
            task_id = item.get("task_id")
            if not task_id:
                print(f"[{idx+1}/{len(samples)}] SKIP missing task_id for params init")
                continue
            params_path = os.path.join(_PARAMS_DIR.value, f"{task_id}_params.pkl")
            if os.path.exists(params_path):
                with open(params_path, "rb") as f:
                    params = pickle.load(f)
                print(f"loaded params from {params_path}")
            else:
                print(f"params not found ({params_path}); continuing without pickle")
                params = {"task_name": base_task_name}
        else:
            params = {"task_name": base_task_name}

        _ensure_dir(save_dir)
        # Keep params next to rollout for debugging / resume.
        with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2, default=str)

        print(
            f"[{idx+1}/{len(samples)}] base_task={base_task_name} "
            f"sample_id={sample_id}"
        )
        try:
            env.initialize_task(base_task_name)
            print("init complete")
            run_episode(
                goal=instruction,
                agent=agent,
                max_n_steps=_MAX_N_STEPS.value,
                start_on_home_screen=True,
                termination_fn=None,
                save_dir=save_dir,
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            with open(os.path.join(save_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(repr(e))
            print(f"  ERROR: {e!r}")
        finally:
            try:
                env.tear_down_task(base_task_name)
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"  tear_down ERROR: {e!r}")

    env.close()


if __name__ == "__main__":
    app.run(_main)
