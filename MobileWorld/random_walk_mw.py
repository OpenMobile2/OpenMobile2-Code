"""
Random walk exploration on MobileWorld, producing the same explore_results
layout as AndroidWorld/random_walk_aw.py so process_explore.py / task_synthesis
can be reused unchanged.

Prerequisites:
  1. MobileWorld backends running, e.g. `uv run mw env run --count 1`
  2. From OpenMobile-Code/AndroidWorld (so android_world imports resolve):
       python ../MobileWorld/random_walk_mw.py --aw_host http://127.0.0.1:6800
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

# Allow importing sibling mw_env and AndroidWorld package.
_HERE = Path(__file__).resolve().parent
_AW_ROOT = _HERE.parent / "AndroidWorld"
if str(_AW_ROOT) not in sys.path:
    sys.path.insert(0, str(_AW_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from absl import app  # noqa: E402
from absl import flags  # noqa: E402
from absl import logging  # noqa: E402

from android_world.agents.t3a import _generate_ui_elements_description_list_full  # noqa: E402
from android_world.env import json_action  # noqa: E402

from mw_env import MobileWorldEnv  # noqa: E402

logging.set_verbosity(logging.WARNING)

_AW_HOST = flags.DEFINE_string(
    "aw_host",
    "http://127.0.0.1:6800",
    "MobileWorld backend URL.",
)
_DEVICE = flags.DEFINE_string("device", "emulator-5554", "ADB device id inside container.")
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    str(_HERE / "explore_results"),
    "Explore output directory (screenshots/trajectories/params).",
)
_TASK_RANDOM_SEED = flags.DEFINE_integer("task_random_seed", 222, "Seed for reproducibility.")
_NUM_STEP = flags.DEFINE_integer("num_step", 10, "Random-walk steps per task.")
_MAX_TASKS = flags.DEFINE_integer("max_tasks", 0, "0 = all GUI-only tasks.")
_ENABLE_MCP = flags.DEFINE_boolean("enable_mcp", False, "Include MCP tasks.")
_ENABLE_USER = flags.DEFINE_boolean(
    "enable_user_interaction", False, "Include user-interaction tasks."
)
_TASK = flags.DEFINE_string("task", None, "Optional single task name.")
_STEP_WAIT = flags.DEFINE_float(
    "step_wait_time",
    0.6,
    "Seconds to wait after actions / before stable screenshot.",
)
_POST_ACTION_SLEEP = flags.DEFINE_float(
    "post_action_sleep",
    0.8,
    "Extra sleep after each explore action before re-reading UI.",
)
_SHARD_INDEX = flags.DEFINE_integer(
    "shard_index",
    0,
    "Worker shard index for parallel explore (0-based).",
)
_NUM_SHARDS = flags.DEFINE_integer(
    "num_shards",
    1,
    "Total parallel explore workers. Tasks are split by hash.",
)


def _derive_instance_seed(task_random_seed: int, task_name: str, instance_id: int) -> int:
    unique_seed_str = f"{task_random_seed}_{task_name}_{instance_id}"
    return int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (2**32)


def generate_text_input(element_list_text, interactive_element, max_retries=3):
    def is_valid_response(response_text):
        return "\n" not in response_text

    def element_to_text(element):
        description = ""
        if getattr(element, "resource_id", None):
            description += f"Resource ID: {element.resource_id}\n"
        if getattr(element, "text", None):
            description += f"Text: {element.text}\n"
        if getattr(element, "content_description", None):
            description += f"Content Description: {element.content_description}\n"
        if getattr(element, "class_name", None):
            description += f"Class Name: {element.class_name}\n"
        if getattr(element, "hint_text", None):
            description += f"Hint Text: {element.hint_text}\n"
        return description or "No additional information."

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Test Input"

    prompt = f"""
    You are an intelligent input assistant. The current UI elements are as follows:
    {element_list_text}
    The selected editable element information is as follows:
    {element_to_text(interactive_element)}
    Based on the above information, please randomly generate a text content that a user might input into this element.
    Please return only the generated text without any additional explanation.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": (
            os.environ.get("OPENAI_TEXT_INPUT_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "gpt-4o-mini-2024-07-18"
        ),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
        "temperature": 0.7,
        "n": 1,
    }
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            text_input = response.json()["choices"][0]["message"]["content"].strip()
            if is_valid_response(text_input):
                return text_input
        except Exception as e:
            print(f"Error generating text input: {e}")
        retries += 1
        time.sleep(1)
    return "Test Input"


def get_state(env_state, logical_screen_size, ui_elements):
    element_list_text = _generate_ui_elements_description_list_full(
        ui_elements,
        logical_screen_size,
    )
    screen = Image.fromarray(env_state.pixels.astype("uint8"))
    return screen, element_list_text


def element_to_identifier(element):
    bbox = getattr(element, "bbox_pixels", None)
    bbox_dict = (
        {
            "x_min": bbox.x_min,
            "x_max": bbox.x_max,
            "y_min": bbox.y_min,
            "y_max": bbox.y_max,
        }
        if bbox
        else None
    )
    return {
        "resource_id": getattr(element, "resource_id", None),
        "text": getattr(element, "text", None),
        "content_description": getattr(element, "content_description", None),
        "class_name": getattr(element, "class_name", None),
        "bbox_pixels": bbox_dict,
        "hint_text": getattr(element, "hint_text", None),
        "is_checkable": getattr(element, "is_checkable", None),
        "is_enabled": getattr(element, "is_enabled", None),
        "is_visible": getattr(element, "is_visible", None),
        "is_clickable": getattr(element, "is_clickable", None),
        "is_editable": getattr(element, "is_editable", None),
        "is_focused": getattr(element, "is_focused", None),
        "is_focusable": getattr(element, "is_focusable", None),
        "is_long_clickable": getattr(element, "is_long_clickable", None),
        "is_scrollable": getattr(element, "is_scrollable", None),
        "is_selected": getattr(element, "is_selected", None),
        "package_name": getattr(element, "package_name", None),
        "resource_name": getattr(element, "resource_name", None),
    }


def filter_interactive_elements(elements, screen_width_height_px, unc_elem_pool):
    """Keep only actionable controls.

    UIAutomator dumps many enabled/visible nodes (labels, containers) that do not
    change the screen when clicked. Require an interactive flag and reject tiny /
    near-fullscreen boxes which are usually noise.
    """
    interactive_elements = []
    screen_width, screen_height = screen_width_height_px
    screen_area = max(1, screen_width * screen_height)
    excluded_packages = {
        "com.google.android.inputmethod.latin",
        "com.android.systemui",
        "com.android.launcher3",
    }
    for index, element in enumerate(elements):
        if element.package_name in excluded_packages:
            continue
        actionable = bool(
            element.is_clickable
            or element.is_editable
            or element.is_long_clickable
            or element.is_checkable
            or element.is_scrollable
        )
        if not (element.is_enabled and element.is_visible and element.bbox_pixels and actionable):
            continue
        x_min = element.bbox_pixels.x_min
        x_max = element.bbox_pixels.x_max
        y_min = element.bbox_pixels.y_min
        y_max = element.bbox_pixels.y_max
        if x_min >= x_max or x_min >= screen_width or x_max <= 0:
            continue
        if y_min >= y_max or y_min >= screen_height or y_max <= 0:
            continue
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        # Tiny hit targets and near-fullscreen wrappers are rarely useful.
        if width < 16 or height < 16 or area < 256:
            continue
        if area / screen_area > 0.85:
            continue
        element_identifier_str = json.dumps(element_to_identifier(element), sort_keys=True)
        if element_identifier_str not in unc_elem_pool:
            interactive_elements.append([index, element])
    return interactive_elements


def sample_action_element(element, element_list_text):
    index, interactive_element = element
    if interactive_element.is_editable:
        text_input = generate_text_input(element_list_text, interactive_element)
        return {"action_type": "input_text", "text": text_input, "index": index}
    actions = [{"action_type": "click", "index": index}]
    if interactive_element.is_long_clickable:
        actions.append({"action_type": "long_press", "index": index})
    if len(actions) == 1:
        return actions[0]
    return random.choices(actions, weights=[9, 1], k=1)[0]


def has_screen_changed(before_elements, after_elements):
    before_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in before_elements
        if elem.package_name != "com.android.systemui"
    )
    after_set = set(
        json.dumps(element_to_identifier(elem), sort_keys=True)
        for elem in after_elements
        if elem.package_name != "com.android.systemui"
    )
    return before_set != after_set


def _pick_app_name(task_meta: dict) -> str:
    apps = task_meta.get("apps") or []
    if not apps:
        return "Home"
    # Prefer a concrete app over MCP placeholders when multiple.
    for app in apps:
        if not str(app).startswith("MCP-"):
            return app
    return apps[0]


def _main(_) -> None:
    base_dir = _OUTPUT_DIR.value
    screen_dir = os.path.join(base_dir, "screenshots")
    traj_dir = os.path.join(base_dir, "trajectories")
    params_dir = os.path.join(base_dir, "params")
    unc_pool_dir = os.path.join(base_dir, "unclickable_elem_pool")
    for d in (base_dir, screen_dir, traj_dir, params_dir, unc_pool_dir):
        os.makedirs(d, exist_ok=True)

    env = MobileWorldEnv(
        base_url=_AW_HOST.value,
        device=_DEVICE.value,
        step_wait_time=_STEP_WAIT.value,
    )
    if not env.health():
        raise RuntimeError(
            f"MobileWorld backend not healthy at {_AW_HOST.value}. "
            "Start it with `uv run mw env run --count 1` first."
        )
    print(f"Connected to MobileWorld at {_AW_HOST.value}")
    env.reset(go_home=True)

    tasks = env.list_tasks(
        enable_mcp=_ENABLE_MCP.value,
        enable_user_interaction=_ENABLE_USER.value,
    )
    if _TASK.value:
        tasks = [t for t in tasks if t["name"] == _TASK.value]
        if not tasks:
            raise ValueError(f"Task not found: {_TASK.value}")

    # Deterministic shard split for multi-worker explore.
    if _NUM_SHARDS.value > 1:
        if not (0 <= _SHARD_INDEX.value < _NUM_SHARDS.value):
            raise ValueError("--shard_index must be in [0, num_shards)")
        filtered = []
        for t in tasks:
            digest = int(hashlib.md5(t["name"].encode("utf-8")).hexdigest(), 16)
            if digest % _NUM_SHARDS.value == _SHARD_INDEX.value:
                filtered.append(t)
        tasks = filtered
        print(
            f"Shard {_SHARD_INDEX.value}/{_NUM_SHARDS.value}: {len(tasks)} tasks after split"
        )

    random.shuffle(tasks)
    if _MAX_TASKS.value > 0:
        tasks = tasks[: _MAX_TASKS.value]
    print(f"Exploring {len(tasks)} tasks")

    for task_id, task_info in enumerate(tasks):
        task_name = task_info["name"]
        task_uuid = str(uuid.uuid4())
        meta = env.get_task_metadata(task_name)
        app_name = _pick_app_name(meta)

        seed = _derive_instance_seed(_TASK_RANDOM_SEED.value, task_name, 0)
        params = {"seed": seed, "task_name": task_name, "apps": meta.get("apps", [])}
        with open(os.path.join(params_dir, task_uuid + "_params.pkl"), "wb") as f:
            pickle.dump(params, f)

        print(f"[{task_id+1}/{len(tasks)}] Explore task={task_name} app={app_name}")
        try:
            if not env.wait_until_healthy(retries=5, sleep_s=3.0):
                print(
                    f"  SKIP {task_name}: backend still unhealthy at {_AW_HOST.value}"
                )
                continue
            env.initialize_task(task_name)
            env.reset(go_home=True)
            if app_name != "Home":
                env.execute_action(
                    json_action.JSONAction(action_type="open_app", app_name=app_name)
                )
                time.sleep(3.0)

            unc_elem_pool_path = os.path.join(unc_pool_dir, f"{app_name}.json")
            if os.path.exists(unc_elem_pool_path):
                try:
                    unc_elem_pool = set(json.load(open(unc_elem_pool_path, "r", encoding="utf-8")))
                except Exception:
                    unc_elem_pool = set()
            else:
                unc_elem_pool = set()

            trajectory = []
            for i in range(_NUM_STEP.value):
                env_state = env.get_state(wait_to_stabilize=True)
                logical_screen_size = env.logical_screen_size
                ui_elements = env_state.ui_elements
                screen, element_list_text = get_state(
                    env_state, logical_screen_size, ui_elements
                )

                interactive_elements = filter_interactive_elements(
                    ui_elements, logical_screen_size, unc_elem_pool
                )
                addition_actions = [
                    {"action_type": "scroll", "direction": "down"},
                    {"action_type": "scroll", "direction": "up"},
                    {"action_type": "navigate_back"},
                ]
                candidates = interactive_elements * 10 + addition_actions * 3
                if not candidates:
                    print("No candidates; stop this task early")
                    break
                action_element = random.choice(candidates)
                if "action_type" in action_element and "index" not in action_element:
                    action_sample = action_element
                else:
                    action_sample = sample_action_element(action_element, element_list_text)

                print(f"  step {i+1}: {action_sample}")
                converted_action = json_action.JSONAction(**action_sample)
                env.execute_action(converted_action)
                time.sleep(_POST_ACTION_SLEEP.value)

                env_state_after = env.get_state(wait_to_stabilize=True)
                ui_elements_after = env_state_after.ui_elements
                if not has_screen_changed(ui_elements, ui_elements_after):
                    print("  screen unchanged")
                    if "index" in action_sample:
                        unc_elem_pool.add(
                            json.dumps(
                                element_to_identifier(ui_elements[action_sample["index"]]),
                                sort_keys=True,
                            )
                        )
                    continue

                screen_before_uuid = str(uuid.uuid4())
                screen_after_uuid = str(uuid.uuid4())
                screen_before_filename = os.path.join(screen_dir, f"{screen_before_uuid}.png")
                screen.save(screen_before_filename)
                screen_after, element_list_text_after = get_state(
                    env_state_after, logical_screen_size, ui_elements_after
                )
                screen_after_filename = os.path.join(screen_dir, f"{screen_after_uuid}.png")
                screen_after.save(screen_after_filename)

                interactive_after = filter_interactive_elements(
                    ui_elements_after, logical_screen_size, unc_elem_pool
                )
                action_element_id = (
                    element_to_identifier(ui_elements[action_sample["index"]])
                    if "index" in action_sample
                    else None
                )
                trajectory.append(
                    {
                        "task_uuid": task_uuid,
                        "task": task_name,
                        "app": app_name,
                        "screen_before": screen_before_filename,
                        "element_list_text_before": element_list_text,
                        "ui_elements_before": [element_to_identifier(e) for e in ui_elements],
                        "interactive_elements_before": [
                            element_to_identifier(e[1]) for e in interactive_elements
                        ],
                        "screen_after": screen_after_filename,
                        "element_list_text_after": element_list_text_after,
                        "ui_elements_after": [
                            element_to_identifier(e) for e in ui_elements_after
                        ],
                        "interactive_elements_after": [
                            element_to_identifier(e[1]) for e in interactive_after
                        ],
                        "action": action_sample,
                        "action_element": action_element_id,
                    }
                )
                print(f"  recorded step {i+1}")

            with open(unc_elem_pool_path, "w", encoding="utf-8") as f:
                json.dump(list(unc_elem_pool), f, indent=2)

            traj_path = os.path.join(traj_dir, f"{task_name}_{uuid.uuid4()}.json")
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2)
            print(f"  saved {traj_path} ({len(trajectory)} steps)")
        except Exception as e:
            print(f"  ERROR on {task_name}: {e!r}")
            # Emulator often dies mid-task; try to bring it back before the next one.
            env.wait_until_healthy(retries=3, sleep_s=5.0)
        finally:
            try:
                env.tear_down_task(task_name)
            except Exception as e:
                print(f"  tear_down ERROR: {e!r}")
            if not env.health():
                env.wait_until_healthy(retries=3, sleep_s=5.0)

    env.close()
    print("Done.")


if __name__ == "__main__":
    app.run(_main)
