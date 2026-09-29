"""MobileWorld → AndroidWorld AsyncEnv adapter.

Talks to a running MobileWorld backend (`mw env run` / `mw server`) over HTTP
and exposes the subset of `android_world.env.interface.AsyncEnv` used by
OpenMobile exploration / rollout agents (screenshot, UI XML, execute_action).
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


class _StubController:
    """Placeholder so agents that touch `env.controller` do not crash."""

    def close(self) -> None:
        return None


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _parse_bounds(bounds: str | None) -> representation_utils.BoundingBox | None:
    if not bounds:
        return None
    try:
        # Format: [x_min,y_min][x_max,y_max]
        x_min, y_min, x_max, y_max = map(
            int, bounds.strip("[]").replace("][", ",").split(",")
        )
        return representation_utils.BoundingBox(x_min, x_max, y_min, y_max)
    except Exception:
        return None


def xml_string_to_ui_elements(xml_string: str) -> list[representation_utils.UIElement]:
    """UIAutomator XML → UIElement list (same shape as AW forest conversion)."""
    elements = representation_utils.xml_dump_to_ui_elements(xml_string)
    # xml_dump_to_ui_elements does not set is_editable; infer from class name.
    for el in elements:
        cls = (el.class_name or "").lower()
        if "edittext" in cls or "autocompletetextview" in cls:
            el.is_editable = True
        elif el.is_editable is None:
            el.is_editable = False
        if el.is_visible is None:
            el.is_visible = True
        if el.is_enabled is None:
            el.is_enabled = True
        if el.is_clickable is None:
            # Treat focusable/checkable nodes without explicit clickable as interactive-ish.
            el.is_clickable = bool(el.is_focusable or el.is_checkable or el.is_editable)
    return elements


class MobileWorldEnv(interface.AsyncEnv):
    """AsyncEnv-compatible wrapper around MobileWorld HTTP backend."""

    interaction_cache = ""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6800",
        device: str = "emulator-5554",
        step_wait_time: float = 1.5,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.device = device
        self.step_wait_time = step_wait_time
        self.timeout = timeout
        self._controller = _StubController()
        self._initialized = False
        self._screen_size: tuple[int, int] | None = None
        self._last_ui_elements: list[representation_utils.UIElement] = []
        self.interaction_cache = ""

    # ---- HTTP helpers -----------------------------------------------------

    def _post(self, path: str, json_body: dict | None = None, **kwargs) -> requests.Response:
        return requests.post(
            f"{self.base_url}{path}",
            json=json_body,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def _get(self, path: str, **kwargs) -> requests.Response:
        return requests.get(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        resp = self._post("/init", {"device": self.device})
        resp.raise_for_status()
        self._initialized = True

    def force_reinit_device(self) -> bool:
        """Re-register device with backend (helps after emulator blips)."""
        self._initialized = False
        try:
            self._ensure_initialized()
            return True
        except Exception as e:
            print(f"force_reinit_device failed: {e}")
            return False

    def wait_until_healthy(self, *, retries: int = 5, sleep_s: float = 3.0) -> bool:
        """Wait for /health to report a live device; re-init between attempts."""
        for i in range(max(1, int(retries))):
            if self.health():
                return True
            print(
                f"Device unhealthy at {self.base_url} "
                f"(attempt {i + 1}/{retries}); re-init and wait {sleep_s}s"
            )
            self.force_reinit_device()
            time.sleep(float(sleep_s))
        return self.health()

    # ---- AsyncEnv API -----------------------------------------------------

    @property
    def controller(self) -> Any:
        return self._controller

    def reset(self, go_home: bool = False) -> interface.State:
        self._ensure_initialized()
        self.interaction_cache = ""
        if go_home:
            self.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_HOME))
            time.sleep(self.step_wait_time)
        return self.get_state(wait_to_stabilize=True)

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        self._ensure_initialized()
        if wait_to_stabilize:
            time.sleep(self.step_wait_time)

        pixels = self._fetch_screenshot_pixels()
        h, w = pixels.shape[:2]
        self._screen_size = (w, h)

        xml_string = self._fetch_xml()
        ui_elements = xml_string_to_ui_elements(xml_string) if xml_string else []
        self._last_ui_elements = ui_elements
        return interface.State(
            pixels=pixels,
            forest=None,
            ui_elements=ui_elements,
            auxiliaries={"xml": xml_string} if xml_string else None,
        )

    def display_message(self, message: str, header: str = "") -> None:
        return None

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> str | None:
        action = json_action.JSONAction(action_type="ask_user", text=question)
        # MobileWorld ask_user is not in AW JSONAction types; send raw via HTTP.
        self._ensure_initialized()
        resp = self._post(
            "/step",
            {
                "device": self.device,
                "action": {"action_type": "ask_user", "text": question},
            },
        )
        if not resp.ok:
            return None
        try:
            return resp.json().get("result")
        except Exception:
            return None

    def execute_action(self, action: json_action.JSONAction) -> None:
        self._ensure_initialized()
        if action.action_type == json_action.ANSWER:
            self.interaction_cache = action.text or ""
            self._post(
                "/step",
                {
                    "device": self.device,
                    "action": {"action_type": "answer", "text": action.text or ""},
                },
            )
            return

        payload = self._aw_action_to_mw_payload(action)
        resp = self._post("/step", {"device": self.device, "action": payload})
        if not resp.ok:
            raise RuntimeError(f"MobileWorld /step failed ({resp.status_code}): {resp.text}")
        time.sleep(self.step_wait_time)

    def hide_automation_ui(self) -> None:
        # No-op on MobileWorld Docker images.
        return None

    @property
    def foreground_activity_name(self) -> str:
        return ""

    @property
    def device_screen_size(self) -> tuple[int, int]:
        if self._screen_size is None:
            self.get_state(wait_to_stabilize=False)
        assert self._screen_size is not None
        return self._screen_size

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        return self.device_screen_size

    def close(self) -> None:
        return None

    @property
    def orientation(self) -> int:
        return 0

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        w, h = self.logical_screen_size
        return (0, 0, w, h)

    # ---- MobileWorld task helpers ----------------------------------------

    def health(self) -> bool:
        try:
            # Register device first; otherwise /health may report ok with empty devices.
            try:
                self._ensure_initialized()
            except Exception:
                pass
            resp = self._get("/health")
            if not (resp.ok and resp.json().get("ok", False)):
                return False
            data = resp.json()
            devices = data.get("devices") or []
            status = data.get("device_status") or {}
            # Backend can be up while the emulator is gone — treat that as unhealthy.
            if not devices:
                return False
            if self.device and status and not status.get(self.device, False):
                return False
            return True
        except Exception:
            return False

    def list_tasks(
        self,
        enable_mcp: bool = False,
        enable_user_interaction: bool = False,
    ) -> list[dict[str, Any]]:
        self._ensure_initialized()
        resp = self._get("/task/list")
        resp.raise_for_status()
        tasks = resp.json()
        out = []
        for t in tasks:
            tags = t.get("tags") or []
            if not enable_mcp and "agent-mcp" in tags:
                continue
            if not enable_user_interaction and "agent-user-interaction" in tags:
                continue
            out.append(t)
        return out

    def get_task_goal(self, task_name: str) -> str:
        self._ensure_initialized()
        resp = self._get("/task/goal", params={"task_name": task_name})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("goal") or data.get("task_goal") or str(data)
        return str(data)

    def get_task_metadata(self, task_name: str) -> dict[str, Any]:
        self._ensure_initialized()
        resp = self._get("/task/metadata", params={"task_name": task_name})
        resp.raise_for_status()
        return resp.json()

    def initialize_task(self, task_name: str) -> interface.State:
        self._ensure_initialized()
        resp = self._post(
            "/task/init",
            {"task_name": task_name, "req_device": self.device},
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(
                f"task/init failed ({resp.status_code}) for {task_name!r} "
                f"on {self.base_url}: {resp.text[:500]}"
            )
        return self.get_state(wait_to_stabilize=True)

    def tear_down_task(self, task_name: str) -> None:
        self._ensure_initialized()
        resp = self._post(
            "/task/tear_down",
            {"task_name": task_name, "req_device": self.device},
            timeout=300,
        )
        # Best-effort; keep explore/rollout loops alive.
        if not resp.ok:
            print(f"tear_down_task warning: {resp.status_code} {resp.text}")

    def get_task_score(self, task_name: str, retries: int = 4) -> tuple[float, str]:
        """Score the current device state. Retry transient /task/eval failures in place.

        HTTP 500 here is not a model failure: the episode already finished. Retry
        while the emulator still holds that state. Verifier crashes that the
        server reports as ``eval_error`` are raised so the caller can skip
        instead of writing a fake 0.
        """
        self._ensure_initialized()
        last_err: Exception | None = None
        timeout = max(float(self.timeout), 180.0)
        for i in range(max(1, int(retries))):
            try:
                resp = self._get(
                    "/task/eval",
                    params={"task_name": task_name, "req_device": self.device},
                    timeout=timeout,
                )
                if not resp.ok:
                    resp = requests.get(
                        f"{self.base_url}/task/eval",
                        json={"task_name": task_name, "req_device": self.device},
                        timeout=timeout,
                    )
                if resp.status_code in {500, 502, 503, 504}:
                    last_err = requests.HTTPError(
                        f"{resp.status_code} Server Error for url: {resp.url}: "
                        f"{(resp.text or '')[:300]}",
                        response=resp,
                    )
                    if i + 1 < retries:
                        time.sleep(min(8.0, 2.0 * (i + 1)))
                        continue
                    raise last_err
                resp.raise_for_status()
                result = resp.json()
                reason = str(result.get("reason") or "")
                if result.get("eval_error") or reason.startswith("eval exception:"):
                    raise RuntimeError(
                        f"task/eval verifier crashed for {task_name}: {reason}"
                    )
                return float(result.get("score", 0.0)), reason
            except RuntimeError:
                raise
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_err = e
                if i + 1 < retries:
                    time.sleep(min(8.0, 2.0 * (i + 1)))
                    continue
                raise
        assert last_err is not None
        raise last_err

    # ---- internals --------------------------------------------------------

    def _fetch_screenshot_pixels(self, *, retries: int = 3, sleep_s: float = 1.0) -> np.ndarray:
        last_err: Exception | None = None
        for i in range(max(1, int(retries))):
            try:
                resp = self._get(
                    "/screenshot",
                    params={"device": self.device, "return_b64": True},
                )
                resp.raise_for_status()
                b64 = resp.json()["b64_png"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(b64)))
                return _pil_to_rgb_array(image)
            except Exception as e:
                last_err = e
                if i + 1 < retries:
                    time.sleep(float(sleep_s) * (i + 1))
                    # UIAutomator/screenshot often fails when device is flaky.
                    if "not healthy" in str(e).lower() or "500" in str(e):
                        self.force_reinit_device()
        assert last_err is not None
        raise last_err

    def _fetch_xml(self, *, retries: int = 3, sleep_s: float = 1.0) -> str:
        last_err = ""
        for i in range(max(1, int(retries))):
            resp = self._get(
                "/xml",
                params={"device": self.device, "mode": "uia", "return_content": True},
            )
            if resp.ok:
                data = resp.json()
                return data.get("content") or ""
            last_err = f"{resp.status_code}: {resp.text[:200]}"
            if i + 1 < retries:
                time.sleep(float(sleep_s) * (i + 1))
        print(f"Failed to fetch XML ({last_err})")
        return ""

    def _resolve_xy(self, action: json_action.JSONAction) -> tuple[int, int]:
        if action.x is not None and action.y is not None:
            return int(action.x), int(action.y)
        if action.index is not None:
            if action.index < 0 or action.index >= len(self._last_ui_elements):
                # Refresh elements once.
                state = self.get_state(wait_to_stabilize=False)
                if action.index < 0 or action.index >= len(state.ui_elements):
                    raise ValueError(f"Invalid element index: {action.index}")
                elements = state.ui_elements
            else:
                elements = self._last_ui_elements
            el = elements[action.index]
            if el.bbox_pixels is None:
                raise ValueError(f"Element {action.index} has no bbox")
            x, y = el.bbox_pixels.center
            return int(x), int(y)
        raise ValueError(f"Action needs index or (x,y): {action}")

    def _aw_action_to_mw_payload(self, action: json_action.JSONAction) -> dict[str, Any]:
        at = action.action_type
        if at in (json_action.CLICK, json_action.DOUBLE_TAP, json_action.LONG_PRESS):
            x, y = self._resolve_xy(action)
            return {"action_type": at, "x": x, "y": y}
        if at == json_action.INPUT_TEXT:
            # Focus target if provided, then type.
            if action.index is not None or (action.x is not None and action.y is not None):
                x, y = self._resolve_xy(action)
                self._post(
                    "/step",
                    {"device": self.device, "action": {"action_type": "click", "x": x, "y": y}},
                )
                time.sleep(0.5)
            return {"action_type": "input_text", "text": action.text or ""}
        if at == json_action.SCROLL:
            payload: dict[str, Any] = {
                "action_type": "scroll",
                "direction": action.direction or "down",
            }
            if action.x is not None and action.y is not None:
                payload["x"] = int(action.x)
                payload["y"] = int(action.y)
            if action.x2 is not None and action.y2 is not None:
                payload["x2"] = int(action.x2)
                payload["y2"] = int(action.y2)
            return payload
        if at == json_action.SWIPE:
            payload: dict[str, Any] = {
                "action_type": "swipe",
                "direction": action.direction or "up",
            }
            if action.x is not None:
                payload["x"] = int(action.x)
            if action.y is not None:
                payload["y"] = int(action.y)
            if action.x2 is not None:
                payload["x2"] = int(action.x2)
            if action.y2 is not None:
                payload["y2"] = int(action.y2)
            return payload
        if at in (
            json_action.NAVIGATE_HOME,
            json_action.NAVIGATE_BACK,
            json_action.KEYBOARD_ENTER,
            json_action.WAIT,
            json_action.STATUS,
            json_action.UNKNOWN,
        ):
            payload = {"action_type": at}
            if at == json_action.STATUS:
                payload["goal_status"] = action.goal_status or "success"
            return payload
        if at == json_action.OPEN_APP:
            return {"action_type": "open_app", "app_name": action.app_name}
        # Fallback: dump known fields.
        payload = {"action_type": at}
        for key in ("text", "direction", "app_name", "goal_status", "x", "y"):
            val = getattr(action, key, None)
            if val is not None:
                payload[key] = val
        return payload
