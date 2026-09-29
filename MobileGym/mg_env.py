"""MobileGym → AndroidWorld AsyncEnv adapter.

Drives a running MobileGym frontend (`npm run dev` / `preview` / nginx gateway)
via sync Playwright and exposes the AsyncEnv surface used by OpenMobile agents
(screenshot + execute_action). Vision agents do not require UI XML.
"""

from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
from PIL import Image

from android_world.env import interface
from android_world.env import json_action


class _StubController:
    def close(self) -> None:
        return None


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


# Subset of MobileGym APP_NAME_MAP (Chinese / English → appId).
APP_NAME_MAP: dict[str, str] = {
    "设置": "settings",
    "Settings": "settings",
    "相册": "gallery",
    "Gallery": "gallery",
    "文件": "file_manager",
    "File Manager": "file_manager",
    "计算器": "calculator",
    "Calculator": "calculator",
    "时钟": "clock",
    "Clock": "clock",
    "通讯录": "contacts",
    "联系人": "contacts",
    "Contacts": "contacts",
    "笔记": "notes",
    "Notes": "notes",
    "短信": "sms",
    "Sms": "sms",
    "SMS": "sms",
    "日历": "calendar",
    "Calendar": "calendar",
    "浏览器": "browser",
    "Browser": "browser",
    "微信": "wechat",
    "WeChat": "wechat",
    "Wechat": "wechat",
    "天气": "weather",
    "Weather": "weather",
    "微信读书": "wechat_reading",
    "WeChat Reading": "wechat_reading",
    "WeRead": "wechat_reading",
    "哔哩哔哩": "bilibili",
    "B站": "bilibili",
    "Bilibili": "bilibili",
    "BilibiliV2": "bilibili",
    "腾讯会议": "tencent_meeting",
    "Tencent Meeting": "tencent_meeting",
    "支付宝": "alipay",
    "Alipay": "alipay",
    "地图": "map",
    "Map": "map",
    "小红书": "redbook",
    "RedNote": "redbook",
    "RedBook": "redbook",
    "Spotify": "spotify",
    "X": "x",
    "淘宝": "taobao",
    "Taobao": "taobao",
    "京东": "jingdong",
    "JD": "jingdong",
    "Jingdong": "jingdong",
    "美团": "meituan",
    "Meituan": "meituan",
    "大众点评": "dianping",
    "Dianping": "dianping",
    "携程": "ctrip",
    "Ctrip": "ctrip",
    "12306": "railway12306",
    "铁路12306": "railway12306",
    "Railway12306": "railway12306",
    "瑞幸": "luckin",
    "Luckin": "luckin",
    "Luckin Coffee": "luckin",
    "BOSS直聘": "boss",
    "Boss": "boss",
    "BOSS Zhipin": "boss",
    "闲鱼": "xianyu",
    "Idle Fish": "xianyu",
    "Xianyu": "xianyu",
    "拼多多": "pinduoduo",
    "Pinduoduo": "pinduoduo",
    "猫眼": "maoyan",
    "Maoyan": "maoyan",
    "腾讯视频": "tencentvideo",
    "Tencent Video": "tencentvideo",
    "Tencentvideo": "tencentvideo",
    "优酷": "youku",
    "Youku": "youku",
    "微博": "weibo",
    "Weibo": "weibo",
    "豆瓣": "douban",
    "Douban": "douban",
    "Reddit": "reddit",
    "eBay": "ebay",
    "Ebay": "ebay",
    "Keep": "keep",
    "菜鸟": "cainiao",
    "Cainiao": "cainiao",
    "苏宁": "suning",
    "Suning": "suning",
    "唯品会": "vipshop",
    "Vipshop": "vipshop",
    "VIP.com": "vipshop",
    "喜马拉雅": "ximalaya",
    "Ximalaya": "ximalaya",
    "播客": "broadcast",
    "Podcasts": "broadcast",
    "Broadcast": "broadcast",
    "番茄ToDo": "fanqietodo",
    "FanqieTodo": "fanqietodo",
    "番茄小说": "fanqiexiaoshuo",
    "FanqieXiaoshuo": "fanqiexiaoshuo",
    "flomo": "flomo",
    "Flomo": "flomo",
    "卡皮记账": "kapi",
    "Kapi": "kapi",
    "百词斩": "baicizhan",
    "Baicizhan": "baicizhan",
    "美柚": "meiyou",
    "Meetyou": "meiyou",
    "Meiyou": "meiyou",
    "小日常": "xiaorichang",
    "Daily": "xiaorichang",
    "Xiaorichang": "xiaorichang",
    "倒数日": "daysmatter",
    "Days Matter": "daysmatter",
    "Daysmatter": "daysmatter",
    "淘宝闪购": "taobaoshangou",
    "TaobaoShangou": "taobaoshangou",
}


class MobileGymEnv(interface.AsyncEnv):
    """AsyncEnv-compatible wrapper around a MobileGym SPA URL."""

    interaction_cache = ""
    FREEFORM_TASK = "MobileGymFreeform"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:3000",
        *,
        headless: bool = True,
        step_wait_time: float = 1.0,
        physical_size: tuple[int, int] = (1080, 2400),
        dpr: float = 3.0,
        browser_type: str = "chromium",
    ):
        self.base_url = base_url.rstrip("/")
        self.headless = headless
        self.step_wait_time = step_wait_time
        self.physical_width, self.physical_height = physical_size
        self.dpr = float(dpr)
        self.css_width = int(round(self.physical_width / self.dpr))
        self.css_height = int(round(self.physical_height / self.dpr))
        self.browser_type = browser_type

        self._controller = _StubController()
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False
        self._screen_size: tuple[int, int] | None = None
        self.interaction_cache = ""

    # ---- Playwright lifecycle --------------------------------------------

    def _ensure_started(self) -> None:
        if self._initialized and self._page is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        browser_launcher = getattr(self._pw, self.browser_type)
        self._browser = browser_launcher.launch(
            headless=self.headless,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        self._context = self._browser.new_context(
            viewport={"width": self.css_width, "height": self.css_height},
            device_scale_factor=self.dpr,
            is_mobile=True,
            has_touch=True,
        )
        self._page = self._context.new_page()
        self._page.goto(self.base_url, wait_until="domcontentloaded")
        self._wait_for_sim(timeout_ms=60_000)
        self._initialized = True

    def _wait_for_sim(self, timeout_ms: int = 60_000) -> None:
        assert self._page is not None
        self._page.wait_for_function(
            "() => Boolean(window.__SIM__ && window.__OS__)",
            timeout=timeout_ms,
        )

    # ---- AsyncEnv API ----------------------------------------------------

    @property
    def controller(self) -> Any:
        return self._controller

    def reset(self, go_home: bool = False) -> interface.State:
        self._ensure_started()
        self.interaction_cache = ""
        self._reset_sim()
        if go_home:
            self._home()
            time.sleep(self.step_wait_time)
        return self.get_state(wait_to_stabilize=True)

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        self._ensure_started()
        if wait_to_stabilize:
            time.sleep(self.step_wait_time)
        pixels = self._fetch_screenshot_pixels()
        h, w = pixels.shape[:2]
        self._screen_size = (w, h)
        return interface.State(
            pixels=pixels,
            forest=None,
            ui_elements=[],
            auxiliaries=None,
        )

    def display_message(self, message: str, header: str = "") -> None:
        return None

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> str | None:
        return None

    def execute_action(self, action: json_action.JSONAction) -> None:
        self._ensure_started()
        at = action.action_type

        if at == json_action.ANSWER:
            self.interaction_cache = action.text or ""
            return

        if at == json_action.STATUS:
            # Agent-declared success/failure — no env side effect.
            return

        if at in (json_action.CLICK, json_action.DOUBLE_TAP, json_action.LONG_PRESS):
            x, y = self._require_xy(action)
            if at == json_action.DOUBLE_TAP:
                self._tap(x, y)
                time.sleep(0.05)
                self._tap(x, y)
            elif at == json_action.LONG_PRESS:
                self._long_press(x, y)
            else:
                self._tap(x, y)
            time.sleep(self.step_wait_time)
            return

        if at == json_action.INPUT_TEXT:
            if action.x is not None and action.y is not None:
                self._tap(float(action.x), float(action.y))
                time.sleep(0.2)
            self._type_text(action.text or "")
            time.sleep(self.step_wait_time)
            return

        if at in (json_action.SCROLL, json_action.SWIPE):
            if (
                action.x is not None
                and action.y is not None
                and action.x2 is not None
                and action.y2 is not None
            ):
                self._swipe(
                    (float(action.x), float(action.y)),
                    (float(action.x2), float(action.y2)),
                    duration_ms=500,
                )
            elif action.x is not None and action.y is not None and action.direction:
                self._scroll_from(
                    float(action.x),
                    float(action.y),
                    (action.direction or "down").lower(),
                )
            else:
                direction = (action.direction or "down").lower()
                self._scroll(direction)
            time.sleep(self.step_wait_time)
            return

        if at == json_action.NAVIGATE_HOME:
            self._home()
            time.sleep(self.step_wait_time)
            return

        if at == json_action.NAVIGATE_BACK:
            self._back()
            time.sleep(self.step_wait_time)
            return

        if at == json_action.KEYBOARD_ENTER:
            assert self._page is not None
            self._page.keyboard.press("Enter")
            time.sleep(self.step_wait_time)
            return

        if at == json_action.OPEN_APP:
            self._open_app(action.app_name or "")
            time.sleep(self.step_wait_time)
            return

        if at == json_action.WAIT:
            time.sleep(max(self.step_wait_time, 1.0))
            return

        # Unknown / no-op
        time.sleep(0.1)

    def hide_automation_ui(self) -> None:
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
        try:
            if self._page:
                self._page.close()
            if self._context:
                self._context.close()
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._pw = self._browser = self._context = self._page = None
        self._initialized = False

    @property
    def orientation(self) -> int:
        return 0

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        w, h = self.logical_screen_size
        return (0, 0, w, h)

    # ---- Task helpers (freeform; no MobileWorld registry) ----------------

    def health(self) -> bool:
        try:
            self._ensure_started()
            assert self._page is not None
            ok = self._page.evaluate("() => Boolean(window.__SIM__ && window.__OS__)")
            return bool(ok)
        except Exception as exc:
            # Surface root cause (missing playwright, page not ready, etc.)
            print(f"[MobileGymEnv.health] failed for {self.base_url}: {exc!r}")
            return False

    def list_tasks(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"name": self.FREEFORM_TASK, "tags": ["freeform"]}]

    def initialize_task(self, task_name: str) -> interface.State:
        del task_name
        return self.reset(go_home=True)

    def tear_down_task(self, task_name: str) -> None:
        del task_name
        try:
            self._home()
        except Exception:
            pass

    # ---- Internals -------------------------------------------------------

    def _require_xy(self, action: json_action.JSONAction) -> tuple[float, float]:
        if action.x is None or action.y is None:
            raise ValueError(f"Action needs x,y: {action}")
        return float(action.x), float(action.y)

    def _p2c(self, x: float, y: float) -> tuple[float, float]:
        """Physical screenshot pixels → CSS viewport coordinates."""
        w, h = self.device_screen_size
        cx = (x / max(1.0, float(w))) * float(self.css_width)
        cy = (y / max(1.0, float(h))) * float(self.css_height)
        return cx, cy

    def _fetch_screenshot_pixels(self) -> np.ndarray:
        assert self._page is not None
        png = self._page.screenshot(type="png")
        image = Image.open(io.BytesIO(png))
        return _pil_to_rgb_array(image)

    def _reset_sim(self) -> None:
        assert self._page is not None
        try:
            self._page.evaluate(
                """async () => {
                    if (window.__SIM__?.resetState) {
                        await window.__SIM__.resetState();
                        return;
                    }
                    try { localStorage.clear(); } catch {}
                    try { sessionStorage.clear(); } catch {}
                }"""
            )
        except Exception:
            pass
        self._page.goto(self.base_url, wait_until="domcontentloaded")
        self._wait_for_sim(timeout_ms=60_000)

    def _tap(self, x: float, y: float) -> None:
        assert self._page is not None
        cx, cy = self._p2c(x, y)
        try:
            used = self._page.evaluate(
                "({x,y}) => { if (window.__SIM_INPUT__?.tap) { window.__SIM_INPUT__.tap(x,y); return true; } return false; }",
                {"x": cx, "y": cy},
            )
            if not used:
                raise RuntimeError("no __SIM_INPUT__")
        except Exception:
            try:
                self._page.touchscreen.tap(cx, cy)
            except Exception:
                self._page.mouse.click(cx, cy)

    def _long_press(self, x: float, y: float, duration_ms: int = 800) -> None:
        assert self._page is not None
        cx, cy = self._p2c(x, y)
        try:
            used = self._page.evaluate(
                """async ({x,y,d}) => {
                    if (window.__SIM_INPUT__?.longPress) {
                        await window.__SIM_INPUT__.longPress(x,y,{ms:d});
                        return true;
                    }
                    return false;
                }""",
                {"x": cx, "y": cy, "d": duration_ms},
            )
            if used:
                return
        except Exception:
            pass
        self._page.mouse.move(cx, cy)
        self._page.mouse.down()
        time.sleep(duration_ms / 1000.0)
        self._page.mouse.up()

    def _type_text(self, text: str) -> None:
        assert self._page is not None
        try:
            used = self._page.evaluate(
                "async ({t}) => { if (window.__SIM_INPUT__?.type) { await window.__SIM_INPUT__.type(t, {clear:false}); return true; } return false; }",
                {"t": text},
            )
            if not used:
                raise RuntimeError("no __SIM_INPUT__")
        except Exception:
            self._page.keyboard.type(text, delay=0)

    def _scroll(self, direction: str) -> None:
        """Match AndroidWorld actuation.scroll: direction = content move.

        AW scrolls from screen center toward the opposite edge of ``direction``
        (scroll down → finger moves up; scroll right → finger moves left).
        See ``android_world/env/actuation.py``.
        """
        w, h = self.device_screen_size
        self._scroll_from(w / 2.0, h / 2.0, direction)

    def _scroll_from(self, start_x: float, start_y: float, direction: str) -> None:
        """Scroll content in ``direction`` starting at an explicit point."""
        w, h = self.device_screen_size
        # Leave a small margin so we don't hit system gesture edges.
        margin_x, margin_y = w * 0.05, h * 0.05
        ends = {
            "down": (start_x, margin_y),          # finger up
            "up": (start_x, h - margin_y),        # finger down
            "left": (w - margin_x, start_y),      # finger right
            "right": (margin_x, start_y),         # finger left
        }
        d = direction if direction in ends else "down"
        self._swipe((start_x, start_y), ends[d], duration_ms=500)

    def _swipe(self, start: tuple[float, float], end: tuple[float, float], duration_ms: int = 400) -> None:
        assert self._page is not None
        x1, y1 = start
        x2, y2 = end
        cx1, cy1 = self._p2c(x1, y1)
        cx2, cy2 = self._p2c(x2, y2)
        try:
            # Disable inertia: with 3+ home pages, post-fling scroll easily
            # overshoots screen_2 and lands on screen_3 (looks like "page 2 is gone").
            used = self._page.evaluate(
                """async ({sx,sy,ex,ey,d}) => {
                    if (window.__SIM_INPUT__?.swipe) {
                        await window.__SIM_INPUT__.swipe(
                          {x:sx,y:sy},{x:ex,y:ey},
                          {ms:d, inertia: false},
                        );
                        return true;
                    }
                    return false;
                }""",
                {"sx": cx1, "sy": cy1, "ex": cx2, "ey": cy2, "d": duration_ms},
            )
            if used:
                return
        except Exception:
            pass
        self._page.mouse.move(cx1, cy1)
        self._page.mouse.down()
        self._page.mouse.move(cx2, cy2, steps=12)
        self._page.mouse.up()

    def _back(self) -> None:
        assert self._page is not None
        try:
            self._page.evaluate(
                """() => {
                    if (window.__OS__?.handleBack) { window.__OS__.handleBack(); return; }
                    history.back();
                }"""
            )
        except Exception:
            pass

    def _home(self) -> None:
        assert self._page is not None
        try:
            used = self._page.evaluate(
                """() => {
                    if (window.__OS__?.goHome) { window.__OS__.goHome(); return true; }
                    return false;
                }"""
            )
            if not used:
                self._page.goto(self.base_url, wait_until="domcontentloaded")
                self._wait_for_sim()
        except Exception:
            self._page.goto(self.base_url, wait_until="domcontentloaded")
            self._wait_for_sim()

    def _open_app(self, app_name: str) -> None:
        assert self._page is not None
        name = (app_name or "").strip()
        if not name:
            return
        if name in APP_NAME_MAP:
            app_id = APP_NAME_MAP[name]
        elif name.lower() in {v.lower() for v in APP_NAME_MAP.values()}:
            app_id = name.lower()
        else:
            # Best-effort: try lowercase camel-stripped id
            app_id = name.replace(" ", "").lower()
        try:
            self._page.wait_for_function(
                "() => Boolean(window.__OS__?.openApp)",
                timeout=30_000,
            )
            self._page.evaluate("({a}) => { window.__OS__.openApp(a); }", {"a": app_id})
        except Exception as e:
            print(f"open_app({app_name!r} -> {app_id!r}) failed: {e!r}")
