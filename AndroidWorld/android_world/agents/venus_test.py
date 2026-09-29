"""Unit tests for Venus parse + JSONAction mapping (no env / LLM)."""

from android_world.agents.venus import parse_venus_llm_output
from android_world.agents.venus import venus_action_to_json
from android_world.agents.session_runtimes import resolve_runtime


def test_parse_click():
    raw = (
        "<think>tap search</think>\n"
        "<action>Click(box=(120,340))</action>\n"
        "<conclusion>ok</conclusion>"
    )
    parsed = parse_venus_llm_output(raw)
    assert parsed["action"] == "Click"
    assert parsed["params"]["box"] == [120, 340]
    assert "tap search" in parsed["think"]


def test_parse_scroll_and_type():
    raw = (
        "<think>x</think>"
        "<action>Scroll(start=(500,800), end=(500,200))</action>"
        "<conclusion>y</conclusion>"
    )
    parsed = parse_venus_llm_output(raw)
    assert parsed["action"] == "Scroll"
    assert parsed["params"]["start"] == [500, 800]
    assert parsed["params"]["end"] == [500, 200]

    raw2 = "<action>Type(content='hello world')</action>"
    parsed2 = parse_venus_llm_output(raw2)
    assert parsed2["action"] == "Type"
    assert parsed2["params"]["content"] == "hello world"


def test_transform_coords():
    click = venus_action_to_json("Click", {"box": [500, 500]}, 1080, 1920)
    assert click["action_type"] == "click"
    assert abs(click["x"] - 540.0) < 1e-6
    assert abs(click["y"] - 960.0) < 1e-6

    swipe = venus_action_to_json(
        "Scroll",
        {"start": [100, 900], "end": [100, 100]},
        1000,
        1000,
    )
    assert swipe["action_type"] == "swipe"
    assert swipe["x"] == 100.0
    assert swipe["y"] == 900.0
    assert swipe["x2"] == 100.0
    assert swipe["y2"] == 100.0


def test_transform_system_and_terminal():
    assert venus_action_to_json("PressBack", {}, 1, 1)["action_type"] == "navigate_back"
    assert venus_action_to_json("PressHome", {}, 1, 1)["action_type"] == "navigate_home"
    assert venus_action_to_json("PressEnter", {}, 1, 1)["action_type"] == "keyboard_enter"
    assert venus_action_to_json("Launch", {"app": "微信"}, 1, 1) == {
        "action_type": "open_app",
        "app_name": "微信",
    }
    assert venus_action_to_json("Finished", {}, 1, 1) == {
        "action_type": "status",
        "goal_status": "complete",
    }
    assert venus_action_to_json("Finished", {"content": "done"}, 1, 1) == {
        "action_type": "answer",
        "text": "done",
    }
    assert venus_action_to_json("CallUser", {"content": "which?"}, 1, 1) == {
        "action_type": "answer",
        "text": "which?",
    }


def test_runtime_preset():
    spec = resolve_runtime("venus")
    assert spec.agent_name == "venus"
    assert "Venus" in spec.description or "venus" in spec.description.lower()
