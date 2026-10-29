import pytest

from synthetic import Sim, make_hand
from virtual_touch.backends.base import parse_hotkey
from virtual_touch.config import os_key, parse_override
from virtual_touch.hand import HandAnalyzer, Pose
from virtual_touch.system_info import Display, Screens


@pytest.mark.parametrize("pose", ["relaxed", "fist", "point", "two", "three", "palm", "pinch", "pinch_middle", "grab"])
def test_pose_classification(pose):
    sim = Sim()
    analyzer = HandAnalyzer(sim.cfg.gestures)
    for _ in range(3):
        features = analyzer.analyze(make_hand(pose))
    assert features.pose == Pose(pose)


@pytest.mark.parametrize("scale", [0.12, 0.22, 0.4])
def test_classification_is_distance_invariant(scale):
    analyzer = HandAnalyzer(Sim().cfg.gestures)
    for _ in range(3):
        f = analyzer.analyze(make_hand("pinch", scale=scale))
    assert f.pose == Pose.PINCH


@pytest.mark.parametrize("pose", ["relaxed", "point", "palm", "three"])
def test_hand_moves_cursor_in_hand_direction(pose):
    sim = Sim()
    x0 = sim.backend.pos[0]
    for i in range(20):
        sim.step(make_hand(pose, center=(0.4 + i * 0.005, 0.5)))
    assert sim.engine.mode == "hover"
    assert sim.backend.pos[0] > x0 + 50
    assert abs(sim.backend.pos[1] - 400) < 5
    assert not sim.backend.of("hotkey")  # slow movement is pointing, not swiping


def test_click_rewinds_cursor_to_pre_pinch_position():
    sim = Sim()
    for i in range(10):
        sim.step(make_hand("relaxed", center=(0.5, 0.5)))
    still = sim.backend.pos
    for i in range(3):  # hand drifts while the pinch closes
        sim.step(make_hand("relaxed", center=(0.5 + (i + 1) * 0.01, 0.5)))
    sim.step(make_hand("pinch", center=(0.53, 0.5)), frames=3)
    sim.step(make_hand("relaxed", center=(0.53, 0.5)), frames=2)
    calls = sim.backend.calls
    click_at = calls.index(("click", "left", 1))
    drifted = max(c[1] for c in calls[:click_at] if c[0] == "move")
    x_at_click = [c for c in calls[:click_at] if c[0] == "move"][-1][1]
    assert x_at_click < drifted - 10  # pulled back toward where the hand was resting
    assert abs(x_at_click - still[0]) < abs(drifted - still[0]) / 2


def test_fist_is_clutch():
    sim = Sim()
    sim.step(make_hand("point", center=(0.4, 0.5)), frames=3)
    before = sim.backend.pos
    for i in range(10):
        sim.step(make_hand("fist", center=(0.4 + i * 0.02, 0.5)))
    assert sim.backend.pos == before


def test_tap_clicks_and_double_tap_double_clicks():
    sim = Sim()
    sim.step(make_hand("point"), frames=3)
    sim.step(make_hand("pinch"), frames=3)
    sim.step(make_hand("relaxed"), frames=3)
    sim.step(make_hand("pinch"), frames=3)
    sim.step(make_hand("relaxed"), frames=2)
    assert sim.backend.of("click") == [("click", "left", 1), ("click", "left", 2)]
    assert not sim.backend.of("press")


def test_hold_pinch_drags():
    sim = Sim()
    sim.step(make_hand("point"), frames=3)
    sim.step(make_hand("pinch", center=(0.5, 0.5)), frames=12)  # 0.4 s hold
    assert sim.backend.of("press") == [("press", "left")]
    start = sim.backend.pos
    for i in range(10):
        sim.step(make_hand("pinch", center=(0.5, 0.5 + i * 0.01)))
    assert sim.backend.pos[1] > start[1] + 20
    sim.step(make_hand("point"), frames=2)
    assert sim.backend.of("release") == [("release", "left")]
    assert not sim.backend.of("click")


def test_drag_released_when_hand_lost():
    sim = Sim()
    sim.step(make_hand("pinch"), frames=12)
    sim.step(frames=10)  # no hands
    assert sim.backend.of("release") == [("release", "left")]


def test_thumb_middle_tap_right_clicks():
    sim = Sim()
    sim.step(make_hand("point"), frames=3)
    sim.step(make_hand("pinch_middle"), frames=4)
    sim.step(make_hand("point"), frames=3)
    assert sim.backend.of("click") == [("click", "right", 1)]


def test_two_finger_scroll_natural_direction():
    sim = Sim(scroll__inertia="false")
    for i in range(15):
        sim.step(make_hand("two", center=(0.5, 0.6 - i * 0.01)))  # fingers move up
    total_dy = sum(c[2] for c in sim.backend.of("scroll"))
    assert total_dy < -20  # content moves up with the fingers


def test_scroll_inertia_continues_after_release():
    sim = Sim()
    for i in range(10):
        sim.step(make_hand("two", center=(0.5, 0.7 - i * 0.03)))
    n = len(sim.backend.of("scroll"))
    sim.step(make_hand("fist"), frames=10)
    assert len(sim.backend.of("scroll")) > n


@pytest.mark.parametrize("dx,dy,name", [(-0.03, 0, "swipe3_left"), (0.03, 0, "swipe3_right"),
                                         (0, -0.03, "swipe3_up"), (0, 0.03, "swipe3_down")])
def test_three_finger_flick_swipes_and_restores_cursor(dx, dy, name):
    sim = Sim()
    sim.step(make_hand("three", center=(0.5, 0.5)), frames=5)
    start = sim.backend.pos
    for i in range(1, 12):
        sim.step(make_hand("three", center=(0.5 + i * dx, 0.5 + i * dy)))
    expected = sim.cfg.actions[os_key()][name]
    assert sim.backend.of("hotkey") == [("hotkey", expected)]
    moves = [c for c in sim.backend.calls if c[0] == "move"]
    restored = sim.backend.calls[sim.backend.calls.index(("hotkey", expected)) - 1]
    assert restored[0] == "move" and abs(restored[1] - start[0]) < 30 and abs(restored[2] - start[1]) < 30
    assert moves


def test_four_finger_swipe():
    sim = Sim()
    for i in range(12):
        sim.step(make_hand("palm", center=(0.5, 0.3 + i * 0.03)))
    assert any(e.startswith("swipe4_down") for e in sim.events())


def test_two_hand_spread_zooms_in_without_clicking():
    sim = Sim()
    for i in range(15):
        right = make_hand("pinch", center=(0.6 + i * 0.01, 0.5), handedness="Right")
        left = make_hand("pinch", center=(0.4 - i * 0.01, 0.5), handedness="Left")
        sim.step(right, left)
    sim.step(make_hand("point"), frames=3)
    assert any(e.startswith("zoom_in") for e in sim.events())
    assert not sim.backend.of("click") and not sim.backend.of("press")


def test_two_hand_rotate():
    sim = Sim()
    for i in range(15):
        right = make_hand("pinch", center=(0.65, 0.5 + i * 0.012), handedness="Right")
        left = make_hand("pinch", center=(0.35, 0.5 - i * 0.012), handedness="Left")
        sim.step(right, left)
    assert any(e.startswith("rotate_cw") for e in sim.events())


def test_grab_push_toward_camera_zooms_in():
    sim = Sim()
    for i in range(15):
        sim.step(make_hand("grab", scale=0.2 + i * 0.006))
    assert any(e.startswith("zoom_in") for e in sim.events())
    assert not sim.backend.of("click")


def test_grab_roll_rotates():
    sim = Sim()
    for i in range(15):
        sim.step(make_hand("grab", roll=-i * 3.0))
    assert any(e.startswith("rotate_ccw") for e in sim.events())


def test_both_palms_toggle_pause():
    sim = Sim()
    right = make_hand("palm", center=(0.7, 0.5), handedness="Right")
    left = make_hand("palm", center=(0.3, 0.5), handedness="Left")
    sim.step(right, left, frames=50)
    assert sim.engine.paused
    sim.step(make_hand("point"), frames=2)
    sim.step(make_hand("pinch"), frames=3)
    sim.step(make_hand("point"), frames=3)
    assert not sim.backend.of("click")


def test_absolute_mode_maps_region_to_screen():
    sim = Sim(pointer__mode="absolute")
    sim.step(make_hand("point", center=(0.2, 0.7)), frames=10)
    left_x = sim.backend.pos[0]
    sim.step(make_hand("point", center=(0.8, 0.7)), frames=10)
    assert sim.backend.pos[0] > left_x + 500


def test_parse_hotkey_and_overrides():
    assert parse_hotkey("Cmd+Shift+Tab") == ["cmd", "shift", "tab"]
    assert parse_hotkey("cmd+=") == ["cmd", "="]
    assert parse_hotkey("ctrl++") == ["ctrl", "="]
    assert parse_override("pointer.speed=1.5") == (["pointer", "speed"], 1.5)


def test_screens_clamp_handles_gaps():
    screens = Screens([Display(0, 0, 1000, 800, primary=True), Display(1000, -200, 1920, 1080)])
    assert screens.clamp(500, 900) == (500, 799)
    assert screens.clamp(1500, -150) == (1500, -150)


def test_edge_push_glides_cursor_while_knuckles_past_edge():
    sim = Sim()
    sim.step(make_hand("relaxed", center=(0.78, 0.5)), frames=5)
    sim.step(make_hand("relaxed", center=(0.92, 0.5)), frames=3)  # knuckles past the right edge
    x_entered = sim.backend.pos[0]
    assert x_entered < 1100
    sim.step(make_hand("relaxed", center=(0.92, 0.5)), frames=30)  # hold still there
    assert sim.backend.pos[0] > x_entered + 100
    assert sim.engine.edge_push[0] > 0
    held = sim.backend.pos[0]
    sim.step(make_hand("relaxed", center=(0.6, 0.5)), frames=2)
    sim.step(make_hand("relaxed", center=(0.6, 0.5)), frames=10)  # back inside: gliding stops
    assert sim.engine.edge_push[0] == 0
    still = sim.backend.pos
    sim.step(make_hand("relaxed", center=(0.6, 0.5)), frames=10)
    assert sim.backend.pos == still and held


def test_edge_push_can_be_disabled():
    sim = Sim(pointer__edge_push="false")
    sim.step(make_hand("relaxed", center=(0.97, 0.5)), frames=10)
    still = sim.backend.pos
    sim.step(make_hand("relaxed", center=(0.97, 0.5)), frames=30)
    assert sim.backend.pos == still


def test_write_user_config_saves_only_updates(tmp_path):
    import yaml
    from virtual_touch.config import load_config, write_user_config

    path = write_user_config({"gestures": {"pinch_close": 0.4}}, tmp_path / "config.yaml")
    assert yaml.safe_load(path.read_text()) == {"gestures": {"pinch_close": 0.4}}
    cfg = load_config(path)
    assert cfg.gestures.pinch_close == 0.4 and cfg.camera.width == "auto"  # defaults still apply
