"""Gesture engine: turns per-frame hand features into trackpad operations.

Pure logic — no camera, no OS calls except through the backend — so it can be
driven by synthetic hands in tests.

One-hand modes (dominant hand):
    idle           tight fist: finger "lifted", cursor stays put (clutch)
    hover          relaxed / open / pointing hand: cursor follows the knuckles;
                   a quick flick with 3 fingers or an open palm = swipe gesture
    pinch_pending  thumb+index just closed: tap -> click, hold/move -> drag
    drag           left button held, pointer follows the pinch
    right          thumb+middle closed: release quickly -> right click
    scroll         index+middle (ring+pinky curled): two-finger scroll (with inertia)
    grab           thumb+index+middle: push/pull = zoom, roll wrist = rotate
Two-hand modes:
    two_hand       both hands pinching: spread = zoom, steer = rotate
    (both palms open and held = pause / resume)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

from .backends.base import Backend
from .filters import OneEuroFilter
from .hand import HOVER_POSES, PINCH_POSES, HandFeatures, Pose
from .system_info import Screens


def _wrap_degrees(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class GestureEngine:
    def __init__(self, cfg, backend: Backend, screens: Screens, actions: dict):
        self.cfg = cfg
        self.g = cfg.gestures
        self.p = cfg.pointer
        self.s = cfg.scroll
        self.backend = backend
        self.screens = screens
        self.actions = dict(actions or {})
        ref = cfg.screen.reference_height
        self.ref_h = float(screens.primary.height if ref in (None, "auto") else ref)

        self.mode = "idle"
        self.paused = False
        self.events: Deque[Tuple[float, str]] = deque(maxlen=8)
        self.dominant: Optional[HandFeatures] = None
        self.other: Optional[HandFeatures] = None

        self._filter = OneEuroFilter(self.p.smoothing_min_cutoff, self.p.smoothing_beta)
        self._anchor: Optional[np.ndarray] = None
        self._anchor_t = 0.0
        self._cursor: Optional[np.ndarray] = None
        self._last_set: Optional[Tuple[float, float]] = None
        self._palm_ema: Optional[float] = None
        self._last_t: Optional[float] = None
        self._now = 0.0

        self._lost_since: Optional[float] = None
        self._suppress_pinch = False
        self._pinch_t0 = 0.0
        self._pinch_start_pt = np.zeros(2)
        self._last_click_t = -1e9
        self._last_click_pos: Optional[np.ndarray] = None
        self._click_count = 0
        self._right_t0 = 0.0
        self._scroll_v = np.zeros(2)
        self._inertia_v: Optional[np.ndarray] = None
        # (t, knuckles, palm, pose, cursor) for the last ~0.5 s: swipe detection + click rewind
        self._history: Deque[tuple] = deque()
        self._swipe_block_until = -1.0
        self._grab_size = 1.0
        self._grab_smooth = 1.0
        self._grab_roll = 0.0
        self._th_d0 = 1.0
        self._th_smooth = 1.0
        self._th_a0 = 0.0
        self.edge_push = np.zeros(2)  # current glide direction/strength, for the overlay
        self._edge_since: Optional[float] = None
        self._last_glide_t: Optional[float] = None
        self._pause_t0: Optional[float] = None
        self._pause_fired = False

    # ------------------------------------------------------------------ public

    def update(self, hands: List[HandFeatures], t: float) -> None:
        dt = 1 / 30 if self._last_t is None else min(max(t - self._last_t, 1e-3), 0.2)
        self._last_t = self._now = t

        dom, other = self._select(hands)
        self.dominant, self.other = dom, other
        if dom is not None:
            self._palm_ema = dom.palm_size if self._palm_ema is None else 0.8 * self._palm_ema + 0.2 * dom.palm_size

        if self._check_pause(dom, other, t) or self.paused:
            return
        if dom is None:
            self._on_lost(t)
            self._inertia_step(dt)
            return
        self._lost_since = None
        if self._two_hand(dom, other, t):
            return
        self._one_hand(dom, t, dt)
        if self.mode == "idle":
            self._inertia_step(dt)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._log("paused" if self.paused else "resumed")
        if self.paused:
            self.reset()

    def reset(self) -> None:
        """Release anything held and return to idle."""
        if self.mode == "drag":
            self.backend.release("left")
        self._inertia_v = None
        self._set_mode("idle")

    # ------------------------------------------------------------------ helpers

    def _log(self, text: str) -> None:
        self.events.append((self._now, text))

    def _fire(self, name: str) -> None:
        combo = self.actions.get(name)
        self._log(f"{name}" + (f" [{combo}]" if combo else " (unmapped)"))
        if combo:
            try:
                self.backend.hotkey(combo)
            except Exception as exc:  # a bad mapping must not crash the loop
                self._log(f"hotkey error: {exc}")

    def _palm(self, h: HandFeatures) -> float:
        return self._palm_ema or h.palm_size

    def _set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        if self.mode == "scroll" and mode == "idle" and self.s.inertia:
            if float(np.linalg.norm(self._scroll_v)) > 300.0:
                self._inertia_v = self._scroll_v.copy()
        if mode != "idle":
            self._inertia_v = None
        self.mode = mode
        self._history.clear()
        self.edge_push = np.zeros(2)
        self._edge_since = None
        self._last_glide_t = None
        self._anchor = None
        self._filter.reset()
        self._scroll_v = np.zeros(2)

    def _select(self, hands: List[HandFeatures]):
        if not hands:
            return None, None
        if len(hands) == 1:
            return hands[0], None
        want = str(self.cfg.hands.dominant).capitalize()
        matches = [h for h in hands if h.handedness == want]
        if len(matches) == 1:
            dom = matches[0]
        else:  # handedness ambiguous: dominant hand is the one on its side of the mirror
            ordered = sorted(hands, key=lambda h: h.palm_center[0])
            dom = ordered[-1] if want == "Right" else ordered[0]
        other = next(h for h in hands if h is not dom)
        return dom, other

    # ------------------------------------------------------------------ cursor

    def _sync_cursor(self) -> None:
        real = np.array(self.backend.position(), dtype=float)
        if self._cursor is None or self._last_set is None or np.linalg.norm(real - self._last_set) > 2.0:
            self._cursor = real  # the physical mouse/trackpad moved the cursor

    def _place_cursor(self, x: float, y: float) -> None:
        x, y = self.screens.clamp(x, y)
        self._cursor = np.array([x, y])
        self._last_set = (x, y)
        self.backend.move_to(x, y)

    def _relative_move(self, pt: np.ndarray, h: HandFeatures, t: float) -> None:
        f = self._filter(pt, t)
        if self._anchor is None:
            self._anchor, self._anchor_t = f.copy(), t
            return
        d = f - self._anchor
        if self.p.distance_compensation:
            d = d * (self.p.reference_palm / self._palm(h))
        mag = float(np.linalg.norm(d))
        if mag < self.p.deadzone:
            return  # keep the anchor so slow movements still accumulate
        dt = max(t - self._anchor_t, 1e-3)
        self._anchor, self._anchor_t = f.copy(), t
        k = min(max((mag / dt - self.p.slow_velocity) / (self.p.fast_velocity - self.p.slow_velocity), 0.0), 1.0)
        gain = (self.p.min_gain + (self.p.max_gain - self.p.min_gain) * k) * self.p.speed
        self._sync_cursor()
        target = self._cursor + d * gain * self.ref_h
        self._place_cursor(float(target[0]), float(target[1]))

    def _absolute_move(self, pt: np.ndarray, h: HandFeatures, t: float) -> None:
        f = self._filter(pt, t)
        x0, y0, x1, y1 = self.p.absolute_region
        u = min(max((f[0] / h.raw.aspect - x0) / (x1 - x0), 0.0), 1.0)
        v = min(max((f[1] - y0) / (y1 - y0), 0.0), 1.0)
        bx0, by0, bx1, by1 = self.screens.bounds
        self._place_cursor(bx0 + u * (bx1 - bx0 - 1), by0 + v * (by1 - by0 - 1))

    def _move(self, pt: np.ndarray, h: HandFeatures, t: float) -> None:
        if self.p.mode == "absolute":
            self._absolute_move(pt, h, t)
        else:
            self._relative_move(pt, h, t)
            self._edge_glide(h, t)

    def _edge_glide(self, h: HandFeatures, t: float) -> None:
        """Edge push: knuckles beyond the edge region keep the cursor moving that way,
        so a small camera field of view still reaches the whole (multi-)screen."""
        self.edge_push = np.zeros(2)
        if not self.p.get("edge_push", False):
            return
        u, v = h.knuckles[0] / h.raw.aspect, h.knuckles[1]
        x0, y0, x1, y1 = self.p.edge_region

        def depth(val: float, lo: float, hi: float) -> float:
            if val < lo:
                return -min((lo - val) / max(lo, 1e-6), 1.0)
            if val > hi:
                return min((val - hi) / max(1.0 - hi, 1e-6), 1.0)
            return 0.0

        push = np.array([depth(u, x0, x1), depth(v, y0, y1)])
        if not push.any():
            self._edge_since = self._last_glide_t = None
            return
        if self._edge_since is None:
            self._edge_since = t
        if t - self._edge_since < self.p.edge_dwell:
            return
        self.edge_push = push
        dt = 1 / 30 if self._last_glide_t is None else min(t - self._last_glide_t, 0.1)
        self._last_glide_t = t
        speed = self.p.edge_speed * self.ref_h * self.p.speed
        step = np.sign(push) * np.abs(push) ** 1.5 * speed * dt
        self._sync_cursor()
        target = self._cursor + step
        self._place_cursor(float(target[0]), float(target[1]))

    # ------------------------------------------------------------------ pause / lost

    def _check_pause(self, dom, other, t: float) -> bool:
        if dom is not None and other is not None and dom.pose == Pose.PALM and other.pose == Pose.PALM:
            if self._pause_t0 is None:
                self._pause_t0, self._pause_fired = t, False
                if not self.paused:
                    self.reset()
            elif not self._pause_fired and t - self._pause_t0 >= self.g.pause_hold_time:
                self._pause_fired = True
                self.toggle_pause()
            return True
        self._pause_t0 = None
        return False

    def _on_lost(self, t: float) -> None:
        if self._lost_since is None:
            self._lost_since = t
        if self.mode == "drag":
            if t - self._lost_since >= self.g.lost_hand_grace:
                self.backend.release("left")
                self._log("drag end (hand lost)")
                self._set_mode("idle")
        elif self.mode != "idle":
            self._set_mode("idle")

    def _inertia_step(self, dt: float) -> None:
        if self._inertia_v is None:
            return
        step = self._inertia_v * dt
        self.backend.scroll(float(step[0]), float(step[1]))
        self._inertia_v = self._inertia_v * math.exp(-dt / self.s.inertia_decay)
        if float(np.linalg.norm(self._inertia_v)) < 40.0:
            self._inertia_v = None

    # ------------------------------------------------------------------ two hands

    def _two_hand(self, dom: HandFeatures, other: Optional[HandFeatures], t: float) -> bool:
        active = other is not None and dom.pose in PINCH_POSES and other.pose in PINCH_POSES
        if self.mode == "two_hand":
            if not active:
                self._set_mode("idle")
                self._suppress_pinch = True  # don't turn the remaining pinch into a click
                return True
        elif active and self.mode != "drag":
            vec = other.pinch_point - dom.pinch_point
            self._set_mode("two_hand")  # cancels a pending click
            self._th_d0 = self._th_smooth = float(np.linalg.norm(vec))
            self._th_a0 = math.degrees(math.atan2(vec[1], vec[0]))
            self._log("two-hand transform")
            return True
        else:
            return False

        vec = other.pinch_point - dom.pinch_point
        self._th_smooth = 0.7 * self._th_smooth + 0.3 * float(np.linalg.norm(vec))
        ratio = self._th_smooth / max(self._th_d0, 1e-6)
        if ratio >= 1 + self.g.zoom_step:
            self._fire("zoom_in")
            self._th_d0 = self._th_smooth
        elif ratio <= 1 / (1 + self.g.zoom_step):
            self._fire("zoom_out")
            self._th_d0 = self._th_smooth
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        da = _wrap_degrees(angle - self._th_a0)
        if abs(da) >= self.g.rotate_step:
            self._fire("rotate_cw" if da > 0 else "rotate_ccw")
            self._th_a0 = angle
        return True

    # ------------------------------------------------------------------ one hand

    def _one_hand(self, h: HandFeatures, t: float, dt: float) -> None:
        pose = h.pose

        if self._suppress_pinch:
            if pose in PINCH_POSES:
                return
            self._suppress_pinch = False

        if self.mode in ("pinch_pending", "drag"):
            if pose in PINCH_POSES:
                self._pinch_continue(h, t)
            else:
                self._pinch_end(t)
            return

        if self.mode == "right" and pose != Pose.PINCH_MIDDLE:
            if pose not in PINCH_POSES and t - self._right_t0 <= self.g.right_click_max_time:
                self.backend.click("right", 1)
                self._log("right click")
            self._set_mode("idle")

        if pose == Pose.PINCH:
            if self.mode == "grab":  # letting go of a grab passes through a pinch
                self._suppress_pinch = True
                self._set_mode("idle")
            else:
                self._pinch_start(h, t)
        elif pose == Pose.GRAB:
            self._grab(h)
        elif pose == Pose.PINCH_MIDDLE:
            if self.mode != "right":
                self._set_mode("right")
                self._right_t0 = t
        elif pose in HOVER_POSES:
            self._hover(h, t)
        elif pose == Pose.TWO:
            self._scroll(h, t, dt)
        else:  # fist
            self._set_mode("idle")

    # pinch: click / double click / drag

    def _hover(self, h: HandFeatures, t: float) -> None:
        self._set_mode("hover")
        self._move(h.knuckles, h, t)
        if self._cursor is not None:
            self._history.append((t, h.knuckles.copy(), self._palm(h), h.pose, self._cursor.copy()))
        while self._history and t - self._history[0][0] > max(self.g.swipe_window, self.g.click_rewind) + 0.1:
            self._history.popleft()
        if h.pose in (Pose.THREE, Pose.PALM):
            self._check_swipe(h, t)

    def _check_swipe(self, h: HandFeatures, t: float) -> None:
        """A swipe is a quick flick (swipe_distance palms within swipe_window s) with the
        swipe pose held the whole time; slower movement just moves the pointer."""
        if t < self._swipe_block_until:
            return
        window = [e for e in self._history if t - e[0] <= self.g.swipe_window]
        if len(window) < 3 or any(e[3] != h.pose for e in window):
            return
        disp = (window[-1][1] - window[0][1]) / self._palm(h)
        ax, ay = abs(disp[0]), abs(disp[1])
        if max(ax, ay) < self.g.swipe_distance or max(ax, ay) < 2 * min(ax, ay):
            return
        if ax > ay:
            direction = "right" if disp[0] > 0 else "left"
        else:
            direction = "down" if disp[1] > 0 else "up"
        start_cursor = window[0][4]
        self._place_cursor(float(start_cursor[0]), float(start_cursor[1]))  # undo the flick's pointer motion
        self._fire(f"swipe{3 if h.pose == Pose.THREE else 4}_{direction}")
        self._swipe_block_until = t + self.g.swipe_cooldown
        self._history.clear()
        self._anchor = None

    def _rewind_cursor(self, t: float) -> None:
        """Put the cursor back where it was just before the pinch began to close,
        so the small hand motion of pinching doesn't make the click miss."""
        target = None
        for e in self._history:
            if t - e[0] <= self.g.click_rewind:
                break
            target = e[4]
        if target is None and self._history:
            target = self._history[0][4]
        if target is not None:
            self._place_cursor(float(target[0]), float(target[1]))

    def _pinch_start(self, h: HandFeatures, t: float) -> None:
        if self.mode == "hover":
            self._rewind_cursor(t)
        self._set_mode("pinch_pending")
        self._pinch_t0 = t
        self._pinch_start_pt = h.pinch_point.copy()

    def _pinch_continue(self, h: HandFeatures, t: float) -> None:
        if self.mode == "pinch_pending":
            if h.pose == Pose.GRAB:
                self._grab(h)
                return
            moved = float(np.linalg.norm(h.pinch_point - self._pinch_start_pt)) / self._palm(h)
            if t - self._pinch_t0 >= self.g.drag_hold_time or moved >= self.g.drag_start_distance:
                self.backend.press("left")
                self._log("drag start")
                self._set_mode("drag")
            return
        self._move(h.knuckles, h, t)

    def _pinch_end(self, t: float) -> None:
        if self.mode == "pinch_pending" and t - self._pinch_t0 <= self.g.tap_max_time:
            self._sync_cursor()
            near = self._last_click_pos is not None and np.linalg.norm(self._cursor - self._last_click_pos) < 8
            if t - self._last_click_t <= self.g.double_click_time and near:
                self._click_count = min(self._click_count + 1, 3)
            else:
                self._click_count = 1
            self.backend.click("left", self._click_count)
            self._last_click_t, self._last_click_pos = t, self._cursor.copy()
            self._log({1: "click", 2: "double click", 3: "triple click"}[self._click_count])
        elif self.mode == "drag":
            self.backend.release("left")
            self._log("drag end")
        self._set_mode("idle")

    # two-finger scroll

    def _scroll(self, h: HandFeatures, t: float, dt: float) -> None:
        self._set_mode("scroll")
        f = self._filter(h.two_point, t)
        if self._anchor is None:
            self._anchor = f.copy()
            return
        d = f - self._anchor
        if float(np.linalg.norm(d)) < self.p.deadzone * 0.5:
            return
        self._anchor = f.copy()
        if self.p.distance_compensation:
            d = d * (self.p.reference_palm / self._palm(h))
        content = d * self.s.speed * self.ref_h
        if not self.s.natural:
            content = -content
        if self.s.invert_x:
            content[0] = -content[0]
        self.backend.scroll(float(content[0]), float(content[1]))
        self._scroll_v = 0.6 * self._scroll_v + 0.4 * (content / dt)

    # one-hand zoom / rotate

    def _grab(self, h: HandFeatures) -> None:
        if self.mode != "grab":
            self._set_mode("grab")
            self._grab_size = self._grab_smooth = h.palm_size
            self._grab_roll = h.roll
            return
        self._grab_smooth = 0.7 * self._grab_smooth + 0.3 * h.palm_size
        ratio = self._grab_smooth / max(self._grab_size, 1e-6)
        if ratio >= 1 + self.g.zoom_step:
            self._fire("zoom_in")
            self._grab_size = self._grab_smooth
        elif ratio <= 1 / (1 + self.g.zoom_step):
            self._fire("zoom_out")
            self._grab_size = self._grab_smooth
        dr = _wrap_degrees(h.roll - self._grab_roll)
        if abs(dr) >= self.g.rotate_step:
            self._fire("rotate_cw" if dr > 0 else "rotate_ccw")
            self._grab_roll = h.roll
