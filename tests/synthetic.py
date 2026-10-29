"""Build plausible 21-point hands for each pose, so the engine can be tested without a camera."""

from __future__ import annotations

import math

import numpy as np

from virtual_touch.backends.base import DryRunBackend
from virtual_touch.config import load_config, os_key
from virtual_touch.gestures import GestureEngine
from virtual_touch.hand import HandAnalyzer, RawHand
from virtual_touch.system_info import Display, Screens

ASPECT = 4 / 3
MCP_X = {"index": -0.3, "middle": -0.1, "ring": 0.1, "pinky": 0.3}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

EXTENDED_FINGERS = {
    "fist": (),
    "point": ("index",),
    "two": ("index", "middle"),
    "three": ("index", "middle", "ring"),
    "palm": ("index", "middle", "ring", "pinky"),
    "pinch": (),
    "pinch_middle": ("index",),
    "grab": (),
    "relaxed": (),
}


def make_hand(pose: str, center=(0.5, 0.5), scale=0.22, handedness="Right", roll=0.0) -> RawHand:
    """`center` is in normalised image coords; `scale` is palm length in frame heights."""
    cx, cy = center[0] * ASPECT, center[1]
    s = scale
    pts = np.zeros((21, 3))
    pts[0] = (cx, cy + 0.5 * s, 0)

    for name, mx in MCP_X.items():
        b = BASE[name]
        mcp = np.array([cx + mx * s, cy - 0.5 * s, 0])
        pts[b] = mcp
        if pose == "relaxed":  # gently bent toward the camera, neither straight nor curled
            pts[b + 1] = mcp + (0, -0.39 * s, -0.05 * s)
            pts[b + 2] = mcp + (0, -0.51 * s, -0.20 * s)
            pts[b + 3] = mcp + (0, -0.43 * s, -0.40 * s)
        elif name in EXTENDED_FINGERS[pose]:
            pts[b + 1] = mcp + (0, -0.40 * s, 0)
            pts[b + 2] = mcp + (0, -0.65 * s, 0)
            pts[b + 3] = mcp + (0, -0.85 * s, 0)
        else:  # curled into the palm
            pts[b + 1] = mcp + (0, -0.30 * s, 0)
            pts[b + 2] = mcp + (0.02 * s, -0.15 * s, 0)
            pts[b + 3] = mcp + (0, 0.05 * s, 0)

    # relaxed thumb
    pts[1] = (cx - 0.35 * s, cy + 0.3 * s, 0)
    pts[2] = (cx - 0.50 * s, cy + 0.1 * s, 0)
    pts[3] = (cx - 0.60 * s, cy - 0.05 * s, 0)
    pts[4] = (cx - 0.65 * s, cy - 0.20 * s, 0)

    def bend_to(base: int, tip):
        mcp = pts[base]
        tip = np.array(tip)
        pts[base + 1] = mcp + (tip - mcp) * 0.45 + (0, -0.12 * s, 0)
        pts[base + 2] = mcp + (tip - mcp) * 0.75 + (0, -0.08 * s, 0)
        pts[base + 3] = tip

    if pose == "relaxed":  # thumb resting close to the index finger
        pts[4] = (cx - 0.38 * s, cy - 0.62 * s, -0.1 * s)
    elif pose == "pinch":
        target = (cx - 0.45 * s, cy - 0.75 * s, 0)
        bend_to(5, target)
        pts[4] = np.array(target) + (-0.03 * s, 0.02 * s, 0)
    elif pose == "pinch_middle":
        target = (cx - 0.25 * s, cy - 0.80 * s, 0)
        bend_to(9, target)
        pts[4] = np.array(target) + (-0.03 * s, 0.02 * s, 0)
    elif pose == "grab":
        target = np.array((cx - 0.30 * s, cy - 0.80 * s, 0))
        bend_to(5, target + (-0.04 * s, 0, 0))
        bend_to(9, target + (0.04 * s, 0, 0))
        pts[4] = target + (0, 0.03 * s, 0)

    if roll:
        a = math.radians(roll)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        pts[:, :2] = (pts[:, :2] - (cx, cy)) @ rot.T + (cx, cy)

    pts[:, 0] /= ASPECT
    return RawHand(pts, handedness, 1.0, ASPECT)


class Sim:
    """Feeds synthetic frames at 30 fps through analyzer + engine."""

    def __init__(self, **overrides):
        sets = [f"{k.replace('__', '.')}={v}" for k, v in overrides.items()]
        self.cfg = _load(sets)
        self.backend = DryRunBackend(start=(500.0, 400.0))
        self.screens = Screens([Display(0, 0, 1440, 900, primary=True)])
        self.analyzer = HandAnalyzer(self.cfg.gestures)
        self.engine = GestureEngine(self.cfg, self.backend, self.screens, self.cfg.actions[os_key()])
        self.t = 0.0

    def step(self, *hands: RawHand, frames: int = 1):
        for _ in range(frames):
            self.t += 1 / 30
            seen = {h.handedness for h in hands}
            for side in ("Left", "Right"):
                if side not in seen:
                    self.analyzer.reset(side)
            self.engine.update([self.analyzer.analyze(h) for h in hands], self.t)
        return self

    def events(self):
        return [text for _, text in self.engine.events]


def _load(sets):
    from pathlib import Path

    # Ignore any user config on the dev machine so tests are deterministic.
    empty = Path(__file__).with_name("empty_config.yaml")
    return load_config(path=empty, overrides=sets)
