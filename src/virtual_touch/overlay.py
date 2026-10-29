"""Preview window: camera image, hand skeletons, current mode and recent actions."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .hand import CONNECTIONS, FINGERS, INDEX_TIP, MIDDLE_TIP, THUMB_TIP, HandFeatures

WINDOW = "Virtual Touch"
HELP = "q quit | p pause | d dry-run | m pointer mode"

MODE_COLORS = {
    "idle": (160, 160, 160), "hover": (80, 220, 80), "pinch_pending": (0, 200, 255),
    "drag": (0, 120, 255), "right": (255, 120, 0), "scroll": (255, 200, 0),
    "grab": (80, 255, 255),
    "two_hand": (80, 255, 255),
}


def _px(pt, w: int, h: int):
    return int(pt[0] * w), int(pt[1] * h)


def _draw_edge_region(out: np.ndarray, region, push) -> None:
    """The 'trackpad' box: knuckles past an edge glide the cursor; the active edge lights up."""
    h, w = out.shape[:2]
    x0, y0, x1, y1 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
    idle, active = (120, 120, 120), (0, 200, 255)
    edges = {
        "left": ((x0, y0), (x0, y1), push[0] < 0), "right": ((x1, y0), (x1, y1), push[0] > 0),
        "top": ((x0, y0), (x1, y0), push[1] < 0), "bottom": ((x0, y1), (x1, y1), push[1] > 0),
    }
    for a, b, on in edges.values():
        cv2.line(out, a, b, active if on else idle, 4 if on else 1, cv2.LINE_AA)


def _leaving_view(hand: HandFeatures) -> str:
    lm = hand.raw.landmarks
    dirs = []
    if lm[:, 1].min() < 0.01:
        dirs.append("down")   # fingers leaving the top: lower your hand
    if lm[:, 1].max() > 0.99:
        dirs.append("up")
    if lm[:, 0].min() < 0.01:
        dirs.append("right")
    if lm[:, 0].max() > 0.99:
        dirs.append("left")
    return " & ".join(dirs)


def draw(frame: np.ndarray, hands: List[HandFeatures], dominant: Optional[HandFeatures],
         mode: str, paused: bool, dry_run: bool, events, now: float, fps: float,
         edge_region=None, edge_push=(0.0, 0.0)) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    color = (128, 128, 128) if paused else MODE_COLORS.get(mode, (255, 255, 255))
    if edge_region is not None:
        _draw_edge_region(out, edge_region, edge_push)

    for hand in hands:
        lm = hand.raw.landmarks
        is_dom = hand is dominant
        bone = color if is_dom else (200, 200, 200)
        for a, b in CONNECTIONS:
            cv2.line(out, _px(lm[a], w, h), _px(lm[b], w, h), bone, 2 if is_dom else 1, cv2.LINE_AA)
        for i, p in enumerate(lm):
            r = 6 if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP) else 3
            cv2.circle(out, _px(p, w, h), r, (255, 255, 255), -1, cv2.LINE_AA)
        if is_dom:  # pointer anchor
            k = hand.knuckles
            cv2.circle(out, (int(k[0] / hand.raw.aspect * w), int(k[1] * h)), 9, color, 2, cv2.LINE_AA)
        wx, wy = _px(lm[0], w, h)
        # finger states: E = extended, C = curled, r = relaxed (index, middle, ring, pinky)
        states = "".join("E" if hand.extended[f] else "C" if hand.curled[f] else "r" for f in FINGERS)
        label = f"{hand.handedness[0]} {hand.pose.value} {states} p={hand.pinch_index:.2f}"
        cv2.putText(out, label, (wx - 40, wy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bone, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (w, 34), (30, 30, 30), -1)
    status = "PAUSED" if paused else mode.upper()
    cv2.putText(out, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    right = f"{fps:4.1f} fps" + ("  DRY-RUN" if dry_run else "")
    cv2.putText(out, right, (w - 170, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    y = 56
    for t, text in list(events)[-5:]:
        if now - t > 4.0:
            continue
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22

    if dominant is not None:
        leaving = _leaving_view(dominant)
        if leaving:
            msg = f"hand leaving camera view - move it {leaving}"
            cv2.putText(out, msg, (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.putText(out, HELP, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return out
