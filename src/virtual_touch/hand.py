"""Per-hand features computed from MediaPipe's 21 landmarks.

Coordinates are converted to "frame-height units" (x scaled by the frame aspect
ratio) so horizontal and vertical distances are comparable, and all gesture
distances are divided by the palm length so they do not depend on how far the
hand is from the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class Pose(str, Enum):
    RELAXED = "relaxed"           # natural, half-open hand — move pointer
    FIST = "fist"                 # all fingers tightly curled — "finger lifted"
    POINT = "point"               # index only — move pointer
    TWO = "two"                   # index + middle, others curled — scroll
    THREE = "three"               # index + middle + ring — move; flick = 3-finger swipe
    PALM = "palm"                 # all four straight — move; flick = 4-finger swipe; both = pause
    PINCH = "pinch"               # thumb + index — click / drag / zoom
    PINCH_MIDDLE = "pinch_middle" # thumb + middle — right click
    GRAB = "grab"                 # thumb + index + middle — one-hand zoom / rotate


PINCH_POSES = (Pose.PINCH, Pose.GRAB)
HOVER_POSES = (Pose.RELAXED, Pose.POINT, Pose.THREE, Pose.PALM)


@dataclass
class RawHand:
    """One detected hand as delivered by the tracker."""

    landmarks: np.ndarray        # (21, 3) normalised image coords (x, y in 0..1, z relative)
    handedness: str              # "Left" / "Right" as seen by the user (frame is mirrored)
    score: float = 1.0
    aspect: float = 4 / 3        # frame width / height


@dataclass
class HandFeatures:
    raw: RawHand
    points: np.ndarray           # (21, 3) in frame-height units
    palm_size: float
    extended: Dict[str, bool]    # clearly straight
    curled: Dict[str, bool]      # clearly folded into the palm (neither = relaxed)
    pinch_index: float           # thumb tip <-> index tip, palm units
    pinch_middle: float          # thumb tip <-> middle tip, palm units
    index_middle: float          # index tip <-> middle tip, palm units
    pinch_index_closed: bool
    pinch_middle_closed: bool
    raw_pose: Pose
    pose: Pose                   # debounced

    @property
    def handedness(self) -> str:
        return self.raw.handedness

    def xy(self, idx: int) -> np.ndarray:
        return self.points[idx, :2]

    @property
    def index_tip(self) -> np.ndarray:
        return self.xy(INDEX_TIP)

    @property
    def knuckles(self) -> np.ndarray:
        """Centre of the four knuckles: barely moves when fingers curl or pinch,
        so it is the pointer anchor (like the contact point on a trackpad)."""
        return self.points[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)

    @property
    def pinch_point(self) -> np.ndarray:
        return (self.xy(THUMB_TIP) + self.xy(INDEX_TIP)) / 2

    @property
    def two_point(self) -> np.ndarray:
        return (self.xy(INDEX_TIP) + self.xy(MIDDLE_TIP)) / 2

    @property
    def palm_center(self) -> np.ndarray:
        return self.points[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)

    @property
    def roll(self) -> float:
        """Hand roll in degrees (direction of the knuckle line, image y points down)."""
        v = self.xy(PINKY_MCP) - self.xy(INDEX_MCP)
        return math.degrees(math.atan2(v[1], v[0]))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def to_frame_units(landmarks: np.ndarray, aspect: float) -> np.ndarray:
    pts = np.array(landmarks, dtype=float, copy=True)
    pts[:, 0] *= aspect
    pts[:, 2] *= aspect  # MediaPipe z uses roughly the same scale as x
    return pts


def finger_straightness(pts: np.ndarray, finger: str) -> float:
    mcp, pip, dip, tip = (pts[i] for i in FINGERS[finger])
    path = _dist(mcp, pip) + _dist(pip, dip) + _dist(dip, tip)
    return _dist(mcp, tip) / path if path > 1e-9 else 0.0


def classify(extended: Dict[str, bool], curled: Dict[str, bool], index_closed: bool, middle_closed: bool,
             pinch_index: float, pinch_middle: float, index_middle: float, close: float) -> Pose:
    if index_closed and middle_closed:
        if index_middle < close * 1.4:
            return Pose.GRAB
        return Pose.PINCH if pinch_index <= pinch_middle else Pose.PINCH_MIDDLE
    if index_closed:
        return Pose.PINCH
    if middle_closed:
        return Pose.PINCH_MIDDLE
    i, m, r, p = (extended[f] for f in ("index", "middle", "ring", "pinky"))
    ci, cm, cr, cp = (curled[f] for f in ("index", "middle", "ring", "pinky"))
    # Command poses need the "other" fingers clearly curled, so a relaxed hand
    # (half-bent fingers) never scrolls or clutches by accident.
    if ci and cm and cr and cp:
        return Pose.FIST
    if i and m and cr and cp:
        return Pose.TWO
    if i and cm and cr and cp:
        return Pose.POINT
    if i and m and r and p:
        return Pose.PALM
    if i and m and r and cp:
        return Pose.THREE
    return Pose.RELAXED


@dataclass
class _HandMemory:
    index_closed: bool = False
    middle_closed: bool = False
    pose: Pose = Pose.RELAXED
    candidate: Pose = Pose.RELAXED
    candidate_frames: int = 0


class HandAnalyzer:
    """Turns RawHand into HandFeatures, keeping per-hand hysteresis state."""

    def __init__(self, gestures_cfg):
        self.cfg = gestures_cfg
        self._memory: Dict[str, _HandMemory] = {}

    def reset(self, handedness: Optional[str] = None) -> None:
        if handedness is None:
            self._memory.clear()
        else:
            self._memory.pop(handedness, None)

    def analyze(self, raw: RawHand) -> HandFeatures:
        cfg = self.cfg
        mem = self._memory.setdefault(raw.handedness, _HandMemory())
        pts = to_frame_units(raw.landmarks, raw.aspect)
        palm = max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)

        extended, curled = {}, {}
        for name, (mcp, pip, _dip, tip) in FINGERS.items():
            straight = finger_straightness(pts, name) >= cfg.finger_straightness
            reach = _dist(pts[WRIST], pts[tip]) / palm
            extended[name] = straight and _dist(pts[WRIST], pts[tip]) > _dist(pts[WRIST], pts[pip])
            curled[name] = not extended[name] and reach < cfg.finger_curl

        pinch_index = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm
        pinch_middle = _dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm
        index_middle = _dist(pts[INDEX_TIP], pts[MIDDLE_TIP]) / palm

        close, open_ = cfg.pinch_close, cfg.pinch_open
        mem.index_closed = pinch_index < (open_ if mem.index_closed else close)
        mem.middle_closed = pinch_middle < (open_ if mem.middle_closed else close)

        raw_pose = classify(extended, curled, mem.index_closed, mem.middle_closed,
                            pinch_index, pinch_middle, index_middle, close)

        # Pinch transitions are already stabilised by hysteresis and must be
        # instant for snappy clicks; finger-count poses are debounced.
        instant = raw_pose in PINCH_POSES or raw_pose == Pose.PINCH_MIDDLE or mem.pose in PINCH_POSES
        if raw_pose == mem.pose:
            mem.candidate, mem.candidate_frames = raw_pose, 0
        elif instant:
            mem.pose, mem.candidate, mem.candidate_frames = raw_pose, raw_pose, 0
        else:
            if raw_pose == mem.candidate:
                mem.candidate_frames += 1
            else:
                mem.candidate, mem.candidate_frames = raw_pose, 1
            if mem.candidate_frames >= cfg.pose_stable_frames:
                mem.pose = raw_pose

        return HandFeatures(
            raw=raw, points=pts, palm_size=palm, extended=extended, curled=curled,
            pinch_index=pinch_index, pinch_middle=pinch_middle, index_middle=index_middle,
            pinch_index_closed=mem.index_closed, pinch_middle_closed=mem.middle_closed,
            raw_pose=raw_pose, pose=mem.pose,
        )
