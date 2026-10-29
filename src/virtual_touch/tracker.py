"""Camera capture + MediaPipe HandLandmarker."""

from __future__ import annotations

import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import cache_dir
from .hand import RawHand
from .system_info import open_camera, probe_cameras

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def ensure_model(model_path) -> Path:
    if model_path not in (None, "auto"):
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"hand model not found: {path}")
        return path
    path = cache_dir() / "hand_landmarker.task"
    if not path.exists():
        print(f"[virtual-touch] downloading hand model to {path} ...", file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
    return path


class Camera:
    """Reads frames on a background thread so inference always gets the newest frame."""

    def __init__(self, cam_cfg):
        index = cam_cfg.index
        if index in (None, "auto"):
            found = probe_cameras()
            if not found:
                raise RuntimeError(
                    "No working camera found. On macOS grant Camera access to your terminal "
                    "(System Settings > Privacy & Security > Camera)."
                )
            index = found[0]["index"]
        self.index = int(index)
        self.mirror = bool(cam_cfg.mirror)
        self.width = int(cam_cfg.get("process_width", 640))
        self.cap = open_camera(self.index, cam_cfg.width, cam_cfg.height, cam_cfg.fps)
        if self.cap is None:
            raise RuntimeError(f"Could not open camera {self.index}")
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            # Some cameras (e.g. MacBook built-ins) ignore the requested size; downscale so
            # inference and drawing stay fast.
            if frame.shape[1] > self.width * 1.1:
                h = round(frame.shape[0] * self.width / frame.shape[1])
                frame = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame, self._seq = frame, self._seq + 1

    def read(self, last_seq: int, timeout: float = 1.0) -> Tuple[Optional[np.ndarray], int]:
        """Block until a frame newer than `last_seq` arrives."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._frame, self._seq
            time.sleep(0.002)
        return None, last_seq

    @property
    def capture_size(self) -> Tuple[int, int]:
        """Native format delivered by the camera."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def size(self) -> Tuple[int, int]:
        """Size of the frames delivered to the tracker (after any downscaling)."""
        frame, _ = self.read(-1)
        if frame is None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame.shape[1], frame.shape[0]

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


class HandTracker:
    def __init__(self, tracking_cfg):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(tracking_cfg.model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(tracking_cfg.max_hands),
            min_hand_detection_confidence=float(tracking_cfg.min_detection_confidence),
            min_hand_presence_confidence=float(tracking_cfg.min_presence_confidence),
            min_tracking_confidence=float(tracking_cfg.min_tracking_confidence),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, t: float) -> List[RawHand]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(t * 1000), self._last_ts + 1)  # timestamps must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)
        hands = []
        for lms, handedness in zip(result.hand_landmarks, result.handedness):
            arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
            cat = handedness[0]
            hands.append(RawHand(arr, cat.category_name, float(cat.score), w / h))
        return hands

    def close(self) -> None:
        self._landmarker.close()
