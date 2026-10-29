"""Interactive calibration: measure *your* open vs. pinched distances and save thresholds."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import write_user_config
from .hand import HandAnalyzer
from .tracker import Camera, HandTracker

STEPS = [
    ("open", "Point with your INDEX finger, thumb relaxed. Move it around a little."),
    ("pinch", "Touch THUMB and INDEX fingertips together and hold."),
]
READY_TIME = 2.5
COLLECT_TIME = 3.0


def calibrate(cfg, config_path=None) -> int:
    camera = Camera(cfg.camera)
    tracker = HandTracker(cfg.tracking)
    analyzer = HandAnalyzer(cfg.gestures)
    samples = {name: [] for name, _ in STEPS}
    seq = 0
    try:
        for name, instruction in STEPS:
            start = time.monotonic()
            while True:
                frame, seq = camera.read(seq)
                if frame is None:
                    continue
                t = time.monotonic()
                elapsed = t - start
                raws = tracker.process(frame, t)
                collecting = elapsed >= READY_TIME
                if raws and collecting:
                    samples[name].append(analyzer.analyze(raws[0]).pinch_index)
                if elapsed >= READY_TIME + COLLECT_TIME:
                    break
                img = frame.copy()
                msg = "get ready..." if not collecting else f"recording {len(samples[name])}"
                cv2.putText(img, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(img, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Virtual Touch calibration", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("calibration cancelled")
                    return 1
    finally:
        camera.close()
        tracker.close()
        cv2.destroyAllWindows()

    if len(samples["open"]) < 10 or len(samples["pinch"]) < 10:
        print("Not enough hand samples — make sure your hand is visible and try again.")
        return 1
    closed = float(np.percentile(samples["pinch"], 90))
    opened = float(np.percentile(samples["open"], 10))
    gap = opened - closed
    if gap < 0.1:
        print(f"Open ({opened:.2f}) and pinched ({closed:.2f}) are too similar — try again "
              "with a clearer pinch and a relaxed thumb.")
        return 1
    result = {
        "pinch_close": round(closed + 0.35 * gap, 3),
        "pinch_open": round(closed + 0.6 * gap, 3),
    }
    path = write_user_config({"gestures": result}, config_path)
    print(f"pinched≈{closed:.2f}  open≈{opened:.2f} (palm units)")
    print(f"saved {result} to {path}")
    return 0
