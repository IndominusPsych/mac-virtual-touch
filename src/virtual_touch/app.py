"""Main loop: camera -> tracker -> analyzer -> gesture engine -> backend (+ preview)."""

from __future__ import annotations

import sys
import threading
import time

from .backends import DryRunBackend, create_backend
from .config import os_key
from .gestures import GestureEngine
from .hand import HandAnalyzer
from .system_info import accessibility_trusted, screens_from_config
from .tracker import Camera, HandTracker


def _start_hotkey(combo, flag: threading.Event):
    if not combo:
        return None
    try:
        from pynput import keyboard

        listener = keyboard.GlobalHotKeys({combo: flag.set})
        listener.daemon = True
        listener.start()
        return listener
    except Exception as exc:
        print(f"[virtual-touch] pause hotkey unavailable: {exc}", file=sys.stderr)
        return None


def run(cfg) -> int:
    if sys.platform == "darwin" and not cfg.app.dry_run and accessibility_trusted(prompt=True) is False:
        print(
            "[virtual-touch] Accessibility permission is missing, so the cursor cannot be moved.\n"
            "  System Settings > Privacy & Security > Accessibility: enable your terminal app\n"
            "  (Terminal, iTerm, VS Code...), then restart it. Continuing in view-only mode.",
            file=sys.stderr,
        )

    screens = screens_from_config(cfg.screen.displays)
    camera = Camera(cfg.camera)
    tracker = HandTracker(cfg.tracking)
    analyzer = HandAnalyzer(cfg.gestures)
    real_backend = None if cfg.app.dry_run else create_backend(cfg)
    dry_backend = DryRunBackend()
    engine = GestureEngine(cfg, real_backend or dry_backend, screens, cfg.actions.get(os_key(), {}))

    pause_flag = threading.Event()
    listener = _start_hotkey(cfg.app.pause_hotkey, pause_flag)

    cw, ch = camera.capture_size
    print(f"[virtual-touch] camera {camera.index} {cw}x{ch} (tracking at {camera.size[0]}x{camera.size[1]}), "
          f"{len(screens.displays)} display(s), backend={engine.backend.name}, "
          f"pointer={cfg.pointer.mode}, dominant={cfg.hands.dominant}")
    print("[virtual-touch] q / Esc to quit (preview window), Ctrl+C in terminal.")

    preview = bool(cfg.app.preview)
    if preview:
        import cv2
        from . import overlay

        cv2.namedWindow(overlay.WINDOW, cv2.WINDOW_NORMAL)

    seq, fps, last = 0, 0.0, time.monotonic()
    try:
        while True:
            frame, seq = camera.read(seq)
            if frame is None:
                continue
            t = time.monotonic()
            raws = tracker.process(frame, t)
            seen = {r.handedness for r in raws}
            for side in ("Left", "Right"):
                if side not in seen:
                    analyzer.reset(side)
            hands = [analyzer.analyze(r) for r in raws]

            if pause_flag.is_set():
                pause_flag.clear()
                engine.toggle_pause()
            engine.update(hands, t)

            fps = 0.9 * fps + 0.1 / max(t - last, 1e-3)
            last = t
            if not preview:
                continue

            img = overlay.draw(frame, hands, engine.dominant, engine.mode, engine.paused,
                               engine.backend is dry_backend, engine.events, t, fps,
                               edge_region=cfg.pointer.edge_region if cfg.pointer.edge_push
                               and cfg.pointer.mode == "relative" else None,
                               edge_push=engine.edge_push)
            cv2.imshow(overlay.WINDOW, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                engine.toggle_pause()
            elif key == ord("d"):
                engine.reset()
                if engine.backend is dry_backend:
                    real_backend = real_backend or create_backend(cfg, dry_run=False)
                    engine.backend = real_backend
                else:
                    engine.backend = dry_backend
            elif key == ord("m"):
                cfg.pointer.mode = "absolute" if cfg.pointer.mode == "relative" else "relative"
                engine._log(f"pointer mode: {cfg.pointer.mode}")
    except KeyboardInterrupt:
        pass
    finally:
        engine.reset()
        engine.backend.close()
        camera.close()
        tracker.close()
        if listener:
            listener.stop()
        if preview:
            cv2.destroyAllWindows()
    return 0
