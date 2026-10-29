"""Detect machine-specific parameters at start-up: displays, cameras, permissions.

Everything here can be overridden in config.yaml, so detection only has to be
right by default, not always.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Display:
    x: float
    y: float
    width: float
    height: float
    scale: float = 1.0          # backing pixels per coordinate unit (2.0 on Retina)
    name: str = ""
    primary: bool = False

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def clamp(self, px: float, py: float) -> Tuple[float, float]:
        return (
            min(max(px, self.x), self.x + self.width - 1),
            min(max(py, self.y), self.y + self.height - 1),
        )


class Screens:
    """The set of displays in the OS's global cursor coordinate space."""

    def __init__(self, displays: List[Display]):
        if not displays:
            displays = [Display(0, 0, 1920, 1080, primary=True, name="fallback")]
        self.displays = displays

    @property
    def primary(self) -> Display:
        return next((d for d in self.displays if d.primary), self.displays[0])

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        x0 = min(d.x for d in self.displays)
        y0 = min(d.y for d in self.displays)
        x1 = max(d.x + d.width for d in self.displays)
        y1 = max(d.y + d.height for d in self.displays)
        return x0, y0, x1, y1

    def clamp(self, px: float, py: float) -> Tuple[float, float]:
        """Keep the cursor on a real display (handles gaps in multi-monitor layouts)."""
        for d in self.displays:
            if d.contains(px, py):
                return px, py
        best = min(
            (d.clamp(px, py) for d in self.displays),
            key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2,
        )
        return best


# --------------------------------------------------------------------------- displays

def _displays_macos() -> List[Display]:
    import Quartz

    err, ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if err:
        return []
    main = Quartz.CGMainDisplayID()
    out = []
    for did in ids[:count]:
        b = Quartz.CGDisplayBounds(did)  # points, global coords, origin top-left of main display
        mode = Quartz.CGDisplayCopyDisplayMode(did)
        px_w = Quartz.CGDisplayModeGetPixelWidth(mode) if mode else b.size.width
        out.append(
            Display(
                x=b.origin.x,
                y=b.origin.y,
                width=b.size.width,
                height=b.size.height,
                scale=px_w / b.size.width if b.size.width else 1.0,
                name="built-in" if Quartz.CGDisplayIsBuiltin(did) else f"display {did}",
                primary=did == main,
            )
        )
    return out


def enable_windows_dpi_awareness() -> None:
    """Make Win32 coordinates physical pixels so detection and cursor agree."""
    if not sys.platform.startswith("win"):
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _displays_windows() -> List[Display]:
    import ctypes
    from ctypes import wintypes

    enable_windows_dpi_awareness()
    user32 = ctypes.windll.user32

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    out: List[Display] = []
    proc = ctypes.WINFUNCTYPE(
        ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )

    def callback(hmon, _hdc, _rect, _lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        r = info.rcMonitor
        scale = 1.0
        try:
            dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
            ctypes.windll.shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y))
            scale = dpi_x.value / 96.0
        except Exception:
            pass
        out.append(
            Display(
                x=r.left, y=r.top, width=r.right - r.left, height=r.bottom - r.top,
                scale=scale, name=info.szDevice, primary=bool(info.dwFlags & 1),
            )
        )
        return 1

    user32.EnumDisplayMonitors(None, None, proc(callback), 0)
    return out


def _displays_linux() -> List[Display]:
    try:
        from screeninfo import get_monitors
    except ImportError:
        return []
    return [
        Display(m.x, m.y, m.width, m.height, name=m.name or "", primary=bool(getattr(m, "is_primary", False)))
        for m in get_monitors()
    ]


def detect_displays() -> List[Display]:
    try:
        if sys.platform == "darwin":
            return _displays_macos()
        if sys.platform.startswith("win"):
            return _displays_windows()
        return _displays_linux()
    except Exception as exc:  # detection must never stop the app
        print(f"[virtual-touch] display detection failed ({exc}); using fallback", file=sys.stderr)
        return []


def screens_from_config(displays_cfg) -> Screens:
    if displays_cfg in (None, "auto"):
        return Screens(detect_displays())
    return Screens([
        Display(
            x=d.get("x", 0), y=d.get("y", 0), width=d["width"], height=d["height"],
            scale=d.get("scale", 1.0), name=d.get("name", f"display {i}"), primary=(i == 0),
        )
        for i, d in enumerate(displays_cfg)
    ])


# --------------------------------------------------------------------------- cameras

def _capture_backend() -> int:
    import cv2

    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def open_camera(index: int, width, height, fps: int):
    """Open a camera. width/height "auto" asks for the largest format the driver offers,
    which on most webcams is the one covering the most sensor area."""
    import cv2

    cap = cv2.VideoCapture(index, _capture_backend())
    if not cap.isOpened():
        cap.release()
        return None
    if sys.platform.startswith("win"):
        # DirectShow defaults to YUY2, which drops large formats to ~5 fps; MJPG keeps 30.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096 if width in (None, "auto") else int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4096 if height in (None, "auto") else int(height))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe_cameras(max_index: int = 5) -> List[dict]:
    """Return cameras that actually deliver frames, with their native resolution."""
    import cv2

    found = []
    for idx in range(max_index):
        cap = open_camera(idx, 640, 480, 30)
        if cap is None:
            break  # indices are contiguous; also avoids repeated permission prompts
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append({
                "index": idx,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0),
            })
        cap.release()
    return found


def camera_names() -> List[str]:
    """Human-readable camera names (macOS only, for `doctor`)."""
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return []
    return [line.strip().rstrip(":") for line in out.splitlines()
            if line.startswith("    ") and not line.startswith("      ") and line.strip().endswith(":")]


# --------------------------------------------------------------------------- permissions

def accessibility_trusted(prompt: bool = False) -> Optional[bool]:
    """macOS: can this process post mouse/keyboard events? None on other OSes."""
    if sys.platform != "darwin":
        return None
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt}))
    except Exception:
        return None


def macos_natural_scrolling() -> Optional[bool]:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(
            ["defaults", "read", "-g", "com.apple.swipescrolldirection"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out != "0"
    except Exception:
        return None


def summary() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "displays": detect_displays(),
        "cameras": probe_cameras(),
        "camera_names": camera_names(),
        "accessibility": accessibility_trusted(),
        "natural_scrolling": macos_natural_scrolling(),
    }
