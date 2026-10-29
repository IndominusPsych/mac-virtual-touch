"""pynput backend — used on Windows and Linux (and as a macOS fallback)."""

from __future__ import annotations

import sys
from typing import List

from .base import Backend, parse_hotkey


def pynput_key(name: str):
    from pynput.keyboard import Key, KeyCode

    special = {
        "cmd": Key.cmd, "win": Key.cmd, "ctrl": Key.ctrl, "alt": Key.alt, "shift": Key.shift,
        "tab": Key.tab, "space": Key.space, "enter": Key.enter, "esc": Key.esc,
        "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
        "home": Key.home, "end": Key.end, "pageup": Key.page_up, "pagedown": Key.page_down,
        "backspace": Key.backspace, "delete": Key.delete,
    }
    if name in special:
        return special[name]
    if name.startswith("f") and name[1:].isdigit():
        return getattr(Key, name)
    if len(name) == 1:
        return KeyCode.from_char(name)
    raise ValueError(f"unknown key {name!r}")


class PynputBackend(Backend):
    name = "pynput"

    def __init__(self, pixels_per_notch: float = 60.0):
        from pynput.keyboard import Controller as KeyboardController
        from pynput.mouse import Button, Controller as MouseController

        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._buttons = {"left": Button.left, "right": Button.right, "middle": Button.middle}
        self.pixels_per_notch = pixels_per_notch
        self._acc_x = 0.0
        self._acc_y = 0.0
        # Windows accepts fractional wheel deltas (smooth scrolling); X11 only whole notches.
        self._fractional = sys.platform.startswith("win")

    def position(self):
        return tuple(self._mouse.position)

    def move_to(self, x, y):
        self._mouse.position = (int(round(x)), int(round(y)))

    def press(self, button="left"):
        self._mouse.press(self._buttons[button])

    def release(self, button="left"):
        self._mouse.release(self._buttons[button])

    def click(self, button="left", count=1):
        # The OS derives double-clicks from timing on Windows/X11, so a single
        # click is sent per tap; `count` is only informational here.
        self._mouse.click(self._buttons[button], 1)

    def scroll(self, content_dx, content_dy):
        # pynput: dy > 0 scrolls up (content moves down); dx > 0 scrolls right (content moves left).
        self._acc_y += content_dy / self.pixels_per_notch
        self._acc_x += -content_dx / self.pixels_per_notch
        if self._fractional:
            # WHEEL_DELTA is 120; send in 1/120 steps.
            sx, sy = int(self._acc_x * 120) / 120, int(self._acc_y * 120) / 120
        else:
            sx, sy = float(int(self._acc_x)), float(int(self._acc_y))
        if sx or sy:
            self._acc_x -= sx
            self._acc_y -= sy
            self._mouse.scroll(sx if self._fractional else int(sx), sy if self._fractional else int(sy))

    def hotkey(self, combo):
        keys: List = [pynput_key(k) for k in parse_hotkey(combo)]
        for k in keys:
            self._keyboard.press(k)
        for k in reversed(keys):
            self._keyboard.release(k)
