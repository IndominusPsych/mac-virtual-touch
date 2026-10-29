"""macOS backend built directly on Quartz CGEvents.

Why not plain pynput: macOS only recognises a double-click when the second
click carries clickState=2, dragging needs *Dragged* events, smooth scrolling
wants pixel units, and Mission Control / Spaces shortcuts ignore synthetic arrow
keys unless the Fn + NumericPad flags are set like a real keyboard does.
"""

from __future__ import annotations

import time
from typing import Tuple

import Quartz

from .base import Backend, parse_hotkey

# ANSI US virtual key codes (Carbon HIToolbox/Events.h).
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47, "`": 50,
    "enter": 36, "tab": 48, "space": 49, "backspace": 51, "esc": 53, "delete": 117,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113,
    "f16": 106, "f17": 64, "f18": 79, "f19": 80, "f20": 90,
}
MODIFIER_FLAGS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "win": Quartz.kCGEventFlagMaskCommand,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "shift": Quartz.kCGEventFlagMaskShift,
}
MODIFIER_KEYCODES = {"cmd": 55, "win": 55, "shift": 56, "alt": 58, "ctrl": 59}
ARROWS = {"left", "right", "up", "down"}
FN_KEYS = {"home", "end", "pageup", "pagedown", "delete"} | {f"f{i}" for i in range(1, 21)}

BUTTONS = {
    "left": (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp,
             Quartz.kCGEventLeftMouseDragged, Quartz.kCGMouseButtonLeft),
    "right": (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp,
              Quartz.kCGEventRightMouseDragged, Quartz.kCGMouseButtonRight),
    "middle": (Quartz.kCGEventOtherMouseDown, Quartz.kCGEventOtherMouseUp,
               Quartz.kCGEventOtherMouseDragged, Quartz.kCGMouseButtonCenter),
}


def _post(event) -> None:
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class MacOSBackend(Backend):
    name = "quartz"

    def __init__(self):
        self._held = None  # button currently held (for drag events)
        self._acc_x = 0.0  # sub-pixel scroll remainders
        self._acc_y = 0.0

    def position(self) -> Tuple[float, float]:
        loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return loc.x, loc.y

    def move_to(self, x, y):
        if self._held:
            _down, _up, drag, btn = BUTTONS[self._held]
            event = Quartz.CGEventCreateMouseEvent(None, drag, (x, y), btn)
        else:
            event = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, (x, y), 0)
        _post(event)

    def _button_event(self, button: str, down: bool, click_state: int = 1) -> None:
        down_t, up_t, _drag, btn = BUTTONS[button]
        event = Quartz.CGEventCreateMouseEvent(None, down_t if down else up_t, self.position(), btn)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_state)
        _post(event)

    def press(self, button="left"):
        self._held = button
        self._button_event(button, True)

    def release(self, button="left"):
        self._button_event(button, False)
        if self._held == button:
            self._held = None

    def click(self, button="left", count=1):
        # The click state tells apps this is the Nth click of a multi-click.
        self._button_event(button, True, count)
        time.sleep(0.004)
        self._button_event(button, False, count)

    def scroll(self, content_dx, content_dy):
        # Quartz: wheel1 > 0 scrolls up (content moves down); wheel2 > 0 scrolls left (content moves right).
        self._acc_x += content_dx
        self._acc_y += content_dy
        dx, dy = int(self._acc_x), int(self._acc_y)
        if dx or dy:
            self._acc_x -= dx
            self._acc_y -= dy
            _post(Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, dy, dx))

    def hotkey(self, combo):
        keys = parse_hotkey(combo)
        mods = [k for k in keys if k in MODIFIER_FLAGS]
        mains = [k for k in keys if k not in MODIFIER_FLAGS]
        flags = 0
        for m in mods:
            flags |= MODIFIER_FLAGS[m]

        for m in mods:  # press modifiers like a person would
            ev = Quartz.CGEventCreateKeyboardEvent(None, MODIFIER_KEYCODES[m], True)
            Quartz.CGEventSetFlags(ev, flags)
            _post(ev)
        for key in mains:
            code = KEYCODES.get(key)
            if code is None:
                raise ValueError(f"unknown key {key!r} in hotkey {combo!r}")
            key_flags = flags
            if key in ARROWS:
                key_flags |= Quartz.kCGEventFlagMaskNumericPad | Quartz.kCGEventFlagMaskSecondaryFn
            elif key in FN_KEYS:
                key_flags |= Quartz.kCGEventFlagMaskSecondaryFn
            for down in (True, False):
                ev = Quartz.CGEventCreateKeyboardEvent(None, code, down)
                Quartz.CGEventSetFlags(ev, key_flags)
                _post(ev)
        for m in reversed(mods):
            ev = Quartz.CGEventCreateKeyboardEvent(None, MODIFIER_KEYCODES[m], False)
            Quartz.CGEventSetFlags(ev, 0)
            _post(ev)

    def close(self):
        if self._held:
            self.release(self._held)
