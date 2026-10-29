"""Backend interface: the only place that touches the real mouse and keyboard."""

from __future__ import annotations

from typing import List, Tuple

MODIFIERS = {"cmd", "ctrl", "alt", "shift", "win"}

ALIASES = {
    "command": "cmd", "meta": "cmd", "super": "win", "windows": "win",
    "control": "ctrl", "option": "alt", "opt": "alt",
    "return": "enter", "escape": "esc", "del": "delete",
    "pgup": "pageup", "pgdn": "pagedown", "plus": "=", "minus": "-",
}


def parse_hotkey(combo: str) -> List[str]:
    """'Cmd+Shift+Tab' -> ['cmd', 'shift', 'tab']. A trailing '+' means the plus key."""
    combo = combo.strip()
    parts = combo[:-1].split("+") + ["="] if combo.endswith("++") else combo.split("+")
    keys = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lowered = part.lower()
        keys.append(ALIASES.get(lowered, lowered))
    return keys


class Backend:
    """Coordinates are the OS global cursor space (points on macOS, pixels on Windows).

    Scroll deltas use "content" semantics: positive dy moves page content down
    (like dragging a page down on a touch screen), positive dx moves it right.
    """

    name = "base"

    def position(self) -> Tuple[float, float]:
        raise NotImplementedError

    def move_to(self, x: float, y: float) -> None:
        raise NotImplementedError

    def press(self, button: str = "left") -> None:
        raise NotImplementedError

    def release(self, button: str = "left") -> None:
        raise NotImplementedError

    def click(self, button: str = "left", count: int = 1) -> None:
        raise NotImplementedError

    def scroll(self, content_dx: float, content_dy: float) -> None:
        raise NotImplementedError

    def hotkey(self, combo: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DryRunBackend(Backend):
    """Records every call instead of acting. Used by --dry-run and the tests."""

    name = "dry-run"

    def __init__(self, start: Tuple[float, float] = (500.0, 500.0)):
        self.pos = tuple(start)
        self.calls: List[tuple] = []
        self.pressed: set = set()

    def position(self):
        return self.pos

    def move_to(self, x, y):
        self.pos = (x, y)
        self.calls.append(("move", round(x, 1), round(y, 1)))

    def press(self, button="left"):
        self.pressed.add(button)
        self.calls.append(("press", button))

    def release(self, button="left"):
        self.pressed.discard(button)
        self.calls.append(("release", button))

    def click(self, button="left", count=1):
        self.calls.append(("click", button, count))

    def scroll(self, content_dx, content_dy):
        self.calls.append(("scroll", content_dx, content_dy))

    def hotkey(self, combo):
        self.calls.append(("hotkey", combo))

    def of(self, kind: str) -> list:
        return [c for c in self.calls if c[0] == kind]
