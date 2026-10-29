"""Pick the right input-injection backend for this OS."""

from __future__ import annotations

import sys

from .base import Backend, DryRunBackend, parse_hotkey


def create_backend(cfg, dry_run=None) -> Backend:
    if cfg.app.dry_run if dry_run is None else dry_run:
        return DryRunBackend()
    if sys.platform == "darwin":
        from .macos import MacOSBackend

        return MacOSBackend()
    from .generic import PynputBackend

    return PynputBackend(pixels_per_notch=cfg.scroll.pixels_per_notch)


__all__ = ["Backend", "DryRunBackend", "create_backend", "parse_hotkey"]
