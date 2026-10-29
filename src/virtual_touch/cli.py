"""Command-line interface: `virtual-touch [run|doctor|calibrate|config|list-cameras]`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet MediaPipe / TFLite / OpenCV native logging (must be set before they load).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import yaml

from . import __version__
from .config import DEFAULT_CONFIG_PATH, load_config, user_config_path, write_user_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="config file (default: user config, see `config --path`)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set pointer.speed=1.5 (repeatable)")
    p.add_argument("--camera", type=int, help="camera index (see list-cameras)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="virtual-touch", description="Webcam hand-tracking trackpad.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start gesture control (default)")
    _common(run)
    run.add_argument("--dry-run", action="store_true", help="recognise gestures without moving the cursor")
    run.add_argument("--no-preview", action="store_true", help="run without the camera window")
    run.add_argument("--hand", choices=["right", "left"], help="dominant hand")
    run.add_argument("--absolute", action="store_true", help="absolute pointer mode instead of trackpad mode")

    cal = sub.add_parser("calibrate", help="measure your pinch and save thresholds")
    _common(cal)

    doc = sub.add_parser("doctor", help="show detected hardware, permissions and effective config")
    _common(doc)

    cfg = sub.add_parser("config", help="manage the user config file")
    cfg.add_argument("--init", action="store_true", help="create the user config from defaults")
    cfg.add_argument("--path", action="store_true", help="print the user config path")
    cfg.add_argument("--show", action="store_true", help="print the effective config")
    cfg.add_argument("--defaults", action="store_true", help="print the packaged defaults")

    sub.add_parser("list-cameras", help="list working camera indices")
    return parser


def _load(args):
    overrides = list(args.overrides)
    if getattr(args, "camera", None) is not None:
        overrides.append(f"camera.index={args.camera}")
    if getattr(args, "dry_run", False):
        overrides.append("app.dry_run=true")
    if getattr(args, "no_preview", False):
        overrides.append("app.preview=false")
    if getattr(args, "hand", None):
        overrides.append(f"hands.dominant={args.hand}")
    if getattr(args, "absolute", False):
        overrides.append("pointer.mode=absolute")
    return load_config(args.config, overrides)


def cmd_doctor(args) -> int:
    from . import system_info

    info = system_info.summary()
    print(f"OS:        {info['os']}")
    print(f"Python:    {info['python']}")
    print("Displays:")
    for d in info["displays"]:
        print(f"  - {d.name:<12} x={d.x:<6g} y={d.y:<6g} {d.width:g}x{d.height:g} "
              f"scale={d.scale:g}{'  (primary)' if d.primary else ''}")
    print("Cameras:")
    for c in info["cameras"] or []:
        print(f"  - index {c['index']}: {c['width']}x{c['height']} @ {c['fps']} fps")
    if not info["cameras"]:
        print("  none working (check camera permission for your terminal)")
    if info["camera_names"]:
        print(f"  names: {', '.join(info['camera_names'])}")
    if info["accessibility"] is not None:
        print(f"Accessibility permission: {'granted' if info['accessibility'] else 'MISSING'}")
    if info["natural_scrolling"] is not None:
        print(f"macOS natural scrolling: {info['natural_scrolling']}")
    path = args.config or user_config_path()
    print(f"Config file: {path} ({'exists' if path.exists() else 'not created, using defaults'})")
    return 0


def cmd_config(args) -> int:
    path = user_config_path()
    if args.init:
        if path.exists():
            print(f"already exists: {path}")
        else:
            print(f"created {write_user_config({}, path)}")
    elif args.show:
        print(yaml.safe_dump(load_config().to_plain(), sort_keys=False))
    elif args.defaults:
        print(DEFAULT_CONFIG_PATH.read_text())
    else:
        print(path)
    return 0


def cmd_list_cameras(_args) -> int:
    from .system_info import camera_names, probe_cameras

    cams = probe_cameras()
    for c in cams:
        print(f"{c['index']}: {c['width']}x{c['height']} @ {c['fps']} fps")
    names = camera_names()
    if names:
        print(f"(macOS cameras: {', '.join(names)}; index order usually matches)")
    return 0 if cams else 1


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    if args.command is None:  # bare `virtual-touch` behaves like `virtual-touch run`
        args = parser.parse_args(["run", *(argv if argv is not None else sys.argv[1:])])

    if command == "config":
        return cmd_config(args)
    if command == "list-cameras":
        return cmd_list_cameras(args)
    if command == "doctor":
        return cmd_doctor(args)

    from .system_info import enable_windows_dpi_awareness

    enable_windows_dpi_awareness()
    cfg = _load(args)
    if command == "calibrate":
        from .calibrate import calibrate

        return calibrate(cfg, args.config)
    from .app import run

    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
