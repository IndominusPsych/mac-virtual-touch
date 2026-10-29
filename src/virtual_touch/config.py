"""Configuration: packaged defaults <- user config.yaml <- CLI overrides."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from platformdirs import user_cache_dir, user_config_dir

APP_NAME = "virtual-touch"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


def user_config_path() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False)) / "config.yaml"


def cache_dir() -> Path:
    path = Path(user_cache_dir(APP_NAME, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def os_key() -> str:
    """Key used for per-OS sections such as `actions`."""
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("win"):
        return "win32"
    return "linux"


class Config(dict):
    """A dict with attribute access for nested sections (cfg.pointer.speed)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @classmethod
    def wrap(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return cls({k: cls.wrap(v) for k, v in data.items()})
        if isinstance(data, list):
            return [cls.wrap(v) for v in data]
        return data

    def to_plain(self) -> dict:
        def unwrap(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: unwrap(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unwrap(x) for x in v]
            return v

        return unwrap(self)


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse `section.key=value` (value is YAML, so numbers/bools/lists work)."""
    if "=" not in item:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    dotted, raw = item.split("=", 1)
    return dotted.strip().split("."), yaml.safe_load(raw)


def set_dotted(data: dict, keys: list[str], value: Any) -> None:
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: Optional[Path] = None, overrides: Iterable[str] = ()) -> Config:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text()) or {}
    user_path = path or user_config_path()
    if user_path.exists():
        data = deep_merge(data, yaml.safe_load(user_path.read_text()) or {})
    elif path is not None:
        raise FileNotFoundError(f"config file not found: {path}")
    for item in overrides:
        keys, value = parse_override(item)
        set_dotted(data, keys, value)
    return Config.wrap(data)


USER_CONFIG_HEADER = (
    "# Virtual Touch user settings. Only values that differ from the packaged defaults\n"
    "# belong here, so new defaults from upgrades still apply.\n"
    "# See all options with: virtual-touch config --defaults\n"
)


def write_user_config(updates: dict, path: Optional[Path] = None) -> Path:
    """Merge `updates` into the user config file.

    A new file holds only `updates` (not a copy of every default), so later
    default changes keep reaching the user. With no updates, the commented
    defaults are written as an editable template (`config --init`).
    """
    target = path or user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = yaml.safe_load(target.read_text()) or {}
        target.write_text(yaml.safe_dump(deep_merge(current, updates), sort_keys=False))
    elif updates:
        target.write_text(USER_CONFIG_HEADER + yaml.safe_dump(updates, sort_keys=False))
    else:
        target.write_text(DEFAULT_CONFIG_PATH.read_text())
    return target
