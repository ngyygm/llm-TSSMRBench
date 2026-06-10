"""Helpers for loading YAML configs with environment-variable expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_env_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml_with_env(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a top-level YAML mapping")
    return _expand_env_value(payload)
