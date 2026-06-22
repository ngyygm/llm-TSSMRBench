"""Helpers for loading YAML configs with environment-variable expansion."""

from __future__ import annotations

import os
from copy import deepcopy
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


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {key: deepcopy(val) for key, val in base.items()}
        for key, val in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], val)
            else:
                merged[key] = deepcopy(val)
        return merged
    return deepcopy(override)


def _resolve_profile_payload(payload: dict[str, Any], profile: str | None) -> dict[str, Any]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return payload

    base = payload.get("base") or {}
    if not isinstance(base, dict):
        raise ValueError("Config `base` section must be a mapping")

    profile_name = profile or payload.get("default_profile")
    if not profile_name:
        raise ValueError("Config defines `profiles` but no profile was requested and no `default_profile` is set")
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown config profile `{profile_name}`; available profiles: {available}")

    def resolve_profile(name: str, stack: tuple[str, ...] = ()) -> dict[str, Any]:
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise ValueError(f"Config profile inheritance cycle detected: {cycle}")
        raw = profiles.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"Config profile `{name}` must be a mapping")
        parent_name = raw.get("inherits")
        resolved_parent: dict[str, Any] = {}
        if parent_name:
            if not isinstance(parent_name, str):
                raise ValueError(f"Config profile `{name}` has non-string `inherits`")
            resolved_parent = resolve_profile(parent_name, (*stack, name))
        body = {key: val for key, val in raw.items() if key != "inherits"}
        return _deep_merge(resolved_parent, body)

    return _deep_merge(base, resolve_profile(str(profile_name)))


# Configs are resolved in two steps: first profile inheritance, then environment
# variable expansion, so checked-in YAML can stay generic and secret-free.
def load_yaml_with_env(path: Path, profile: str | None = None) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a top-level YAML mapping")
    resolved = _resolve_profile_payload(payload, profile)
    return _expand_env_value(resolved)
