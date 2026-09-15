"""Configuration loading and validation for the gate.

Reads a YAML config file (if any) and applies environment overrides, then
validates the result. All failures raise :class:`ConfigError` with a clear
message so startup fails fast.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "/etc/gate/config.yaml"

# Known top-level config keys, mapped to the GateConfig field they populate.
_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "vllm_host",
        "vllm_port",
        "listen_host",
        "listen_port",
        "metrics_poll_interval_s",
        "stale_after_s",
        "chars_per_token",
        "default_max_tokens",
        "thresholds",
        "target_kv_cache_pct",
        "retry_min_s",
        "retry_max_s",
        "token_margin",
    }
)

_THRESHOLD_KEYS: frozenset[str] = frozenset({"kv_pct", "max_context", "timeout_s"})


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class Threshold:
    """One tier of the KV-cache policy.

    ``kv_pct`` is a percentage (0-100); the vLLM metric is a fraction (0-1),
    so the gate compares ``usage * 100 >= kv_pct``.
    """

    kv_pct: float
    max_context: int
    timeout_s: int


@dataclass(frozen=True)
class GateConfig:
    """Fully validated gate configuration."""

    vllm_host: str = "vllm"
    vllm_port: int = 8000
    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    metrics_poll_interval_s: float = 2.0
    stale_after_s: float = 6.0
    chars_per_token: int = 4
    default_max_tokens: int = 256
    thresholds: tuple[Threshold, ...] = ()
    target_kv_cache_pct: float = 85.0
    retry_min_s: int = 5
    retry_max_s: int = 60
    token_margin: float = 1.25


def _is_int(value: Any) -> bool:
    """True for real ints (bool is excluded on purpose)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """True for int or float (bool is excluded on purpose)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_thresholds(raw: Any, source: str) -> list[Threshold]:
    """Parse a ``thresholds`` value (list of objects) into Thresholds."""
    if not isinstance(raw, list):
        raise ConfigError(f"{source}: 'thresholds' must be a list of objects")
    parsed: list[Threshold] = []
    for i, entry in enumerate(raw):
        where = f"{source}: thresholds[{i}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: entry must be an object")
        keys = frozenset(entry)
        if keys != _THRESHOLD_KEYS:
            raise ConfigError(
                f"{where}: entry must have exactly the keys "
                f"{sorted(_THRESHOLD_KEYS)}, got {sorted(keys)}"
            )
        kv_pct = entry["kv_pct"]
        max_context = entry["max_context"]
        timeout_s = entry["timeout_s"]
        if not _is_number(kv_pct):
            raise ConfigError(f"{where}: 'kv_pct' must be a number")
        if not _is_int(max_context):
            raise ConfigError(f"{where}: 'max_context' must be an integer")
        if not _is_int(timeout_s):
            raise ConfigError(f"{where}: 'timeout_s' must be an integer")
        parsed.append(Threshold(kv_pct=float(kv_pct), max_context=max_context, timeout_s=timeout_s))
    return parsed


def _load_file(path: str) -> dict[str, Any]:
    """Read and parse a YAML config file; raise ConfigError on any problem."""
    if not os.path.isfile(path):
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"config file {path} is not valid YAML: {e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"config file {path} is not valid UTF-8 text: {e}") from e
    except OSError as e:
        raise ConfigError(f"could not read config file {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping")
    return raw


def _merge_file(data: GateConfig, raw: dict[str, Any], source: str) -> GateConfig:
    """Merge known keys from a parsed config mapping over the defaults."""
    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise ConfigError(f"{source}: unknown config key(s): {sorted(unknown)}")
    updates: dict[str, Any] = {}
    if "vllm_host" in raw:
        if not isinstance(raw["vllm_host"], str):
            raise ConfigError(f"{source}: 'vllm_host' must be a string")
        updates["vllm_host"] = raw["vllm_host"]
    if "listen_host" in raw:
        if not isinstance(raw["listen_host"], str):
            raise ConfigError(f"{source}: 'listen_host' must be a string")
        updates["listen_host"] = raw["listen_host"]
    for key in ("vllm_port", "listen_port", "chars_per_token", "default_max_tokens"):
        if key in raw:
            if not _is_int(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be an integer")
            updates[key] = raw[key]
    for key in ("metrics_poll_interval_s", "stale_after_s"):
        if key in raw:
            if not _is_number(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be a number")
            updates[key] = float(raw[key])
    if "thresholds" in raw:
        updates["thresholds"] = tuple(_parse_thresholds(raw["thresholds"], source))
    for key in ("retry_min_s", "retry_max_s"):
        if key in raw:
            if not _is_int(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be an integer")
            updates[key] = raw[key]
    for key in ("target_kv_cache_pct", "token_margin"):
        if key in raw:
            if not _is_number(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be a number")
            updates[key] = float(raw[key])
    return replace(data, **updates)


def _parse_env_int(env: Mapping[str, str], name: str) -> int | None:
    """Parse an env var as an int; None if unset, ConfigError if malformed."""
    value = env.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ConfigError(f"environment variable {name}={value!r} is not a valid integer") from None


def _parse_env_float(env: Mapping[str, str], name: str) -> float | None:
    """Parse an env var as a float; None if unset, ConfigError if malformed."""
    value = env.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"environment variable {name}={value!r} is not a valid number") from None


def _apply_env(data: GateConfig, env: Mapping[str, str]) -> GateConfig:
    """Apply environment overrides (env wins over file and defaults)."""
    updates: dict[str, Any] = {}
    host = env.get("VLLM_HOST")
    if host is not None:
        updates["vllm_host"] = host
    listen_host = env.get("LISTEN_HOST")
    if listen_host is not None:
        updates["listen_host"] = listen_host
    vllm_port = _parse_env_int(env, "VLLM_PORT")
    if vllm_port is not None:
        updates["vllm_port"] = vllm_port
    listen_port = _parse_env_int(env, "LISTEN_PORT")
    if listen_port is not None:
        updates["listen_port"] = listen_port
    thresholds_raw = env.get("THRESHOLDS_JSON")
    if thresholds_raw is not None:
        try:
            raw = json.loads(thresholds_raw)
        except json.JSONDecodeError as e:
            raise ConfigError(f"THRESHOLDS_JSON is not valid JSON: {e}") from e
        updates["thresholds"] = tuple(_parse_thresholds(raw, "THRESHOLDS_JSON"))
    retry_min_s = _parse_env_int(env, "RETRY_MIN_S")
    if retry_min_s is not None:
        updates["retry_min_s"] = retry_min_s
    retry_max_s = _parse_env_int(env, "RETRY_MAX_S")
    if retry_max_s is not None:
        updates["retry_max_s"] = retry_max_s
    target_kv_cache_pct = _parse_env_float(env, "TARGET_KV_CACHE_PCT")
    if target_kv_cache_pct is not None:
        updates["target_kv_cache_pct"] = target_kv_cache_pct
    token_margin = _parse_env_float(env, "TOKEN_MARGIN")
    if token_margin is not None:
        updates["token_margin"] = token_margin
    return replace(data, **updates)


def _validate(data: GateConfig) -> None:
    """Validate the final config; raise ConfigError on any violation."""
    # thresholds are optional: an empty/absent tiered policy just means the
    # tiered layer always allows (zero-config is valid).
    seen_kv: set[float] = set()
    for i, t in enumerate(data.thresholds):
        where = f"thresholds[{i}]"
        if not 0 <= t.kv_pct <= 100:
            raise ConfigError(f"{where}: kv_pct must be in [0, 100], got {t.kv_pct}")
        if t.max_context < 1:
            raise ConfigError(f"{where}: max_context must be >= 1, got {t.max_context}")
        if t.timeout_s < 1:
            raise ConfigError(f"{where}: timeout_s must be >= 1, got {t.timeout_s}")
        if t.kv_pct in seen_kv:
            raise ConfigError(f"{where}: duplicate kv_pct {t.kv_pct}")
        seen_kv.add(t.kv_pct)
    for name, port in (("vllm_port", data.vllm_port), ("listen_port", data.listen_port)):
        if not 1 <= port <= 65535:
            raise ConfigError(f"{name} must be in [1, 65535], got {port}")
    if data.chars_per_token < 1:
        raise ConfigError(f"chars_per_token must be >= 1, got {data.chars_per_token}")
    if data.default_max_tokens < 0:
        raise ConfigError(f"default_max_tokens must be >= 0, got {data.default_max_tokens}")
    # NaN slips past the <= 0 checks (NaN <= 0 is False), so require finiteness.
    if not math.isfinite(data.metrics_poll_interval_s) or data.metrics_poll_interval_s <= 0:
        raise ConfigError(
            f"metrics_poll_interval_s must be a finite number > 0, "
            f"got {data.metrics_poll_interval_s}"
        )
    if not math.isfinite(data.stale_after_s) or data.stale_after_s <= 0:
        raise ConfigError(f"stale_after_s must be a finite number > 0, got {data.stale_after_s}")
    if not math.isfinite(data.target_kv_cache_pct) or not 0 < data.target_kv_cache_pct <= 100:
        raise ConfigError(
            f"target_kv_cache_pct must be a finite number in (0, 100], "
            f"got {data.target_kv_cache_pct}"
        )
    if not _is_int(data.retry_min_s) or data.retry_min_s < 1:
        raise ConfigError(f"retry_min_s must be an integer >= 1, got {data.retry_min_s}")
    if not _is_int(data.retry_max_s) or data.retry_max_s < data.retry_min_s:
        raise ConfigError(
            f"retry_max_s must be an integer >= retry_min_s ({data.retry_min_s}), "
            f"got {data.retry_max_s}"
        )
    if not math.isfinite(data.token_margin) or data.token_margin < 1.0:
        raise ConfigError(f"token_margin must be a finite number >= 1.0, got {data.token_margin}")


def load_config(path: str | None = None, env: Mapping[str, str] | None = None) -> GateConfig:
    """Load and validate the gate configuration.

    Precedence (lowest to highest): defaults < config file < environment.

    - ``path``: explicit config file; must exist and be valid YAML.
    - ``path is None``: use ``DEFAULT_CONFIG_PATH`` if it exists, else defaults.
    - ``env``: overrides applied last; defaults to ``os.environ``.

    Raises :class:`ConfigError` on any invalid input or validation failure.
    """
    if env is None:
        env = os.environ

    data = GateConfig()

    if path is not None:
        data = _merge_file(data, _load_file(path), path)
    elif os.path.isfile(DEFAULT_CONFIG_PATH):
        data = _merge_file(data, _load_file(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH)

    data = _apply_env(data, env)
    _validate(data)
    return data
