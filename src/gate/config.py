"""Configuration loading and validation for the gate.

Reads a YAML config file (if any) and applies environment overrides, then
validates the result. All failures raise :class:`ConfigError` with a clear
message so startup fails fast.

Multi-backend (breaking change, docs/plans/multi-backend.md decision 6):
the ``backends:`` list (file) / ``BACKENDS_JSON`` (env, wins over the file
entirely, mirroring ``THRESHOLDS_JSON``) configures the vLLM backends and is
**required** — at least one backend (name + host + port) must always be
configured, because the gate has no other way to learn its upstream. Each
backend may override the tiered ``thresholds`` and the four autoconfig
knobs; ``None`` means "use the global default". The legacy
``vllm_host``/``vllm_port`` keys and the ``VLLM_HOST``/``VLLM_PORT`` env
vars are **removed**: a config containing them fails with an unknown-key
:class:`ConfigError` naming the migration to ``backends:``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, TypeGuard

import yaml

DEFAULT_CONFIG_PATH = "/etc/gate/config.yaml"

# Known top-level config keys, mapped to the GateConfig field they populate.
# ``vllm_host``/``vllm_port`` are deliberately absent: they were removed in
# the multi-backend breaking change (decision 6) and now fail with the
# unknown-key error (see :func:`_merge_file` for the migration hint).
_KNOWN_KEYS: frozenset[str] = frozenset(
    {
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
        "backends",
        "model_refresh_interval_s",
    }
)

_THRESHOLD_KEYS: frozenset[str] = frozenset({"kv_pct", "max_context", "timeout_s"})

# Known keys inside one ``backends:`` entry.
_BACKEND_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "host",
        "port",
        "models",
        "default",
        "thresholds",
        "target_kv_cache_pct",
        "token_margin",
        "retry_min_s",
        "retry_max_s",
    }
)


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
class Backend:
    """One configured vLLM backend (multi-backend routing, design plan §2.1).

    ``models`` is an optional mnemonic + validation anchor; an empty tuple
    means the owned model set is auto-adopted from the backend's
    ``GET /v1/models``. The four autoconfig knobs and ``thresholds`` are
    ``None`` when the backend uses the global default value.
    """

    name: str
    host: str
    port: int
    models: tuple[str, ...] = ()
    default: bool = False
    thresholds: tuple[Threshold, ...] | None = None
    target_kv_cache_pct: float | None = None
    token_margin: float | None = None
    retry_min_s: int | None = None
    retry_max_s: int | None = None


@dataclass(frozen=True)
class GateConfig:
    """Fully validated gate configuration.

    ``backends`` is the REQUIRED upstream list (decision 6): the gate has no
    other way to learn its upstream, so an empty ``backends`` tuple is a
    validation error, not a valid zero-config state.
    """

    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    metrics_poll_interval_s: float = 2.0
    stale_after_s: float = 6.0
    model_refresh_interval_s: float = 30.0
    chars_per_token: int = 4
    default_max_tokens: int = 256
    thresholds: tuple[Threshold, ...] = ()
    target_kv_cache_pct: float = 85.0
    retry_min_s: int = 5
    retry_max_s: int = 60
    token_margin: float = 1.25
    backends: tuple[Backend, ...] = ()


def _is_int(value: Any) -> TypeGuard[int]:
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


def _parse_backend_entry(raw: Any, source: str, index: int) -> Backend:
    """Parse one ``backends:`` entry (mapping) into a Backend.

    ``None`` knob/threshold fields mean "use the global default". Type errors
    raise :class:`ConfigError`; range/uniqueness checks happen in
    :func:`_validate` (same split as the scalar keys).
    """
    where = f"{source}: backends[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: entry must be an object")
    unknown = set(raw) - _BACKEND_KEYS
    if unknown:
        raise ConfigError(f"{where}: unknown backend key(s): {sorted(unknown)}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: 'name' must be a non-empty string")
    host = raw.get("host")
    if not isinstance(host, str) or not host:
        raise ConfigError(f"{where}: 'host' must be a non-empty string")
    port = raw.get("port")
    if not _is_int(port):
        raise ConfigError(f"{where}: 'port' must be an integer")
    models: tuple[str, ...] = ()
    if "models" in raw:
        if not isinstance(raw["models"], list):
            raise ConfigError(f"{where}: 'models' must be a list of strings")
        seen_models: set[str] = set()
        for i, m in enumerate(raw["models"]):
            if not isinstance(m, str) or not m:
                raise ConfigError(f"{where}: models[{i}] must be a non-empty string")
            if m in seen_models:
                raise ConfigError(f"{where}: duplicate model id {m!r} in 'models'")
            seen_models.add(m)
        models = tuple(raw["models"])
    default = raw.get("default", False)
    if not isinstance(default, bool):
        raise ConfigError(f"{where}: 'default' must be a boolean")
    thresholds: tuple[Threshold, ...] | None = None
    if "thresholds" in raw:
        thresholds = tuple(_parse_thresholds(raw["thresholds"], where))
    target_kv_cache_pct: float | None = None
    if "target_kv_cache_pct" in raw:
        if not _is_number(raw["target_kv_cache_pct"]):
            raise ConfigError(f"{where}: 'target_kv_cache_pct' must be a number")
        target_kv_cache_pct = float(raw["target_kv_cache_pct"])
    token_margin: float | None = None
    if "token_margin" in raw:
        if not _is_number(raw["token_margin"]):
            raise ConfigError(f"{where}: 'token_margin' must be a number")
        token_margin = float(raw["token_margin"])
    retry_min_s: int | None = None
    if "retry_min_s" in raw:
        if not _is_int(raw["retry_min_s"]):
            raise ConfigError(f"{where}: 'retry_min_s' must be an integer")
        retry_min_s = raw["retry_min_s"]
    retry_max_s: int | None = None
    if "retry_max_s" in raw:
        if not _is_int(raw["retry_max_s"]):
            raise ConfigError(f"{where}: 'retry_max_s' must be an integer")
        retry_max_s = raw["retry_max_s"]
    return Backend(
        name=name,
        host=host,
        port=port,
        models=models,
        default=default,
        thresholds=thresholds,
        target_kv_cache_pct=target_kv_cache_pct,
        token_margin=token_margin,
        retry_min_s=retry_min_s,
        retry_max_s=retry_max_s,
    )


def _parse_backends(raw: Any, source: str) -> list[Backend]:
    """Parse a ``backends`` value (list of objects) into Backends."""
    if not isinstance(raw, list):
        raise ConfigError(f"{source}: 'backends' must be a list of objects")
    return [_parse_backend_entry(entry, source, i) for i, entry in enumerate(raw)]


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
    """Merge known keys from a parsed config mapping over the defaults.

    The legacy ``vllm_host``/``vllm_port`` keys are no longer known keys
    (decision 6); a config containing them fails with a migration hint
    instead of the generic unknown-key message.
    """
    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        legacy = sorted(unknown & {"vllm_host", "vllm_port"})
        if legacy:
            verb = "is" if len(legacy) == 1 else "are"
            raise ConfigError(
                f"{source}: {', '.join(legacy)} {verb} no longer supported; "
                f"use the 'backends' list (or the BACKENDS_JSON env var)"
            )
        raise ConfigError(f"{source}: unknown config key(s): {sorted(unknown)}")
    updates: dict[str, Any] = {}
    if "listen_host" in raw:
        if not isinstance(raw["listen_host"], str):
            raise ConfigError(f"{source}: 'listen_host' must be a string")
        updates["listen_host"] = raw["listen_host"]
    for key in ("listen_port", "chars_per_token", "default_max_tokens"):
        if key in raw:
            if not _is_int(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be an integer")
            updates[key] = raw[key]
    for key in ("metrics_poll_interval_s", "stale_after_s", "model_refresh_interval_s"):
        if key in raw:
            if not _is_number(raw[key]):
                raise ConfigError(f"{source}: '{key}' must be a number")
            updates[key] = float(raw[key])
    if "thresholds" in raw:
        updates["thresholds"] = tuple(_parse_thresholds(raw["thresholds"], source))
    if "backends" in raw:
        updates["backends"] = tuple(_parse_backends(raw["backends"], source))
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
    """Apply environment overrides (env wins over file and defaults).

    ``VLLM_HOST``/``VLLM_PORT`` are no longer read (decision 6): they were
    the single-backend shorthand, removed with ``vllm_host``/``vllm_port``.
    """
    updates: dict[str, Any] = {}
    listen_host = env.get("LISTEN_HOST")
    if listen_host is not None:
        updates["listen_host"] = listen_host
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
    backends_raw = env.get("BACKENDS_JSON")
    if backends_raw is not None:
        try:
            raw = json.loads(backends_raw)
        except json.JSONDecodeError as e:
            raise ConfigError(f"BACKENDS_JSON is not valid JSON: {e}") from e
        # Env wins over the file entirely: this REPLACES any file backends.
        updates["backends"] = tuple(_parse_backends(raw, "BACKENDS_JSON"))
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
    model_refresh_interval_s = _parse_env_float(env, "MODEL_REFRESH_INTERVAL_S")
    if model_refresh_interval_s is not None:
        updates["model_refresh_interval_s"] = model_refresh_interval_s
    return replace(data, **updates)


def _validate(data: GateConfig) -> None:
    """Validate the final config; raise ConfigError on any violation."""
    # thresholds are optional: an empty/absent tiered policy just means the
    # tiered layer always allows (backends, however, are required — decision 6).
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
    if not 1 <= data.listen_port <= 65535:
        raise ConfigError(f"listen_port must be in [1, 65535], got {data.listen_port}")
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
    if not math.isfinite(data.model_refresh_interval_s) or data.model_refresh_interval_s <= 0:
        raise ConfigError(
            f"model_refresh_interval_s must be a finite number > 0, "
            f"got {data.model_refresh_interval_s}"
        )
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
    _validate_backends(data)


def _validate_backends(data: GateConfig) -> None:
    """Validate the REQUIRED ``backends`` list.

    ``backends`` must be non-empty (decision 6): the gate has no other way
    to learn its upstream, so an empty list is a startup error. When
    present: names are unique and non-empty, ports are in 1-65535, model ids
    are globally unique across backends, at most one backend is
    ``default: true``, and each per-backend knob, when present, satisfies the
    same ranges as the global default.
    """
    if not data.backends:
        raise ConfigError(
            "at least one backend is required (file key 'backends' or the BACKENDS_JSON env var)"
        )
    seen_names: set[str] = set()
    seen_models: dict[str, str] = {}  # model id -> backend name
    defaults = 0
    for i, b in enumerate(data.backends):
        where = f"backends[{i}] ({b.name})"
        if not b.name:
            raise ConfigError(f"{where}: backend name must be a non-empty string")
        if b.name in seen_names:
            raise ConfigError(f"{where}: duplicate backend name {b.name!r}")
        seen_names.add(b.name)
        if not 1 <= b.port <= 65535:
            raise ConfigError(f"{where}: port must be in [1, 65535], got {b.port}")
        for m in b.models:
            if m in seen_models:
                raise ConfigError(
                    f"{where}: model id {m!r} is already owned by backend "
                    f"{seen_models[m]!r}; model ids must be unique across backends"
                )
            seen_models[m] = b.name
        if b.default:
            defaults += 1
        if b.thresholds is not None:
            seen_kv: set[float] = set()
            for j, t in enumerate(b.thresholds):
                twhere = f"{where}: thresholds[{j}]"
                if not 0 <= t.kv_pct <= 100:
                    raise ConfigError(f"{twhere}: kv_pct must be in [0, 100], got {t.kv_pct}")
                if t.max_context < 1:
                    raise ConfigError(f"{twhere}: max_context must be >= 1, got {t.max_context}")
                if t.timeout_s < 1:
                    raise ConfigError(f"{twhere}: timeout_s must be >= 1, got {t.timeout_s}")
                if t.kv_pct in seen_kv:
                    raise ConfigError(f"{twhere}: duplicate kv_pct {t.kv_pct}")
                seen_kv.add(t.kv_pct)
        if b.target_kv_cache_pct is not None and (
            not math.isfinite(b.target_kv_cache_pct) or not 0 < b.target_kv_cache_pct <= 100
        ):
            raise ConfigError(
                f"{where}: target_kv_cache_pct must be a finite number in (0, 100], "
                f"got {b.target_kv_cache_pct}"
            )
        if b.token_margin is not None and (
            not math.isfinite(b.token_margin) or b.token_margin < 1.0
        ):
            raise ConfigError(
                f"{where}: token_margin must be a finite number >= 1.0, got {b.token_margin}"
            )
        if b.retry_min_s is not None and (not _is_int(b.retry_min_s) or b.retry_min_s < 1):
            raise ConfigError(f"{where}: retry_min_s must be an integer >= 1, got {b.retry_min_s}")
        # The effective pair (per-backend override falling back to the global
        # default) must satisfy the same max >= min rule as the globals.
        effective_min = b.retry_min_s if b.retry_min_s is not None else data.retry_min_s
        effective_max = b.retry_max_s if b.retry_max_s is not None else data.retry_max_s
        if not _is_int(effective_max) or effective_max < effective_min:
            raise ConfigError(
                f"{where}: retry_max_s must be an integer >= retry_min_s "
                f"({effective_min}), got {effective_max}"
            )
    if defaults > 1:
        raise ConfigError(f"at most one backend may have 'default: true', got {defaults}")


def load_config(path: str | None = None, env: Mapping[str, str] | None = None) -> GateConfig:
    """Load and validate the gate configuration.

    Precedence (lowest to highest): defaults < config file < environment.

    - ``path``: explicit config file; must exist and be valid YAML.
    - ``path is None``: use ``DEFAULT_CONFIG_PATH`` if it exists, else defaults.
    - ``env``: overrides applied last; defaults to ``os.environ``.
      ``BACKENDS_JSON`` (a JSON list of backend objects) REPLACES the file's
      ``backends:`` entirely, mirroring ``THRESHOLDS_JSON``.

    The ``backends`` list is required (decision 6): a config with no
    ``backends`` (file or ``BACKENDS_JSON``) raises :class:`ConfigError`.

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
