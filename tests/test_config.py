"""Tests for gate.config: file loading, env overrides, validation."""

from __future__ import annotations

import json

import pytest
import yaml

from gate import config
from gate.config import Backend, ConfigError, GateConfig, RoutingEntry, Threshold, load_config
from gate.routing import RoutingSpec


def _write(tmp_path, data: object, name: str = "config.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


# --- happy paths -----------------------------------------------------------


def test_load_from_valid_file(tmp_config_file):
    cfg = load_config(path=tmp_config_file, env={})
    assert cfg.listen_host == "127.0.0.1"
    assert cfg.listen_port == 8080
    assert cfg.metrics_poll_interval_s == 1.5
    assert cfg.stale_after_s == 4.0
    assert cfg.chars_per_token == 3
    assert cfg.default_max_tokens == 128
    assert cfg.thresholds == (
        Threshold(kv_pct=50.0, max_context=4096, timeout_s=15),
        Threshold(kv_pct=80.0, max_context=1024, timeout_s=30),
    )


def test_defaults_fill_missing_file_keys(tmp_path):
    """Only some keys in the file; the rest come from defaults."""
    path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 1}],
            "thresholds": [{"kv_pct": 10, "max_context": 1, "timeout_s": 1}],
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.listen_port == 8000
    assert cfg.chars_per_token == 4
    assert cfg.default_max_tokens == 256


def test_env_overrides_win_over_file(tmp_config_file):
    env = {
        "LISTEN_HOST": "0.0.0.0",
        "LISTEN_PORT": "7000",
        "THRESHOLDS_JSON": json.dumps([{"kv_pct": 90, "max_context": 512, "timeout_s": 5}]),
    }
    cfg = load_config(path=tmp_config_file, env=env)
    assert cfg.listen_host == "0.0.0.0"
    assert cfg.listen_port == 7000
    assert cfg.thresholds == (Threshold(kv_pct=90.0, max_context=512, timeout_s=5),)


def test_vllm_host_port_env_vars_no_longer_honored(tmp_config_file, tmp_path, monkeypatch):
    """VLLM_HOST/VLLM_PORT are removed (decision 6): they no longer affect the
    config, and a config with only those env vars (no backends) fails."""
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", str(tmp_path / "does-not-exist.yaml"))
    env = {"VLLM_HOST": "other-host", "VLLM_PORT": "9999"}
    # The sample file's backends are unaffected by the dead env vars.
    cfg = load_config(path=tmp_config_file, env=env)
    assert cfg.backends == (Backend(name="vllm", host="vllm", port=9000, default=True),)
    # No file and no backends: the dead env vars cannot supply an upstream.
    with pytest.raises(ConfigError, match="at least one backend is required"):
        load_config(path=None, env=env)


def test_no_file_no_env_requires_backends(tmp_path, monkeypatch):
    """No file and no env: the gate has no upstream to learn, so it fails
    (decision 6 supersedes the old zero-config property)."""
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", str(tmp_path / "does-not-exist.yaml"))
    with pytest.raises(ConfigError, match="at least one backend is required"):
        load_config(path=None, env={})


def test_minimal_valid_config_is_backends_only(tmp_path):
    """The minimal valid config is a single backends entry; everything else
    comes from defaults."""
    path = _write(tmp_path, {"backends": [{"name": "b1", "host": "h1", "port": 8000}]})
    cfg = load_config(path=path, env={})
    assert cfg.backends == (Backend(name="b1", host="h1", port=8000),)
    assert cfg.thresholds == ()
    assert cfg.target_kv_cache_pct == 85.0
    assert cfg.retry_min_s == 5
    assert cfg.retry_max_s == 60
    assert cfg.token_margin == 1.25


def test_no_thresholds_file_is_valid(tmp_path):
    """A file without a thresholds key loads fine (thresholds are optional)."""
    path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.thresholds == ()
    assert cfg.target_kv_cache_pct == 85.0
    assert cfg.retry_min_s == 5
    assert cfg.retry_max_s == 60
    assert cfg.token_margin == 1.25


def test_default_config_path_used_when_present(tmp_path, monkeypatch):
    """path=None falls back to DEFAULT_CONFIG_PATH when that file exists."""
    default_path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 5, "max_context": 2, "timeout_s": 2}],
        },
    )
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", default_path)
    cfg = load_config(path=None, env={})
    assert cfg.thresholds == (Threshold(kv_pct=5.0, max_context=2, timeout_s=2),)


def test_valid_multi_threshold_config_is_tuple():
    """A valid multi-threshold config loads; thresholds is a tuple of Threshold."""
    raw = [
        {"kv_pct": 0, "max_context": 1, "timeout_s": 1},
        {"kv_pct": 100, "max_context": 2, "timeout_s": 2},
    ]
    env = {
        "THRESHOLDS_JSON": json.dumps(raw),
        "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
    }
    cfg = load_config(path=None, env=env)
    assert isinstance(cfg, GateConfig)
    assert isinstance(cfg.thresholds, tuple)
    assert all(isinstance(t, Threshold) for t in cfg.thresholds)
    assert len(cfg.thresholds) == 2


# --- file errors ------------------------------------------------------------


def test_bad_file_path_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=str(tmp_path / "missing.yaml"), env={})


def test_invalid_yaml_file_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("key: [unterminated", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path=str(p), env={})


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
            "bogus_key": True,
        },
    )
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(path=path, env={})


def test_legacy_vllm_host_key_fails_with_migration_hint(tmp_path):
    """The removed vllm_host key fails with a helpful migration message
    (decision 6), not the generic unknown-key error."""
    path = _write(
        tmp_path,
        {
            "vllm_host": "my-vllm",
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
        },
    )
    with pytest.raises(ConfigError, match=r"vllm_host is no longer supported"):
        load_config(path=path, env={})


def test_legacy_vllm_port_key_fails_with_migration_hint(tmp_path):
    path = _write(
        tmp_path,
        {
            "vllm_port": 9999,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
        },
    )
    with pytest.raises(ConfigError, match=r"vllm_port is no longer supported"):
        load_config(path=path, env={})


def test_legacy_vllm_host_without_backends_fails(tmp_path):
    """A legacy single-backend config (vllm_host/vllm_port, no backends) now
    fails: the migration hint fires first, and even after migrating, no
    backends means 'at least one backend is required'."""
    path = _write(tmp_path, {"vllm_host": "my-vllm", "vllm_port": 9000})
    with pytest.raises(ConfigError, match=r"vllm_host, vllm_port are no longer supported"):
        load_config(path=path, env={})


def test_empty_backends_list_raises(tmp_path):
    """An explicit empty backends list is not a valid zero-config state."""
    path = _write(tmp_path, {"backends": []})
    with pytest.raises(ConfigError, match="at least one backend is required"):
        load_config(path=path, env={})


def test_file_must_be_yaml_mapping(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n- 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(path=str(p), env={})


# --- threshold validation ----------------------------------------------------


def _env_with_thresholds(thresholds: list[dict]) -> dict[str, str]:
    """Thresholds plus a minimal backend (backends are required, decision 6)."""
    return {
        "THRESHOLDS_JSON": json.dumps(thresholds),
        "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
    }


def test_kv_pct_above_100_raises():
    env = _env_with_thresholds([{"kv_pct": 150, "max_context": 1, "timeout_s": 1}])
    with pytest.raises(ConfigError, match="kv_pct"):
        load_config(path=None, env=env)


def test_kv_pct_negative_raises():
    env = _env_with_thresholds([{"kv_pct": -1, "max_context": 1, "timeout_s": 1}])
    with pytest.raises(ConfigError, match="kv_pct"):
        load_config(path=None, env=env)


def test_max_context_zero_raises():
    env = _env_with_thresholds([{"kv_pct": 50, "max_context": 0, "timeout_s": 1}])
    with pytest.raises(ConfigError, match="max_context"):
        load_config(path=None, env=env)


def test_timeout_s_zero_raises():
    env = _env_with_thresholds([{"kv_pct": 50, "max_context": 1, "timeout_s": 0}])
    with pytest.raises(ConfigError, match="timeout_s"):
        load_config(path=None, env=env)


def test_duplicate_kv_pct_raises():
    env = _env_with_thresholds(
        [
            {"kv_pct": 50, "max_context": 1, "timeout_s": 1},
            {"kv_pct": 50, "max_context": 2, "timeout_s": 2},
        ]
    )
    with pytest.raises(ConfigError, match="duplicate kv_pct"):
        load_config(path=None, env=env)


def test_threshold_entry_wrong_shape_raises():
    env = _env_with_thresholds([{"kv_pct": 50, "max_context": 1}])  # missing timeout_s
    with pytest.raises(ConfigError, match="exactly the keys"):
        load_config(path=None, env=env)


def test_threshold_entry_extra_key_raises():
    env = _env_with_thresholds([{"kv_pct": 50, "max_context": 1, "timeout_s": 1, "extra": 3}])
    with pytest.raises(ConfigError, match="exactly the keys"):
        load_config(path=None, env=env)


def test_threshold_entry_not_object_raises():
    with pytest.raises(ConfigError, match="entry must be an object"):
        load_config(path=None, env={"THRESHOLDS_JSON": "[42]"})


# --- env errors ---------------------------------------------------------------


def test_bad_int_env_raises():
    with pytest.raises(ConfigError, match="RETRY_MAX_S"):
        load_config(path=None, env={"RETRY_MAX_S": "abc", "THRESHOLDS_JSON": "[]"})


def test_bad_listen_port_env_raises():
    with pytest.raises(ConfigError, match="LISTEN_PORT"):
        load_config(path=None, env={"LISTEN_PORT": "not-a-port"})


def test_bad_thresholds_json_not_list_raises():
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(path=None, env={"THRESHOLDS_JSON": '{"kv_pct": 50}'})


def test_bad_thresholds_json_invalid_json_raises():
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path=None, env={"THRESHOLDS_JSON": "{oops"})


def test_empty_thresholds_list_is_valid():
    """An empty thresholds list is valid: the tiered layer simply always allows."""
    cfg = load_config(
        path=None,
        env={
            "THRESHOLDS_JSON": "[]",
            "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
        },
    )
    assert cfg.thresholds == ()


# --- scalar validation ----------------------------------------------------------


def test_listen_port_out_of_range_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "listen_port": 70000,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="listen_port"):
        load_config(path=path, env={})


def test_chars_per_token_zero_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "chars_per_token": 0,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="chars_per_token"):
        load_config(path=path, env={})


def test_negative_default_max_tokens_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "default_max_tokens": -1,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="default_max_tokens"):
        load_config(path=path, env={})


def test_zero_poll_interval_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "metrics_poll_interval_s": 0,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="metrics_poll_interval_s"):
        load_config(path=path, env={})


def test_zero_stale_after_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "stale_after_s": 0,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="stale_after_s"):
        load_config(path=path, env={})


# --- boundary conditions ---------------------------------------------------------


def test_boundary_values_accepted():
    """kv_pct exactly 0 and 100, max_context exactly 1, ports 1 and 65535."""
    env = {
        "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 1}]),
        "LISTEN_PORT": "65535",
        "THRESHOLDS_JSON": json.dumps(
            [
                {"kv_pct": 0, "max_context": 1, "timeout_s": 1},
                {"kv_pct": 100, "max_context": 1, "timeout_s": 1},
            ]
        ),
    }
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].port == 1
    assert cfg.listen_port == 65535
    assert cfg.thresholds[0].kv_pct == 0.0
    assert cfg.thresholds[1].kv_pct == 100.0


def test_kv_pct_accepts_float():
    env = _env_with_thresholds([{"kv_pct": 50.5, "max_context": 1, "timeout_s": 1}])
    cfg = load_config(path=None, env=env)
    assert cfg.thresholds[0].kv_pct == 50.5


def test_nan_poll_interval_raises(tmp_path):
    """NaN is representable in YAML (.nan) but must be rejected by > 0."""
    p = tmp_path / "nan.yaml"
    p.write_text(
        "backends:\n"
        "  - name: b1\n"
        "    host: h1\n"
        "    port: 8000\n"
        "metrics_poll_interval_s: .nan\n"
        "thresholds:\n"
        "  - kv_pct: 50\n"
        "    max_context: 1\n"
        "    timeout_s: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="metrics_poll_interval_s"):
        load_config(path=str(p), env={})


def test_non_utf8_file_raises(tmp_path):
    p = tmp_path / "binary.yaml"
    p.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(path=str(p), env={})


def test_wrong_type_values_raise(tmp_path):
    """Type mismatches in the file (string port, bool int) are rejected."""
    path = _write(
        tmp_path,
        {
            "listen_port": "8000",
            "chars_per_token": True,
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="listen_port"):
        load_config(path=path, env={})


# --- autoconfig knobs ---------------------------------------------------------


def test_knob_defaults():
    cfg = load_config(
        path=None,
        env={"BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}])},
    )
    assert cfg.target_kv_cache_pct == 85.0
    assert cfg.retry_min_s == 5
    assert cfg.retry_max_s == 60
    assert cfg.token_margin == 1.25


def test_knobs_from_file(tmp_path):
    path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "target_kv_cache_pct": 70.5,
            "retry_min_s": 10,
            "retry_max_s": 120,
            "token_margin": 2.0,
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.target_kv_cache_pct == 70.5
    assert cfg.retry_min_s == 10
    assert cfg.retry_max_s == 120
    assert cfg.token_margin == 2.0


def test_knobs_env_override_wins_over_file(tmp_config_file):
    env = {
        "TARGET_KV_CACHE_PCT": "90.5",
        "RETRY_MIN_S": "7",
        "RETRY_MAX_S": "45",
        "TOKEN_MARGIN": "1.5",
    }
    cfg = load_config(path=tmp_config_file, env=env)
    assert cfg.target_kv_cache_pct == 90.5
    assert cfg.retry_min_s == 7
    assert cfg.retry_max_s == 45
    assert cfg.token_margin == 1.5


def test_target_kv_cache_pct_zero_raises():
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(
            path=None,
            env={
                "TARGET_KV_CACHE_PCT": "0",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_target_kv_cache_pct_negative_raises(tmp_path):
    path = _write(
        tmp_path,
        {"backends": [{"name": "b1", "host": "h1", "port": 8000}], "target_kv_cache_pct": -5},
    )
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=path, env={})


def test_target_kv_cache_pct_above_100_raises():
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(
            path=None,
            env={
                "TARGET_KV_CACHE_PCT": "101",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_target_kv_cache_pct_at_100_is_valid():
    cfg = load_config(
        path=None,
        env={
            "TARGET_KV_CACHE_PCT": "100",
            "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
        },
    )
    assert cfg.target_kv_cache_pct == 100.0


def test_target_kv_cache_pct_nan_raises(tmp_path):
    p = tmp_path / "nan.yaml"
    p.write_text(
        "backends:\n  - name: b1\n    host: h1\n    port: 8000\ntarget_kv_cache_pct: .nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=str(p), env={})


def test_target_kv_cache_pct_inf_raises(tmp_path):
    p = tmp_path / "inf.yaml"
    p.write_text(
        "backends:\n  - name: b1\n    host: h1\n    port: 8000\ntarget_kv_cache_pct: .inf\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=str(p), env={})


def test_token_margin_below_one_raises():
    with pytest.raises(ConfigError, match="token_margin"):
        load_config(
            path=None,
            env={
                "TOKEN_MARGIN": "0.9",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_token_margin_at_one_is_valid():
    cfg = load_config(
        path=None,
        env={
            "TOKEN_MARGIN": "1.0",
            "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
        },
    )
    assert cfg.token_margin == 1.0


def test_token_margin_nan_raises(tmp_path):
    p = tmp_path / "nan.yaml"
    p.write_text(
        "backends:\n  - name: b1\n    host: h1\n    port: 8000\ntoken_margin: .nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="token_margin"):
        load_config(path=str(p), env={})


def test_retry_min_s_zero_raises():
    with pytest.raises(ConfigError, match="retry_min_s"):
        load_config(
            path=None,
            env={
                "RETRY_MIN_S": "0",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_retry_min_s_negative_raises(tmp_path):
    path = _write(tmp_path, {"retry_min_s": -3})
    with pytest.raises(ConfigError, match="retry_min_s"):
        load_config(path=path, env={})


def test_retry_max_s_below_min_raises():
    with pytest.raises(ConfigError, match="retry_max_s"):
        load_config(
            path=None,
            env={
                "RETRY_MIN_S": "10",
                "RETRY_MAX_S": "5",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_retry_max_s_equal_to_min_is_valid():
    cfg = load_config(
        path=None,
        env={
            "RETRY_MIN_S": "10",
            "RETRY_MAX_S": "10",
            "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
        },
    )
    assert cfg.retry_min_s == 10
    assert cfg.retry_max_s == 10


def test_retry_min_s_wrong_file_type_raises(tmp_path):
    path = _write(
        tmp_path, {"backends": [{"name": "b1", "host": "h1", "port": 8000}], "retry_min_s": "five"}
    )
    with pytest.raises(ConfigError, match="retry_min_s"):
        load_config(path=path, env={})


def test_retry_max_s_bool_file_type_raises(tmp_path):
    path = _write(
        tmp_path, {"backends": [{"name": "b1", "host": "h1", "port": 8000}], "retry_max_s": True}
    )
    with pytest.raises(ConfigError, match="retry_max_s"):
        load_config(path=path, env={})


def test_target_kv_cache_pct_wrong_file_type_raises(tmp_path):
    path = _write(
        tmp_path,
        {"backends": [{"name": "b1", "host": "h1", "port": 8000}], "target_kv_cache_pct": "85"},
    )
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=path, env={})


def test_token_margin_wrong_file_type_raises(tmp_path):
    path = _write(
        tmp_path, {"backends": [{"name": "b1", "host": "h1", "port": 8000}], "token_margin": True}
    )
    with pytest.raises(ConfigError, match="token_margin"):
        load_config(path=path, env={})


def test_bad_float_env_raises():
    with pytest.raises(ConfigError, match="TARGET_KV_CACHE_PCT"):
        load_config(
            path=None,
            env={
                "TARGET_KV_CACHE_PCT": "not-a-number",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_bad_float_env_token_margin_raises():
    with pytest.raises(ConfigError, match="TOKEN_MARGIN"):
        load_config(
            path=None,
            env={
                "TOKEN_MARGIN": "abc",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_bad_int_env_retry_raises():
    with pytest.raises(ConfigError, match="RETRY_MIN_S"):
        load_config(
            path=None,
            env={
                "RETRY_MIN_S": "abc",
                "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
            },
        )


def test_thresholds_still_validated_when_present():
    """Thresholds present alongside the knobs are parsed and validated as before."""
    env = {
        "THRESHOLDS_JSON": json.dumps([{"kv_pct": 50, "max_context": 4096, "timeout_s": 15}]),
        "TARGET_KV_CACHE_PCT": "60",
        "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
    }
    cfg = load_config(path=None, env=env)
    assert cfg.thresholds == (Threshold(kv_pct=50.0, max_context=4096, timeout_s=15),)
    assert cfg.target_kv_cache_pct == 60.0
    with pytest.raises(ConfigError, match="kv_pct"):
        load_config(
            path=None,
            env=_env_with_thresholds([{"kv_pct": 150, "max_context": 1, "timeout_s": 1}]),
        )


# --- backends (multi-backend, additive) ---------------------------------------


def _backend_env(backends: list[dict]) -> dict[str, str]:
    return {"BACKENDS_JSON": json.dumps(backends)}


def test_backends_from_yaml(tmp_path):
    path = _write(
        tmp_path,
        {
            "backends": [
                {
                    "name": "qwen",
                    "host": "vllm-qwen",
                    "port": 8000,
                    "models": ["qwen3-32b"],
                    "default": True,
                    "target_kv_cache_pct": 80.0,
                },
                {"name": "llama", "host": "vllm-llama", "port": 8001},
            ]
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.backends == (
        Backend(
            name="qwen",
            host="vllm-qwen",
            port=8000,
            models=("qwen3-32b",),
            default=True,
            target_kv_cache_pct=80.0,
        ),
        Backend(name="llama", host="vllm-llama", port=8001),
    )


def test_backends_required_when_absent():
    """backends is required (decision 6): no file, no BACKENDS_JSON -> error."""
    with pytest.raises(ConfigError, match="at least one backend is required"):
        load_config(path=None, env={})


def test_backends_json_env_wins_over_file(tmp_path):
    path = _write(
        tmp_path,
        {"backends": [{"name": "from-file", "host": "file-host", "port": 1111, "default": True}]},
    )
    env = _backend_env([{"name": "from-env", "host": "env-host", "port": 2222}])
    cfg = load_config(path=path, env=env)
    assert cfg.backends == (Backend(name="from-env", host="env-host", port=2222),)


def test_backends_per_backend_overrides_parsed():
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "models": ["m1"],
                "default": True,
                "thresholds": [{"kv_pct": 50, "max_context": 1024, "timeout_s": 10}],
                "target_kv_cache_pct": 70.0,
                "token_margin": 1.5,
                "retry_min_s": 10,
                "retry_max_s": 120,
            }
        ]
    )
    cfg = load_config(path=None, env=env)
    b = cfg.backends[0]
    assert b.thresholds == (Threshold(kv_pct=50.0, max_context=1024, timeout_s=10),)
    assert b.target_kv_cache_pct == 70.0
    assert b.token_margin == 1.5
    assert b.retry_min_s == 10
    assert b.retry_max_s == 120


def test_backends_omitted_knobs_are_none():
    """Per-backend knobs/thresholds omitted parse as None (use global default)."""
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1}])
    cfg = load_config(path=None, env=env)
    b = cfg.backends[0]
    assert b.thresholds is None
    assert b.target_kv_cache_pct is None
    assert b.token_margin is None
    assert b.retry_min_s is None
    assert b.retry_max_s is None
    assert b.models == ()
    assert b.default is False


def test_backends_from_file_parse(tmp_config_file):
    """The sample file's single backends entry parses (decision 6)."""
    cfg = load_config(path=tmp_config_file, env={})
    assert cfg.backends == (Backend(name="vllm", host="vllm", port=9000, default=True),)
    assert cfg.model_refresh_interval_s == 30.0


def test_model_refresh_interval_s_from_file_and_env(tmp_path):
    path = _write(
        tmp_path,
        {
            "backends": [{"name": "b1", "host": "h1", "port": 8000}],
            "model_refresh_interval_s": 12.5,
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.model_refresh_interval_s == 12.5
    cfg = load_config(
        path=None,
        env={
            "MODEL_REFRESH_INTERVAL_S": "45.5",
            "BACKENDS_JSON": json.dumps([{"name": "b1", "host": "h1", "port": 8000}]),
        },
    )
    assert cfg.model_refresh_interval_s == 45.5


def test_model_refresh_interval_s_zero_raises(tmp_path):
    path = _write(tmp_path, {"model_refresh_interval_s": 0})
    with pytest.raises(ConfigError, match="model_refresh_interval_s"):
        load_config(path=path, env={})


def test_model_refresh_interval_s_nan_raises(tmp_path):
    p = tmp_path / "nan.yaml"
    p.write_text("model_refresh_interval_s: .nan\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="model_refresh_interval_s"):
        load_config(path=str(p), env={})


def test_model_refresh_interval_s_wrong_type_raises(tmp_path):
    path = _write(tmp_path, {"model_refresh_interval_s": "30"})
    with pytest.raises(ConfigError, match="model_refresh_interval_s"):
        load_config(path=path, env={})


def test_backends_json_malformed_json_raises():
    with pytest.raises(ConfigError, match="BACKENDS_JSON is not valid JSON"):
        load_config(path=None, env={"BACKENDS_JSON": "{oops"})


def test_backends_json_not_a_list_raises():
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(path=None, env={"BACKENDS_JSON": '{"name": "x"}'})


def test_backends_entry_not_an_object_raises():
    with pytest.raises(ConfigError, match="entry must be an object"):
        load_config(path=None, env={"BACKENDS_JSON": "[42]"})


def test_backends_unknown_key_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "bogus": True}])
    with pytest.raises(ConfigError, match="unknown backend key"):
        load_config(path=None, env=env)


def test_backends_missing_name_raises():
    with pytest.raises(ConfigError, match="'name'"):
        load_config(path=None, env={"BACKENDS_JSON": '[{"host": "h1", "port": 1}]'})


def test_backends_empty_name_raises():
    env = _backend_env([{"name": "", "host": "h1", "port": 1}])
    with pytest.raises(ConfigError, match="'name'"):
        load_config(path=None, env=env)


def test_backends_non_string_name_raises():
    env = _backend_env([{"name": 7, "host": "h1", "port": 1}])
    with pytest.raises(ConfigError, match="'name'"):
        load_config(path=None, env=env)


def test_backends_missing_host_raises():
    with pytest.raises(ConfigError, match="'host'"):
        load_config(path=None, env={"BACKENDS_JSON": '[{"name": "b1", "port": 1}]'})


def test_backends_bad_port_type_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": "8000"}])
    with pytest.raises(ConfigError, match="'port'"):
        load_config(path=None, env=env)


def test_backends_port_out_of_range_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 70000}])
    with pytest.raises(ConfigError, match="port must be in \\[1, 65535\\]"):
        load_config(path=None, env=env)


def test_backends_port_zero_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 0}])
    with pytest.raises(ConfigError, match="port must be in \\[1, 65535\\]"):
        load_config(path=None, env=env)


def test_backends_port_boundaries_accepted():
    """Port exactly 1 and 65535 are accepted (inclusive bounds)."""
    env = _backend_env(
        [
            {"name": "a", "host": "h1", "port": 1},
            {"name": "b", "host": "h2", "port": 65535},
        ]
    )
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].port == 1
    assert cfg.backends[1].port == 65535


def test_backends_per_backend_target_at_100_is_valid():
    """Per-backend target_kv_cache_pct exactly 100 is accepted."""
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "target_kv_cache_pct": 100}])
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].target_kv_cache_pct == 100.0


def test_backends_per_backend_margin_at_one_is_valid():
    """Per-backend token_margin exactly 1.0 is accepted."""
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "token_margin": 1.0}])
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].token_margin == 1.0


def test_backends_per_backend_retry_max_equal_min_is_valid():
    """Per-backend retry_max_s == retry_min_s is accepted."""
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "retry_min_s": 10,
                "retry_max_s": 10,
            }
        ]
    )
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].retry_min_s == 10
    assert cfg.backends[0].retry_max_s == 10


def test_backends_duplicate_name_raises():
    env = _backend_env(
        [
            {"name": "same", "host": "h1", "port": 1},
            {"name": "same", "host": "h2", "port": 2},
        ]
    )
    with pytest.raises(ConfigError, match="duplicate backend name"):
        load_config(path=None, env=env)


def test_backends_model_id_duplicated_across_backends_is_legal():
    """Duplicate model ids across backends are LEGAL (decision 11 supersedes
    the old global-uniqueness rule): the routing: section decides which
    backend gets a request, so the config must load."""
    env = _backend_env(
        [
            {"name": "a", "host": "h1", "port": 1, "models": ["shared"]},
            {"name": "b", "host": "h2", "port": 2, "models": ["shared"]},
        ]
    )
    cfg = load_config(path=None, env=env)
    assert cfg.backends[0].models == ("shared",)
    assert cfg.backends[1].models == ("shared",)


def test_backends_model_id_duplicated_within_backend_raises():
    env = _backend_env(
        [
            {"name": "a", "host": "h1", "port": 1, "models": ["m", "m"]},
            {"name": "b", "host": "h2", "port": 2, "models": ["other"]},
        ]
    )
    with pytest.raises(ConfigError, match="duplicate model id"):
        load_config(path=None, env=env)


def test_backends_models_not_a_list_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "models": "m1"}])
    with pytest.raises(ConfigError, match="'models'"):
        load_config(path=None, env=env)


def test_backends_models_empty_string_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "models": [""]}])
    with pytest.raises(ConfigError, match="models\\[0\\]"):
        load_config(path=None, env=env)


def test_backends_models_non_string_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "models": [7]}])
    with pytest.raises(ConfigError, match="models\\[0\\]"):
        load_config(path=None, env=env)


def test_backends_two_defaults_raises():
    env = _backend_env(
        [
            {"name": "a", "host": "h1", "port": 1, "default": True},
            {"name": "b", "host": "h2", "port": 2, "default": True},
        ]
    )
    with pytest.raises(ConfigError, match="at most one backend"):
        load_config(path=None, env=env)


def test_backends_one_default_is_valid():
    env = _backend_env(
        [
            {"name": "a", "host": "h1", "port": 1},
            {"name": "b", "host": "h2", "port": 2, "default": True},
        ]
    )
    cfg = load_config(path=None, env=env)
    assert cfg.backends[1].default is True


def test_backends_default_not_bool_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "default": "yes"}])
    with pytest.raises(ConfigError, match="'default'"):
        load_config(path=None, env=env)


def test_backends_per_backend_target_out_of_range_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "target_kv_cache_pct": 101}])
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=None, env=env)


def test_backends_per_backend_target_zero_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "target_kv_cache_pct": 0}])
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=None, env=env)


def test_backends_per_backend_target_nan_raises(tmp_path):
    p = tmp_path / "nan.yaml"
    p.write_text(
        "backends:\n  - name: b1\n    host: h1\n    port: 1\n    target_kv_cache_pct: .nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="target_kv_cache_pct"):
        load_config(path=str(p), env={})


def test_backends_per_backend_margin_below_one_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "token_margin": 0.9}])
    with pytest.raises(ConfigError, match="token_margin"):
        load_config(path=None, env=env)


def test_backends_per_backend_retry_min_zero_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "retry_min_s": 0}])
    with pytest.raises(ConfigError, match="retry_min_s"):
        load_config(path=None, env=env)


def test_backends_per_backend_retry_max_below_min_raises():
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "retry_min_s": 30,
                "retry_max_s": 10,
            }
        ]
    )
    with pytest.raises(ConfigError, match="retry_max_s"):
        load_config(path=None, env=env)


def test_backends_per_backend_retry_max_below_global_min_raises():
    """retry_max_s override below the global retry_min_s is rejected."""
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "retry_max_s": 3,
            }
        ]
    )
    with pytest.raises(ConfigError, match="retry_max_s"):
        load_config(path=None, env=env)


def test_backends_per_backend_retry_min_above_global_max_raises():
    """retry_min_s override above the global retry_max_s is rejected."""
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "retry_min_s": 120,
            }
        ]
    )
    with pytest.raises(ConfigError, match="retry_max_s"):
        load_config(path=None, env=env)


def test_backends_per_backend_threshold_kv_pct_out_of_range_raises():
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "thresholds": [{"kv_pct": 150, "max_context": 1, "timeout_s": 1}],
            }
        ]
    )
    with pytest.raises(ConfigError, match="kv_pct"):
        load_config(path=None, env=env)


def test_backends_per_backend_threshold_duplicate_kv_pct_raises():
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "thresholds": [
                    {"kv_pct": 50, "max_context": 1, "timeout_s": 1},
                    {"kv_pct": 50, "max_context": 2, "timeout_s": 2},
                ],
            }
        ]
    )
    with pytest.raises(ConfigError, match="duplicate kv_pct"):
        load_config(path=None, env=env)


def test_backends_per_backend_threshold_bad_shape_raises():
    env = _backend_env(
        [
            {
                "name": "b1",
                "host": "h1",
                "port": 1,
                "thresholds": [{"kv_pct": 50, "max_context": 1}],
            }
        ]
    )
    with pytest.raises(ConfigError, match="exactly the keys"):
        load_config(path=None, env=env)


def test_backends_knob_wrong_type_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": 1, "token_margin": "1.5"}])
    with pytest.raises(ConfigError, match="token_margin"):
        load_config(path=None, env=env)


def test_backends_port_bool_raises():
    env = _backend_env([{"name": "b1", "host": "h1", "port": True}])
    with pytest.raises(ConfigError, match="'port'"):
        load_config(path=None, env=env)


# --- routing (multi-backend, decision 9/15) -----------------------------------


def _routing_file(tmp_path, routing: object) -> str:
    """A config file with two backends plus the given routing section
    (routing is file-only — there is no env override)."""
    return _write(
        tmp_path,
        {
            "backends": [
                {"name": "a", "host": "h1", "port": 1},
                {"name": "b", "host": "h2", "port": 2},
            ],
            "routing": routing,
        },
    )


def test_routing_valid_two_entries(tmp_path):
    """A valid routing mapping parses to RoutingEntry/RoutingSpec exactly."""
    path = _routing_file(
        tmp_path,
        {
            "m1": {"policy": "fill", "order": ["a", "b"]},
            "m2": {"policy": "large_small", "order": ["b", "a"], "threshold_tokens": 4096},
        },
    )
    cfg = load_config(path=path, env={})
    assert cfg.routing == (
        RoutingEntry(model="m1", spec=RoutingSpec(policy="fill", order=("a", "b"))),
        RoutingEntry(
            model="m2",
            spec=RoutingSpec(policy="large_small", order=("b", "a"), threshold_tokens=4096),
        ),
    )


def test_routing_omitted_is_empty_tuple(tmp_path):
    path = _write(tmp_path, {"backends": [{"name": "b1", "host": "h1", "port": 8000}]})
    cfg = load_config(path=path, env={})
    assert cfg.routing == ()


def test_routing_empty_mapping_is_empty_tuple(tmp_path):
    path = _routing_file(tmp_path, {})
    cfg = load_config(path=path, env={})
    assert cfg.routing == ()


def test_routing_not_a_mapping_raises(tmp_path):
    path = _routing_file(tmp_path, [{"policy": "fill", "order": ["a"]}])
    with pytest.raises(ConfigError, match="must be a mapping of model id to entry"):
        load_config(path=path, env={})


def test_routing_non_string_model_key_raises(tmp_path):
    path = _routing_file(tmp_path, {123: {"policy": "fill", "order": ["a"]}})
    with pytest.raises(ConfigError, match="model id must be a non-empty string"):
        load_config(path=path, env={})


def test_routing_empty_model_key_raises(tmp_path):
    path = _routing_file(tmp_path, {"": {"policy": "fill", "order": ["a"]}})
    with pytest.raises(ConfigError, match="model id must be a non-empty string"):
        load_config(path=path, env={})


def test_routing_entry_not_a_dict_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": "fill"})
    with pytest.raises(ConfigError, match="entry must be an object"):
        load_config(path=path, env={})


def test_routing_unknown_key_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": ["a"], "bogus": 1}})
    with pytest.raises(ConfigError, match="unknown routing key"):
        load_config(path=path, env={})


def test_routing_missing_policy_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"order": ["a"]}})
    with pytest.raises(ConfigError, match="'policy'"):
        load_config(path=path, env={})


def test_routing_non_string_policy_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": 7, "order": ["a"]}})
    with pytest.raises(ConfigError, match="'policy'"):
        load_config(path=path, env={})


def test_routing_unknown_policy_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "greedy", "order": ["a"]}})
    with pytest.raises(ConfigError, match="unknown routing policy"):
        load_config(path=path, env={})


def test_routing_missing_order_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill"}})
    with pytest.raises(ConfigError, match="'order'"):
        load_config(path=path, env={})


def test_routing_empty_order_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": []}})
    with pytest.raises(ConfigError, match="'order'"):
        load_config(path=path, env={})


def test_routing_order_not_a_list_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": "a"}})
    with pytest.raises(ConfigError, match="'order'"):
        load_config(path=path, env={})


def test_routing_order_non_string_entry_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": [7]}})
    with pytest.raises(ConfigError, match="order\\[0\\]"):
        load_config(path=path, env={})


def test_routing_order_empty_string_entry_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": [""]}})
    with pytest.raises(ConfigError, match="order\\[0\\]"):
        load_config(path=path, env={})


def test_routing_order_duplicate_entry_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": ["a", "a"]}})
    with pytest.raises(ConfigError, match="duplicate backend name"):
        load_config(path=path, env={})


@pytest.mark.parametrize("policy", ["round_robin", "primary_fallback", "even", "fill"])
def test_routing_threshold_tokens_on_non_large_small_raises(tmp_path, policy):
    path = _routing_file(
        tmp_path, {"m1": {"policy": policy, "order": ["a", "b"], "threshold_tokens": 100}}
    )
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_threshold_tokens_absent_on_large_small_raises(tmp_path):
    path = _routing_file(tmp_path, {"m1": {"policy": "large_small", "order": ["a", "b"]}})
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_threshold_tokens_zero_raises(tmp_path):
    path = _routing_file(
        tmp_path, {"m1": {"policy": "large_small", "order": ["a", "b"], "threshold_tokens": 0}}
    )
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_threshold_tokens_negative_raises(tmp_path):
    path = _routing_file(
        tmp_path, {"m1": {"policy": "large_small", "order": ["a", "b"], "threshold_tokens": -5}}
    )
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_threshold_tokens_bool_raises(tmp_path):
    path = _routing_file(
        tmp_path, {"m1": {"policy": "large_small", "order": ["a", "b"], "threshold_tokens": True}}
    )
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_threshold_tokens_float_raises(tmp_path):
    path = _routing_file(
        tmp_path, {"m1": {"policy": "large_small", "order": ["a", "b"], "threshold_tokens": 1.5}}
    )
    with pytest.raises(ConfigError, match="threshold_tokens"):
        load_config(path=path, env={})


def test_routing_order_unknown_backend_raises(tmp_path):
    """Every order name must be a configured backend (decision 15)."""
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": ["a", "ghost"]}})
    with pytest.raises(ConfigError, match="unknown backend 'ghost'"):
        load_config(path=path, env={})


def test_routing_order_single_backend_is_valid(tmp_path):
    """A one-candidate order is legal (e.g. an explicit pin)."""
    path = _routing_file(tmp_path, {"m1": {"policy": "primary_fallback", "order": ["a"]}})
    cfg = load_config(path=path, env={})
    assert cfg.routing == (
        RoutingEntry(model="m1", spec=RoutingSpec(policy="primary_fallback", order=("a",))),
    )


def test_routing_is_file_only_no_env_override(tmp_path):
    """There is no routing env var: no code reads one, so an env var named
    like one (e.g. ROUTING_JSON) is simply never consulted — the file's
    routing section is unaffected by it."""
    path = _routing_file(tmp_path, {"m1": {"policy": "fill", "order": ["a"]}})
    cfg = load_config(
        path=path, env={"ROUTING_JSON": json.dumps({"m9": {"policy": "fill", "order": ["a"]}})}
    )
    assert cfg.routing == (RoutingEntry(model="m1", spec=RoutingSpec(policy="fill", order=("a",))),)
