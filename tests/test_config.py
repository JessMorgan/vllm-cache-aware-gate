"""Tests for gate.config: file loading, env overrides, validation."""

from __future__ import annotations

import json

import pytest

from gate import config
from gate.config import ConfigError, GateConfig, Threshold, load_config


def _write(tmp_path, data: object, name: str = "config.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# --- happy paths -----------------------------------------------------------


def test_load_from_valid_file(tmp_config_file):
    cfg = load_config(path=tmp_config_file, env={})
    assert cfg.vllm_host == "vllm"
    assert cfg.vllm_port == 9000
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
    path = _write(tmp_path, {"thresholds": [{"kv_pct": 10, "max_context": 1, "timeout_s": 1}]})
    cfg = load_config(path=path, env={})
    assert cfg.vllm_host == "vllm"
    assert cfg.vllm_port == 8000
    assert cfg.listen_port == 8000
    assert cfg.chars_per_token == 4
    assert cfg.default_max_tokens == 256


def test_env_overrides_win_over_file(tmp_config_file):
    env = {
        "VLLM_HOST": "other-host",
        "VLLM_PORT": "9999",
        "LISTEN_HOST": "0.0.0.0",
        "LISTEN_PORT": "7000",
        "THRESHOLDS_JSON": json.dumps([{"kv_pct": 90, "max_context": 512, "timeout_s": 5}]),
    }
    cfg = load_config(path=tmp_config_file, env=env)
    assert cfg.vllm_host == "other-host"
    assert cfg.vllm_port == 9999
    assert cfg.listen_host == "0.0.0.0"
    assert cfg.listen_port == 7000
    assert cfg.thresholds == (Threshold(kv_pct=90.0, max_context=512, timeout_s=5),)


def test_no_file_no_env_defaults_are_invalid(tmp_path, monkeypatch):
    """Defaults have zero thresholds, which is invalid: must raise ConfigError."""
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", str(tmp_path / "does-not-exist.json"))
    with pytest.raises(ConfigError, match="at least one threshold"):
        load_config(path=None, env={})


def test_default_config_path_used_when_present(tmp_path, monkeypatch):
    """path=None falls back to DEFAULT_CONFIG_PATH when that file exists."""
    default_path = _write(
        tmp_path, {"thresholds": [{"kv_pct": 5, "max_context": 2, "timeout_s": 2}]}
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
    env = {"THRESHOLDS_JSON": json.dumps(raw)}
    cfg = load_config(path=None, env=env)
    assert isinstance(cfg, GateConfig)
    assert isinstance(cfg.thresholds, tuple)
    assert all(isinstance(t, Threshold) for t in cfg.thresholds)
    assert len(cfg.thresholds) == 2


# --- file errors ------------------------------------------------------------


def test_bad_file_path_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(path=str(tmp_path / "missing.json"), env={})


def test_invalid_json_file_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path=str(p), env={})


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
            "bogus_key": True,
        },
    )
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(path=path, env={})


def test_file_must_be_json_object(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a JSON object"):
        load_config(path=str(p), env={})


# --- threshold validation ----------------------------------------------------


def _env_with_thresholds(thresholds: list[dict]) -> dict[str, str]:
    return {"THRESHOLDS_JSON": json.dumps(thresholds)}


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
    with pytest.raises(ConfigError, match="VLLM_PORT"):
        load_config(path=None, env={"VLLM_PORT": "abc", "THRESHOLDS_JSON": "[]"})


def test_bad_listen_port_env_raises():
    with pytest.raises(ConfigError, match="LISTEN_PORT"):
        load_config(path=None, env={"LISTEN_PORT": "not-a-port"})


def test_bad_thresholds_json_not_list_raises():
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(path=None, env={"THRESHOLDS_JSON": '{"kv_pct": 50}'})


def test_bad_thresholds_json_invalid_json_raises():
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path=None, env={"THRESHOLDS_JSON": "{oops"})


def test_empty_thresholds_list_invalid():
    with pytest.raises(ConfigError, match="at least one threshold"):
        load_config(path=None, env={"THRESHOLDS_JSON": "[]"})


# --- scalar validation ----------------------------------------------------------


def test_listen_port_out_of_range_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "listen_port": 70000,
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="listen_port"):
        load_config(path=path, env={})


def test_vllm_port_zero_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "vllm_port": 0,
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="vllm_port"):
        load_config(path=path, env={})


def test_chars_per_token_zero_raises(tmp_path):
    path = _write(
        tmp_path,
        {
            "chars_per_token": 0,
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
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="stale_after_s"):
        load_config(path=path, env={})


# --- boundary conditions ---------------------------------------------------------


def test_boundary_values_accepted():
    """kv_pct exactly 0 and 100, max_context exactly 1, ports 1 and 65535."""
    env = {
        "VLLM_PORT": "1",
        "LISTEN_PORT": "65535",
        "THRESHOLDS_JSON": json.dumps(
            [
                {"kv_pct": 0, "max_context": 1, "timeout_s": 1},
                {"kv_pct": 100, "max_context": 1, "timeout_s": 1},
            ]
        ),
    }
    cfg = load_config(path=None, env=env)
    assert cfg.vllm_port == 1
    assert cfg.listen_port == 65535
    assert cfg.thresholds[0].kv_pct == 0.0
    assert cfg.thresholds[1].kv_pct == 100.0


def test_kv_pct_accepts_float():
    env = _env_with_thresholds([{"kv_pct": 50.5, "max_context": 1, "timeout_s": 1}])
    cfg = load_config(path=None, env=env)
    assert cfg.thresholds[0].kv_pct == 50.5


def test_nan_poll_interval_raises(tmp_path):
    """NaN is valid JSON but must be rejected by the > 0 validation."""
    p = tmp_path / "nan.json"
    p.write_text(
        json.dumps(
            {
                "metrics_poll_interval_s": float("nan"),
                "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="metrics_poll_interval_s"):
        load_config(path=str(p), env={})


def test_non_utf8_file_raises(tmp_path):
    p = tmp_path / "binary.json"
    p.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ConfigError, match="not valid UTF-8"):
        load_config(path=str(p), env={})


def test_wrong_type_values_raise(tmp_path):
    """Type mismatches in the file (string port, bool int) are rejected."""
    path = _write(
        tmp_path,
        {
            "vllm_port": "8000",
            "chars_per_token": True,
            "thresholds": [{"kv_pct": 50, "max_context": 1, "timeout_s": 1}],
        },
    )
    with pytest.raises(ConfigError, match="vllm_port"):
        load_config(path=path, env={})
