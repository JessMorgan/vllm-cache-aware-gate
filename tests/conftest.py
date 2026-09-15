"""Shared fixtures for gate tests (kept minimal; units add their own)."""

from __future__ import annotations

import json

import pytest

SAMPLE_CONFIG: dict = {
    "vllm_host": "vllm",
    "vllm_port": 9000,
    "listen_host": "127.0.0.1",
    "listen_port": 8080,
    "metrics_poll_interval_s": 1.5,
    "stale_after_s": 4.0,
    "chars_per_token": 3,
    "default_max_tokens": 128,
    "thresholds": [
        {"kv_pct": 50, "max_context": 4096, "timeout_s": 15},
        {"kv_pct": 80, "max_context": 1024, "timeout_s": 30},
    ],
}


@pytest.fixture
def sample_config_dict() -> dict:
    """A valid config dict with two thresholds."""
    return json.loads(json.dumps(SAMPLE_CONFIG))


@pytest.fixture
def tmp_config_file(tmp_path, sample_config_dict) -> str:
    """Write the sample config to a temp file and yield its path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(sample_config_dict), encoding="utf-8")
    return str(path)
