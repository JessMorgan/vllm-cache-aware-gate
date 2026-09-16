"""Tests for gate.stats: GateStats metrics and Prometheus rendering.

Covers the multi-backend additions (docs/plans/multi-backend.md §2.5): the
``backend`` label on the feed/remaining gauges, the
``gate_backend_capacity_unavailable`` gauge, and the ``gate_config_info``
backend-structure serialization (which omits the ``models:`` lists).
"""

from __future__ import annotations

import json
from dataclasses import replace

from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from gate.config import GateConfig, Threshold
from gate.stats import GateStats


def make_config() -> GateConfig:
    """A GateConfig with two tiers (50/4096/15, 80/1024/30)."""
    return GateConfig(
        vllm_host="vllm",
        vllm_port=9000,
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=60.0,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
    )


def make_stats(cfg: GateConfig | None = None) -> GateStats:
    """A fresh GateStats with its own registry (never shared across tests)."""
    return GateStats(cfg if cfg is not None else make_config(), registry=CollectorRegistry())


def _config_labels(stats: GateStats) -> dict[str, str]:
    """The label mapping of the single ``gate_config_info`` sample."""
    for family in text_string_to_metric_families(stats.render().decode()):
        if family.name == "gate_config_info":
            for sample in family.samples:
                return sample.labels
    raise AssertionError("gate_config_info family not found in rendered output")


class TestRecordForwarded:
    def test_increments_counter(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500)
        text = stats.render().decode()
        assert (
            'gate_requests_total{endpoint="chat_completions",model="m",result="forwarded"} 1.0'
            in text
        )

    def test_observes_histogram(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500)
        text = stats.render().decode()
        assert 'gate_request_ctx_tokens_count{model="m",result="forwarded"} 1.0' in text
        assert 'gate_request_ctx_tokens_sum{model="m",result="forwarded"} 500.0' in text
        assert 'gate_request_ctx_tokens_bucket{le="512.0",model="m",result="forwarded"} 1.0' in text

    def test_none_skips_histogram(self) -> None:
        # A fail-open unparseable request (no estimate) increments the counter
        # but must not skew the size distribution.
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500)
        stats.record_forwarded("chat_completions", "m", None)
        text = stats.render().decode()
        assert (
            'gate_requests_total{endpoint="chat_completions",model="m",result="forwarded"} 2.0'
            in text
        )
        assert 'gate_request_ctx_tokens_count{model="m",result="forwarded"} 1.0' in text


class TestRecordRejected:
    def test_increments_counter(self) -> None:
        stats = make_stats()
        stats.record_rejected("completions", "n", 9000)
        text = stats.render().decode()
        assert 'gate_requests_total{endpoint="completions",model="n",result="rejected"} 1.0' in text

    def test_observes_histogram(self) -> None:
        stats = make_stats()
        stats.record_rejected("completions", "n", 9000)
        text = stats.render().decode()
        assert 'gate_request_ctx_tokens_count{model="n",result="rejected"} 1.0' in text
        assert 'gate_request_ctx_tokens_sum{model="n",result="rejected"} 9000.0' in text


class TestPerModelSeparation:
    def test_distinct_model_series(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 100)
        stats.record_forwarded("chat_completions", "z", 200)
        text = stats.render().decode()
        assert 'gate_request_ctx_tokens_sum{model="m",result="forwarded"} 100.0' in text
        assert 'gate_request_ctx_tokens_sum{model="z",result="forwarded"} 200.0' in text


class TestKvUsage:
    def test_sets_percent(self) -> None:
        # Fraction 0.5 -> 50.0 percent.
        stats = make_stats()
        stats.set_kv_usage({"A": 0.5})
        text = stats.render().decode()
        assert 'gate_kv_cache_usage_pct{model_name="A"} 50.0' in text

    def test_removes_stale_models(self) -> None:
        stats = make_stats()
        stats.set_kv_usage({"A": 0.5, "B": 0.2})
        stats.set_kv_usage({"B": 0.3})
        text = stats.render().decode()
        # A dropped out of the feed -> its series must be gone.
        assert 'model_name="A"' not in text
        assert 'gate_kv_cache_usage_pct{model_name="B"} 30.0' in text

    def test_empty_mapping_removes_all(self) -> None:
        stats = make_stats()
        stats.set_kv_usage({"A": 0.5})
        stats.set_kv_usage({})
        text = stats.render().decode()
        assert 'model_name="A"' not in text


class TestFreshness:
    def test_fresh(self) -> None:
        stats = make_stats()
        stats.set_freshness("qwen", True, 1.2)
        text = stats.render().decode()
        assert 'gate_metrics_fresh{backend="qwen"} 1.0' in text
        assert 'gate_metrics_age_s{backend="qwen"} 1.2' in text

    def test_stale_never_fetched(self) -> None:
        # age_s=None (never fetched) must render as NaN, not 0.0.
        stats = make_stats()
        stats.set_freshness("qwen", False, None)
        text = stats.render().decode()
        assert 'gate_metrics_fresh{backend="qwen"} 0.0' in text
        assert 'gate_metrics_age_s{backend="qwen"} NaN' in text

    def test_per_backend_series_are_independent(self) -> None:
        # Each backend gets its own series; one backend's staleness does not
        # affect the other's.
        stats = make_stats()
        stats.set_freshness("qwen", True, 1.2)
        stats.set_freshness("llama", False, None)
        text = stats.render().decode()
        assert 'gate_metrics_fresh{backend="qwen"} 1.0' in text
        assert 'gate_metrics_fresh{backend="llama"} 0.0' in text
        assert 'gate_metrics_age_s{backend="qwen"} 1.2' in text
        assert 'gate_metrics_age_s{backend="llama"} NaN' in text

    def test_initial_state_before_set_freshness(self) -> None:
        # Before any set_freshness, no per-backend SAMPLE exists yet (the
        # family's HELP/TYPE lines always render). Each backend is set on
        # every /metrics render.
        stats = make_stats()
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name in ("gate_metrics_fresh", "gate_metrics_age_s"):
                assert list(family.samples) == []


class TestRemaining:
    def test_sets_value(self) -> None:
        stats = make_stats()
        stats.set_remaining("qwen", 123)
        text = stats.render().decode()
        assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 123.0' in text

    def test_sets_negative_value(self) -> None:
        # An over-committed counter renders its negative value.
        stats = make_stats()
        stats.set_remaining("qwen", -5)
        text = stats.render().decode()
        assert 'gate_kv_cache_remaining_tokens{backend="qwen"} -5.0' in text

    def test_none_renders_nan(self) -> None:
        # None (never anchored) must render as NaN, not 0.0.
        stats = make_stats()
        stats.set_remaining("qwen", 123)
        stats.set_remaining("qwen", None)
        text = stats.render().decode()
        assert 'gate_kv_cache_remaining_tokens{backend="qwen"} NaN' in text

    def test_per_backend_series_are_independent(self) -> None:
        stats = make_stats()
        stats.set_remaining("qwen", 123)
        stats.set_remaining("llama", None)
        text = stats.render().decode()
        assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 123.0' in text
        assert 'gate_kv_cache_remaining_tokens{backend="llama"} NaN' in text

    def test_initial_state_before_set_remaining(self) -> None:
        # Before any set_remaining, no per-backend sample exists yet (the
        # family's HELP/TYPE lines always render).
        stats = make_stats()
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name == "gate_kv_cache_remaining_tokens":
                assert list(family.samples) == []


class TestBackendCapacityUnavailable:
    def test_set_and_clear(self) -> None:
        stats = make_stats()
        stats.set_backend_capacity_unavailable("qwen", True)
        text = stats.render().decode()
        assert 'gate_backend_capacity_unavailable{backend="qwen"} 1.0' in text
        stats.set_backend_capacity_unavailable("qwen", False)
        text = stats.render().decode()
        assert 'gate_backend_capacity_unavailable{backend="qwen"} 0.0' in text

    def test_per_backend_independent(self) -> None:
        stats = make_stats()
        stats.set_backend_capacity_unavailable("qwen", True)
        stats.set_backend_capacity_unavailable("llama", False)
        text = stats.render().decode()
        assert 'gate_backend_capacity_unavailable{backend="qwen"} 1.0' in text
        assert 'gate_backend_capacity_unavailable{backend="llama"} 0.0' in text

    def test_initial_state_before_set(self) -> None:
        # The app sets every backend on each render; before that the family
        # has no samples (the HELP/TYPE lines always render).
        stats = make_stats()
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name == "gate_backend_capacity_unavailable":
                assert list(family.samples) == []


class TestConfigInfo:
    def test_labels_present(self) -> None:
        stats = make_stats()
        text = stats.render().decode()
        assert "gate_config_info{" in text
        assert 'vllm_host="vllm"' in text
        assert "thresholds_json=" in text

    def test_label_values(self) -> None:
        stats = make_stats()
        labels = _config_labels(stats)
        assert labels["vllm_host"] == "vllm"
        assert labels["vllm_port"] == "9000"
        assert labels["listen_host"] == "127.0.0.1"
        assert labels["listen_port"] == "8080"
        assert labels["metrics_poll_interval_s"] == "1.5"
        assert labels["stale_after_s"] == "60.0"
        assert labels["chars_per_token"] == "4"
        assert labels["default_max_tokens"] == "256"
        # Compact JSON of the two thresholds as [kv_pct, max_context, timeout_s].
        assert labels["thresholds_json"] == json.dumps([[50.0, 4096, 15], [80.0, 1024, 30]])

    def test_backends_json_legacy_single_backend(self) -> None:
        # cfg.backends empty -> the legacy single backend implied by
        # vllm_host/vllm_port is serialized (name = "host:port", default).
        stats = make_stats()
        labels = _config_labels(stats)
        backends = json.loads(labels["backends_json"])
        assert backends == [
            {
                "name": "vllm:9000",
                "host": "vllm",
                "port": 9000,
                "default": True,
                "overrides": [],
            }
        ]

    def test_backends_json_serializes_structure_and_overrides(self) -> None:
        from gate.config import Backend

        cfg = make_config()
        cfg = replace(
            cfg,
            backends=(
                Backend(
                    name="qwen",
                    host="vllm-qwen",
                    port=8000,
                    models=("qwen3-32b",),
                    default=True,
                    target_kv_cache_pct=80.0,
                    token_margin=1.5,
                ),
                Backend(name="llama", host="vllm-llama", port=8001),
            ),
        )
        stats = GateStats(cfg, registry=CollectorRegistry())
        labels = _config_labels(stats)
        backends = json.loads(labels["backends_json"])
        assert backends == [
            {
                "name": "qwen",
                "host": "vllm-qwen",
                "port": 8000,
                "default": True,
                "overrides": ["target_kv_cache_pct", "token_margin"],
            },
            {
                "name": "llama",
                "host": "vllm-llama",
                "port": 8001,
                "default": False,
                "overrides": [],
            },
        ]

    def test_backends_json_omits_model_lists(self) -> None:
        from gate.config import Backend

        cfg = make_config()
        cfg = replace(
            cfg,
            backends=(Backend(name="qwen", host="h", port=1, models=("secret-model",)),),
        )
        stats = GateStats(cfg, registry=CollectorRegistry())
        text = stats.render().decode()
        # The models: lists are discovered at runtime and must not appear in
        # the startup-static info metric.
        assert "secret-model" not in text


class TestRender:
    def test_returns_bytes_with_gate_header(self) -> None:
        stats = make_stats()
        out = stats.render()
        assert isinstance(out, bytes)
        first_line = out.decode().splitlines()[0]
        assert first_line.startswith("# HELP gate_") or first_line.startswith("# TYPE gate_")


class TestRegistryIsolation:
    def test_recording_on_one_does_not_affect_other(self) -> None:
        a = GateStats(make_config(), registry=CollectorRegistry())
        b = GateStats(make_config(), registry=CollectorRegistry())
        a.record_forwarded("chat_completions", "m", 500)
        a.set_kv_usage({"A": 0.5})
        a_text = a.render().decode()
        b_text = b.render().decode()
        assert 'model="m"' in a_text
        assert 'model="m"' not in b_text
        assert 'model_name="A"' in a_text
        assert 'model_name="A"' not in b_text
