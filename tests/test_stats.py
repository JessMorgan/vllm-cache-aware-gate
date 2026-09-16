"""Tests for gate.stats: GateStats metrics and Prometheus rendering."""

from __future__ import annotations

import json

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
        stats.set_freshness(True, 1.2)
        text = stats.render().decode()
        assert "gate_metrics_fresh 1.0" in text
        assert "gate_metrics_age_s 1.2" in text

    def test_stale_never_fetched(self) -> None:
        # age_s=None (never fetched) must render as NaN, not 0.0.
        stats = make_stats()
        stats.set_freshness(False, None)
        text = stats.render().decode()
        assert "gate_metrics_fresh 0.0" in text
        assert "gate_metrics_age_s NaN" in text

    def test_initial_state_before_set_freshness(self) -> None:
        # Before any set_freshness, age must be NaN (not 0.0) and fresh must be 0.0.
        stats = make_stats()
        text = stats.render().decode()
        assert "gate_metrics_fresh 0.0" in text
        assert "gate_metrics_age_s NaN" in text


class TestRemaining:
    def test_sets_value(self) -> None:
        stats = make_stats()
        stats.set_remaining(123)
        text = stats.render().decode()
        assert "gate_kv_cache_remaining_tokens 123.0" in text

    def test_sets_negative_value(self) -> None:
        # An over-committed counter renders its negative value.
        stats = make_stats()
        stats.set_remaining(-5)
        text = stats.render().decode()
        assert "gate_kv_cache_remaining_tokens -5.0" in text

    def test_none_renders_nan(self) -> None:
        # None (never anchored) must render as NaN, not 0.0.
        stats = make_stats()
        stats.set_remaining(123)
        stats.set_remaining(None)
        text = stats.render().decode()
        assert "gate_kv_cache_remaining_tokens NaN" in text

    def test_initial_state_before_set_remaining(self) -> None:
        # Before any set_remaining, the gauge must be NaN (not 0.0).
        stats = make_stats()
        text = stats.render().decode()
        assert "gate_kv_cache_remaining_tokens NaN" in text


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
