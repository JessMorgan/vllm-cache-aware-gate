"""Tests for gate.stats: GateStats metrics and Prometheus rendering.

Covers the multi-backend additions (docs/plans/multi-backend.md §2.5): the
``backend`` label on the feed/remaining gauges, the
``gate_backend_capacity_unavailable`` gauge, the ``backend`` label on the
request counters (decision 13 — with duplicate model ids legal, the backend
is no longer recoverable from the model; the sentinel ``"none"`` on 502
exhaustion), the ``gate_routing_failovers_total`` counter (decision 12), the SGLang
additions (docs/plans/sglang-backend.md §2.5: the ``gate_backend_engine``
detected-engine gauge and the ``gate_metrics_endpoint_unavailable`` alert
flag), and the ``gate_config_info`` backend-structure + ``routing:``
serialization (which omits the ``models:`` lists).
"""

from __future__ import annotations

import json
from dataclasses import replace

from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from gate.config import Backend, GateConfig, Threshold
from gate.stats import GateStats


def make_config() -> GateConfig:
    """A single-backend GateConfig (decision 6) with two tiers (50/4096/15,
    80/1024/30)."""
    return GateConfig(
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=60.0,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
        backends=(Backend(name="vllm", host="vllm", port=9000, default=True),),
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


# Prometheus renders labels in ALPHABETICAL order regardless of the definition
# order, so the rendered-text assertions below use the alphabetical label order
# (backend, endpoint, model, result) for gate_requests_total and (backend,
# model, result) for the ctx_tokens histogram.


class TestRecordForwarded:
    def test_increments_counter(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500, backend="qwen")
        text = stats.render().decode()
        assert (
            'gate_requests_total{backend="qwen",endpoint="chat_completions",model="m",'
            'result="forwarded"} 1.0' in text
        )

    def test_observes_histogram(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500, backend="qwen")
        text = stats.render().decode()
        assert (
            'gate_request_ctx_tokens_count{backend="qwen",model="m",result="forwarded"} 1.0' in text
        )
        assert (
            'gate_request_ctx_tokens_sum{backend="qwen",model="m",result="forwarded"} 500.0' in text
        )
        assert (
            'gate_request_ctx_tokens_bucket{backend="qwen",le="512.0",model="m",'
            'result="forwarded"} 1.0' in text
        )

    def test_none_skips_histogram(self) -> None:
        # A fail-open unparseable request (no estimate) increments the counter
        # but must not skew the size distribution.
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500, backend="qwen")
        stats.record_forwarded("chat_completions", "m", None, backend="qwen")
        text = stats.render().decode()
        assert (
            'gate_requests_total{backend="qwen",endpoint="chat_completions",model="m",'
            'result="forwarded"} 2.0' in text
        )
        assert (
            'gate_request_ctx_tokens_count{backend="qwen",model="m",result="forwarded"} 1.0' in text
        )


class TestRecordRejected:
    def test_increments_counter(self) -> None:
        stats = make_stats()
        stats.record_rejected("completions", "n", 9000, backend="llama")
        text = stats.render().decode()
        assert (
            'gate_requests_total{backend="llama",endpoint="completions",model="n",'
            'result="rejected"} 1.0' in text
        )

    def test_observes_histogram(self) -> None:
        stats = make_stats()
        stats.record_rejected("completions", "n", 9000, backend="llama")
        text = stats.render().decode()
        assert (
            'gate_request_ctx_tokens_count{backend="llama",model="n",result="rejected"} 1.0' in text
        )
        assert (
            'gate_request_ctx_tokens_sum{backend="llama",model="n",result="rejected"} 9000.0'
            in text
        )

    def test_502_exhaustion_uses_none_sentinel(self) -> None:
        # On 502 exhaustion (no rejector exists) the backend label is the
        # sentinel "none" (decision 13).
        stats = make_stats()
        stats.record_rejected("completions", "n", 9000, backend="none")
        text = stats.render().decode()
        assert (
            'gate_requests_total{backend="none",endpoint="completions",model="n",'
            'result="rejected"} 1.0' in text
        )


class TestRequestCounterLabels:
    def test_requests_total_label_set(self) -> None:
        # The label set is exactly {endpoint, model, result, backend} — the
        # existing labels are preserved and backend is added LAST (decision 13).
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500, backend="qwen")
        labels: dict[str, str] = {}
        for family in text_string_to_metric_families(stats.render().decode()):
            # prometheus_client strips the _total suffix: the family is named
            # "gate_requests", the sample "gate_requests_total".
            if family.name == "gate_requests":
                for sample in family.samples:
                    if sample.name == "gate_requests_total":
                        labels = sample.labels
        assert labels == {
            "endpoint": "chat_completions",
            "model": "m",
            "result": "forwarded",
            "backend": "qwen",
        }

    def test_ctx_tokens_label_set(self) -> None:
        # The label set is exactly {model, result, backend} (decision 13).
        # The histogram's _count/_sum/_bucket children render under the base
        # family name, so match the sample name, not the family name.
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 500, backend="qwen")
        labels: dict[str, str] = {}
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name == "gate_request_ctx_tokens":
                for sample in family.samples:
                    if sample.name == "gate_request_ctx_tokens_count":
                        labels = sample.labels
        assert labels == {"model": "m", "result": "forwarded", "backend": "qwen"}

    def test_distinct_backends_are_separate_series(self) -> None:
        # The same model on two backends (duplicate model ids, decision 11)
        # yields distinct series — the backend label is what separates them.
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 100, backend="qwen")
        stats.record_forwarded("chat_completions", "m", 200, backend="llama")
        text = stats.render().decode()
        assert (
            'gate_requests_total{backend="qwen",endpoint="chat_completions",model="m",'
            'result="forwarded"} 1.0' in text
        )
        assert (
            'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
            'result="forwarded"} 1.0' in text
        )


class TestRoutingFailovers:
    def test_increments_per_call(self) -> None:
        stats = make_stats()
        stats.record_failover("m", "qwen", "llama", "transport")
        stats.record_failover("m", "qwen", "llama", "upstream_5xx")
        stats.record_failover("m", "qwen", "llama", "reject")
        text = stats.render().decode()
        # Alphabetical label order: from, model, reason, to.
        assert (
            'gate_routing_failovers_total{from="qwen",model="m",reason="transport",to="llama"} 1.0'
            in text
        )
        assert (
            'gate_routing_failovers_total{from="qwen",model="m",'
            'reason="upstream_5xx",to="llama"} 1.0' in text
        )
        assert (
            'gate_routing_failovers_total{from="qwen",model="m",reason="reject",to="llama"} 1.0'
            in text
        )

    def test_distinct_reasons_are_separate_series(self) -> None:
        stats = make_stats()
        stats.record_failover("m", "a", "b", "reject")
        stats.record_failover("m", "a", "b", "reject")
        stats.record_failover("m", "a", "b", "transport")
        text = stats.render().decode()
        assert 'gate_routing_failovers_total{from="a",model="m",reason="reject",to="b"} 2.0' in text
        assert (
            'gate_routing_failovers_total{from="a",model="m",reason="transport",to="b"} 1.0' in text
        )


class TestPerModelSeparation:
    def test_distinct_model_series(self) -> None:
        stats = make_stats()
        stats.record_forwarded("chat_completions", "m", 100, backend="qwen")
        stats.record_forwarded("chat_completions", "z", 200, backend="qwen")
        text = stats.render().decode()
        assert (
            'gate_request_ctx_tokens_sum{backend="qwen",model="m",result="forwarded"} 100.0' in text
        )
        assert (
            'gate_request_ctx_tokens_sum{backend="qwen",model="z",result="forwarded"} 200.0' in text
        )


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


class TestBackendEngine:
    def test_set_detected_engine(self) -> None:
        # The detected engine's series is 1.0 and the other two are
        # explicitly 0.0 (not absent) so the gauge is always fully
        # populated (docs/plans/sglang-backend.md §2.5). Prometheus renders
        # labels alphabetically: backend, engine.
        stats = make_stats()
        stats.set_backend_engine("qwen", "sglang")
        text = stats.render().decode()
        assert 'gate_backend_engine{backend="qwen",engine="sglang"} 1.0' in text
        assert 'gate_backend_engine{backend="qwen",engine="vllm"} 0.0' in text
        assert 'gate_backend_engine{backend="qwen",engine="unknown"} 0.0' in text

    def test_per_backend_independent(self) -> None:
        # Each backend's detected engine is its own set of series.
        stats = make_stats()
        stats.set_backend_engine("qwen", "vllm")
        stats.set_backend_engine("llama", "sglang")
        text = stats.render().decode()
        assert 'gate_backend_engine{backend="qwen",engine="vllm"} 1.0' in text
        assert 'gate_backend_engine{backend="qwen",engine="sglang"} 0.0' in text
        assert 'gate_backend_engine{backend="llama",engine="sglang"} 1.0' in text
        assert 'gate_backend_engine{backend="llama",engine="vllm"} 0.0' in text

    def test_initial_state_before_set(self) -> None:
        # The app sets every backend on each render; before that the family
        # has no samples (the HELP/TYPE lines always render).
        stats = make_stats()
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name == "gate_backend_engine":
                assert list(family.samples) == []


class TestMetricsEndpointUnavailable:
    def test_set_and_clear(self) -> None:
        stats = make_stats()
        stats.set_backend_metrics_unavailable("qwen", True)
        text = stats.render().decode()
        assert 'gate_metrics_endpoint_unavailable{backend="qwen"} 1.0' in text
        stats.set_backend_metrics_unavailable("qwen", False)
        text = stats.render().decode()
        assert 'gate_metrics_endpoint_unavailable{backend="qwen"} 0.0' in text

    def test_per_backend_independent(self) -> None:
        stats = make_stats()
        stats.set_backend_metrics_unavailable("qwen", True)
        stats.set_backend_metrics_unavailable("llama", False)
        text = stats.render().decode()
        assert 'gate_metrics_endpoint_unavailable{backend="qwen"} 1.0' in text
        assert 'gate_metrics_endpoint_unavailable{backend="llama"} 0.0' in text

    def test_initial_state_before_set(self) -> None:
        # The app sets every backend on each render; before that the family
        # has no samples (the HELP/TYPE lines always render).
        stats = make_stats()
        for family in text_string_to_metric_families(stats.render().decode()):
            if family.name == "gate_metrics_endpoint_unavailable":
                assert list(family.samples) == []


class TestConfigInfo:
    def test_labels_present(self) -> None:
        stats = make_stats()
        text = stats.render().decode()
        assert "gate_config_info{" in text
        assert "thresholds_json=" in text
        assert "backends_json=" in text

    def test_label_values(self) -> None:
        stats = make_stats()
        labels = _config_labels(stats)
        assert labels["listen_host"] == "127.0.0.1"
        assert labels["listen_port"] == "8080"
        assert labels["metrics_poll_interval_s"] == "1.5"
        assert labels["stale_after_s"] == "60.0"
        assert labels["chars_per_token"] == "4"
        assert labels["default_max_tokens"] == "256"
        # Compact JSON of the two thresholds as [kv_pct, max_context, timeout_s].
        assert labels["thresholds_json"] == json.dumps([[50.0, 4096, 15], [80.0, 1024, 30]])
        # The removed vllm_host/vllm_port labels are gone (decision 6).
        assert "vllm_host" not in labels
        assert "vllm_port" not in labels

    def test_backends_json_single_backend(self) -> None:
        # make_config() has one backends entry; it is serialized with its
        # name/host/port/default flag and no overrides.
        stats = make_stats()
        labels = _config_labels(stats)
        backends = json.loads(labels["backends_json"])
        assert backends == [
            {
                "name": "vllm",
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

    def test_routing_json_empty(self) -> None:
        # No routing: section -> the label is the empty JSON list, and the
        # backends_json is still present (unchanged).
        stats = make_stats()
        labels = _config_labels(stats)
        assert labels["routing_json"] == "[]"
        assert "backends_json" in labels

    def test_routing_json_serializes_entries(self) -> None:
        from gate.config import Backend, RoutingEntry
        from gate.routing import RoutingSpec

        cfg = make_config()
        cfg = replace(
            cfg,
            backends=(
                Backend(name="qwen", host="h1", port=1, default=True),
                Backend(name="llama", host="h2", port=2),
            ),
            routing=(
                RoutingEntry(
                    model="qwen3-32b",
                    spec=RoutingSpec(policy="fill", order=("qwen", "llama")),
                ),
                RoutingEntry(
                    model="big-model",
                    spec=RoutingSpec(
                        policy="large_small", order=("qwen", "llama"), threshold_tokens=8000
                    ),
                ),
            ),
        )
        stats = GateStats(cfg, registry=CollectorRegistry())
        labels = _config_labels(stats)
        routing = json.loads(labels["routing_json"])
        assert routing == [
            ["qwen3-32b", "fill", ["qwen", "llama"]],
            ["big-model", "large_small", ["qwen", "llama"]],
        ]
        # The backends_json is still serialized alongside (unchanged).
        backends = json.loads(labels["backends_json"])
        assert [b["name"] for b in backends] == ["qwen", "llama"]


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
        a.record_forwarded("chat_completions", "m", 500, backend="qwen")
        a.set_kv_usage({"A": 0.5})
        a_text = a.render().decode()
        b_text = b.render().decode()
        assert 'model="m"' in a_text
        assert 'model="m"' not in b_text
        assert 'model_name="A"' in a_text
        assert 'model_name="A"' not in b_text
