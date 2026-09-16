"""Tests for gate.metrics: parsing, caching, and the fetch helper."""

from __future__ import annotations

import math

import httpx
import pytest

from gate.metrics import (
    CapacityCache,
    KvRemaining,
    MetricsCache,
    MetricsSample,
    fetch_metrics,
    fetch_usage,
    fetch_usage_by_model,
    parse_kv_cache_capacity,
    parse_kv_cache_usage,
    parse_kv_cache_usage_by_model,
)

SINGLE_SERIES = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.42\n"
)

LABELLED_SERIES = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="llama-3-8b"} 0.42\n'
)

MULTI_SERIES = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="llama-3-8b"} 0.42\n'
    'vllm:kv_cache_usage_perc{model_name="qwen-2-7b"} 0.87\n'
)

OTHER_METRICS = (
    "# HELP vllm:num_requests_running Number of requests in running state.\n"
    "# TYPE vllm:num_requests_running gauge\n"
    "vllm:num_requests_running 3\n"
    "# HELP vllm:gpu_cache_usage_perc Fraction of GPU memory in use.\n"
    "# TYPE vllm:gpu_cache_usage_perc gauge\n"
    "vllm:gpu_cache_usage_perc 0.99\n"
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.42\n"
)

NAN_ONLY = "# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc NaN\n"

NAN_PLUS_REAL = (
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="a"} NaN\n'
    'vllm:kv_cache_usage_perc{model_name="b"} 0.55\n'
)

REALISTIC_SAMPLE = (
    "# HELP python_info Python platform information\n"
    "# TYPE python_info gauge\n"
    'python_info{implementation="CPython", platform_version="3.11.9"} 1.0\n'
    "# HELP vllm:num_requests_waiting Number of requests in waiting state.\n"
    "# TYPE vllm:num_requests_waiting gauge\n"
    "vllm:num_requests_waiting 0\n"
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="llama-3-8b"} 0.42\n'
    'vllm:kv_cache_usage_perc{model_name="qwen-2-7b"} 0.87\n'
    "# HELP vllm:e2e_request_latency_seconds E2E request latency in seconds.\n"
    "# TYPE vllm:e2e_request_latency_seconds histogram\n"
    'vllm:e2e_request_latency_seconds_bucket{le="0.5"} 10\n'
    'vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 12\n'
)


class TestParseKvCacheUsage:
    def test_single_unlabelled_series(self) -> None:
        assert parse_kv_cache_usage(SINGLE_SERIES) == pytest.approx(0.42)

    def test_labelled_series(self) -> None:
        assert parse_kv_cache_usage(LABELLED_SERIES) == pytest.approx(0.42)

    def test_multiple_series_takes_max(self) -> None:
        assert parse_kv_cache_usage(MULTI_SERIES) == pytest.approx(0.87)

    def test_metric_absent_returns_none(self) -> None:
        text = "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 3\n"
        assert parse_kv_cache_usage(text) is None

    def test_empty_text_returns_none(self) -> None:
        assert parse_kv_cache_usage("") is None

    def test_other_vllm_metrics_ignored(self) -> None:
        # gpu_cache_usage_perc is 0.99 — must not leak into the kv gauge result.
        assert parse_kv_cache_usage(OTHER_METRICS) == pytest.approx(0.42)

    def test_nan_only_returns_none(self) -> None:
        assert parse_kv_cache_usage(NAN_ONLY) is None

    def test_nan_plus_real_returns_real(self) -> None:
        assert parse_kv_cache_usage(NAN_PLUS_REAL) == pytest.approx(0.55)

    def test_unparseable_garbage_returns_none(self) -> None:
        assert parse_kv_cache_usage("this is not a prometheus exposition !!!") is None

    def test_realistic_multi_line_sample(self) -> None:
        # Max across the two labelled series; HELP/TYPE/histogram lines ignored.
        assert parse_kv_cache_usage(REALISTIC_SAMPLE) == pytest.approx(0.87)

    def test_result_is_finite(self) -> None:
        value = parse_kv_cache_usage(MULTI_SERIES)
        assert value is not None and math.isfinite(value)


SHARED_MODEL = (
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="a"} 0.40\n'
    'vllm:kv_cache_usage_perc{model_name="a"} 0.65\n'
)


class TestParseKvCacheUsageByModel:
    def test_unlabelled_series_maps_to_default(self) -> None:
        assert parse_kv_cache_usage_by_model(SINGLE_SERIES) == {"default": pytest.approx(0.42)}

    def test_single_labelled_series(self) -> None:
        assert parse_kv_cache_usage_by_model(LABELLED_SERIES) == {"llama-3-8b": pytest.approx(0.42)}

    def test_multiple_labelled_series(self) -> None:
        assert parse_kv_cache_usage_by_model(MULTI_SERIES) == {
            "llama-3-8b": pytest.approx(0.42),
            "qwen-2-7b": pytest.approx(0.87),
        }

    def test_shared_model_name_takes_per_model_max(self) -> None:
        assert parse_kv_cache_usage_by_model(SHARED_MODEL) == {"a": pytest.approx(0.65)}

    def test_metric_absent_returns_none(self) -> None:
        text = "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 3\n"
        assert parse_kv_cache_usage_by_model(text) is None

    def test_empty_text_returns_none(self) -> None:
        assert parse_kv_cache_usage_by_model("") is None

    def test_nan_only_returns_none(self) -> None:
        assert parse_kv_cache_usage_by_model(NAN_ONLY) is None

    def test_nan_plus_real_returns_real(self) -> None:
        assert parse_kv_cache_usage_by_model(NAN_PLUS_REAL) == {"b": pytest.approx(0.55)}

    def test_unparseable_garbage_returns_none(self) -> None:
        assert parse_kv_cache_usage_by_model("this is not a prometheus exposition !!!") is None

    def test_other_vllm_metrics_ignored(self) -> None:
        # gpu_cache_usage_perc is 0.99 — must not leak into the kv gauge result.
        assert parse_kv_cache_usage_by_model(OTHER_METRICS) == {"default": pytest.approx(0.42)}

    def test_realistic_multi_line_sample(self) -> None:
        # Per-model values across the two labelled series; other metrics ignored.
        assert parse_kv_cache_usage_by_model(REALISTIC_SAMPLE) == {
            "llama-3-8b": pytest.approx(0.42),
            "qwen-2-7b": pytest.approx(0.87),
        }


class TestMetricsCache:
    def test_initial_state(self) -> None:
        cache = MetricsCache()
        assert cache.value() is None
        assert cache.age() is None
        assert cache.is_stale(6.0) is True

    def test_update_stores_value_and_timestamp(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        assert cache.value() == pytest.approx(0.5)
        assert cache.age(now=103.0) == pytest.approx(3.0)
        assert cache.is_stale(6.0, now=103.0) is False

    def test_update_none_keeps_value_and_timestamp(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        cache.update(None, now=200.0)
        assert cache.value() == pytest.approx(0.5)
        # Failed fetch must NOT refresh the timestamp: age is still 100.0-based.
        assert cache.age(now=200.0) == pytest.approx(100.0)

    def test_staleness_boundary_not_stale_at_exact_age(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        # age == stale_after_s exactly → not stale (strictly greater required).
        assert cache.is_stale(5.0, now=105.0) is False

    def test_staleness_boundary_stale_just_after(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        assert cache.is_stale(5.0, now=105.0 + 1e-9) is True

    def test_update_overwrites_value_and_timestamp(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        cache.update(0.9, now=150.0)
        assert cache.value() == pytest.approx(0.9)
        assert cache.age(now=150.0) == pytest.approx(0.0)


class TestMetricsCacheByModel:
    def test_initial_by_model_is_empty(self) -> None:
        cache = MetricsCache()
        assert cache.by_model() == {}

    def test_update_by_model_stores_mapping_and_max(self) -> None:
        cache = MetricsCache()
        cache.update_by_model({"A": 0.5, "B": 0.7}, now=100.0)
        assert cache.by_model() == {"A": pytest.approx(0.5), "B": pytest.approx(0.7)}
        assert cache.value() == pytest.approx(0.7)  # overall max

    def test_update_by_model_none_is_noop(self) -> None:
        cache = MetricsCache()
        cache.update_by_model({"A": 0.5}, now=100.0)
        cache.update_by_model(None, now=200.0)
        assert cache.by_model() == {"A": pytest.approx(0.5)}
        assert cache.value() == pytest.approx(0.5)
        # Failed fetch must NOT refresh the timestamp: age is still 100.0-based.
        assert cache.age(now=200.0) == pytest.approx(100.0)

    def test_update_by_model_empty_is_noop(self) -> None:
        cache = MetricsCache()
        cache.update_by_model({"A": 0.5}, now=100.0)
        cache.update_by_model({}, now=200.0)
        assert cache.by_model() == {"A": pytest.approx(0.5)}
        assert cache.value() == pytest.approx(0.5)
        assert cache.age(now=200.0) == pytest.approx(100.0)

    def test_update_by_model_refreshes_timestamp(self) -> None:
        cache = MetricsCache()
        cache.update_by_model({"A": 0.5}, now=100.0)
        cache.update_by_model({"A": 0.9}, now=150.0)
        assert cache.by_model() == {"A": pytest.approx(0.9)}
        assert cache.value() == pytest.approx(0.9)
        assert cache.age(now=150.0) == pytest.approx(0.0)

    def test_by_model_returns_a_copy(self) -> None:
        cache = MetricsCache()
        cache.update_by_model({"A": 0.5}, now=100.0)
        snapshot = cache.by_model()
        snapshot["A"] = 0.99
        assert cache.by_model() == {"A": pytest.approx(0.5)}

    def test_legacy_update_still_sets_value(self) -> None:
        cache = MetricsCache()
        cache.update(0.5, now=100.0)
        assert cache.value() == pytest.approx(0.5)


class TestFetchUsage:
    async def test_200_with_metric_returns_parsed_value(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=MULTI_SERIES))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await fetch_usage(client, "http://vllm:9000/metrics") == pytest.approx(0.87)

    async def test_200_without_metric_returns_none(self) -> None:
        text = "vllm:num_requests_running 3\n"
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await fetch_usage(client, "http://vllm:9000/metrics") is None

    async def test_500_returns_none(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(500, text="boom"))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await fetch_usage(client, "http://vllm:9000/metrics") is None

    async def test_connection_error_propagates(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.ConnectError):
                await fetch_usage(client, "http://vllm:9000/metrics")


CAP_SINGLE = (
    "# HELP vllm:kv_cache_size_tokens Total KV cache capacity in tokens.\n"
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    "vllm:kv_cache_size_tokens 100000\n"
)

CAP_LABELLED = (
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    'vllm:kv_cache_size_tokens{model_name="llama-3-8b"} 100000\n'
)

CAP_MULTI = (
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    'vllm:kv_cache_size_tokens{model_name="llama-3-8b"} 100000\n'
    'vllm:kv_cache_size_tokens{model_name="qwen-2-7b"} 250000\n'
)

CAP_NAN_ONLY = "# TYPE vllm:kv_cache_size_tokens gauge\nvllm:kv_cache_size_tokens NaN\n"

CAP_NON_POSITIVE = (
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    "vllm:kv_cache_size_tokens 0.0\n"
    'vllm:kv_cache_size_tokens{model_name="neg"} -5\n'
)

CAP_USAGE_BOTH = (
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="llama-3-8b"} 0.42\n'
    "vllm:kv_cache_size_tokens 100000\n"
)

CAP_ONLY = "# TYPE vllm:kv_cache_size_tokens gauge\nvllm:kv_cache_size_tokens 100000\n"

USAGE_ONLY = "# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc 0.42\n"

# The shape current vLLM actually emits (vllm-project/vllm PR #42206): the
# capacity is a *label* on the vllm:cache_config_info info gauge, not a
# standalone vllm:kv_cache_size_tokens gauge.
CONFIG_INFO_LABEL_ONLY = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="123433",'
    'num_gpu_blocks="14835"} 1.0\n'
)

CONFIG_INFO_LABEL_NONE = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="None",'
    'num_gpu_blocks="0"} 1.0\n'
)

CONFIG_INFO_TWO_MODELS = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="50000",'
    'num_gpu_blocks="6000"} 1.0\n'
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="200000",'
    'num_gpu_blocks="24000"} 1.0\n'
)

CONFIG_INFO_NON_NUMERIC = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="abc",'
    'num_gpu_blocks="0"} 1.0\n'
)

CONFIG_INFO_NON_POSITIVE = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="0",'
    'num_gpu_blocks="0"} 1.0\n'
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="-5",'
    'num_gpu_blocks="0"} 1.0\n'
)

CONFIG_INFO_LABEL_MISSING = (
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",num_gpu_blocks="0"} 1.0\n'
)

CONFIG_INFO_BOTH_SOURCES = (
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    "vllm:kv_cache_size_tokens 100000\n"
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="999999",'
    'num_gpu_blocks="120000"} 1.0\n'
)

CONFIG_INFO_USAGE_FALLBACK = (
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="llama-3-8b"} 0.42\n'
    "# TYPE vllm:cache_config_info gauge\n"
    'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="77777",'
    'num_gpu_blocks="9000"} 1.0\n'
)


class TestParseKvCacheCapacity:
    def test_single_unlabelled_series(self) -> None:
        assert parse_kv_cache_capacity(CAP_SINGLE) == 100000

    def test_labelled_series(self) -> None:
        assert parse_kv_cache_capacity(CAP_LABELLED) == 100000

    def test_multiple_series_takes_max(self) -> None:
        assert parse_kv_cache_capacity(CAP_MULTI) == 250000

    def test_metric_absent_returns_none(self) -> None:
        text = "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 3\n"
        assert parse_kv_cache_capacity(text) is None

    def test_empty_text_returns_none(self) -> None:
        assert parse_kv_cache_capacity("") is None

    def test_nan_only_returns_none(self) -> None:
        assert parse_kv_cache_capacity(CAP_NAN_ONLY) is None

    def test_non_positive_returns_none(self) -> None:
        # 0.0 and negative samples are not usable capacities.
        assert parse_kv_cache_capacity(CAP_NON_POSITIVE) is None

    def test_unparseable_garbage_returns_none(self) -> None:
        assert parse_kv_cache_capacity("this is not a prometheus exposition !!!") is None

    def test_result_is_int(self) -> None:
        value = parse_kv_cache_capacity(CAP_MULTI)
        assert isinstance(value, int)
        assert value == 250000

    def test_config_info_label_only(self) -> None:
        # The real-world vLLM shape (PR #42206): no standalone gauge, only the
        # kv_cache_size_tokens label on vllm:cache_config_info.
        assert parse_kv_cache_capacity(CONFIG_INFO_LABEL_ONLY) == 123433

    def test_config_info_label_none_string(self) -> None:
        # Attention-free model: the label is the literal string "None".
        assert parse_kv_cache_capacity(CONFIG_INFO_LABEL_NONE) is None

    def test_config_info_label_missing(self) -> None:
        # A cache_config_info sample that lacks the kv_cache_size_tokens label
        # is skipped; with no other source the capacity is unknown.
        assert parse_kv_cache_capacity(CONFIG_INFO_LABEL_MISSING) is None

    def test_config_info_two_samples_takes_max(self) -> None:
        assert parse_kv_cache_capacity(CONFIG_INFO_TWO_MODELS) == 200000

    def test_config_info_non_numeric_label(self) -> None:
        assert parse_kv_cache_capacity(CONFIG_INFO_NON_NUMERIC) is None

    def test_config_info_non_positive_labels(self) -> None:
        # "0" and "-5" are not usable capacities.
        assert parse_kv_cache_capacity(CONFIG_INFO_NON_POSITIVE) is None

    def test_standalone_gauge_preferred_over_label(self) -> None:
        # Both sources present: the standalone gauge wins.
        assert parse_kv_cache_capacity(CONFIG_INFO_BOTH_SOURCES) == 100000

    def test_fallback_when_standalone_absent(self) -> None:
        # Usage gauge + config-info label, no standalone gauge.
        assert parse_kv_cache_capacity(CONFIG_INFO_USAGE_FALLBACK) == 77777


class TestFetchUsageByModel:
    async def test_200_with_metric_returns_mapping(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=MULTI_SERIES))
        async with httpx.AsyncClient(transport=transport) as client:
            result = await fetch_usage_by_model(client, "http://vllm:9000/metrics")
            assert result == {
                "llama-3-8b": pytest.approx(0.42),
                "qwen-2-7b": pytest.approx(0.87),
            }

    async def test_200_without_metric_returns_none(self) -> None:
        text = "vllm:num_requests_running 3\n"
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await fetch_usage_by_model(client, "http://vllm:9000/metrics") is None

    async def test_500_returns_none(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(500, text="boom"))
        async with httpx.AsyncClient(transport=transport) as client:
            assert await fetch_usage_by_model(client, "http://vllm:9000/metrics") is None

    async def test_connection_error_propagates(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.ConnectError):
                await fetch_usage_by_model(client, "http://vllm:9000/metrics")


class TestFetchMetrics:
    async def test_200_with_both_gauges_populates_all(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=CAP_USAGE_BOTH))
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await fetch_metrics(client, "http://vllm:9000/metrics")
        assert sample.observed is True
        assert sample.usage_frac == pytest.approx(0.42)
        assert sample.by_model == {"llama-3-8b": pytest.approx(0.42)}
        assert sample.capacity_tokens == 100000

    async def test_200_without_capacity(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=USAGE_ONLY))
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await fetch_metrics(client, "http://vllm:9000/metrics")
        assert sample.observed is True
        assert sample.usage_frac == pytest.approx(0.42)
        assert sample.capacity_tokens is None

    async def test_200_without_usage(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=CAP_ONLY))
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await fetch_metrics(client, "http://vllm:9000/metrics")
        assert sample.observed is True
        assert sample.usage_frac is None
        assert sample.by_model is None
        assert sample.capacity_tokens == 100000

    async def test_200_capacity_from_config_info_label(self) -> None:
        # Regression test for the [major] defect: a live vLLM body that carries
        # the usage gauge and the kv_cache_size_tokens *label* on
        # vllm:cache_config_info (no standalone gauge) must yield a capacity.
        transport = httpx.MockTransport(
            lambda _req: httpx.Response(200, text=CONFIG_INFO_USAGE_FALLBACK)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await fetch_metrics(client, "http://vllm:9000/metrics")
        assert sample.observed is True
        assert sample.usage_frac == pytest.approx(0.42)
        assert sample.capacity_tokens == 77777

    async def test_500_returns_unobserved_sample(self) -> None:
        transport = httpx.MockTransport(lambda _req: httpx.Response(500, text="boom"))
        async with httpx.AsyncClient(transport=transport) as client:
            sample = await fetch_metrics(client, "http://vllm:9000/metrics")
        assert sample == MetricsSample(False, None, None, None)

    async def test_connection_error_propagates(self) -> None:
        def handler(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.ConnectError):
                await fetch_metrics(client, "http://vllm:9000/metrics")


class TestCapacityCache:
    def test_initial_state(self) -> None:
        cache = CapacityCache()
        assert cache.value() is None
        assert cache.age() is None

    def test_update_stores_value_and_timestamp(self) -> None:
        cache = CapacityCache()
        cache.update(100000, now=100.0)
        assert cache.value() == 100000
        assert cache.age(now=103.0) == pytest.approx(3.0)

    def test_update_none_keeps_value_and_timestamp(self) -> None:
        cache = CapacityCache()
        cache.update(100000, now=100.0)
        cache.update(None, now=200.0)
        assert cache.value() == 100000
        # Failed fetch must NOT refresh the timestamp: age is still 100.0-based.
        assert cache.age(now=200.0) == pytest.approx(100.0)

    def test_update_overwrites_value_and_timestamp(self) -> None:
        cache = CapacityCache()
        cache.update(100000, now=100.0)
        cache.update(250000, now=150.0)
        assert cache.value() == 250000
        assert cache.age(now=150.0) == pytest.approx(0.0)


class TestKvRemaining:
    def test_initial_value_is_none(self) -> None:
        counter = KvRemaining()
        assert counter.value() is None

    def test_reanchor_sets_value(self) -> None:
        counter = KvRemaining()
        counter.reanchor(35000)
        assert counter.value() == 35000

    def test_subtract_decrements(self) -> None:
        counter = KvRemaining()
        counter.reanchor(35000)
        counter.subtract(25000)
        assert counter.value() == 10000

    def test_subtract_when_none_is_noop(self) -> None:
        counter = KvRemaining()
        counter.subtract(25000)
        assert counter.value() is None

    def test_subtract_can_go_negative(self) -> None:
        # Over-committed signal: subtraction is not clamped.
        counter = KvRemaining()
        counter.reanchor(10000)
        counter.subtract(31250)
        assert counter.value() == -21250

    def test_reanchor_resets_value(self) -> None:
        counter = KvRemaining()
        counter.reanchor(35000)
        counter.subtract(25000)
        counter.reanchor(10000)
        assert counter.value() == 10000

    def test_add_increments_value(self) -> None:
        counter = KvRemaining()
        counter.reanchor(100)
        counter.add(25)
        assert counter.value() == 125

    def test_add_when_none_is_noop(self) -> None:
        counter = KvRemaining()
        counter.add(25)
        assert counter.value() is None

    def test_add_subtract_round_trip(self) -> None:
        counter = KvRemaining()
        counter.reanchor(100)
        counter.subtract(30)
        counter.add(30)
        assert counter.value() == 100

    def test_add_on_negative_value(self) -> None:
        counter = KvRemaining()
        counter.reanchor(10)
        counter.subtract(50)
        assert counter.value() == -40
        counter.add(50)
        assert counter.value() == 10
