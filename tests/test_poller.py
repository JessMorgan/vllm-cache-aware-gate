"""Tests for gate.poller: the background metrics poller + model discovery.

Covers the load-bearing invariants: a successful poll updates the cache, a
transport error keeps all state (caches AND counter — fail-open is driven by
staleness, never by clearing), a non-200 response keeps all state, an
observed (HTTP 200) body missing ``vllm:kv_cache_size_tokens`` is NO LONGER
fatal (no exception, the ``capacity_unavailable`` callback is invoked, the
counter is unanchored — including a previously anchored one — and the task
keeps running), a good poll re-anchors
the ``KvRemaining`` counter to ``anchor_remaining_tokens(...)`` exactly,
model discovery fires on the first tick and then on the ``model_refresh_s``
cadence (explicit ``models`` win over the discovered set; failures keep the
last known map and warn only on state change), and cancellation exits cleanly
with only ``CancelledError`` escaping.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gate.metrics import CapacityCache, KvRemaining, MetricsCache
from gate.models import ModelRegistry
from gate.poller import fetch_models, run_poller
from gate.router import anchor_remaining_tokens

#: Usage-only body (no capacity gauge) — the per-backend fail-open fixture.
METRICS_055 = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.55\n"
)

CAPACITY_BLOCK = (
    "# HELP vllm:kv_cache_size_tokens Total KV cache capacity in tokens.\n"
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    "vllm:kv_cache_size_tokens 100000\n"
)

#: Body carrying BOTH the usage and the capacity gauge.
METRICS_BOTH = METRICS_055 + CAPACITY_BLOCK

METRICS_MODEL_A_055 = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="A"} 0.55\n' + CAPACITY_BLOCK
)

#: Capacity gauge present, usage gauge absent.
METRICS_CAPACITY_ONLY = CAPACITY_BLOCK

MODELS_BODY = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]})


def _models_body(*ids: str) -> str:
    return json.dumps({"data": [{"id": i} for i in ids]})


async def test_happy_path_updates_cache() -> None:
    """A successful poll stores the parsed value in the cache."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_BOTH))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.value() == pytest.approx(0.55)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_error_keeps_last_value() -> None:
    """A transport error must NOT clear the cache; the last good value stays."""
    cache = MetricsCache()
    cache.update(0.5)

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.value() == pytest.approx(0.5)  # not cleared
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_non_200_keeps_last_value() -> None:
    """A non-200 response is non-observed; the last value stays."""
    cache = MetricsCache()
    cache.update(0.5)
    transport = httpx.MockTransport(lambda _req: httpx.Response(500, text="boom"))
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.value() == pytest.approx(0.5)  # not cleared
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_cancellation_is_clean() -> None:
    """Cancelling the task raises only CancelledError; nothing else escapes."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_BOTH))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_successful_poll_stores_by_model() -> None:
    """A successful poll stores the per-model mapping and the overall max."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_MODEL_A_055))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.by_model() == {"A": pytest.approx(0.55)}
        assert cache.value() == pytest.approx(0.55)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_error_keeps_by_model_unchanged() -> None:
    """A transport error must NOT clear the per-model mapping (fail-open)."""
    cache = MetricsCache()
    cache.update_by_model({"A": 0.5})

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.by_model() == {"A": pytest.approx(0.5)}  # not cleared
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_observed_without_capacity_is_not_fatal() -> None:
    """An observed (200) body missing capacity: no exception, callback fired, loop continues.

    The counter stays unanchored (autoconfig fails open), the
    ``capacity_unavailable`` callback is invoked, and the task keeps running.
    """
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_055))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        counter = KvRemaining()
        calls = 0

        def on_unavailable() -> None:
            nonlocal calls
            calls += 1

        task = asyncio.create_task(
            run_poller(
                client,
                cache,
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                capacity_unavailable=on_unavailable,
            )
        )
        await asyncio.sleep(0.05)  # several capacity-missing ticks; no raise
        assert not task.done()  # the loop is still running
        assert calls >= 1  # the alert callback was invoked
        assert counter.value() is None  # never anchored (autoconfig fails open)
        assert cache.value() == pytest.approx(0.55)  # usage still recorded
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_capacity_missing_unanchors_previously_anchored_counter() -> None:
    """A capacity-missing observed body unanchors a PREVIOUSLY anchored counter.

    A stale anchor from an earlier good poll must not survive the body losing
    the capacity gauge: ``decision_auto`` would otherwise make a real
    (fail-closed) decision off stale data. After the flip, the counter is
    unanchored (autoconfig fails open) and the task keeps running.
    """
    body = {"text": METRICS_BOTH}  # mutable flag: HAS capacity -> capacity-missing

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body["text"])

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        counter = KvRemaining()
        task = asyncio.create_task(
            run_poller(
                client,
                cache,
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
            )
        )
        await asyncio.sleep(0.05)  # several good ticks
        assert counter.value() is not None  # anchored by the capacity-carrying body
        body["text"] = METRICS_055  # flip: usage present, capacity gauge gone
        await asyncio.sleep(0.05)  # several capacity-missing ticks
        assert counter.value() is None  # unanchored -> autoconfig fails open
        assert not task.done()  # the loop is still running
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_transport_error_keeps_caches_and_counter() -> None:
    """A transport error keeps ALL state: caches AND the counter are untouched."""
    cache = MetricsCache()
    cache.update_by_model({"A": 0.5})
    counter = KvRemaining()
    counter.reanchor(1000)
    capacity_cache = CapacityCache()

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            run_poller(
                client,
                cache,
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                capacity_cache=capacity_cache,
            )
        )
        await asyncio.sleep(0.05)  # several failed ticks; no raise
        assert cache.value() == pytest.approx(0.5)
        assert cache.by_model() == {"A": pytest.approx(0.5)}
        assert counter.value() == 1000  # not re-anchored, not cleared
        assert capacity_cache.value() is None  # never updated
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_good_poll_reanchors_counter_and_updates_caches() -> None:
    """A good poll re-anchors the counter exactly and updates both caches."""
    capacity = 100000
    target_frac = 0.85
    usage = 0.55
    expected = anchor_remaining_tokens(capacity, target_frac, usage)

    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_BOTH))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        counter = KvRemaining()
        capacity_cache = CapacityCache()
        task = asyncio.create_task(
            run_poller(
                client,
                cache,
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=target_frac,
                capacity_cache=capacity_cache,
            )
        )
        await asyncio.sleep(0.05)
        assert counter.value() == expected
        assert cache.value() == pytest.approx(usage)
        assert cache.by_model() == {"default": pytest.approx(usage)}
        assert capacity_cache.value() == capacity
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_capacity_without_usage_no_reanchor_no_raise() -> None:
    """Capacity present but usage absent: no reanchor, no raise; capacity cached."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_CAPACITY_ONLY))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        counter = KvRemaining()
        counter.reanchor(1000)
        capacity_cache = CapacityCache()
        task = asyncio.create_task(
            run_poller(
                client,
                cache,
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                capacity_cache=capacity_cache,
            )
        )
        await asyncio.sleep(0.05)
        assert counter.value() == 1000  # last anchor persists
        assert cache.value() is None  # no usage to store
        assert capacity_cache.value() == 100000
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_legacy_call_without_counter_updates_caches_only() -> None:
    """The 4-positional-arg call (no counter) still works: caches updated, no raise."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_BOTH))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        await asyncio.sleep(0.05)
        assert cache.value() == pytest.approx(0.55)
        assert cache.by_model() == {"default": pytest.approx(0.55)}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# on_reanchor callback
# ---------------------------------------------------------------------------


async def test_on_reanchor_fires_once_per_reanchoring_tick() -> None:
    """A good poll (usage + capacity) fires the callback exactly once per tick."""
    metrics_calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal metrics_calls
        metrics_calls += 1
        return httpx.Response(200, text=METRICS_BOTH)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        counter = KvRemaining()
        reanchor_calls = 0

        def on_reanchor() -> None:
            nonlocal reanchor_calls
            reanchor_calls += 1

        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                on_reanchor=on_reanchor,
            )
        )
        await asyncio.sleep(0.05)  # several good ticks
        assert metrics_calls >= 3  # the loop actually ticked
        assert reanchor_calls == metrics_calls  # exactly one callback per reanchor
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_on_reanchor_not_fired_without_capacity() -> None:
    """A poll with usage but no capacity does not reanchor: no callback."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_055))
    async with httpx.AsyncClient(transport=transport) as client:
        counter = KvRemaining()
        reanchor_calls = 0

        def on_reanchor() -> None:
            nonlocal reanchor_calls
            reanchor_calls += 1

        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                on_reanchor=on_reanchor,
            )
        )
        await asyncio.sleep(0.05)  # several capacity-missing ticks
        assert reanchor_calls == 0
        assert counter.value() is None  # never anchored
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_on_reanchor_not_fired_on_transport_error() -> None:
    """A transport-error tick does not reanchor: no callback."""
    counter = KvRemaining()
    counter.reanchor(1000)
    reanchor_calls = 0

    def on_reanchor() -> None:
        nonlocal reanchor_calls
        reanchor_calls += 1

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=0.85,
                on_reanchor=on_reanchor,
            )
        )
        await asyncio.sleep(0.05)  # several failed ticks
        assert reanchor_calls == 0
        assert counter.value() == 1000  # last anchor persists
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_on_reanchor_fires_synchronously_with_reanchor() -> None:
    """At the moment the callback runs, the counter is already re-anchored.

    The callback records ``counter.value()``; every recorded value must equal
    the freshly anchored value — proving no await (and no re-entrant tick)
    separates the re-anchor from the callback.
    """
    capacity = 100000
    target_frac = 0.85
    usage = 0.55
    expected = anchor_remaining_tokens(capacity, target_frac, usage)

    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_BOTH))
    async with httpx.AsyncClient(transport=transport) as client:
        counter = KvRemaining()
        observed: list[int | None] = []

        def on_reanchor() -> None:
            observed.append(counter.value())

        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                counter=counter,
                target_frac=target_frac,
                on_reanchor=on_reanchor,
            )
        )
        await asyncio.sleep(0.05)  # several re-anchoring ticks
        assert len(observed) >= 3
        assert all(v == expected for v in observed)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# routing order ⊆ serving check
# ---------------------------------------------------------------------------


async def test_routing_order_dead_candidate_warns_once_and_keeps_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A backend named in a model's order but NOT serving it (dead candidate) warns once.

    The warning fires on the first successful discovery tick, does NOT repeat
    on later ticks, and is not fatal: the poller keeps running and syncing.
    """
    # "b" serves only m2; m1's order names "b" but "b" does not serve m1.
    transport = _routing_transport(METRICS_BOTH, _models_body("m2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,  # several discovery ticks over the window
                registry=registry,
                backend_name="b",
                routing_order={"m1": ("b", "other")},  # dead candidate for m1
            )
        )
        with caplog.at_level("WARNING", logger="gate.poller"):
            await asyncio.sleep(0.05)  # first discovery tick
            violations = [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
            assert len(violations) == 1
            assert "m1" in violations[0].getMessage()
            assert "b" in violations[0].getMessage()
            assert "dead candidate" in violations[0].getMessage()
            # The poller kept running: discovery synced the map...
            assert registry.resolve("m2") == "b"
            assert registry.resolve("m1") is None  # "b" does not serve m1
            # ...and a second discovery tick does NOT re-log the same pair.
            caplog.clear()
            await asyncio.sleep(0.1)  # at least one more discovery tick
            assert not [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
            assert not task.done()  # still running (logged, not fatal)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_routing_order_dead_candidate_applies_to_explicit_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With explicit ``models``, the owned set is the explicit list — same check."""
    transport = _routing_transport(METRICS_BOTH, _models_body("discovered1"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=30.0,
                registry=registry,
                backend_name="b",
                explicit_models=("explicit1",),  # owned set = {explicit1}
                routing_order={"ghost": ("b",)},  # "b" named for "ghost" but does not serve it
            )
        )
        with caplog.at_level("WARNING", logger="gate.poller"):
            await asyncio.sleep(0.05)
            violations = [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
            assert len(violations) == 1
            assert "ghost" in violations[0].getMessage()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_routing_order_serving_backend_in_order_warns_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A backend that serves a model and is in its order: no warning."""
    transport = _routing_transport(METRICS_BOTH, _models_body("m1", "m2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
                routing_order={"m1": ("b", "other")},  # "b" serves m1 and is in the order
            )
        )
        with caplog.at_level("WARNING", logger="gate.poller"):
            await asyncio.sleep(0.1)  # several discovery ticks
            assert not [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
            assert registry.resolve("m1") == "b"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_routing_order_serving_backend_scoped_out_warns_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A serving backend deliberately scoped OUT of a model's order: no warning.

    The order defines the candidate set; a serving backend absent from the
    order is a deliberate scoping choice, not a violation.
    """
    transport = _routing_transport(METRICS_BOTH, _models_body("m1", "m2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
                routing_order={"m1": ("other",)},  # "b" serves m1 but is not in the order
            )
        )
        with caplog.at_level("WARNING", logger="gate.poller"):
            await asyncio.sleep(0.1)  # several discovery ticks
            assert not [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
            assert registry.resolve("m1") == "b"  # still serves it
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_routing_order_none_is_default_no_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With ``routing_order`` unset (default), no check runs and no warnings appear."""
    transport = _routing_transport(METRICS_BOTH, _models_body("m1", "m2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
            )
        )
        with caplog.at_level("WARNING", logger="gate.poller"):
            await asyncio.sleep(0.1)
            assert not [
                r
                for r in caplog.records
                if r.levelname == "WARNING" and "routing entry for model" in r.getMessage()
            ]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# fetch_models
# ---------------------------------------------------------------------------


async def test_fetch_models_200_valid_returns_list() -> None:
    """A 200 with a valid body yields the ordered id list."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=MODELS_BODY))
    async with httpx.AsyncClient(transport=transport) as client:
        assert await fetch_models(client, "http://x/v1/models") == ["m1", "m2"]


async def test_fetch_models_200_malformed_returns_none() -> None:
    """A 200 with a malformed body yields None (parse failure)."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text="not json"))
    async with httpx.AsyncClient(transport=transport) as client:
        assert await fetch_models(client, "http://x/v1/models") is None


async def test_fetch_models_non_200_returns_none() -> None:
    """A non-200 response yields None."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(503, text="down"))
    async with httpx.AsyncClient(transport=transport) as client:
        assert await fetch_models(client, "http://x/v1/models") is None


async def test_fetch_models_transport_error_propagates() -> None:
    """A transport error propagates (same contract as fetch_metrics)."""

    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_req)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.ConnectError):
            await fetch_models(client, "http://x/v1/models")


# ---------------------------------------------------------------------------
# model discovery
# ---------------------------------------------------------------------------


def _routing_transport(metrics_text: str, models_text: str) -> httpx.MockTransport:
    """A mock transport serving both the /metrics and /v1/models endpoints."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/v1/models"):
            return httpx.Response(200, text=models_text)
        return httpx.Response(200, text=metrics_text)

    return httpx.MockTransport(handler)


async def test_discovery_first_tick_fetches_immediately() -> None:
    """With discovery wired, the FIRST tick fetches /v1/models immediately."""
    transport = _routing_transport(METRICS_BOTH, _models_body("m1", "m2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=30.0,
                registry=registry,
                backend_name="b",
            )
        )
        await asyncio.sleep(0.05)  # several ticks, but only the first may discover
        assert registry.resolve("m1") == "b"
        assert registry.resolve("m2") == "b"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_discovery_explicit_models_win_over_discovered() -> None:
    """An explicit ``models`` list defines the owned set, not the discovered one."""
    transport = _routing_transport(METRICS_BOTH, _models_body("discovered1", "discovered2"))
    async with httpx.AsyncClient(transport=transport) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=30.0,
                registry=registry,
                backend_name="b",
                explicit_models=("explicit1", "explicit2"),
            )
        )
        await asyncio.sleep(0.05)
        assert registry.resolve("explicit1") == "b"
        assert registry.resolve("explicit2") == "b"
        assert registry.resolve("discovered1") is None  # not adopted
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_discovery_respects_cadence() -> None:
    """Discovery fires on the first tick and refetches on the cadence — not every tick.

    ``model_refresh_s`` is set to several ``interval_s`` so that, over the
    observed window, a cadence-respecting poller issues fewer ``/v1/models``
    fetches than ``/metrics`` ticks while still refetching more than once
    (proving it is not a one-shot first-tick-only fetch).
    """
    metrics_calls = 0
    model_calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal metrics_calls, model_calls
        if req.url.path.endswith("/v1/models"):
            model_calls += 1
            return httpx.Response(200, text=_models_body("m1"))
        metrics_calls += 1
        return httpx.Response(200, text=METRICS_BOTH)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,  # 5 ticks between discoveries
                registry=registry,
                backend_name="b",
            )
        )
        await asyncio.sleep(0.25)  # ~25 ticks
        assert metrics_calls >= 5  # the loop actually ticked
        assert model_calls >= 2  # first tick + at least one cadence-driven refetch
        assert model_calls < metrics_calls  # not every tick (cadence respected)
        assert registry.resolve("m1") == "b"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_discovery_failure_keeps_last_map_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed discovery keeps the last known map and warns only on state change."""
    failing = True

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/v1/models"):
            if failing:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, text=_models_body("m2"))
        return httpx.Response(200, text=METRICS_BOTH)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        registry.sync("b", {"m1"})  # last known map
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
            )
        )
        with caplog.at_level("INFO", logger="gate.poller"):
            await asyncio.sleep(0.15)  # several failed discovery ticks
            assert registry.resolve("m1") == "b"  # last map kept
            assert registry.resolve("m2") is None  # nothing adopted
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warnings) == 1  # one warning per failure streak
            # Recovery: the map is replaced and an info (not a new warning) is
            # logged. Clear first so the streak's warning is not counted again.
            caplog.clear()
            failing = False
            await asyncio.sleep(0.1)
            assert registry.resolve("m2") == "b"
            assert registry.resolve("m1") is None
            assert not [r for r in caplog.records if r.levelname == "WARNING"]
            assert any(
                r.levelname == "INFO" and "recovered" in r.getMessage() for r in caplog.records
            )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_discovery_transport_error_keeps_last_map() -> None:
    """A transport error on /v1/models keeps the last known map; never fatal."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/v1/models"):
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, text=METRICS_BOTH)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        registry.sync("b", {"m1"})
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
            )
        )
        await asyncio.sleep(0.1)
        assert not task.done()  # discovery failure is never fatal
        assert registry.resolve("m1") == "b"  # last map kept
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_explicit_models_registered_when_models_fetch_fails() -> None:
    """Explicit models are synced unconditionally — even when /v1/models fails.

    Explicit ``models`` are known operator data (they define the owned set),
    so they must route regardless of the discovery fetch's reachability.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/v1/models"):
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text=METRICS_BOTH)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        registry = ModelRegistry()
        registry.register_backend("b", is_default=True)
        task = asyncio.create_task(
            run_poller(
                client,
                MetricsCache(),
                "http://x/metrics",
                0.01,
                model_url="http://x/v1/models",
                model_refresh_s=0.05,
                registry=registry,
                backend_name="b",
                explicit_models=("e1", "e2"),
            )
        )
        await asyncio.sleep(0.1)  # several failed discovery ticks
        assert not task.done()  # discovery failure is never fatal
        assert registry.resolve("e1") == "b"  # explicit models route anyway
        assert registry.resolve("e2") == "b"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
