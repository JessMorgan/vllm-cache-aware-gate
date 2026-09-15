"""Tests for gate.poller: the background metrics poller.

Covers the load-bearing invariants: a successful poll updates the cache, a
transport error keeps all state (caches AND counter — fail-open is driven by
staleness, never by clearing), a non-200 response keeps all state, an
observed (HTTP 200) body missing ``vllm:kv_cache_size_tokens`` raises
``CapacityUnavailableError`` unconditionally, a good poll re-anchors the
``KvRemaining`` counter to ``anchor_remaining_tokens(...)`` exactly, and
cancellation exits cleanly with only ``CancelledError`` escaping.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gate.metrics import CapacityCache, CapacityUnavailableError, KvRemaining, MetricsCache
from gate.poller import run_poller
from gate.router import anchor_remaining_tokens

#: Usage-only body (no capacity gauge) — the unconditional-fatal fixture.
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


async def test_observed_without_capacity_raises_unconditionally() -> None:
    """An observed (200) body missing the capacity gauge is fatal — no flag."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_055))
    async with httpx.AsyncClient(transport=transport) as client:
        cache = MetricsCache()
        task = asyncio.create_task(run_poller(client, cache, "http://x/metrics", 0.01))
        with pytest.raises(CapacityUnavailableError):
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
