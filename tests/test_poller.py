"""Tests for gate.poller: the background metrics poller.

Covers the load-bearing invariants: a successful poll updates the cache, a
transport error keeps the last value (fail-open is driven by staleness, never
by clearing the cache), a non-200 response keeps the last value, and
cancellation exits cleanly with only ``CancelledError`` escaping.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gate.metrics import MetricsCache
from gate.poller import run_poller

METRICS_055 = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.55\n"
)

METRICS_MODEL_A_055 = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    'vllm:kv_cache_usage_perc{model_name="A"} 0.55\n'
)


async def test_happy_path_updates_cache() -> None:
    """A successful poll stores the parsed value in the cache."""
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_055))
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
    """A non-200 response yields None from fetch_usage; the last value stays."""
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
    transport = httpx.MockTransport(lambda _req: httpx.Response(200, text=METRICS_055))
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
