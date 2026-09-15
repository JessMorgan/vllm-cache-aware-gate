"""Background poller that keeps the KV-cache metrics fresh.

Thin edge: a single asyncio task that repeatedly fetches vLLM's ``/metrics``
endpoint with ONE ``GET`` per tick (:func:`~gate.metrics.fetch_metrics`) and,
on an observed (HTTP 200) body:

- updates the :class:`~gate.metrics.MetricsCache` with the usage fraction and
  per-model breakdown,
- re-anchors the :class:`~gate.metrics.KvRemaining` counter to
  ``anchor_remaining_tokens(capacity, target_frac, usage_frac)`` (when a
  counter and target are wired in), and
- updates the :class:`~gate.metrics.CapacityCache` with the capacity gauge.

**Fail open:** a transport error (``httpx.HTTPError``) is logged and **all**
state is kept — caches AND counter untouched — so fail-open behavior is driven
purely by staleness in the app layer, never by a failed fetch. A non-200
response is likewise non-fatal (``observed`` is ``False`` and all state is
kept).

**Fail closed (unconditional):** an observed (HTTP 200) body that lacks a
usable ``vllm:kv_cache_size_tokens`` gauge cannot anchor the counter, so the
poller logs an error and raises
:class:`~gate.metrics.CapacityUnavailableError`. There is no mode or flag:
autoconfig is always on, and the app's fatal callback exits the process.

The loop runs until the task is cancelled; cancellation is logged at debug and
re-raised so the awaiting caller observes it.

Invariants (see AGENTS.md "Known gotchas" #1):

- A metrics outage must never block traffic. A failed fetch keeps all state;
  the value then ages out and the app layer fails open via
  :meth:`MetricsCache.is_stale`.
- The loop never crashes on a transport error. The ONLY fatal path is an
  observed 200 body missing the capacity gauge.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from gate.metrics import (
    CapacityCache,
    CapacityUnavailableError,
    KvRemaining,
    MetricsCache,
    fetch_metrics,
)
from gate.router import anchor_remaining_tokens

log = logging.getLogger("gate.poller")


async def run_poller(
    client: httpx.AsyncClient,
    cache: MetricsCache,
    url: str,
    interval_s: float,
    *,
    counter: KvRemaining | None = None,
    target_frac: float | None = None,
    capacity_cache: CapacityCache | None = None,
) -> None:
    """Poll ``url`` every ``interval_s`` seconds, updating all wired state.

    Runs until the task is cancelled. On each iteration one ``GET`` is made
    via :func:`fetch_metrics`:

    - ``httpx.HTTPError`` (transport failure) -> log a warning and keep ALL
      state (caches and counter untouched; do NOT clear anything — fail-open
      is driven by staleness).
    - ``sample.observed`` (HTTP 200):
      - ``capacity_tokens is None`` -> log an error and raise
        ``CapacityUnavailableError`` **unconditionally** (autoconfig is
        always on; a live vLLM without the capacity gauge cannot anchor the
        counter).
      - ``usage_frac is not None`` -> ``cache.update(usage_frac)``; a
        truthy ``by_model`` -> ``cache.update_by_model(by_model)``.
      - ``counter`` and ``target_frac`` and ``usage_frac`` all present ->
        ``counter.reanchor(anchor_remaining_tokens(capacity_tokens,
        target_frac, usage_frac))``.
      - ``capacity_cache`` present -> ``capacity_cache.update(
        capacity_tokens)``.
      - A body with capacity but no usage -> no reanchor (nothing to anchor
        against), not fatal; the last anchor persists.
    - non-observed (non-200) -> keep all state (no-op), never fatal.

    On ``asyncio.CancelledError`` the loop exits cleanly: it is logged at
    debug and allowed to propagate so the awaiting caller observes the
    cancellation.
    """
    try:
        while True:
            try:
                sample = await fetch_metrics(client, url)
            except httpx.HTTPError:
                log.warning("metrics fetch failed for %s; keeping all state", url)
            else:
                if sample.observed:
                    if sample.capacity_tokens is None:
                        log.error(
                            "autoconfig requires vllm:kv_cache_size_tokens but the "
                            "observed body at %s does not provide it; the process "
                            "will exit",
                            url,
                        )
                        raise CapacityUnavailableError(
                            f"observed /metrics body at {url} lacks vllm:kv_cache_size_tokens"
                        )
                    if sample.usage_frac is not None:
                        cache.update(sample.usage_frac)
                    if sample.by_model:
                        cache.update_by_model(sample.by_model)
                    if (
                        counter is not None
                        and target_frac is not None
                        and sample.usage_frac is not None
                    ):
                        counter.reanchor(
                            anchor_remaining_tokens(
                                sample.capacity_tokens, target_frac, sample.usage_frac
                            )
                        )
                    if capacity_cache is not None and sample.capacity_tokens is not None:
                        capacity_cache.update(sample.capacity_tokens)
                    log.debug(
                        "polled %s -> usage=%s capacity=%s",
                        url,
                        sample.usage_frac,
                        sample.capacity_tokens,
                    )
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        log.debug("poller cancelled")
        raise
