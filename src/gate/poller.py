"""Background poller that keeps the KV-cache usage cache fresh.

Thin edge: a single asyncio task that repeatedly fetches vLLM's ``/metrics``
endpoint and updates the :class:`~gate.metrics.MetricsCache`. It **fails
open**: a transport error is logged and the last good value is kept (the
cache is NOT cleared), so fail-open behavior is driven purely by staleness in
the app layer, never by a failed fetch. The loop runs until the task is
cancelled.

Invariants (see AGENTS.md "Known gotchas" #1):

- A metrics outage must never block traffic. A failed fetch keeps the last
  value; the value then ages out and the app layer fails open via
  :meth:`MetricsCache.is_stale`.
- The loop never crashes on a transport error.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from gate.metrics import MetricsCache, fetch_usage

log = logging.getLogger("gate.poller")


async def run_poller(
    client: httpx.AsyncClient,
    cache: MetricsCache,
    url: str,
    interval_s: float,
) -> None:
    """Poll ``url`` every ``interval_s`` seconds, updating ``cache``.

    Runs until the task is cancelled. On each iteration:

    - ``fetch_usage`` succeeds -> ``cache.update(value)``. A ``None`` value
      (e.g. a non-200 response) is a no-op that keeps the last good value and
      does not refresh its timestamp.
    - ``fetch_usage`` raises ``httpx.HTTPError`` -> log a warning and keep the
      last value (do NOT clear the cache; fail-open is driven by staleness).

    On ``asyncio.CancelledError`` the loop exits cleanly: it is logged at
    debug and allowed to propagate so the awaiting caller observes the
    cancellation.
    """
    try:
        while True:
            try:
                value = await fetch_usage(client, url)
                cache.update(value)
            except httpx.HTTPError:
                log.warning("metrics fetch failed for %s; keeping last value", url)
            else:
                log.debug("polled %s -> %s", url, value)
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        log.debug("poller cancelled")
        raise
