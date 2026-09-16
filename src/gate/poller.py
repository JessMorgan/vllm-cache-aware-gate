"""Background poller that keeps the KV-cache metrics (and model discovery) fresh.

Thin edge: a single asyncio task that repeatedly fetches vLLM's ``/metrics``
endpoint with ONE ``GET`` per tick (:func:`~gate.metrics.fetch_metrics`) and,
on an observed (HTTP 200) body:

- updates the :class:`~gate.metrics.MetricsCache` with the usage fraction and
  per-model breakdown,
- re-anchors the :class:`~gate.metrics.KvRemaining` counter to
  ``anchor_remaining_tokens(capacity, target_frac, usage_frac)`` (when a
  counter and target are wired in), and
- updates the :class:`~gate.metrics.CapacityCache` with the capacity gauge.

When ``model_url``, ``registry``, and ``backend_name`` are all wired in, the
tick also performs **model discovery** (:func:`fetch_models`): the first tick
fetches immediately, then again whenever at least ``model_refresh_s``
(monotonic) has elapsed since the last attempt. When ``explicit_models`` is
non-empty it is known operator data and is synced via
``registry.sync(backend_name, owned)`` **unconditionally** — on every
discovery tick, success or failure of the fetch — so explicit models route
regardless of ``/v1/models`` reachability. Otherwise a 200 + parseable list
adopts the discovered served set; a failed fetch or an unparseable body keeps
the last known map (discovery is auxiliary and never fatal) and logs a
warning only on a state change (first failure after a success, and the
recovery), not on every tick.

**Fail open:** a transport error (``httpx.HTTPError``) is logged and **all**
state is kept — caches AND counter untouched — so fail-open behavior is driven
purely by staleness in the app layer, never by a failed fetch. A non-200
response is likewise non-fatal (``observed`` is ``False`` and all state is
kept).

**Capacity-missing (per-backend fail-open):** an observed (HTTP 200) body that
lacks a usable KV-cache capacity (the ``vllm:kv_cache_size_tokens`` gauge or
the ``kv_cache_size_tokens`` label on ``vllm:cache_config_info``) cannot
anchor the counter. This is NO LONGER fatal: the poller logs an error,
**unanchors the counter** — a previously anchored counter is reset to the
never-anchored state so the backend's autoconfig layer fails open instead of
deciding off a stale anchor — and invokes the ``capacity_unavailable``
callback (idempotent by design — the app wires it to flip its per-backend
alert flag, surfaced by the ``gate_backend_capacity_unavailable`` gauge). The
loop CONTINUES; the ONLY way the task ends is cancellation.

The loop runs until the task is cancelled; cancellation is logged at debug and
re-raised so the awaiting caller observes it.

Invariants (see AGENTS.md "Known gotchas" #1 and docs/plans/multi-backend.md
§2.3):

- A metrics outage must never block traffic. A failed fetch keeps all state;
  the value then ages out and the app layer fails open via
  :meth:`MetricsCache.is_stale`.
- The loop never crashes on a transport error or on a capacity-missing body.
  The ONLY way the task ends is cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import httpx

from gate.metrics import (
    CapacityCache,
    KvRemaining,
    MetricsCache,
    fetch_metrics,
)
from gate.models import ModelRegistry, parse_v1_models
from gate.router import anchor_remaining_tokens

log = logging.getLogger("gate.poller")


async def fetch_models(client: httpx.AsyncClient, url: str) -> list[str] | None:
    """Fetch ``url`` (a ``/v1/models`` endpoint) and parse the model id list.

    Returns the ordered model id list on a 200 response, or ``None`` on a
    non-200 response or an unparseable body. No logging here — the poller
    logs. Transport errors (``httpx.HTTPError``) propagate to the caller (same
    contract as :func:`gate.metrics.fetch_metrics`).
    """
    resp = await client.get(url)
    if resp.status_code != 200:
        return None
    return parse_v1_models(resp.text)


async def run_poller(
    client: httpx.AsyncClient,
    cache: MetricsCache,
    url: str,
    interval_s: float,
    *,
    counter: KvRemaining | None = None,
    target_frac: float | None = None,
    capacity_cache: CapacityCache | None = None,
    model_url: str | None = None,
    model_refresh_s: float = 30.0,
    registry: ModelRegistry | None = None,
    backend_name: str | None = None,
    explicit_models: tuple[str, ...] | None = None,
    capacity_unavailable: Callable[[], None] | None = None,
) -> None:
    """Poll ``url`` every ``interval_s`` seconds, updating all wired state.

    Runs until the task is cancelled. On each iteration one ``GET`` is made
    via :func:`fetch_metrics`:

    - ``httpx.HTTPError`` (transport failure) -> log a warning and keep ALL
      state (caches and counter untouched; do NOT clear anything — fail-open
      is driven by staleness).
     - ``sample.observed`` (HTTP 200):
       - ``capacity_tokens is None`` -> log an error, **unanchor the counter**
         (``counter.unanchor()`` — a previously anchored counter is reset to
         the never-anchored state, so the autoconfig layer fails open instead
         of deciding off a stale anchor), and invoke the
         ``capacity_unavailable`` callback (when wired) so the app's
         per-backend alert flag is set. **Not fatal**: the loop continues.
      - ``usage_frac is not None`` -> ``cache.update(usage_frac)``; a
        truthy ``by_model`` -> ``cache.update_by_model(by_model)``.
      - ``counter`` and ``target_frac`` and ``usage_frac`` and
        ``capacity_tokens`` all present -> ``counter.reanchor(
        anchor_remaining_tokens(capacity_tokens, target_frac, usage_frac))``.
      - ``capacity_cache`` present -> ``capacity_cache.update(
        capacity_tokens)``.
      - A body with capacity but no usage -> no reanchor (nothing to anchor
        against), not fatal; the last anchor persists.
    - non-observed (non-200) -> keep all state (no-op), never fatal.

    When ``model_url``, ``registry``, and ``backend_name`` are all given, the
    tick also does model discovery: the first tick fetches immediately, then
    again whenever at least ``model_refresh_s`` monotonic seconds have passed
    since the last attempt. A non-empty ``explicit_models`` tuple is known
    operator data and is synced via ``registry.sync(backend_name,
    set(explicit_models))`` **unconditionally** — on every discovery tick,
    success or failure of the fetch — so explicit models route regardless of
    ``/v1/models`` reachability. Otherwise (auto-adopt; an empty tuple is
    treated as absent — the config semantics where an empty ``models:`` means
    auto-adopt) a 200 + parseable list calls ``registry.sync(backend_name,
    set(discovered))``. A failed fetch or ``None`` parse keeps the last known
    map (never fatal) and logs a warning only on a state change (first
    failure after a success, and the recovery) — not on every tick.

    On ``asyncio.CancelledError`` the loop exits cleanly: it is logged at
    debug and allowed to propagate so the awaiting caller observes the
    cancellation.
    """
    last_discovery: float | None = None  # monotonic time of the last discovery attempt
    discovery_ok: bool = True  # was the last discovery attempt a 200? (warning on state change)
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
                            "autoconfig requires a usable KV-cache capacity (the "
                            "vllm:kv_cache_size_tokens gauge or the "
                            "kv_cache_size_tokens label on vllm:cache_config_info) "
                            "but the observed body at %s does not provide it; the "
                            "backend's autoconfig layer will fail open and the "
                            "gate_backend_capacity_unavailable gauge will be set",
                            url,
                        )
                        if capacity_unavailable is not None:
                            capacity_unavailable()
                        if counter is not None:
                            # Unanchor so a stale anchor from an earlier good
                            # poll cannot drive a real (fail-closed) decision:
                            # the autoconfig layer fails open while unanchored.
                            counter.unanchor()
                    if sample.usage_frac is not None:
                        cache.update(sample.usage_frac)
                    if sample.by_model:
                        cache.update_by_model(sample.by_model)
                    if (
                        counter is not None
                        and target_frac is not None
                        and sample.usage_frac is not None
                        and sample.capacity_tokens is not None
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
            if model_url is not None and registry is not None and backend_name is not None:
                now = time.monotonic()
                if last_discovery is None or now - last_discovery >= model_refresh_s:
                    last_discovery = now
                    try:
                        discovered = await fetch_models(client, model_url)
                    except httpx.HTTPError:
                        discovered = None
                    if explicit_models:
                        # Explicit models are known operator data: they define
                        # the owned set and must route regardless of
                        # /v1/models reachability, so sync unconditionally.
                        registry.sync(backend_name, set(explicit_models))
                    elif discovered is not None:
                        registry.sync(backend_name, set(discovered))
                    # The fetch state (and its warning/recovery logging)
                    # tracks the /v1/models fetch itself, independently of
                    # whether a sync ran this tick.
                    if discovered is not None:
                        if not discovery_ok:
                            log.info(
                                "model discovery recovered for backend %s at %s",
                                backend_name,
                                model_url,
                            )
                        discovery_ok = True
                    else:
                        # Keep the last known map (fail-open); warn only on the
                        # state change, not on every failed tick.
                        if discovery_ok:
                            log.warning(
                                "model discovery failed for backend %s at %s; "
                                "keeping last known model map",
                                backend_name,
                                model_url,
                            )
                        discovery_ok = False
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        log.debug("poller cancelled")
        raise
