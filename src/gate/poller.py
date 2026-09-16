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

**Anchor-sequence callback:** when ``on_reanchor`` is wired, it is invoked
synchronously with every re-anchor — immediately after the
``counter.reanchor(...)`` call, with no ``await`` between them (the same
single-asyncio-loop contract as the re-anchor itself). The app wires it to
increment the backend's monotonic anchor sequence; because the callback runs
atomically with the re-anchor, a reader of the sequence plus a failover
charge-rollback ``add`` critical section is atomic with respect to
re-anchoring (docs/plans/multi-backend.md decision 12).

**Routing order ⊆ serving check:** when ``routing_order`` is wired, every
successful discovery tick verifies the decision-15 invariant that a model's
``routing:`` ``order`` names **only** backends that serve the model — for
this backend, the per-backend form is: if ``backend_name`` is named in a
model's ``order`` but the model is NOT in this backend's owned set, then this
backend is a **dead candidate** for that model (the failover walk will reach
it and fail on it) and a WARNING is logged **once per (model, backend) pair**
(never fatal — the walk fails open; decision 15). A serving backend that is
deliberately scoped OUT of a model's order is fine and stays silent (the
order defines the candidate set).

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


def _warn_routing_order_violations(
    routing_order: dict[str, tuple[str, ...]],
    owned: set[str],
    backend_name: str,
    warned: set[tuple[str, str]],
) -> None:
    """Log a WARNING for each model whose routing order names this backend as a dead candidate.

    Implements the decision-15 invariant that a model's ``routing:`` ``order``
    contains **only** backends that serve the model (order ⊆ serving): if
    ``backend_name`` is named in a model's ``order`` but the model is NOT in
    this backend's owned set, this backend is a dead candidate for that model
    — the failover walk will reach it and fail on it. A serving backend that
    is deliberately scoped out of a model's order is fine and stays silent.
    Logged, not fatal (the walk fails open); each (model, backend) pair is
    warned at most once (tracked in ``warned``).
    """
    for model in sorted(routing_order):
        order = routing_order[model]
        if backend_name not in order:
            continue  # this backend is not a candidate for model — not its concern
        if model in owned:
            continue  # this backend serves model — fine
        pair = (model, backend_name)
        if pair in warned:
            continue
        warned.add(pair)
        log.warning(
            "routing entry for model %r names backend %s in its order but this "
            "backend does not serve %r; it is a dead candidate the failover "
            "walk will reach and fail on",
            model,
            backend_name,
            model,
        )


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
    on_reanchor: Callable[[], None] | None = None,
    routing_order: dict[str, tuple[str, ...]] | None = None,
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
         anchor_remaining_tokens(capacity_tokens, target_frac, usage_frac))``,
         then ``on_reanchor()`` (when wired) is invoked **synchronously with
         the re-anchor** (no ``await`` between them) so the app's anchor
         sequence increments atomically with it.
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

    When ``routing_order`` is given, each successful discovery tick also
    verifies the routing order ⊆ serving invariant (decision 15): a model's
    ``order`` must name **only** backends that serve it. For this backend,
    the per-backend form is: if ``backend_name`` is named in a model's
    ``order`` but the model is NOT in this backend's owned set, this backend
    is a **dead candidate** for that model (the failover walk will reach it
    and fail on it) and a WARNING is logged **once per (model, backend) pair**
    — logged, not fatal (the walk fails open). A serving backend deliberately
    scoped out of a model's order is fine and stays silent. The check also
    applies when ``explicit_models`` is non-empty (the owned set is then the
    explicit list, and the check catches operator typos in the mnemonic).

    On ``asyncio.CancelledError`` the loop exits cleanly: it is logged at
    debug and allowed to propagate so the awaiting caller observes the
    cancellation.
    """
    last_discovery: float | None = None  # monotonic time of the last discovery attempt
    discovery_ok: bool = True  # was the last discovery attempt a 200? (warning on state change)
    # (model, backend) pairs for which the routing-order warning already fired
    # (decision 15: log once per pair, not on every tick).
    warned_routing_order: set[tuple[str, str]] = set()
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
                        if on_reanchor is not None:
                            # Synchronous with the re-anchor: no await between
                            # the two, so the app's anchor-sequence increment
                            # is atomic with respect to re-anchoring.
                            on_reanchor()
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
                        owned = set(explicit_models)
                        registry.sync(backend_name, owned)
                    elif discovered is not None:
                        owned = set(discovered)
                        registry.sync(backend_name, owned)
                    else:
                        owned = None
                    # The fetch state (and its warning/recovery logging)
                    # tracks the /v1/models fetch itself, independently of
                    # whether a sync ran this tick.
                    if owned is not None and routing_order:
                        _warn_routing_order_violations(
                            routing_order, owned, backend_name, warned_routing_order
                        )
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
