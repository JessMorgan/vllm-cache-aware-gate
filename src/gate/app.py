"""FastAPI app factory wiring the pure core to the HTTP edges.

This is the assembly layer: it builds the FastAPI app, owns one
:class:`BackendState` per configured vLLM backend (each with its own
:class:`MetricsCache`, :class:`CapacityCache`, :class:`KvRemaining`
remaining-KV counter, and a monotonic ``anchor_seq`` — the charge-rollback
guard for the failover walk), the shared
:class:`~gate.models.ModelRegistry` (model → candidate-set routing map), the
:class:`RoutingState` (per-model round-robin index), the :class:`GateStats`,
and the upstream :class:`httpx.AsyncClient`. The lifespan starts one
background poller per backend.

**Per-model routing (decisions 9/11).** The request body's ``model`` field
yields a *candidate set* — the backends that serve the model
(:meth:`ModelRegistry.resolve_candidates`; duplicate model ids are legal,
decision 11) — and a *routing policy* picks which candidate to attempt first
(:func:`gate.routing.select_backend`, pure). A model owned by no backend
(including the unparseable ``"unknown"``) falls to the default backend
(decision 4), and vLLM returns its own canonical "model not found" error.
The policy comes from the config's ``routing:`` section (decision 9); a
model with no entry defaults to ``round_robin`` over the registry order
(decision 11). Candidates are filtered to the spec's ``order``, to serving
backends only — a backend named in ``order`` that does not serve the model
is a dead candidate the walk skips (the poller separately warns the operator
about such entries, decision 15).

**Pre-stream failover walk (decision 12).** The walk starts at the selected
candidate and advances through the remaining candidates (wrapping) until one
forwards. Per candidate: the staleness gate (a stale/never-fetched feed
fails open in BOTH layers — a stale backend is **not** skipped; failing open
is the point), then the two AND-combined admission layers (the optional
tiered policy :func:`gate.router.decision` and the always-on autoconfig
layer :func:`gate.router.decision_auto`, combined by
:func:`gate.router.combine_decisions`) evaluated on *that candidate's*
thresholds, remaining-KV counter, and autoconfig policy. A reject is a
**skip** (skip-on-reject — this is what makes ``fill`` ≈
``primary_fallback``, invariant 19), not a stop. An allow charges the
candidate's counter (``ceil(ctx * token_margin)``, ``0`` when ctx is
unknown) **before** the proxy await, then proxies to that candidate's
``base_url``. A pre-stream transport failure or upstream 5xx is also a
**skip**: the charge is rolled back via ``KvRemaining.add`` **only if the
candidate's anchor sequence is unchanged since the charge** (a re-anchor in
between makes the fresh anchor authoritative — the old charge is dropped);
the next candidate is then tried. Once a streamed response has started,
errors surface as-is (gotcha #6 preserved in-flight) — no walk. The walk
never short-circuits: every remaining candidate is attempted before
exhaustion is declared.

**Exhaustion rule (decision 12).** If the walk completes with no success:
at least one candidate 429'd → a single 429 with ``Retry-After = max`` over
the candidates' ``Decision.retry_after`` values (the reported rejector is
the max-timeout backend; tie → first in walk order — mirroring
:func:`gate.router.combine_decisions`' rule extended across candidates);
every candidate failed at transport (none 429'd, no upstream 5xx answer) →
**502** (never 200, never 429 — the request never got a capacity answer;
the body names the candidate backends tried). A pre-stream upstream 5xx on
the **last** candidate (no next candidate to skip to) is propagated as-is
instead (gotcha #6 — upstream status is never masked; this is also the
single-backend behavior, where the 5xx is the answer); the request is
recorded as ``forwarded`` with that backend — the gate's decision was
"forward", and the 5xx is the upstream's answer, not the gate's rejection
(baseline stats semantics).

**Recording (decision 13).** ``stats.record_forwarded`` /
``record_rejected`` carry the ``backend`` label — the final backend (the
forwarder, the max-timeout rejector on 429 exhaustion, or the sentinel
``"none"`` on 502 exhaustion). Each failover skip is logged (a
``routing_failover`` line) and counted by
``gate_routing_failovers_total{model, from, to, reason}``.

**Per-backend admission state.** All admission state is scoped per backend,
not per model (docs/plans/multi-backend.md decision 1): models on one
backend share its KV pool (single-engine vLLM), so a shared-KV backend is
treated the same across its models.

**Staleness gate (fail-open override, per candidate).** If a candidate's
usage feed is stale or never fetched, BOTH layers fail open for that
candidate regardless of its counter's value (reason
``metrics_unavailable``): a metrics outage never blocks traffic (invariant
#1). The counter still takes its subtraction (bookkeeping continues) but
cannot reject while the feed is stale. Other candidates are unaffected.

**Capacity-missing (per-backend fail-open).** A backend whose observed
(HTTP 200) ``/metrics`` body lacks a usable KV-cache capacity no longer
kills the process (docs/plans/multi-backend.md decision 18 — killing the
whole proxy because one backend is misconfigured would take down the healthy
ones). Its poller logs an error, leaves its counter unanchored (its
autoconfig layer fails open), and flips the backend's
``capacity_unavailable`` flag, surfaced by the
``gate_backend_capacity_unavailable{backend}`` gauge. The poller task keeps
looping; the fatal done-callback (:func:`_poller_fatal`) now fires only on an
unexpected task death (any other exception), never on a capacity condition.

Invariants (see AGENTS.md "Known gotchas"):

- Fail open when a candidate's metric is stale or never fetched (gotcha #1)
  — the staleness gate outranks the counter, per candidate.
- The vLLM usage metric is a fraction (0-1); it is converted to a percentage
  with ``frac * 100.0`` for the tiered layer (gotcha #2). The autoconfig
  path's single percentage→fraction step is
  ``target_kv_cache_pct / 100.0`` at wiring time, per backend (gotcha #14).
- A 429 carries the max ``Retry-After`` over the rejecting candidates'
  combined retries and a rejector-aware message naming the max-timeout
  backend and the model.
- Upstream status is propagated unmasked and ``stream: true`` SSE is not
  buffered (gotcha #6), to the forwarded candidate's ``base_url``.

The app also exposes ``GET /metrics`` (Prometheus ``gate_*`` stats from
:mod:`gate.stats`, with per-backend labels on the feed/remaining gauges, the
per-backend ``gate_backend_engine{backend,engine}`` and
``gate_metrics_endpoint_unavailable{backend}`` gauges (the detected engine and
the reachable-but-unrecognizable flag, docs/plans/sglang-backend.md §2.5), and
the ``backend`` label on the request counters), ``GET /v1/models``
(gate-local aggregate of the registry's known models with an ``owned_by``
extension — a list when 2+ backends serve the model, a scalar when
single-owner; always 200, never proxied), and ``GET /healthz`` (status plus
a per-backend array and the known models with their policy in effect). Each
allow/reject decision is recorded into the stats layer as a pure side effect
— recording never alters the decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from gate.config import GateConfig, Threshold
from gate.metrics import ENGINE_UNKNOWN, CapacityCache, KvRemaining, MetricsCache
from gate.models import ModelRegistry
from gate.poller import run_poller
from gate.proxy import proxy_request
from gate.router import (
    REASON_AUTO_EXCEEDS_HEADROOM,
    REASON_METRICS_UNAVAILABLE,
    AutoPolicy,
    Decision,
    combine_decisions,
    decision,
    decision_auto,
)
from gate.routing import POLICY_ROUND_ROBIN, CandidateView, RoutingSpec, select_backend
from gate.stats import GateStats
from gate.tokens import estimate_context_tokens, extract_request_model

log = logging.getLogger("gate.app")

# Shared upstream client timeout: connect fast, no read timeout (SSE streams
# can be long-lived), bounded write/pool.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)

# Content type for the Prometheus text exposition format (version 0.0.4).
_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class BackendState:
    """Per-backend admission state (docs/plans/multi-backend.md §2.4).

    One instance per configured backend (``cfg.backends`` is always
    non-empty — decision 6). ``thresholds`` and ``auto_policy`` are the
    *resolved* values (per-backend override, else the global default);
    ``target_frac`` is the single percentage→fraction conversion (gotcha
    #14) done once at wiring. ``models`` is the explicit mnemonic tuple
    (empty = auto-adopt from ``/v1/models``). ``capacity_unavailable`` is
    the per-backend alert flag flipped by the poller's
    ``capacity_unavailable`` callback (decision 18). ``anchor_seq`` is the
    per-backend monotonic anchor sequence, incremented by the poller's
    ``on_reanchor`` callback on every re-anchor — the failover
    charge-rollback guard (decision 12) compares it to decide whether a
    rolled-back charge is still valid. ``engine`` is the currently detected
    engine (``ENGINE_UNKNOWN`` until the first detection), set by the app's
    ``on_engine`` closure; ``metrics_unavailable`` is the per-backend
    reachable-but-unrecognizable flag, set to whatever the app's
    ``on_metrics_unavailable`` closure is invoked with (the new state, either
    direction) — the ``False`` recovery transition rides that state callback,
    NOT the engine setter (docs/plans/sglang-backend.md §2.4).
    """

    name: str
    host: str
    port: int
    base_url: str
    is_default: bool
    models: tuple[str, ...]
    cache: MetricsCache
    capacity_cache: CapacityCache
    counter: KvRemaining
    thresholds: tuple[Threshold, ...]
    target_pct: float
    target_frac: float
    auto_policy: AutoPolicy
    capacity_unavailable: bool = False
    anchor_seq: int = 0
    engine: str = ENGINE_UNKNOWN
    metrics_unavailable: bool = False


class RoutingState:
    """The per-model stateful ``round_robin`` index (decision 16).

    Single-asyncio-loop contract (same as :class:`KvRemaining`): only request
    handlers read-modify-write it, synchronously with no ``await`` between
    the read and the write, so no locking is needed. The index advances
    **only on a successful forward** and wraps modulo the candidate count;
    ``select_backend`` returns ``0`` at selection time for ``round_robin``
    and this offset rotates the starting candidate.
    """

    def __init__(self) -> None:
        self._rr: dict[str, int] = {}  # model -> current index

    def rr_offset(self, model: str, n: int) -> int:
        """The stored index modulo ``n`` (``0`` when absent or ``n <= 0``)."""
        if n <= 0:
            return 0
        return self._rr.get(model, 0) % n

    def advance(self, model: str, n: int) -> None:
        """Increment the stored index modulo ``n`` (no-op when ``n <= 0``)."""
        if n <= 0:
            return
        self._rr[model] = (self._rr.get(model, 0) + 1) % n


def _capacity_flag_setter(bs: BackendState) -> Callable[[], None]:
    """A zero-arg callback that flips one backend's capacity alert flag.

    The poller invokes it when an observed (HTTP 200) ``/metrics`` body
    lacks a usable KV-cache capacity (decision 18). One closure is created
    per backend so the flag set is the right backend's.
    """

    def _set() -> None:
        bs.capacity_unavailable = True

    return _set


def _on_reanchor_setter(bs: BackendState) -> Callable[[], None]:
    """A zero-arg callback that increments one backend's anchor sequence.

    The poller invokes it synchronously with every re-anchor (no ``await``
    between the re-anchor and this increment — the same single-asyncio-loop
    contract as the re-anchor itself), so the failover charge-rollback guard
    (decision 12) reads a sequence that is atomic with respect to
    re-anchoring. One closure is created per backend so the right backend's
    sequence is the one incremented.
    """

    def _bump() -> None:
        bs.anchor_seq += 1

    return _bump


async def _release_streaming_response(resp: Response) -> None:
    """Release the upstream connection behind a discarded streaming response.

    Used when a pre-stream upstream 5xx is *discarded* (skipped) rather than
    propagated — either to walk to the next candidate or because the
    429-exhaustion path returns the 429 instead. The proxy's ``_stream()`` is
    an async generator whose ``finally`` closes the upstream response;
    ``aclose()`` on an UNSTARTED async generator is a no-op (it does not run
    the finally), so prime the generator first (start it, reading the first
    chunk of the discarded body) and then close it — that runs the finally and
    releases the upstream connection back to the shared httpx pool. A
    non-streaming response needs nothing (its body is already buffered and the
    connection is not held open). starlette types ``body_iterator`` as a bare
    AsyncIterable, so cast to the generator type to reach ``__anext__`` /
    ``aclose``.
    """
    if not isinstance(resp, StreamingResponse):
        return
    iterator = cast(AsyncGenerator[bytes, None], resp.body_iterator)
    try:
        await iterator.__anext__()
    except StopAsyncIteration:
        pass  # empty body: exhaustion already ran the finally
    await iterator.aclose()


def _on_engine_setter(bs: BackendState) -> Callable[[str], None]:
    """A callback that records one backend's detected engine.

    The poller invokes it when the detected engine differs from the last
    detected one (first detection and any change, including to/from
    unknown). One closure is created per backend so the right backend's
    engine is the one recorded (docs/plans/sglang-backend.md section 2.4).
    """

    def _set(engine: str) -> None:
        bs.engine = engine

    return _set


def _metrics_unavailable_setter(bs: BackendState) -> Callable[[bool], None]:
    """A callback that records one backend's metrics-endpoint-unavailable state.

    The poller invokes it with the NEW state whenever the unavailable state
    changes (either direction): True for a non-200 response or a 200 body
    detected unknown, False for a 200 body detected vllm/sglang. The app
    owns the flag; the poller only signals. The False recovery transition
    rides THIS callback (not the engine setter) — so a 404 -> healthy-vllm
    recovery (engine unchanged) still clears the flag
    (docs/plans/sglang-backend.md section 2.4).
    """

    def _set(unavailable: bool) -> None:
        bs.metrics_unavailable = unavailable

    return _set


def _is_streaming(body: bytes) -> bool:
    """True iff ``body`` parses to a JSON object with ``"stream": true``.

    Any parse error (invalid JSON, non-object) is treated as non-stream.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("stream") is True


def _reject_message(
    dec: Decision,
    usage_pct: float | None,
    ctx_tokens: int | None,
    *,
    backend: str,
    model: str,
    remaining: int | None = None,
    capacity: int | None = None,
    target_pct: float | None = None,
) -> str:
    """Single-line human-readable 429 message, rejector-aware.

    Names the routed ``backend`` and the request ``model``.

    - ``dec.active_tier`` present (the tiered layer is the reported
      rejector) -> the tiered text, as before.
    - ``dec.reason`` is the autoconfig reject reason -> the autoconfig text,
      naming the usage %, the estimated remaining tokens, the capacity, and
      the target usage %.
    - Otherwise a generic message (defensive — the message must not crash if
      the rejector shape ever changes).
    """
    tier = dec.active_tier
    if tier is not None:
        usage_str = f"{usage_pct:.1f}" if usage_pct is not None else "?"
        ctx_str = f"~{ctx_tokens}" if ctx_tokens is not None else "?"
        return (
            f"backend {backend} (model {model}): KV cache too full for a "
            f"{ctx_str}-token request "
            f"(usage {usage_str}% >= {tier.kv_pct:g}% tier; "
            f"max {tier.max_context}). "
            f"Retry in {dec.retry_after}s."
        )
    if dec.reason == REASON_AUTO_EXCEEDS_HEADROOM:
        usage_str = f"{usage_pct:.1f}" if usage_pct is not None else "?"
        remaining_str = str(remaining) if remaining is not None else "?"
        capacity_str = str(capacity) if capacity is not None else "?"
        target_str = f"{target_pct:g}" if target_pct is not None else "?"
        retry = dec.retry_after if dec.retry_after is not None else 0
        return (
            f"backend {backend} (model {model}): KV cache headroom exhausted "
            f"for a ~{ctx_tokens}-token request "
            f"(usage {usage_str}%, ~{remaining_str} of {capacity_str} tokens "
            f"remaining up to the {target_str}% target). "
            f"Retry in {retry}s."
        )
    retry = dec.retry_after if dec.retry_after is not None else 0
    return f"backend {backend} (model {model}): KV cache too full. Retry in {retry}s."


def _poller_fatal(task: asyncio.Task[None]) -> None:
    """Done-callback for a poller task: exit 1 on a pre-shutdown crash.

    If the task finished with an exception other than
    ``asyncio.CancelledError``, the error is logged and the process exits
    with status 1 (``os._exit(1)``). A clean shutdown (cancellation) must NOT
    trigger the exit. Note the capacity-missing path no longer raises (it
    fails open per backend, decision 18), so this callback now fires only on
    an unexpected task death.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("poller task died: %s; exiting", exc, exc_info=exc)
        os._exit(1)


def _build_backend_states(cfg: GateConfig) -> list[BackendState]:
    """Build the per-backend state list from the config.

    One :class:`BackendState` per configured backend, with fresh
    caches/counter and the resolved per-backend-or-global knobs.
    ``cfg.backends`` is always non-empty (config validation, decision 6).
    """
    # At most one backend is flagged default: true (config validation);
    # the default backend is the fallback for unknown models (decision 4),
    # falling back to the first entry when none is flagged.
    default_name = next((b.name for b in cfg.backends if b.default), cfg.backends[0].name)
    return [
        BackendState(
            name=b.name,
            host=b.host,
            port=b.port,
            base_url=f"http://{b.host}:{b.port}",
            is_default=(b.name == default_name),
            models=b.models,
            cache=MetricsCache(),
            capacity_cache=CapacityCache(),
            counter=KvRemaining(),
            thresholds=b.thresholds if b.thresholds is not None else cfg.thresholds,
            target_pct=b.target_kv_cache_pct
            if b.target_kv_cache_pct is not None
            else cfg.target_kv_cache_pct,
            # The single percentage→fraction step (gotcha #14), done
            # once at wiring per backend.
            target_frac=(
                b.target_kv_cache_pct
                if b.target_kv_cache_pct is not None
                else cfg.target_kv_cache_pct
            )
            / 100.0,
            auto_policy=AutoPolicy(
                b.token_margin if b.token_margin is not None else cfg.token_margin,
                b.retry_min_s if b.retry_min_s is not None else cfg.retry_min_s,
                b.retry_max_s if b.retry_max_s is not None else cfg.retry_max_s,
            ),
        )
        for b in cfg.backends
    ]


def create_app(
    cfg: GateConfig,
    *,
    upstream: httpx.AsyncClient | None = None,
    start_poller: bool = True,
    stats: GateStats | None = None,
) -> FastAPI:
    """Build the gate FastAPI app.

    Args:
        cfg: Validated gate configuration.
        upstream: Shared :class:`httpx.AsyncClient` for proxying and polling;
            a fresh one is created when None.
        start_poller: When True one poller task per backend is started in the
            lifespan. Tests pass False so the pollers do not interfere.
        stats: A :class:`GateStats` for the ``/metrics`` endpoint; a fresh one
            is created when None.

    The per-backend admission state (``MetricsCache`` / ``CapacityCache`` /
    ``KvRemaining`` per backend, resolved thresholds and autoconfig knobs) is
    built internally from ``cfg`` (see :func:`_build_backend_states`); tests
    drive it through the app's ``state.backends`` list. The per-model routing
    specs (``state.routing_specs``) and the stateful round-robin index
    (``state.routing_state``) are built from ``cfg.routing``.

    The pollers are started by the lifespan (which uvicorn / TestClient
    trigger), so :func:`gate.main.main` does not start them separately.
    """
    if upstream is None:
        upstream = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    if stats is None:
        stats = GateStats(cfg)

    backends = _build_backend_states(cfg)
    registry = ModelRegistry()
    for bs in backends:
        registry.register_backend(bs.name, is_default=bs.is_default)
    default_name = next((bs.name for bs in backends if bs.is_default), backends[0].name)
    by_name = {bs.name: bs for bs in backends}

    # Per-model routing specs (decision 9): model -> RoutingSpec. A model with
    # no entry defaults to round_robin over the registry order at request time
    # (decision 11) — the spec is synthesized there, not stored here.
    routing_specs: dict[str, RoutingSpec] = {e.model: e.spec for e in cfg.routing}
    # The full model -> candidate-order mapping, handed to each poller so it
    # can scope the order ⊆ serving check (decision 15) to its own backend.
    routing_order: dict[str, tuple[str, ...]] = {e.model: e.spec.order for e in cfg.routing}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_poller:
            # One poller task per backend, each with its own caches/counter/
            # target_frac plus the model-discovery wiring (shared registry).
            # The capacity-missing path no longer raises (per-backend
            # fail-open, decision 18): the callback only flips the backend's
            # alert flag, and the fatal done-callback fires only on an
            # unexpected task death. The on_reanchor callback increments the
            # backend's anchor sequence synchronously with the re-anchor
            # (decision 12 — the failover charge-rollback guard).
            poller_tasks: list[asyncio.Task[None]] = []
            for bs in backends:
                poller_tasks.append(
                    asyncio.create_task(
                        run_poller(
                            upstream,
                            bs.cache,
                            bs.base_url + "/metrics",
                            cfg.metrics_poll_interval_s,
                            counter=bs.counter,
                            target_frac=bs.target_frac,
                            capacity_cache=bs.capacity_cache,
                            model_url=bs.base_url + "/v1/models",
                            model_refresh_s=cfg.model_refresh_interval_s,
                            registry=registry,
                            backend_name=bs.name,
                            explicit_models=bs.models,
                            capacity_unavailable=_capacity_flag_setter(bs),
                            on_reanchor=_on_reanchor_setter(bs),
                            on_engine=_on_engine_setter(bs),
                            on_metrics_unavailable=_metrics_unavailable_setter(bs),
                            routing_order=routing_order,
                        )
                    )
                )
                poller_tasks[-1].add_done_callback(_poller_fatal)
            app.state.poller_tasks = poller_tasks
        else:
            app.state.poller_tasks = []
        try:
            yield
        finally:
            for task in app.state.poller_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await upstream.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.cfg = cfg
    app.state.upstream = upstream
    app.state.stats = stats
    app.state.backends = backends
    app.state.registry = registry
    app.state.default_backend = default_name
    app.state.routing_specs = routing_specs
    app.state.routing_state = RoutingState()

    async def _handle_generation(request: Request, endpoint: str) -> Response:
        body = await request.body()
        model = extract_request_model(body)
        # Computed BEFORE routing: size-driven policies need it (decision 10).
        ctx_tokens = estimate_context_tokens(body, cfg)
        is_stream = _is_streaming(body)

        # --- Candidate resolution (decisions 4/9/11) ------------------------
        # The registry's candidate set (all backends serving the model, in
        # config order), else the default backend (unknown/unparseable models
        # fall to the default — decision 4; vLLM returns its own canonical
        # "model not found"). Both the registry and the default are read via
        # app.state so a test that swaps in a pre-populated registry also
        # swaps the default consistently (the app never swaps them in
        # production).
        serving = app.state.registry.resolve_candidates(model)
        if not serving:
            # Unowned model (including the unparseable "unknown"): the default
            # backend is the sole candidate (decision 4).
            candidates: tuple[str, ...] = (app.state.default_backend,)
            spec = RoutingSpec(policy=POLICY_ROUND_ROBIN, order=candidates, threshold_tokens=None)
        else:
            # The routing spec for the model (decision 9), or the default
            # round_robin over the registry order (decision 11).
            spec = app.state.routing_specs.get(model)
            if spec is None:
                spec = RoutingSpec(policy=POLICY_ROUND_ROBIN, order=serving, threshold_tokens=None)
            # Filter the spec's order to serving backends only (§2.2b): a
            # backend named in order that does not serve the model is a dead
            # candidate the walk skips (the poller separately warns the
            # operator about such entries, decision 15). If the filter drops
            # every candidate, fall back to the default backend.
            candidates = tuple(b for b in spec.order if b in serving)
            if not candidates:
                candidates = (app.state.default_backend,)
                spec = RoutingSpec(
                    policy=POLICY_ROUND_ROBIN, order=candidates, threshold_tokens=None
                )

        # Per-candidate usage views for the pure selection (fraction/token
        # space; usage_frac is None when stale or never-fetched — the routing
        # contract's "unknown" feed).
        usage_views: dict[str, CandidateView] = {}
        for name in candidates:
            bs_view = by_name[name]
            view_frac = bs_view.cache.value()
            view_stale = view_frac is None or bs_view.cache.is_stale(cfg.stale_after_s)
            usage_views[name] = CandidateView(
                usage_frac=None if view_stale else view_frac,
                remaining_tokens=bs_view.counter.value(),
                capacity_tokens=bs_view.capacity_cache.value(),
                token_margin=bs_view.auto_policy.token_margin,
            )

        # --- Selection -------------------------------------------------------
        i0 = select_backend(spec, candidates, ctx_tokens, usage_views)
        if i0 is None:
            # fill all-full: no candidate is proxied, but the walk still runs
            # (without proxying) to collect the per-candidate retry_after
            # values for the max-Retry-After 429 (decision 12).
            i0 = 0
        if spec.policy == POLICY_ROUND_ROBIN:
            # The stateful per-model index offsets the stateless selection
            # (decision 16); it advances only on a successful forward below.
            i0 = (i0 + app.state.routing_state.rr_offset(model, len(candidates))) % len(candidates)

        # --- Failover walk (decision 12) -------------------------------------
        # Walk i = i0, i0+1, ... wrapping through the candidates. A reject, a
        # pre-stream transport failure, and a pre-stream upstream 5xx are all
        # SKIPS, not stops: the walk never short-circuits. A pre-stream 5xx on
        # the LAST candidate (no next candidate to skip to) is propagated
        # as-is (gotcha #6 — upstream status is never masked; this is also
        # the single-backend behavior, where the 5xx is the answer).
        rejects: list[tuple[str, Decision, float | None]] = []
        # The last candidate's pre-stream upstream 5xx (the response and the
        # backend that answered it), remembered so the exhaustion path can
        # propagate it as-is and record the gate's "forwarded" decision.
        last_upstream_5xx: tuple[Response, str] | None = None
        n = len(candidates)
        for offset in range(n):
            i = (i0 + offset) % n
            name = candidates[i]
            bs = by_name[name]
            next_name = candidates[(i0 + offset + 1) % n] if offset + 1 < n else None

            frac = bs.cache.value()
            if frac is None or bs.cache.is_stale(cfg.stale_after_s):
                # Staleness gate (fail-open override, per candidate): a stale
                # or never-fetched feed fails open in BOTH layers for that
                # candidate regardless of its counter's value — a metrics
                # outage never blocks traffic (invariant #1). A stale backend
                # is NOT skipped: failing open is the point. The counter still
                # takes its subtraction below (bookkeeping continues); the
                # decision is what fails open.
                usage_pct: float | None = None
                tier_dec = decision(None, ctx_tokens, bs.thresholds)
                auto_dec = Decision(
                    allow=True,
                    retry_after=None,
                    active_tier=None,
                    reason=REASON_METRICS_UNAVAILABLE,
                )
            else:
                usage_pct = frac * 100.0  # fraction (0-1) -> percentage (0-100)
                tier_dec = decision(usage_pct, ctx_tokens, bs.thresholds)
                auto_dec = decision_auto(bs.counter.value(), ctx_tokens, bs.auto_policy)

            dec = combine_decisions(tier_dec, auto_dec)

            if not dec.allow:
                # Skip-on-reject (the rule that makes fill ≈ primary_fallback,
                # invariant 19): record the reject for the final 429 and move
                # to the next candidate. Rejected candidates charge nothing.
                rejects.append((name, dec, usage_pct))
                log.info(
                    "reject backend=%s reason=%s usage_pct=%s ctx_tokens=%s retry_after=%s",
                    name,
                    dec.reason,
                    usage_pct,
                    ctx_tokens,
                    dec.retry_after,
                )
                if next_name is not None:
                    _record_failover(model, name, next_name, "reject")
                    log.info(
                        "routing_failover model=%s from=%s to=%s reason=reject",
                        model,
                        name,
                        next_name,
                    )
                continue

            # Allow: charge at admission time (before the proxy await) — the
            # same effective size the autoconfig layer compared (0 when ctx
            # is unknown). Subtraction is unclamped and a no-op when never
            # anchored. The anchor sequence is remembered alongside the
            # charge: the rollback below re-adds the charge only if no
            # re-anchor happened in between (decision 12).
            charge = (
                int(math.ceil(ctx_tokens * bs.auto_policy.token_margin))
                if ctx_tokens is not None
                else 0
            )
            seq_before = bs.anchor_seq
            bs.counter.subtract(charge)
            try:
                resp = await proxy_request(upstream, request, bs.base_url, stream=is_stream)
            except httpx.TransportError:
                # Pre-stream transport failure: roll back the charge (guarded
                # by the anchor sequence — a re-anchor in between makes the
                # fresh anchor authoritative and the old charge is dropped)
                # and skip to the next candidate.
                if bs.anchor_seq == seq_before:
                    bs.counter.add(charge)
                if next_name is not None:
                    _record_failover(model, name, next_name, "transport")
                    log.info(
                        "routing_failover model=%s from=%s to=%s reason=transport",
                        model,
                        name,
                        next_name,
                    )
                continue

            if resp.status_code >= 500:
                # Pre-stream upstream 5xx: roll back the charge (anchor-
                # sequence guarded). When a next candidate exists the 5xx body
                # is discarded (skip to the next candidate) and the upstream
                # connection is released immediately; when this is the last
                # candidate the response is propagated as-is (gotcha #6) and
                # the connection closes via the normal response lifecycle.
                if bs.anchor_seq == seq_before:
                    bs.counter.add(charge)
                if next_name is not None:
                    # Discarding the 5xx body to skip to the next candidate:
                    # release the upstream connection now (the proxy's
                    # _stream() finally closes it; prime-then-close so the
                    # finally runs even on an unstarted generator).
                    await _release_streaming_response(resp)
                    _record_failover(model, name, next_name, "upstream_5xx")
                    log.info(
                        "routing_failover model=%s from=%s to=%s reason=upstream_5xx",
                        model,
                        name,
                        next_name,
                    )
                else:
                    # Last candidate: no next candidate to skip to. Do NOT
                    # prime/close the generator — the 5xx body must propagate
                    # as-is (gotcha #6 — the upstream status and body are the
                    # answer, never masked by a 502), and a primed-then-closed
                    # generator would re-iterate empty when starlette streams
                    # the returned response. The upstream connection closes
                    # via the normal response lifecycle (the generator's
                    # finally runs when starlette finishes streaming). Remember
                    # the response (and the backend it came from) so the
                    # exhaustion path propagates it as-is and records the
                    # gate's "forwarded" decision with that backend (the 5xx is
                    # the upstream's answer, not the gate's rejection — baseline
                    # stats semantics).
                    last_upstream_5xx = (resp, name)
                continue

            # Success: advance the RR index (only on a successful forward)
            # and record the forward with the final backend (decision 13).
            app.state.routing_state.advance(model, n)
            try:
                stats.record_forwarded(endpoint, model, ctx_tokens, backend=name)
            except Exception:  # noqa: BLE001 - stats must never break the request path
                log.warning("stats.record_forwarded failed", exc_info=True)
            log.debug(
                "allow backend=%s reason=%s usage_pct=%s ctx_tokens=%s stream=%s",
                name,
                dec.reason,
                usage_pct,
                ctx_tokens,
                is_stream,
            )
            return resp

        # --- All candidates exhausted (decision 12) --------------------------
        if rejects:
            # A single 429: Retry-After = max over the candidates' combined
            # retry_after values; the reported rejector is the max-timeout
            # backend (tie -> the first in walk order, i.e. the first appended
            # to rejects with that value) — mirroring combine_decisions'
            # max-timeout/tie rule extended across candidates.
            # A reject always carries a non-None retry_after (decision's
            # contract), so the max is over ints.
            max_retry = max(dec.retry_after for _, dec, _ in rejects if dec.retry_after is not None)
            rejector_name, rejector_dec, rejector_pct = next(
                (name, dec, pct) for name, dec, pct in rejects if dec.retry_after == max_retry
            )
            rejector_bs = by_name[rejector_name]
            try:
                stats.record_rejected(endpoint, model, ctx_tokens, backend=rejector_name)
            except Exception:  # noqa: BLE001 - stats must never break the request path
                log.warning("stats.record_rejected failed", exc_info=True)
            # The 429-exhaustion check runs before last-5xx propagation, so a
            # last-candidate pre-stream 5xx (remembered in last_upstream_5xx)
            # is NOT propagated here — it is discarded. If it was a streaming
            # response, release its upstream connection now (the proxy's
            # _stream() finally closes it; prime-then-close so the finally runs
            # even on an unstarted generator) instead of leaking it back to the
            # pool.
            if last_upstream_5xx is not None:
                await _release_streaming_response(last_upstream_5xx[0])
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(max_retry)},
                content={
                    "error": {
                        "message": _reject_message(
                            rejector_dec,
                            rejector_pct,
                            ctx_tokens,
                            backend=rejector_name,
                            model=model,
                            remaining=rejector_bs.counter.value(),
                            capacity=rejector_bs.capacity_cache.value(),
                            target_pct=rejector_bs.target_pct,
                        ),
                        "type": "cache_pressure",
                        "code": "kv_cache_too_full",
                    }
                },
            )

        if last_upstream_5xx is not None:
            # Every candidate was attempted, none 429'd, and the last
            # candidate answered with a pre-stream upstream 5xx: propagate it
            # as-is (gotcha #6 — upstream status is never masked; the single-
            # backend case is exactly this). The gate's decision was "forward
            # to this backend", so the request is recorded as forwarded with
            # that backend (the 5xx is the upstream's answer, not the gate's
            # rejection — baseline stats semantics).
            resp, backend = last_upstream_5xx
            try:
                stats.record_forwarded(endpoint, model, ctx_tokens, backend=backend)
            except Exception:  # noqa: BLE001 - stats must never break the request path
                log.warning("stats.record_forwarded failed", exc_info=True)
            return resp

        # Every candidate failed at transport (none 429'd, no upstream 5xx
        # answer): 502 — never 200, never 429, the request never got a
        # capacity answer. No rejector exists, so the backend label is the
        # sentinel "none" (decision 13).
        try:
            stats.record_rejected(endpoint, model, ctx_tokens, backend="none")
        except Exception:  # noqa: BLE001 - stats must never break the request path
            log.warning("stats.record_rejected failed", exc_info=True)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"all backends failed for model {model}: {', '.join(candidates)}",
                    "type": "upstream_failure",
                    "code": "all_backends_failed",
                }
            },
        )

    def _record_failover(model: str, from_backend: str, to_backend: str, reason: str) -> None:
        """Record a failover skip (stats must never break the request path)."""
        try:
            stats.record_failover(model, from_backend, to_backend, reason)
        except Exception:  # noqa: BLE001 - stats must never break the request path
            log.warning("stats.record_failover failed", exc_info=True)

    def _owned_by(model: str) -> str | list[str]:
        """The model's owning backend(s) (decision 14): a scalar name when
        single-owner, a list of names when 2+ backends serve the model."""
        owners = app.state.registry.resolve_candidates(model)
        return owners[0] if len(owners) == 1 else list(owners)

    def _policy_in_effect(model: str) -> str:
        """The routing policy in effect for the model (decision 14): the
        ``routing:`` entry's policy, or ``round_robin`` by default (decision
        11)."""
        spec = app.state.routing_specs.get(model)
        return spec.policy if spec is not None else POLICY_ROUND_ROBIN

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle_generation(request, "chat_completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _handle_generation(request, "completions")

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """Gate-local aggregate of the registry's known models (always 200).

        One entry per model with an ``owned_by`` extension (decision 14): a
        **list** of backend names when the model is served by 2+ backends
        (duplicate model ids are legal, decision 11), a **scalar** name when
        single-owner (backward-compatible shape). Never proxied; the
        backends' own ``/v1/models`` endpoints are not reachable through the
        gate.
        """
        return JSONResponse(
            status_code=200,
            content={
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "owned_by": _owned_by(model)}
                    for model, _ in app.state.registry.items()
                ],
            },
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "backends": [
                    {
                        "name": bs.name,
                        "metrics_age_s": bs.cache.age(),
                        "kv_usage": bs.cache.value(),
                        "kv_cache_capacity_tokens": bs.capacity_cache.value(),
                        "kv_cache_remaining_tokens": bs.counter.value(),
                    }
                    for bs in backends
                ],
                "models": [
                    {
                        "id": model,
                        "owned_by": _owned_by(model),
                        "routing": _policy_in_effect(model),
                    }
                    for model, _ in app.state.registry.items()
                ],
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        # Union all backends' by-model usage. Model ids may now be duplicated
        # across backends (decision 11), and the synthetic "default" (unlabeled
        # usage series) can repeat across backends; take the max on a repeated
        # key (conservative, gotcha #5) so one backend cannot silently
        # overwrite the other's gauge value.
        merged: dict[str, float] = {}
        for bs in backends:
            for model_name, frac in bs.cache.by_model().items():
                current = merged.get(model_name)
                merged[model_name] = frac if current is None else max(current, frac)
        stats.set_kv_usage(merged)
        for bs in backends:
            stats.set_freshness(bs.name, not bs.cache.is_stale(cfg.stale_after_s), bs.cache.age())
            stats.set_remaining(bs.name, bs.counter.value())
            stats.set_backend_capacity_unavailable(bs.name, bs.capacity_unavailable)
            stats.set_backend_engine(bs.name, bs.engine)
            stats.set_backend_metrics_unavailable(bs.name, bs.metrics_unavailable)
        return Response(content=stats.render(), media_type=_METRICS_CONTENT_TYPE)

    return app
