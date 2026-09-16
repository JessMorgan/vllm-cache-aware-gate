"""FastAPI app factory wiring the pure core to the HTTP edges.

This is the assembly layer: it builds the FastAPI app, owns one
:class:`BackendState` per configured vLLM backend (each with its own
:class:`MetricsCache`, :class:`CapacityCache`, and :class:`KvRemaining`
remaining-KV counter), the shared :class:`~gate.models.ModelRegistry`
(model → backend routing map), the :class:`GateStats`, and the upstream
:class:`httpx.AsyncClient`. The lifespan starts one background poller per
backend, and the two generation endpoints route by the request body's
``model`` field to exactly one backend, whose KV state feeds the two
AND-combined admission layers (the optional tiered policy
:func:`gate.router.decision` and the always-on autoconfig layer
:func:`gate.router.decision_auto`, combined by
:func:`gate.router.combine_decisions`) before the transparent proxy
(:func:`gate.proxy.proxy_request`).

**Per-backend admission state.** All admission state is scoped per backend,
not per model (docs/plans/multi-backend.md decision 1): models on one
backend share its KV pool (single-engine vLLM), so a shared-KV backend is
treated the same across its models. The request body's ``model`` field
selects the backend via the :class:`~gate.models.ModelRegistry`; a model
owned by no backend (including the unparseable ``"unknown"``) falls to the
default backend (decision 4), and vLLM returns its own canonical
"model not found" error.

**Two admission layers, AND-combined (per backend).** A request forwards iff
the tiered layer (optional; no tiers ⇒ always allow) AND the autoconfig layer
both allow, evaluated on the routed backend's thresholds, remaining-KV
counter, and autoconfig policy. The autoconfig layer compares the request's
estimated size (``ceil(ctx * token_margin)``) against the backend's
:class:`KvRemaining` counter — the gate's live estimate of the KV tokens
still available up to the target usage %, re-anchored by that backend's
poller on every good poll. On every forward the counter is charged that same
amount **before** the proxy await; rejected requests charge nothing.

**Staleness gate (fail-open override, per backend).** If a backend's usage
feed is stale or never fetched, BOTH layers fail open for that backend
regardless of its counter's value (reason ``metrics_unavailable``): a
metrics outage never blocks traffic (invariant #1). The counter still takes
its subtraction (bookkeeping continues) but cannot reject while the feed is
stale. Other backends are unaffected.

**Capacity-missing (per-backend fail-open).** A backend whose observed
(HTTP 200) ``/metrics`` body lacks a usable KV-cache capacity no longer
kills the process (docs/plans/multi-backend.md decision 10 — killing the
whole proxy because one backend is misconfigured would take down the healthy
ones). Its poller logs an error, leaves its counter unanchored (its
autoconfig layer fails open), and flips the backend's
``capacity_unavailable`` flag, surfaced by the
``gate_backend_capacity_unavailable{backend}`` gauge. The poller task keeps
looping; the fatal done-callback (:func:`_poller_fatal`) now fires only on an
unexpected task death (any other exception), never on a capacity condition.

Invariants (see AGENTS.md "Known gotchas"):

- Fail open when a backend's metric is stale or never fetched (gotcha #1) —
  the staleness gate outranks the counter, per backend.
- The vLLM usage metric is a fraction (0-1); it is converted to a percentage
  with ``frac * 100.0`` for the tiered layer (gotcha #2). The autoconfig
  path's single percentage→fraction step is
  ``target_kv_cache_pct / 100.0`` at wiring time, per backend (gotcha #14).
- A 429 carries the combined ``Retry-After`` (the max of the rejecting
  layers' retries, from the routed backend's policy) and a rejector-aware
  message naming the backend and model.
- Upstream status is propagated unmasked and ``stream: true`` SSE is not
  buffered (gotcha #6), to the routed backend's ``base_url``.

The app also exposes ``GET /metrics`` (Prometheus ``gate_*`` stats from
:mod:`gate.stats`, with per-backend labels on the feed/remaining gauges),
``GET /v1/models`` (gate-local aggregate of the registry's known models with
an ``owned_by`` extension; always 200, never proxied), and ``GET /healthz``
(status plus a per-backend array and the known models). Each allow/reject
decision is recorded into the stats layer as a pure side effect — recording
never alters the decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from gate.config import GateConfig, Threshold
from gate.metrics import CapacityCache, KvRemaining, MetricsCache
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

    One instance per configured backend (or one synthesized from the legacy
    ``vllm_host``/``vllm_port`` when ``cfg.backends`` is empty). ``thresholds``
    and ``auto_policy`` are the *resolved* values (per-backend override, else
    the global default); ``target_frac`` is the single percentage→fraction
    conversion (gotcha #14) done once at wiring. ``models`` is the explicit
    mnemonic tuple (empty = auto-adopt from ``/v1/models``).
    ``capacity_unavailable`` is the per-backend alert flag flipped by the
    poller's ``capacity_unavailable`` callback (decision 10).
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


def _capacity_flag_setter(bs: BackendState) -> Callable[[], None]:
    """A zero-arg callback that flips one backend's capacity alert flag.

    The poller invokes it when an observed (HTTP 200) ``/metrics`` body
    lacks a usable KV-cache capacity (decision 10). One closure is created
    per backend so the flag set is the right backend's.
    """

    def _set() -> None:
        bs.capacity_unavailable = True

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
            f"backend {backend} (model {model}): vLLM KV cache too full for a "
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
            f"backend {backend} (model {model}): vLLM KV cache headroom exhausted "
            f"for a ~{ctx_tokens}-token request "
            f"(usage {usage_str}%, ~{remaining_str} of {capacity_str} tokens "
            f"remaining up to the {target_str}% target). "
            f"Retry in {retry}s."
        )
    retry = dec.retry_after if dec.retry_after is not None else 0
    return f"backend {backend} (model {model}): vLLM KV cache too full. Retry in {retry}s."


def _poller_fatal(task: asyncio.Task[None]) -> None:
    """Done-callback for a poller task: exit 1 on a pre-shutdown crash.

    If the task finished with an exception other than
    ``asyncio.CancelledError``, the error is logged and the process exits
    with status 1 (``os._exit(1)``). A clean shutdown (cancellation) must NOT
    trigger the exit. Note the capacity-missing path no longer raises (it
    fails open per backend, decision 10), so this callback now fires only on
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

    When ``cfg.backends`` is non-empty, one :class:`BackendState` per
    configured backend, with fresh caches/counter and the resolved
    per-backend-or-global knobs. When it is empty (legacy config), ONE
    synthesized state from ``vllm_host``/``vllm_port`` (name
    ``f"{host}:{port}"``, default, the global thresholds and knobs) — the
    legacy path is removed in a later segment.
    """
    states: list[BackendState] = []
    if cfg.backends:
        # At most one backend is flagged default: true (config validation);
        # the default backend is the fallback for unknown models (decision 4),
        # falling back to the first entry when none is flagged.
        default_name = next((b.name for b in cfg.backends if b.default), cfg.backends[0].name)
        for b in cfg.backends:
            states.append(
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
            )
    else:
        states.append(
            BackendState(
                name=f"{cfg.vllm_host}:{cfg.vllm_port}",
                host=cfg.vllm_host,
                port=cfg.vllm_port,
                base_url=f"http://{cfg.vllm_host}:{cfg.vllm_port}",
                is_default=True,
                models=(),
                cache=MetricsCache(),
                capacity_cache=CapacityCache(),
                counter=KvRemaining(),
                thresholds=cfg.thresholds,
                target_pct=cfg.target_kv_cache_pct,
                target_frac=cfg.target_kv_cache_pct / 100.0,
                auto_policy=AutoPolicy(cfg.token_margin, cfg.retry_min_s, cfg.retry_max_s),
            )
        )
    return states


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
    drive it through the app's ``state.backends`` list.

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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_poller:
            # One poller task per backend, each with its own caches/counter/
            # target_frac plus the model-discovery wiring (shared registry).
            # The capacity-missing path no longer raises (per-backend
            # fail-open, decision 10): the callback only flips the backend's
            # alert flag, and the fatal done-callback fires only on an
            # unexpected task death.
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

    async def _handle_generation(request: Request, endpoint: str) -> Response:
        body = await request.body()
        model = extract_request_model(body)
        ctx_tokens = estimate_context_tokens(body, cfg)

        # Route by model: the registry's owner, else the default backend
        # (unknown/unparseable models fall to the default — decision 4).
        # Both the registry and the default are read via app.state so a test
        # that swaps in a pre-populated registry also swaps the default
        # consistently (the app never swaps them in production).
        backend_name = app.state.registry.resolve(model) or app.state.default_backend
        bs = by_name[backend_name]

        frac = bs.cache.value()
        if frac is None or bs.cache.is_stale(cfg.stale_after_s):
            # Staleness gate (fail-open override, per backend): a stale or
            # never-fetched feed fails open in BOTH layers for that backend
            # regardless of the counter's value — a metrics outage never
            # blocks traffic (invariant #1). The counter still takes its
            # subtraction below (bookkeeping continues); the decision is
            # what fails open.
            usage_pct: float | None = None
            tier_dec = decision(None, ctx_tokens, bs.thresholds)
            auto_dec = Decision(
                allow=True, retry_after=None, active_tier=None, reason=REASON_METRICS_UNAVAILABLE
            )
        else:
            usage_pct = frac * 100.0  # fraction (0-1) -> percentage (0-100)
            tier_dec = decision(usage_pct, ctx_tokens, bs.thresholds)
            auto_dec = decision_auto(bs.counter.value(), ctx_tokens, bs.auto_policy)

        dec = combine_decisions(tier_dec, auto_dec)

        if dec.allow:
            # Charge at admission time (before the proxy await): the same
            # effective size the autoconfig layer compared (0 when ctx is
            # unknown). Subtraction is unclamped and a no-op when never
            # anchored. Rejected requests charge nothing.
            if ctx_tokens is not None:
                bs.counter.subtract(int(math.ceil(ctx_tokens * bs.auto_policy.token_margin)))
            try:
                stats.record_forwarded(endpoint, model, ctx_tokens)
            except Exception:  # noqa: BLE001 - stats must never break the request path
                log.warning("stats.record_forwarded failed", exc_info=True)
            is_stream = _is_streaming(body)
            log.debug(
                "allow backend=%s reason=%s usage_pct=%s ctx_tokens=%s stream=%s",
                backend_name,
                dec.reason,
                usage_pct,
                ctx_tokens,
                is_stream,
            )
            return await proxy_request(upstream, request, bs.base_url, stream=is_stream)

        log.info(
            "reject backend=%s reason=%s usage_pct=%s ctx_tokens=%s retry_after=%s",
            backend_name,
            dec.reason,
            usage_pct,
            ctx_tokens,
            dec.retry_after,
        )
        try:
            stats.record_rejected(endpoint, model, ctx_tokens)
        except Exception:  # noqa: BLE001 - stats must never break the request path
            log.warning("stats.record_rejected failed", exc_info=True)
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(dec.retry_after)},
            content={
                "error": {
                    "message": _reject_message(
                        dec,
                        usage_pct,
                        ctx_tokens,
                        backend=backend_name,
                        model=model,
                        remaining=bs.counter.value(),
                        capacity=bs.capacity_cache.value(),
                        target_pct=bs.target_pct,
                    ),
                    "type": "cache_pressure",
                    "code": "kv_cache_too_full",
                }
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle_generation(request, "chat_completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _handle_generation(request, "completions")

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """Gate-local aggregate of the registry's known models (always 200).

        One entry per model with an ``owned_by`` extension naming the
        resolved owning backend (a collided model appears once, under its
        resolved owner). Never proxied; the backends' own ``/v1/models``
        endpoints are not reachable through the gate.
        """
        return JSONResponse(
            status_code=200,
            content={
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "owned_by": owner}
                    for model, owner in app.state.registry.items()
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
                    {"id": model, "owned_by": owner} for model, owner in app.state.registry.items()
                ],
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        # Union all backends' by-model usage. Model ids are globally unique
        # across backends, so a key can only repeat for the synthetic
        # "default" (unlabeled usage series); take the max in that case
        # (conservative, gotcha #5) so one backend cannot silently overwrite
        # the other's gauge value.
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
        return Response(content=stats.render(), media_type=_METRICS_CONTENT_TYPE)

    return app
