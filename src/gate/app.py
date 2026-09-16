"""FastAPI app factory wiring the pure core to the HTTP edges.

This is the assembly layer: it builds the FastAPI app, owns the shared
:class:`MetricsCache`, :class:`CapacityCache`, and :class:`KvRemaining`
remaining-KV counter, the :class:`GateStats`, and the upstream
:class:`httpx.AsyncClient`, starts the background poller in the lifespan, and
routes the two generation endpoints through the two AND-combined admission
layers (the optional tiered policy :func:`gate.router.decision` and the
always-on autoconfig layer :func:`gate.router.decision_auto`, combined by
:func:`gate.router.combine_decisions`) and the transparent proxy
(:func:`gate.proxy.proxy_request`).

The decision logic lives in :mod:`gate.router`, token estimation in
:mod:`gate.tokens`, and proxying in :mod:`gate.proxy` — this module only wires
them together and owns the I/O.

**Two admission layers, AND-combined.** A request forwards iff the tiered
layer (optional; no tiers ⇒ always allow) AND the autoconfig layer both
allow. The autoconfig layer compares the request's estimated size
(``ceil(ctx * token_margin)``) against the :class:`KvRemaining` counter —
the gate's live estimate of the KV tokens still available up to the target
usage %, re-anchored by the poller on every good poll. On every forward the
counter is charged that same amount **before** the proxy await; rejected
requests charge nothing.

**Staleness gate (fail-open override).** If the usage feed is stale or never
fetched, BOTH layers fail open regardless of the counter's value (reason
``metrics_unavailable``): a metrics outage never blocks traffic (invariant
#1). The counter still takes its subtraction (bookkeeping continues) but
cannot reject while the feed is stale.

**Unconditional fail-closed.** The poller raises
:class:`~gate.metrics.CapacityUnavailableError` when an observed (HTTP 200)
``/metrics`` body lacks a usable KV-cache capacity — the standalone
``vllm:kv_cache_size_tokens`` gauge or the ``kv_cache_size_tokens`` label on
``vllm:cache_config_info``; the app's fatal done-callback
(:func:`_poller_fatal`) logs it and exits the process with status 1
(``os._exit(1)``). A merely unreachable vLLM is never fatal.

Invariants (see AGENTS.md "Known gotchas"):

- Fail open when the metric is stale or never fetched (gotcha #1) — the
  staleness gate outranks the counter.
- The vLLM usage metric is a fraction (0-1); it is converted to a percentage
  with ``frac * 100.0`` for the tiered layer (gotcha #2). The autoconfig
  path's single percentage→fraction step is
  ``target_kv_cache_pct / 100.0`` at poller start.
- A 429 carries the combined ``Retry-After`` (the max of the rejecting
  layers' retries) and a rejector-aware message: the tiered text when the
  tier is the reported rejector, the autoconfig text (usage %, remaining,
  capacity, target) otherwise.
- Upstream status is propagated unmasked and ``stream: true`` SSE is not
  buffered (gotcha #6).

The app also exposes ``GET /metrics``, a Prometheus endpoint serving
in-memory ``gate_*`` stats from :mod:`gate.stats` (including
``gate_kv_cache_remaining_tokens``, set at scrape time from the counter).
Each allow/reject decision is recorded into the stats layer as a pure side
effect — recording never alters the decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from gate.config import GateConfig
from gate.metrics import CapacityCache, KvRemaining, MetricsCache
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
    remaining: int | None = None,
    capacity: int | None = None,
    target_pct: float | None = None,
) -> str:
    """Single-line human-readable 429 message, rejector-aware.

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
            f"vLLM KV cache too full for a {ctx_str}-token request "
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
            f"vLLM KV cache headroom exhausted for a ~{ctx_tokens}-token request "
            f"(usage {usage_str}%, ~{remaining_str} of {capacity_str} tokens "
            f"remaining up to the {target_str}% target). "
            f"Retry in {retry}s."
        )
    retry = dec.retry_after if dec.retry_after is not None else 0
    return f"vLLM KV cache too full. Retry in {retry}s."


def _poller_fatal(task: asyncio.Task[None]) -> None:
    """Done-callback for the poller task: exit 1 on a pre-shutdown crash.

    If the task finished with an exception other than
    ``asyncio.CancelledError`` (e.g. the poller's unconditional
    :class:`~gate.metrics.CapacityUnavailableError`), the error is logged and
    the process exits with status 1 (``os._exit(1)``). A clean shutdown
    (cancellation) must NOT trigger the exit.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error("poller task died: %s; exiting", exc, exc_info=exc)
        os._exit(1)


def create_app(
    cfg: GateConfig,
    *,
    cache: MetricsCache | None = None,
    upstream: httpx.AsyncClient | None = None,
    start_poller: bool = True,
    stats: GateStats | None = None,
    counter: KvRemaining | None = None,
    capacity_cache: CapacityCache | None = None,
) -> FastAPI:
    """Build the gate FastAPI app.

    Args:
        cfg: Validated gate configuration.
        cache: Shared :class:`MetricsCache`; a fresh one is created when None.
        upstream: Shared :class:`httpx.AsyncClient` for proxying and polling;
            a fresh one is created when None.
        start_poller: When True the poller task is started in the lifespan.
            Tests pass False so the poller does not interfere.
        stats: A :class:`GateStats` for the ``/metrics`` endpoint; a fresh one
            is created when None.
        counter: Shared :class:`KvRemaining` remaining-KV counter for the
            autoconfig layer; a fresh one is created when None.
        capacity_cache: Shared :class:`CapacityCache` for the KV-cache
            capacity gauge; a fresh one is created when None.

    The poller is started by the lifespan (which uvicorn / TestClient trigger),
    so :func:`gate.main.main` does not start it separately.
    """
    if cache is None:
        cache = MetricsCache()
    if upstream is None:
        upstream = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    if stats is None:
        stats = GateStats(cfg)
    if counter is None:
        counter = KvRemaining()
    if capacity_cache is None:
        capacity_cache = CapacityCache()
    base_url = f"http://{cfg.vllm_host}:{cfg.vllm_port}"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_poller:
            # The single percentage→fraction step in the autoconfig path:
            # target_kv_cache_pct (0-100) -> fraction (0-1) at anchor time.
            poller_task = asyncio.create_task(
                run_poller(
                    upstream,
                    cache,
                    base_url + "/metrics",
                    cfg.metrics_poll_interval_s,
                    counter=counter,
                    target_frac=cfg.target_kv_cache_pct / 100.0,
                    capacity_cache=capacity_cache,
                )
            )
            poller_task.add_done_callback(_poller_fatal)
            app.state.poller_task = poller_task
        else:
            app.state.poller_task = None
        try:
            yield
        finally:
            task = app.state.poller_task
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await upstream.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.cfg = cfg
    app.state.cache = cache
    app.state.upstream = upstream
    app.state.base_url = base_url
    app.state.stats = stats
    app.state.counter = counter
    app.state.capacity_cache = capacity_cache

    async def _handle_generation(request: Request, endpoint: str) -> Response:
        body = await request.body()
        model = extract_request_model(body)
        ctx_tokens = estimate_context_tokens(body, cfg)

        frac = cache.value()
        if frac is None or cache.is_stale(cfg.stale_after_s):
            # Staleness gate (fail-open override): a stale or never-fetched
            # feed fails open in BOTH layers regardless of the counter's
            # value — a metrics outage never blocks traffic (invariant #1).
            # The counter still takes its subtraction below (bookkeeping
            # continues); the decision is what fails open.
            usage_pct: float | None = None
            tier_dec = decision(None, ctx_tokens, cfg.thresholds)
            auto_dec = Decision(
                allow=True, retry_after=None, active_tier=None, reason=REASON_METRICS_UNAVAILABLE
            )
        else:
            usage_pct = frac * 100.0  # fraction (0-1) -> percentage (0-100)
            tier_dec = decision(usage_pct, ctx_tokens, cfg.thresholds)
            auto_dec = decision_auto(
                counter.value(),
                ctx_tokens,
                AutoPolicy(cfg.token_margin, cfg.retry_min_s, cfg.retry_max_s),
            )

        dec = combine_decisions(tier_dec, auto_dec)

        if dec.allow:
            # Charge at admission time (before the proxy await): the same
            # effective size the autoconfig layer compared (0 when ctx is
            # unknown). Subtraction is unclamped and a no-op when never
            # anchored. Rejected requests charge nothing.
            if ctx_tokens is not None:
                counter.subtract(int(math.ceil(ctx_tokens * cfg.token_margin)))
            try:
                stats.record_forwarded(endpoint, model, ctx_tokens)
            except Exception:  # noqa: BLE001 - stats must never break the request path
                log.warning("stats.record_forwarded failed", exc_info=True)
            is_stream = _is_streaming(body)
            log.debug(
                "allow reason=%s usage_pct=%s ctx_tokens=%s stream=%s",
                dec.reason,
                usage_pct,
                ctx_tokens,
                is_stream,
            )
            return await proxy_request(upstream, request, base_url, stream=is_stream)

        log.info(
            "reject reason=%s usage_pct=%s ctx_tokens=%s retry_after=%s",
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
                        remaining=counter.value(),
                        capacity=capacity_cache.value(),
                        target_pct=cfg.target_kv_cache_pct,
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

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "metrics_age_s": cache.age(),
                "kv_usage": cache.value(),
                "kv_cache_capacity_tokens": capacity_cache.value(),
                "kv_cache_remaining_tokens": counter.value(),
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        stats.set_kv_usage(cache.by_model())
        stats.set_freshness(not cache.is_stale(cfg.stale_after_s), cache.age())
        stats.set_remaining(counter.value())
        return Response(content=stats.render(), media_type=_METRICS_CONTENT_TYPE)

    return app
