"""FastAPI app factory wiring the pure core to the HTTP edges.

This is the assembly layer: it builds the FastAPI app, owns the shared
:meth:`MetricsCache` and the upstream :class:`httpx.AsyncClient`, starts the
background poller in the lifespan, and routes the two generation endpoints
through the pure decision logic (:func:`gate.router.decision`) and the
transparent proxy (:func:`gate.proxy.proxy_request`).

The decision logic lives in :mod:`gate.router`, token estimation in
:mod:`gate.tokens`, and proxying in :mod:`gate.proxy` — this module only wires
them together and owns the I/O.

Invariants (see AGENTS.md "Known gotchas"):

- Fail open when the metric is stale or never fetched (gotcha #1).
- The vLLM metric is a fraction (0-1); it is converted to a percentage with
  ``frac * 100.0`` before :func:`gate.router.decision` (gotcha #2).
- A 429 uses the governing tier's ``timeout_s`` as ``Retry-After`` (gotcha #3).
- Upstream status is propagated unmasked and ``stream: true`` SSE is not
  buffered (gotcha #6).

The app also exposes ``GET /metrics``, a Prometheus endpoint serving in-memory
``gate_*`` stats from :mod:`gate.stats`. Each allow/reject decision is recorded
into the stats layer as a pure side effect — recording never alters the
decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from gate.config import GateConfig
from gate.metrics import MetricsCache
from gate.poller import run_poller
from gate.proxy import proxy_request
from gate.router import Decision, decision
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


def _reject_message(dec: Decision, usage_pct: float | None, ctx_tokens: int | None) -> str:
    """Single-line human-readable 429 message.

    Uses the governing tier's details when available; falls back to a generic
    message when ``dec.active_tier`` is None (defensive — a reject always has
    a governing tier in :func:`gate.router.decision`, but the message must not
    crash if that ever changes).
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
    retry = dec.retry_after if dec.retry_after is not None else 0
    return f"vLLM KV cache too full. Retry in {retry}s."


def create_app(
    cfg: GateConfig,
    *,
    cache: MetricsCache | None = None,
    upstream: httpx.AsyncClient | None = None,
    start_poller: bool = True,
    stats: GateStats | None = None,
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

    The poller is started by the lifespan (which uvicorn / TestClient trigger),
    so :func:`gate.main.main` does not start it separately.
    """
    if cache is None:
        cache = MetricsCache()
    if upstream is None:
        upstream = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    if stats is None:
        stats = GateStats(cfg)
    base_url = f"http://{cfg.vllm_host}:{cfg.vllm_port}"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if start_poller:
            app.state.poller_task = asyncio.create_task(
                run_poller(upstream, cache, base_url + "/metrics", cfg.metrics_poll_interval_s)
            )
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

    async def _handle_generation(request: Request, endpoint: str) -> Response:
        body = await request.body()
        model = extract_request_model(body)
        ctx_tokens = estimate_context_tokens(body, cfg)

        frac = cache.value()
        if frac is None or cache.is_stale(cfg.stale_after_s):
            usage_pct: float | None = None  # fail open
        else:
            usage_pct = frac * 100.0

        dec = decision(usage_pct, ctx_tokens, cfg.thresholds)

        if dec.allow:
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
                    "message": _reject_message(dec, usage_pct, ctx_tokens),
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
            },
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        stats.set_kv_usage(cache.by_model())
        stats.set_freshness(not cache.is_stale(cfg.stale_after_s), cache.age())
        return Response(content=stats.render(), media_type=_METRICS_CONTENT_TYPE)

    return app
