"""Stats edge: in-memory ``gate_*`` metrics in Prometheus exposition format.

This module is the "stats edge" of the gate. It owns a per-app
:class:`prometheus_client.CollectorRegistry` and exposes ``gate_*`` metrics
that the app layer records into as it makes decisions and as the poller
refreshes the vLLM metrics feed. :meth:`GateStats.render` serializes the
registry in Prometheus text exposition format for a ``/metrics`` endpoint.

Invariants (see AGENTS.md "Known gotchas"):

- Stats are **in-memory only** and reset on process restart. This is by
  design: Prometheus ``rate()``/``increase()``/``histogram_quantile()``
  handle restarts correctly, so no persistence is needed.
- Recording is a **pure side effect that must never alter the decision**.
  The decision is computed by :mod:`gate.router`; the stats layer only
  observes it. A stats failure must not change allow/reject.
- Two model namespaces, kept distinct: ``model`` is the request body's
  ``model`` field (the routing key); ``model_name`` is vLLM's
  served-model label (the ``model_name`` label on
  ``vllm:kv_cache_usage_perc``, ``"default"`` when unlabeled).

- Multi-backend (see docs/plans/multi-backend.md §2.5): the feed gauges
  (``gate_metrics_fresh``, ``gate_metrics_age_s``) and the remaining-KV gauge
  (``gate_kv_cache_remaining_tokens``) carry a ``backend`` label — each
  backend is set on every ``/metrics`` render, so a per-backend series is
  always present once the app has backends. ``gate_kv_cache_usage_pct`` stays
  model-keyed (keyed by vLLM's ``model_name`` label — the two-namespace rule
  still holds). The request counters (``gate_requests_total``,
  ``gate_request_ctx_tokens``) carry a ``backend`` label **alongside**
  ``model`` (decision 13): with duplicate model ids legal (decision 11), the
  backend is no longer recoverable from the model. The label is the final
  backend — the one that forwarded, the max-timeout rejector on 429
  exhaustion, or the sentinel ``"none"`` on 502 exhaustion (no rejector
  exists). ``gate_routing_failovers_total{model, from, to, reason}`` counts
  pre-stream failover skips (reason ∈ ``transport`` | ``upstream_5xx`` |
   ``reject``; decision 12). ``gate_backend_capacity_unavailable{backend}`` is
   the per-backend alert flag for a live-but-capacity-less ``/metrics`` body
   (per-backend fail-open, decision 10). ``gate_backend_engine{backend,
   engine}`` (engine ∈ ``vllm`` | ``sglang`` | ``unknown``) is the per-backend
   detected-engine gauge — ``1`` for the currently detected engine, ``0`` for
   the others (``unknown=1`` is the initial state before any detection;
   docs/plans/sglang-backend.md §2.5). ``gate_metrics_endpoint_unavailable{
   backend}`` is the per-backend alert flag for a reachable-but-unrecognizable
   ``/metrics`` endpoint (non-200, or a 200 body with no recognizable KV-cache
   gauge — e.g. SGLang launched without ``--enable-metrics``); ``0`` once a
   real engine is detected. ``gate_config_info`` serializes the backend
   structure and the ``routing:`` section (``routing_json``:
    ``[[model, policy, [order...]], ...]``, startup-static for operator audit).
    ``gate_rejections_total{model, backend, reason, tier_kv_pct}`` counts
    rejected requests by refusal reason (``exceeds_tier`` = the tiered
    "max context" layer, ``auto_exceeds_headroom`` = the autoconfig headroom
    layer, ``all_backends_failed`` = every candidate transport-failed on 502
    exhaustion); ``tier_kv_pct`` is the governing tier's ``kv_pct``
    percentage when the tiered layer is the rejector, else ``"0"``.

- A per-app registry is required: the app creates one ``GateStats`` per
  process, and tests create many apps, so the global default registry must
  never be used (it would leak state across apps and tests).
"""

from __future__ import annotations

import json

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from gate.config import Backend, GateConfig

#: Histogram buckets for estimated context-token sizes, covering typical LLM
#: contexts (128 tokens up to 256k).
CTX_TOKENS_BUCKETS: tuple[int, ...] = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)

#: The engine label values for ``gate_backend_engine`` (docs/plans/sglang-backend.md §2.5).
#: Kept local (not imported from gate.metrics) so this module stays independent.
_ENGINE_LABELS: tuple[str, ...] = ("vllm", "sglang", "unknown")


class GateStats:
    """Per-app ``gate_*`` metrics backed by a private ``CollectorRegistry``.

    All metrics live in a single per-app registry (never the global default),
    so multiple app instances (and tests) can coexist without cross-talk.
    """

    def __init__(self, cfg: GateConfig, *, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry if registry is not None else CollectorRegistry()

        self._requests_total = Counter(
            "gate_requests_total",
            "Generation requests decided by the gate, by endpoint, model, result "
            "(forwarded/rejected), and backend (the final backend: the forwarder, "
            "the max-timeout rejector on 429 exhaustion, or 'none' on 502 "
            "exhaustion).",
            ["endpoint", "model", "result", "backend"],
            registry=self._registry,
        )
        self._ctx_tokens = Histogram(
            "gate_request_ctx_tokens",
            "Estimated context tokens (prompt + headroom) of requests, by model, "
            "result, and backend. Median via histogram_quantile(0.5, ...).",
            ["model", "result", "backend"],
            registry=self._registry,
            buckets=CTX_TOKENS_BUCKETS,
        )
        self._routing_failovers = Counter(
            "gate_routing_failovers_total",
            "Pre-stream failover skips: a candidate rejected (429), transport-failed, "
            "or returned a pre-stream upstream 5xx, so the walk moved to the next "
            "candidate (reason: reject | transport | upstream_5xx).",
            ["model", "from", "to", "reason"],
            registry=self._registry,
        )
        self._rejections_total = Counter(
            "gate_rejections_total",
            "Rejected requests by the gate, by model, backend, and refusal reason "
            "(exceeds_tier = the tiered 'max context' layer; auto_exceeds_headroom = the "
            "autoconfig headroom layer; all_backends_failed = every candidate "
            "transport-failed). The tier_kv_pct label is the governing tier's kv_pct "
            "percentage when the tiered layer is the rejector, else '0'.",
            ["model", "backend", "reason", "tier_kv_pct"],
            registry=self._registry,
        )
        self._kv_usage = Gauge(
            "gate_kv_cache_usage_pct",
            "Current KV-cache usage percentage (0-100) per served model "
            "(vLLM model_name label; 'default' when unlabeled).",
            ["model_name"],
            registry=self._registry,
        )
        self._config_info = Info(
            "gate_config",
            "Loaded gate configuration (set once at startup).",
            registry=self._registry,
        )
        self._fresh = Gauge(
            "gate_metrics_fresh",
            "1 if the vLLM metrics feed is fresh, 0 if stale or never fetched.",
            ["backend"],
            registry=self._registry,
        )
        self._age = Gauge(
            "gate_metrics_age_s",
            "Seconds since the last successful vLLM /metrics fetch (NaN if never fetched).",
            ["backend"],
            registry=self._registry,
        )
        self._remaining = Gauge(
            "gate_kv_cache_remaining_tokens",
            "The gate's own live estimate of remaining KV-cache tokens up to the target "
            "(post-subtraction, pre-reanchor). NaN until anchored.",
            ["backend"],
            registry=self._registry,
        )
        self._capacity_unavailable = Gauge(
            "gate_backend_capacity_unavailable",
            "1 if the backend's observed /metrics body lacks a usable KV-cache "
            "capacity (its autoconfig layer fails open), 0 otherwise.",
            ["backend"],
            registry=self._registry,
        )
        self._backend_engine = Gauge(
            "gate_backend_engine",
            "The currently detected inference engine per backend (1 for the detected "
            "engine, 0 for the others; engine is auto-detected from the /metrics body).",
            ["backend", "engine"],
            registry=self._registry,
        )
        self._metrics_endpoint_unavailable = Gauge(
            "gate_metrics_endpoint_unavailable",
            "1 if the backend's /metrics endpoint is unavailable (a non-200 response, "
            "e.g. SGLang launched without --enable-metrics, or a 200 body with no "
            "recognizable KV-cache gauge), 0 once a real engine is detected.",
            ["backend"],
            registry=self._registry,
        )

        # Tracks which model_name label series are currently set, so stale
        # models can be removed when they disappear from the feed.
        self._kv_models: set[str] = set()

        self._set_config(cfg)

    def record_forwarded(
        self, endpoint: str, model: str, ctx_tokens: int | None, *, backend: str
    ) -> None:
        """Count a forwarded (allowed) request.

        ``backend`` is the backend the request was forwarded to (decision 13:
        with duplicate model ids legal, the backend is no longer recoverable
        from the model). The request counter always increments. The
        context-token histogram is observed only when ``ctx_tokens`` is known
        — fail-open unparseable requests have no estimate and must not skew
        the size distribution.
        """
        self._requests_total.labels(endpoint, model, "forwarded", backend).inc()
        if ctx_tokens is not None:
            self._ctx_tokens.labels(model, "forwarded", backend).observe(ctx_tokens)

    def record_rejected(
        self, endpoint: str, model: str, ctx_tokens: int | None, *, backend: str
    ) -> None:
        """Count a rejected request (429 exhaustion, or 502 exhaustion).

        ``backend`` is the reported rejector's backend on 429 exhaustion, or
        the sentinel ``"none"`` on 502 exhaustion (no rejector exists). In
        practice rejected requests always have a ``ctx_tokens`` estimate, but
        the ``None`` guard is kept for symmetry and robustness.
        """
        self._requests_total.labels(endpoint, model, "rejected", backend).inc()
        if ctx_tokens is not None:
            self._ctx_tokens.labels(model, "rejected", backend).observe(ctx_tokens)

    def record_rejection(
        self,
        model: str,
        backend: str,
        reason: str,
        *,
        tier_kv_pct: str = "0",
    ) -> None:
        """Count one rejected request by its refusal reason.

        ``reason`` is the reported rejector's machine-readable code
        (``exceeds_tier`` / ``auto_exceeds_headroom``) or the sentinel
        ``all_backends_failed`` on 502 exhaustion. ``tier_kv_pct`` is the
        governing tier's ``kv_pct`` (percentage) as a string when the tiered
        layer is the rejector, else ``"0"``. Pure side effect — never alters
        the decision.
        """
        self._rejections_total.labels(model, backend, reason, tier_kv_pct).inc()

    def record_failover(self, model: str, from_backend: str, to_backend: str, reason: str) -> None:
        """Count one pre-stream failover skip (decision 12).

        ``reason`` is one of ``"reject"`` (the candidate 429'd — skip-on-
        reject), ``"transport"`` (the candidate's proxy transport-failed
        pre-stream), or ``"upstream_5xx"`` (the candidate returned a 5xx
        before the first response byte).
        """
        self._routing_failovers.labels(model, from_backend, to_backend, reason).inc()

    def set_kv_usage(self, by_model: dict[str, float]) -> None:
        """Set the per-model fill gauges from the cache's by-model fractions.

        ``by_model`` maps ``model_name`` to a usage **fraction (0-1)**; each is
        converted to a percentage (``frac * 100.0``) for the gauge. Series for
        models no longer present in the feed are removed so the exposition
        does not accumulate stale label values.
        """
        new = set(by_model)
        for m in self._kv_models - new:
            self._kv_usage.remove(m)
        for m, frac in by_model.items():
            self._kv_usage.labels(m).set(frac * 100.0)
        self._kv_models = new

    def set_freshness(self, backend: str, fresh: bool, age_s: float | None) -> None:
        """Set one backend's freshness/age gauges.

        ``age_s=None`` (never fetched) renders as ``NaN`` because an unset
        labeled series would otherwise not render at all. The app calls this
        for every backend on each ``/metrics`` render.
        """
        self._fresh.labels(backend).set(1.0 if fresh else 0.0)
        self._age.labels(backend).set(age_s if age_s is not None else float("nan"))

    def set_remaining(self, backend: str, value: int | None) -> None:
        """Set one backend's estimated-remaining-KV gauge.

        ``None`` (never anchored) renders ``NaN`` so a never-anchored backend
        is distinguishable from a genuinely empty pool.
        """
        self._remaining.labels(backend).set(float(value) if value is not None else float("nan"))

    def set_backend_capacity_unavailable(self, backend: str, unavailable: bool) -> None:
        """Set one backend's capacity-unavailable alert flag (0/1).

        The poller's per-backend callback flips the app's flag when an
        observed (HTTP 200) ``/metrics`` body lacks a usable KV-cache
        capacity; the app surfaces the flag here on every ``/metrics``
        render (decision 10 of docs/plans/multi-backend.md).
        """
        self._capacity_unavailable.labels(backend).set(1.0 if unavailable else 0.0)

    def set_backend_engine(self, backend: str, engine: str) -> None:
        """Set one backend's detected-engine gauge (docs/plans/sglang-backend.md §2.5).

        Sets the currently detected ``engine`` series to ``1.0`` and the other
        engine series to ``0.0`` so the gauge is always fully populated and
        scrape-able (``unknown=1`` is the initial state before any detection).
        The app calls this for every backend on each ``/metrics`` render.
        """
        for label in _ENGINE_LABELS:
            self._backend_engine.labels(backend, label).set(1.0 if label == engine else 0.0)

    def set_backend_metrics_unavailable(self, backend: str, unavailable: bool) -> None:
        """Set one backend's metrics-endpoint-unavailable alert flag (0/1).

        ``1`` while the backend's ``/metrics`` is reachable-but-unrecognizable
        (non-200, or a 200 body with no recognizable KV gauge); ``0`` once a real
        engine is detected. The app calls this for every backend on each
        ``/metrics`` render (docs/plans/sglang-backend.md §2.5).
        """
        self._metrics_endpoint_unavailable.labels(backend).set(1.0 if unavailable else 0.0)

    def render(self) -> bytes:
        """Render the current metrics in Prometheus exposition format."""
        return generate_latest(self._registry)

    def _set_config(self, cfg: GateConfig) -> None:
        """Populate ``gate_config_info`` from the loaded config.

        Static for the process lifetime; set once at startup.

        The ``backends_json`` label serializes the backend structure: per
        backend the name, host, port, default flag, and which knobs are
        overridden per-backend (vs the global default). The ``models:`` lists
        are deliberately omitted — they are discovered at runtime and would
        go stale (keep the info metric small and startup-static).
        ``cfg.backends`` is always non-empty (config validation, decision 6).

        The ``routing_json`` label serializes the ``routing:`` section
        (decision 9/15): a JSON list of ``[model, policy, [order...]]`` per
        entry — startup-static (the config is loaded once), useful for
        operator audit. An empty ``routing:`` section renders as ``[]``.
        """
        thresholds_json = json.dumps(
            [[t.kv_pct, t.max_context, t.timeout_s] for t in cfg.thresholds]
        )
        backends_json = json.dumps([_backend_info(b) for b in cfg.backends], separators=(",", ":"))
        routing_json = json.dumps(
            [[e.model, e.spec.policy, list(e.spec.order)] for e in cfg.routing],
            separators=(",", ":"),
        )
        labels: dict[str, str] = {
            "listen_host": str(cfg.listen_host),
            "listen_port": str(cfg.listen_port),
            "metrics_poll_interval_s": str(cfg.metrics_poll_interval_s),
            "stale_after_s": str(cfg.stale_after_s),
            "model_refresh_interval_s": str(cfg.model_refresh_interval_s),
            "chars_per_token": str(cfg.chars_per_token),
            "default_max_tokens": str(cfg.default_max_tokens),
            "thresholds_json": thresholds_json,
            "backends_json": backends_json,
            "routing_json": routing_json,
        }
        self._config_info.info(labels)


def _backend_info(b: Backend) -> dict[str, object]:
    """One backend's startup-static config info (no ``models`` lists).

    ``overrides`` names the knobs the backend sets per-backend instead of
    inheriting the global default (``None`` on the dataclass = global).
    """
    overrides = [
        name
        for name, value in (
            ("thresholds", b.thresholds),
            ("target_kv_cache_pct", b.target_kv_cache_pct),
            ("token_margin", b.token_margin),
            ("retry_min_s", b.retry_min_s),
            ("retry_max_s", b.retry_max_s),
        )
        if value is not None
    ]
    return {
        "name": b.name,
        "host": b.host,
        "port": b.port,
        "default": b.default,
        "overrides": overrides,
    }
