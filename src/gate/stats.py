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
  ``model`` field (the future routing key); ``model_name`` is vLLM's
  served-model label (the ``model_name`` label on
  ``vllm:kv_cache_usage_perc``, ``"default"`` when unlabeled).

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

from gate.config import GateConfig

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


class GateStats:
    """Per-app ``gate_*`` metrics backed by a private ``CollectorRegistry``.

    All metrics live in a single per-app registry (never the global default),
    so multiple app instances (and tests) can coexist without cross-talk.
    """

    def __init__(self, cfg: GateConfig, *, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry if registry is not None else CollectorRegistry()

        self._requests_total = Counter(
            "gate_requests_total",
            "Generation requests decided by the gate, by endpoint, model, and result "
            "(forwarded/rejected).",
            ["endpoint", "model", "result"],
            registry=self._registry,
        )
        self._ctx_tokens = Histogram(
            "gate_request_ctx_tokens",
            "Estimated context tokens (prompt + headroom) of requests, by model and "
            "result. Median via histogram_quantile(0.5, ...).",
            ["model", "result"],
            registry=self._registry,
            buckets=CTX_TOKENS_BUCKETS,
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
            registry=self._registry,
        )
        self._age = Gauge(
            "gate_metrics_age_s",
            "Seconds since the last successful vLLM /metrics fetch (NaN if never fetched).",
            registry=self._registry,
        )

        # Tracks which model_name label series are currently set, so stale
        # models can be removed when they disappear from the feed.
        self._kv_models: set[str] = set()

        self._set_config(cfg)

        # Initialize the freshness gauges so a never-fetched feed renders
        # fresh=0.0 and age=NaN (not 0.0) before the first set_freshness call.
        self._fresh.set(0.0)
        self._age.set(float("nan"))

    def record_forwarded(self, endpoint: str, model: str, ctx_tokens: int | None) -> None:
        """Count a forwarded (allowed) request.

        The request counter always increments. The context-token histogram is
        observed only when ``ctx_tokens`` is known — fail-open unparseable
        requests have no estimate and must not skew the size distribution.
        """
        self._requests_total.labels(endpoint, model, "forwarded").inc()
        if ctx_tokens is not None:
            self._ctx_tokens.labels(model, "forwarded").observe(ctx_tokens)

    def record_rejected(self, endpoint: str, model: str, ctx_tokens: int | None) -> None:
        """Count a rejected (429) request.

        In practice rejected requests always have a ``ctx_tokens`` estimate,
        but the ``None`` guard is kept for symmetry and robustness.
        """
        self._requests_total.labels(endpoint, model, "rejected").inc()
        if ctx_tokens is not None:
            self._ctx_tokens.labels(model, "rejected").observe(ctx_tokens)

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

    def set_freshness(self, fresh: bool, age_s: float | None) -> None:
        """Set the freshness/age gauges.

        ``age_s=None`` (never fetched) renders as ``NaN`` because an unset
        unlabeled gauge would otherwise render as ``0.0``.
        """
        self._fresh.set(1.0 if fresh else 0.0)
        self._age.set(age_s if age_s is not None else float("nan"))

    def render(self) -> bytes:
        """Render the current metrics in Prometheus exposition format."""
        return generate_latest(self._registry)

    def _set_config(self, cfg: GateConfig) -> None:
        """Populate ``gate_config_info`` from the loaded config.

        Static for the process lifetime; set once at startup.
        """
        thresholds_json = json.dumps(
            [[t.kv_pct, t.max_context, t.timeout_s] for t in cfg.thresholds]
        )
        labels: dict[str, str] = {
            "vllm_host": str(cfg.vllm_host),
            "vllm_port": str(cfg.vllm_port),
            "listen_host": str(cfg.listen_host),
            "listen_port": str(cfg.listen_port),
            "metrics_poll_interval_s": str(cfg.metrics_poll_interval_s),
            "stale_after_s": str(cfg.stale_after_s),
            "chars_per_token": str(cfg.chars_per_token),
            "default_max_tokens": str(cfg.default_max_tokens),
            "thresholds_json": thresholds_json,
        }
        self._config_info.info(labels)
