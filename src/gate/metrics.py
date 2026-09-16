"""Parsing and caching of vLLM's KV-cache metrics.

Pure core: :func:`parse_kv_cache_usage`, :func:`parse_kv_cache_usage_by_model`,
and :func:`parse_kv_cache_capacity` are dependency-free (aside from the
``prometheus_client`` text parser) functions over a Prometheus text
exposition. :class:`MetricsCache` remembers the last good usage value, its
per-model breakdown, and its fetch time so the poller can drive fail-open
behavior via staleness; :class:`CapacityCache` remembers the last good KV-cache
capacity (tokens). :func:`fetch_usage`, :func:`fetch_usage_by_model`, and
:func:`fetch_metrics` are the thin edges that pull ``/metrics`` from vLLM.
:class:`KvRemaining` is the in-memory estimated-remaining-KV counter used by
the autoconfig admission layer.

Invariants (see AGENTS.md "Known gotchas"):

- The usage metric is a **fraction (0-1)**; the gate's config uses a
  percentage. The capacity metric is a **token count** (no percentage
  conversion applies).
- :func:`parse_kv_cache_usage` and :func:`parse_kv_cache_capacity` take the
  **MAX across all series** (vLLM may emit one series per ``model_name``);
  :func:`parse_kv_cache_usage_by_model` keys by ``model_name`` (or
  ``"default"`` when unlabeled) and keeps the per-model max.
- Any parse failure, absent metric, or all-non-finite values yield ``None``;
  the poller then fails open (usage) or treats the capacity as unknown.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

#: Exact family name of the vLLM KV-cache usage gauge.
KV_CACHE_USAGE_METRIC = "vllm:kv_cache_usage_perc"

#: Exact family name of the vLLM KV-cache capacity gauge (total tokens).
KV_CACHE_CAPACITY_METRIC = "vllm:kv_cache_size_tokens"

#: Family name of the vLLM cache-config info gauge. Current vLLM exposes the
#: KV-cache capacity (in tokens) as the ``kv_cache_size_tokens`` *label* on
#: this gauge rather than as a standalone gauge (vllm-project/vllm PR #42206).
KV_CACHE_CONFIG_INFO_METRIC = "vllm:cache_config_info"

#: Label carrying the KV-cache capacity (in tokens) on ``vllm:cache_config_info``.
KV_CACHE_SIZE_TOKENS_LABEL = "kv_cache_size_tokens"


def parse_kv_cache_usage(text: str) -> float | None:
    """Extract the max ``vllm:kv_cache_usage_perc`` sample from a text exposition.

    Returns the maximum finite sample value across all label sets (series) of
    the ``vllm:kv_cache_usage_perc`` family. Returns ``None`` when the text is
    unparseable, the metric is absent, it has no samples, or every sample is
    non-finite (NaN/Inf).
    """
    try:
        families = list(text_string_to_metric_families(text))
    except ValueError:
        return None
    values: list[float] = []
    for family in families:
        if family.name != KV_CACHE_USAGE_METRIC:
            continue
        for sample in family.samples:
            if math.isfinite(sample.value):
                values.append(sample.value)
    if not values:
        return None
    return max(values)


def parse_kv_cache_usage_by_model(text: str) -> dict[str, float] | None:
    """Extract per-model ``vllm:kv_cache_usage_perc`` samples from a text exposition.

    Pure. Returns a mapping of ``model_name`` label value to the maximum
    finite sample value for that model; unlabeled (or empty-label) series map
    to ``"default"``. Returns ``None`` on the same failure conditions as
    :func:`parse_kv_cache_usage` (unparseable text, metric absent, or no
    finite samples). This is the per-model counterpart of
    :func:`parse_kv_cache_usage`, which still returns the overall max.
    """
    try:
        families = list(text_string_to_metric_families(text))
    except ValueError:
        return None
    by_model: dict[str, float] = {}
    for family in families:
        if family.name != KV_CACHE_USAGE_METRIC:
            continue
        for sample in family.samples:
            if not math.isfinite(sample.value):
                continue
            model = sample.labels.get("model_name") or "default"
            current = by_model.get(model)
            if current is None or sample.value > current:
                by_model[model] = sample.value
    if not by_model:
        return None
    return by_model


def _finite_samples(text: str, family_name: str) -> list[float]:
    """Collect the finite sample values of ``family_name`` from a text exposition.

    Returns ``[]`` when the text is unparseable or the family has no finite
    samples. Shared by the gauge parsers so "finite" has one definition.
    """
    try:
        families = list(text_string_to_metric_families(text))
    except ValueError:
        return []
    values: list[float] = []
    for family in families:
        if family.name != family_name:
            continue
        for sample in family.samples:
            if math.isfinite(sample.value):
                values.append(sample.value)
    return values


def _capacity_from_config_info_labels(text: str) -> int | None:
    """Read the ``kv_cache_size_tokens`` label off ``vllm:cache_config_info``.

    Current vLLM (PR #42206) exposes the KV-cache capacity as a *label* on the
    ``vllm:cache_config_info`` info gauge (sample value ``1.0``) rather than as
    a standalone ``vllm:kv_cache_size_tokens`` gauge. Returns the maximum
    positive integer found across all samples' ``kv_cache_size_tokens`` labels,
    or ``None`` when the family is absent, the label is missing or non-numeric
    (e.g. ``"None"``), or no positive value is present.
    """
    try:
        families = list(text_string_to_metric_families(text))
    except ValueError:
        return None
    values: list[int] = []
    for family in families:
        if family.name != KV_CACHE_CONFIG_INFO_METRIC:
            continue
        for sample in family.samples:
            raw = sample.labels.get(KV_CACHE_SIZE_TOKENS_LABEL)
            if raw is None:
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                values.append(value)
    if not values:
        return None
    return max(values)


def parse_kv_cache_capacity(text: str) -> int | None:
    """Extract the KV-cache capacity (total tokens) from a text exposition.

    Tries the standalone ``vllm:kv_cache_size_tokens`` gauge first (maximum
    finite **positive** sample value across all series, as an ``int``). When
    that gauge is absent or has no positive sample, falls back to the
    ``kv_cache_size_tokens`` **label** on the ``vllm:cache_config_info`` info
    gauge — the form current vLLM actually emits (PR #42206). Returns ``None``
    when neither source yields a positive integer.
    """
    values = [v for v in _finite_samples(text, KV_CACHE_CAPACITY_METRIC) if v > 0]
    if values:
        return int(max(values))
    return _capacity_from_config_info_labels(text)


@dataclass(frozen=True)
class MetricsSample:
    """One observed ``/metrics`` body: usage fraction, per-model breakdown, and capacity."""

    observed: bool
    usage_frac: float | None
    by_model: dict[str, float] | None
    capacity_tokens: int | None


class MetricsCache:
    """Last-known-good cache of the KV-cache usage fraction.

    Single-threaded asyncio usage: only the poller task writes and request
    handlers read, so no locking is needed.

    ``update`` with ``None`` is a no-op — the last good value is kept and its
    timestamp is NOT refreshed. Fail-open is driven by :meth:`is_stale`, not by
    clearing the value on a failed fetch. ``update_by_model`` stores the
    per-model breakdown alongside the overall value; ``None``/empty input is
    likewise a no-op.
    """

    def __init__(self) -> None:
        self._value: float | None = None
        self._fetched_at: float | None = None
        self._by_model: dict[str, float] = {}

    def update(self, value: float | None, *, now: float | None = None) -> None:
        """Store ``value`` with a fetch timestamp; ``None`` is a no-op.

        Note: this updates only the single-value (MAX) path and does NOT touch
        the per-model ``_by_model`` mapping (use :meth:`update_by_model` for
        that).
        """
        if value is None:
            return
        self._value = value
        self._fetched_at = time.monotonic() if now is None else now

    def update_by_model(
        self, mapping: dict[str, float] | None, *, now: float | None = None
    ) -> None:
        """Store per-model usage fractions with a fetch timestamp.

        ``None`` or an empty mapping is a no-op that keeps the last good
        values and does not refresh the timestamp (fail-open is driven by
        staleness, not by clearing).
        """
        if not mapping:
            return
        self._by_model = dict(mapping)
        self._value = max(mapping.values())
        self._fetched_at = time.monotonic() if now is None else now

    def by_model(self) -> dict[str, float]:
        """Snapshot of the per-model usage fractions (a copy)."""
        return dict(self._by_model)

    def value(self) -> float | None:
        """Last stored value, or ``None`` if never updated with a real value."""
        return self._value

    def age(self, *, now: float | None = None) -> float | None:
        """Seconds since the last successful update, or ``None`` if never fetched."""
        if self._fetched_at is None:
            return None
        return (time.monotonic() if now is None else now) - self._fetched_at

    def is_stale(self, stale_after_s: float, *, now: float | None = None) -> bool:
        """True if never fetched or strictly older than ``stale_after_s``."""
        age = self.age(now=now)
        if age is None:
            return True
        return age > stale_after_s


class CapacityCache:
    """Last-known-good cache of the KV-cache capacity (total tokens).

    Same shape and single-asyncio-loop contract as :class:`MetricsCache`, but
    for an ``int | None`` token count. Capacity is never treated as *stale* —
    only as unknown (``None``) — so there is no :meth:`is_stale`.

    ``update`` with ``None`` is a no-op — the last good value is kept and its
    timestamp is NOT refreshed.
    """

    def __init__(self) -> None:
        self._value: int | None = None
        self._fetched_at: float | None = None

    def update(self, value: int | None, *, now: float | None = None) -> None:
        """Store ``value`` with a fetch timestamp; ``None`` is a no-op."""
        if value is None:
            return
        self._value = value
        self._fetched_at = time.monotonic() if now is None else now

    def value(self) -> int | None:
        """Last stored value, or ``None`` if never updated with a real value."""
        return self._value

    def age(self, *, now: float | None = None) -> float | None:
        """Seconds since the last successful update, or ``None`` if never fetched."""
        if self._fetched_at is None:
            return None
        return (time.monotonic() if now is None else now) - self._fetched_at


class KvRemaining:
    """In-memory estimate of the KV tokens still available up to the target.

    Re-anchored by the poller on every successful observed poll (reset to the
    real-derived value) and decremented by the estimated size of each
    forwarded request. Subtraction is **not clamped** — the value may go
    negative (an over-committed signal). :meth:`add` is the inverse of
    :meth:`subtract`: the app re-adds a charge when a pre-stream failover
    rolls back a forwarded-but-failed request's charge.

    Single-asyncio-loop contract: only the poller reanchors and request
    handlers subtract (or re-add on a failover charge rollback); the
    read-modify-write is synchronous with no ``await`` between the read and
    the write, so no locking is needed — the same contract as
    :class:`MetricsCache`.
    """

    def __init__(self) -> None:
        self._value: int | None = None

    def reanchor(self, tokens: int) -> None:
        """Reset the estimate to ``tokens`` (poller, each good poll)."""
        self._value = tokens

    def unanchor(self) -> None:
        """Reset to the never-anchored state (poller, on a capacity-missing observed body).

        The autoconfig layer fails open while the counter is unanchored
        (``decision_auto`` allows with reason ``auto_unanchored``), so a stale
        anchor from an earlier good poll can never drive a real (fail-closed)
        decision off stale data.
        """
        self._value = None

    def subtract(self, tokens: int) -> None:
        """Charge ``tokens`` to the estimate (app, on every forwarded request).

        No-op when never anchored (``None``). Not clamped: the value may go
        negative.
        """
        if self._value is not None:
            self._value -= tokens

    def add(self, tokens: int) -> None:
        """Re-add ``tokens`` to the estimate (app, failover charge rollback after a
        pre-stream transport failure).

        No-op when never anchored (``None``). Same single-asyncio-loop
        contract as :meth:`subtract`.
        """
        if self._value is not None:
            self._value += tokens

    def value(self) -> int | None:
        """Current estimate, or ``None`` if never anchored."""
        return self._value


async def fetch_usage(client: httpx.AsyncClient, url: str) -> float | None:
    """Fetch ``url`` and parse the KV-cache usage from the response body.

    Non-200 responses yield ``None`` (no logging here — the poller logs).
    Transport errors (``httpx.HTTPError``) propagate to the caller.
    """
    resp = await client.get(url)
    if resp.status_code != 200:
        return None
    return parse_kv_cache_usage(resp.text)


async def fetch_usage_by_model(client: httpx.AsyncClient, url: str) -> dict[str, float] | None:
    """Fetch ``url`` and parse the per-model KV-cache usage from the body.

    Returns the ``{model: fraction}`` mapping, or ``None`` for a non-200
    response or an unparseable/absent metric. No logging here — the poller
    logs. Transport errors (``httpx.HTTPError``) propagate to the caller.
    """
    resp = await client.get(url)
    if resp.status_code != 200:
        return None
    return parse_kv_cache_usage_by_model(resp.text)


async def fetch_metrics(client: httpx.AsyncClient, url: str) -> MetricsSample:
    """Fetch ``url`` once and parse usage, per-model usage, and capacity.

    A single ``GET``: a non-200 response yields
    ``MetricsSample(False, None, None, None)``; a 200 response parses all three
    gauges from the same body (each ``None`` when that gauge is absent or
    unparseable). No logging here — the poller logs. Transport errors
    (``httpx.HTTPError``) propagate to the caller.
    """
    resp = await client.get(url)
    if resp.status_code != 200:
        return MetricsSample(False, None, None, None)
    return MetricsSample(
        True,
        parse_kv_cache_usage(resp.text),
        parse_kv_cache_usage_by_model(resp.text),
        parse_kv_cache_capacity(resp.text),
    )
