"""Parsing and caching of vLLM's ``vllm:kv_cache_usage_perc`` metric.

Pure core: :func:`parse_kv_cache_usage` is a dependency-free (aside from the
``prometheus_client`` text parser) function over a Prometheus text exposition.
:class:`MetricsCache` remembers the last good value and its fetch time so the
poller can drive fail-open behavior via staleness. :func:`fetch_usage` is the
thin edge that pulls ``/metrics`` from vLLM.

Invariants (see AGENTS.md "Known gotchas"):

- The metric is a **fraction (0-1)**; the gate's config uses a percentage.
- v1 takes the **MAX across all series** (vLLM may emit one series per
  ``model_name``); v2 will key by ``model_name``.
- Any parse failure, absent metric, or all-non-finite values yield ``None``;
  the poller then fails open.
"""

from __future__ import annotations

import math
import time

import httpx
from prometheus_client.parser import text_string_to_metric_families

#: Exact family name of the vLLM KV-cache usage gauge.
KV_CACHE_USAGE_METRIC = "vllm:kv_cache_usage_perc"


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


class MetricsCache:
    """Last-known-good cache of the KV-cache usage fraction.

    Single-threaded asyncio usage: only the poller task writes and request
    handlers read, so no locking is needed.

    ``update`` with ``None`` is a no-op — the last good value is kept and its
    timestamp is NOT refreshed. Fail-open is driven by :meth:`is_stale`, not by
    clearing the value on a failed fetch.
    """

    def __init__(self) -> None:
        self._value: float | None = None
        self._fetched_at: float | None = None

    def update(self, value: float | None, *, now: float | None = None) -> None:
        """Store ``value`` with a fetch timestamp; ``None`` is a no-op."""
        if value is None:
            return
        self._value = value
        self._fetched_at = time.monotonic() if now is None else now

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


async def fetch_usage(client: httpx.AsyncClient, url: str) -> float | None:
    """Fetch ``url`` and parse the KV-cache usage from the response body.

    Non-200 responses yield ``None`` (no logging here — the poller logs).
    Transport errors (``httpx.HTTPError``) propagate to the caller.
    """
    resp = await client.get(url)
    if resp.status_code != 200:
        return None
    return parse_kv_cache_usage(resp.text)
