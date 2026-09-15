"""Pure decision logic for the gate.

Given the current KV-cache usage and the estimated context tokens of an
incoming request, decide ALLOW (forward to vLLM) or REJECT (HTTP 429 with a
``Retry-After``). This module is pure and stdlib-only: no I/O, no state, no
side effects — trivially unit-testable.

.. warning:: UNIT CONVENTION — the single most likely bug in this project.

    ``usage_pct`` is a **percentage in the range 0-100** (e.g. ``85.0`` means
    the KV cache is 85% full). The vLLM Prometheus metric
    ``vllm:kv_cache_usage_perc`` is a **fraction in the range 0-1** (e.g.
    ``0.85``). The *caller* (the app layer) must convert with
    ``frac * 100.0`` **before** calling :func:`decision`. This function does
    **not** accept fractions silently: a caller that passes ``0.85`` thinking
    it is 85% will be treated as 0.85% (below every tier) and the gate will
    never reject. Do not pass the raw metric value here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gate.config import Threshold

# Stable, machine-readable reason codes. These are part of the public
# contract; callers and tests match on them.
REASON_METRICS_UNAVAILABLE = "metrics_unavailable"
REASON_PROMPT_UNPARSEABLE = "prompt_unparseable"
REASON_BELOW_ALL_TIERS = "below_all_tiers"
REASON_WITHIN_TIER = "within_tier"
REASON_EXCEEDS_TIER = "exceeds_tier"


@dataclass(frozen=True)
class Decision:
    """Outcome of :func:`decision`.

    ``allow`` is True when the request should be forwarded. ``retry_after``
    is set (to the governing tier's ``timeout_s``) only when ``allow`` is
    False. ``active_tier`` is the governing tier, or None when no tier
    governs (no tier crossed, or a fail-open path). ``reason`` is a stable
    machine-readable code naming why the decision was made.
    """

    allow: bool
    retry_after: int | None
    active_tier: Threshold | None
    reason: str


def _clamp_usage(usage_pct: float) -> float:
    """Defensively clamp a percentage into [0.0, 100.0].

    Values outside the range are treated as the nearest bound. (Config
    validation already constrains ``kv_pct`` to [0, 100], but the *observed*
    usage can transiently read out of range from a flapping metric; clamping
    keeps the pure function total and never crashing.)
    """
    if usage_pct < 0.0:
        return 0.0
    if usage_pct > 100.0:
        return 100.0
    return usage_pct


def _governing_tier(usage_pct: float, thresholds: Sequence[Threshold]) -> Threshold | None:
    """Return the highest-tier-only governing threshold, or None.

    The governing tier is the one with the **largest ``kv_pct``** among those
    whose ``kv_pct <= usage_pct`` (inclusive). Lower crossed tiers are ignored
    — we do not sum, average, or use the lowest crossed tier. Returns None
    when no tier is crossed (including an empty ``thresholds`` sequence).
    """
    active: Threshold | None = None
    for tier in thresholds:
        if tier.kv_pct <= usage_pct:
            if active is None or tier.kv_pct > active.kv_pct:
                active = tier
    return active


def decision(
    usage_pct: float | None,
    ctx_tokens: int | None,
    thresholds: Sequence[Threshold],
) -> Decision:
    """Decide whether to allow or reject a request.

    Args:
        usage_pct: Current KV-cache usage as a **percentage in 0-100**. See
            the module docstring warning: the caller must convert the vLLM
            fraction (0-1) to a percentage with ``frac * 100.0`` first.
            ``None`` means the metric is unavailable (fail open).
        ctx_tokens: Estimated context tokens for the request (prompt +
            ``max_tokens`` headroom). ``None`` means the prompt could not be
            parsed / estimated (fail open).
        thresholds: The configured tiers. May be empty (defensive; config
            validation requires at least one, but this pure function must not
            crash on an empty sequence).

    Returns:
        A :class:`Decision`.

    Decision rules (in order):

    1. ``usage_pct is None`` -> allow, ``reason="metrics_unavailable"``
       (fail open: a metrics outage must never block traffic).
    2. ``ctx_tokens is None`` -> allow, ``reason="prompt_unparseable"``
       (fail open: an estimation failure must not reject).
    3. Clamp ``usage_pct`` into [0, 100].
    4. Governing tier = the tier with the largest ``kv_pct`` among those with
       ``kv_pct <= usage_pct`` (highest-tier-only). No such tier -> allow,
       ``reason="below_all_tiers"``.
    5. Governing tier and ``ctx_tokens <= max_context`` -> allow,
       ``reason="within_tier"`` (inclusive: exactly equal is allowed).
    6. Governing tier and ``ctx_tokens > max_context`` -> reject,
       ``reason="exceeds_tier"``, ``retry_after = max_context tier's
       timeout_s``.
    """
    if usage_pct is None:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_METRICS_UNAVAILABLE,
        )
    if ctx_tokens is None:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_PROMPT_UNPARSEABLE,
        )

    clamped = _clamp_usage(usage_pct)
    active = _governing_tier(clamped, thresholds)
    if active is None:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_BELOW_ALL_TIERS,
        )

    # Inclusive boundary: exactly ctx_tokens == max_context is ALLOWED; only
    # ctx_tokens > max_context rejects.
    if ctx_tokens <= active.max_context:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=active,
            reason=REASON_WITHIN_TIER,
        )
    return Decision(
        allow=False,
        retry_after=active.timeout_s,
        active_tier=active,
        reason=REASON_EXCEEDS_TIER,
    )
