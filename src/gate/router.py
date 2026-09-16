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

    The autoconfig functions (:func:`anchor_remaining_tokens`,
    :func:`decision_auto`, :func:`scaled_retry_after`) work in the opposite
    space: **fractions (0-1) and token counts**. The only percentage in the
    whole autoconfig feature is ``target_kv_cache_pct`` (0-100), and the
    *caller* converts it to a fraction with ``pct / 100.0`` at anchor time —
    :func:`anchor_remaining_tokens` receives the fraction.
    :func:`decision_auto` and :func:`scaled_retry_after` never see a
    percentage at all.
"""

from __future__ import annotations

import math
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

# Autoconfig (always-on) reason codes. The autoconfig layer admits each
# request by comparing its estimated size against an in-memory estimate of the
# KV space still available up to a target usage %.
REASON_AUTO_UNANCHORED = "auto_unanchored"
REASON_AUTO_WITHIN_HEADROOM = "auto_within_headroom"
REASON_AUTO_EXCEEDS_HEADROOM = "auto_exceeds_headroom"

# Reasons that mean "the layer could not evaluate" (fail-open). When both
# admission layers allow, the more informative of these wins the combined
# reason (for logging); otherwise the combined reason is "allowed".
_FAIL_OPEN_REASONS = frozenset(
    {
        REASON_METRICS_UNAVAILABLE,
        REASON_PROMPT_UNPARSEABLE,
        REASON_AUTO_UNANCHORED,
    }
)


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
        thresholds: The configured tiers. May be empty — the tiered policy is
            optional (autoconfig is always on), and with no tiers this layer
            allows every request.

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


# --- autoconfig (always-on) layer -------------------------------------------
#
# The autoconfig layer admits each request by comparing its estimated size
# against an in-memory estimate of the KV space still available up to a target
# usage %. All of the math below is pure and lives in fraction/token space:
# the only percentage in the feature is ``target_kv_cache_pct`` (0-100), and
# the caller converts it to a fraction (``pct / 100.0``) at anchor time before
# calling :func:`anchor_remaining_tokens`. :func:`decision_auto` and
# :func:`scaled_retry_after` never see a percentage.


@dataclass(frozen=True)
class AutoPolicy:
    """Tunables for the autoconfig admission layer.

    ``token_margin`` is the conservatism multiplier applied to the estimated
    context tokens (>= 1.0). ``retry_min_s`` / ``retry_max_s`` bound the
    ``Retry-After`` seconds for a rejected request. The target KV-cache
    percentage is deliberately NOT a field here: it lives at *anchor time*
    (the caller converts it to a fraction and passes it to
    :func:`anchor_remaining_tokens`), not at decision time.
    """

    token_margin: float
    retry_min_s: int
    retry_max_s: int


def anchor_remaining_tokens(capacity_tokens: int, target_frac: float, usage_frac: float) -> int:
    """Estimate the KV tokens remaining up to the target usage.

    ``floor(capacity_tokens * max(target_frac - clamp(usage_frac, 0, 1), 0))``.

    ``capacity_tokens`` is the total KV-cache size in tokens (from the
    ``vllm:kv_cache_size_tokens`` gauge, or the ``kv_cache_size_tokens``
    label on ``vllm:cache_config_info`` when the gauge is absent).
    ``target_frac`` and ``usage_frac``
    are **fractions in 0-1** (the caller converts the target percentage with
    ``pct / 100.0``). Usage at or above the target yields 0 — the cache is
    "full" for admission and every non-trivial request rejects until usage
    drops. Out-of-range usage is clamped into [0, 1] so the function is total.
    """
    clamped = min(max(usage_frac, 0.0), 1.0)
    headroom = max(target_frac - clamped, 0.0)
    return math.floor(capacity_tokens * headroom)


def scaled_retry_after(effective: int, remaining: int, min_s: int, max_s: int) -> int:
    """Scale a ``Retry-After`` (seconds) by how far the request over-runs.

    ``effective`` is the request's estimated token cost (``ceil(ctx *
    token_margin)``); ``remaining`` is the counter's current value (may be
    zero or negative — an over-committed cache). The result is an integer
    number of seconds, always within ``[min_s, max_s]``:

    - ``remaining <= 0`` (cache full / over-committed) -> ``max_s``.
    - Otherwise ``ceil(clamp(min_s * (1 + effective / remaining), min_s,
      max_s))`` — a small over-run yields ≈ ``2 * min_s``; a large over-run
      caps at ``max_s``.

    ``ceil`` is used (never round a backoff down).
    """
    if remaining <= 0:
        return max_s
    raw = min_s * (1 + effective / remaining)
    clamped = min(max(raw, min_s), max_s)
    return int(math.ceil(clamped))


def decision_auto(
    remaining_tokens: int | None,
    ctx_tokens: int | None,
    policy: AutoPolicy,
) -> Decision:
    """Decide whether the autoconfig layer allows or rejects a request.

    Args:
        remaining_tokens: The counter's current estimated remaining KV tokens.
            ``None`` means the counter has never been anchored (fail open).
        ctx_tokens: Estimated context tokens for the request. ``None`` means
            the prompt could not be parsed / estimated (fail open).
        policy: The autoconfig tunables (see :class:`AutoPolicy`).

    Returns:
        A :class:`Decision`. ``active_tier`` is always ``None`` here — the
        autoconfig layer has no tiers.

    Decision rules (in order):

    1. ``remaining_tokens is None`` -> allow, ``reason="auto_unanchored"``
       (defensive: a fresh usage without an anchor is only a cold-start
       sliver; an observed body with usage but no capacity is fatal upstream).
    2. ``ctx_tokens is None`` -> allow, ``reason="prompt_unparseable"``
       (fail open: an estimation failure must not reject).
    3. ``effective = ceil(ctx_tokens * token_margin)``:
       - ``effective <= remaining_tokens`` -> allow,
         ``reason="auto_within_headroom"`` (inclusive: exactly equal allows).
       - ``effective > remaining_tokens`` -> reject,
         ``reason="auto_exceeds_headroom"``, ``retry_after`` from
         :func:`scaled_retry_after`.
    """
    if remaining_tokens is None:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_AUTO_UNANCHORED,
        )
    if ctx_tokens is None:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_PROMPT_UNPARSEABLE,
        )

    effective = int(math.ceil(ctx_tokens * policy.token_margin))
    # Inclusive boundary: effective == remaining_tokens is ALLOWED.
    if effective <= remaining_tokens:
        return Decision(
            allow=True,
            retry_after=None,
            active_tier=None,
            reason=REASON_AUTO_WITHIN_HEADROOM,
        )
    retry = scaled_retry_after(effective, remaining_tokens, policy.retry_min_s, policy.retry_max_s)
    return Decision(
        allow=False,
        retry_after=retry,
        active_tier=None,
        reason=REASON_AUTO_EXCEEDS_HEADROOM,
    )


def combine_decisions(tier: Decision, auto: Decision) -> Decision:
    """AND-combine the tiered and autoconfig layer decisions.

    The request forwards only if **both** layers allow.

    - Both allow -> ``allow=True``, ``retry_after=None``, ``active_tier=None``;
      ``reason`` is the more informative layer reason (a fail-open reason
      wins, for logging; otherwise ``"allowed"``).
    - Exactly one rejects -> that layer's :class:`Decision` is returned
      verbatim.
    - Both reject -> ``retry_after = max(tier.retry_after, auto.retry_after)``;
      the **reported rejector** is the tier if ``tier.retry_after >=
      auto.retry_after`` (tie -> tiered) else the autoconfig layer;
      ``reason`` is the reported layer's reason and ``active_tier`` is the
      tier iff it is the reported rejector (drives the 429 message text).
    """
    if tier.allow and auto.allow:
        if tier.reason in _FAIL_OPEN_REASONS:
            reason = tier.reason
        elif auto.reason in _FAIL_OPEN_REASONS:
            reason = auto.reason
        else:
            reason = "allowed"
        return Decision(allow=True, retry_after=None, active_tier=None, reason=reason)

    if not tier.allow and auto.allow:
        return tier
    if tier.allow and not auto.allow:
        return auto

    # Both reject: both retry_after values are non-None ints, so max() is safe.
    tier_retry = tier.retry_after
    auto_retry = auto.retry_after
    if tier_retry is None or auto_retry is None:  # pragma: no cover - defensive
        return tier
    retry_after = max(tier_retry, auto_retry)
    if tier_retry >= auto_retry:
        # Tie (or tier higher) -> the tiered layer is the reported rejector.
        return Decision(
            allow=False,
            retry_after=retry_after,
            active_tier=tier.active_tier,
            reason=tier.reason,
        )
    return Decision(allow=False, retry_after=retry_after, active_tier=None, reason=auto.reason)
