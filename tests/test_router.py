"""Tests for gate.router: the pure allow/reject decision functions.

Covers the load-bearing invariants: fail-open on missing inputs,
highest-tier-only selection, the inclusive ``ctx_tokens <= max_context``
boundary, the percentage-vs-fraction convention, clamping, and a randomized
property test of the governing-tier invariant. Also covers the autoconfig
(always-on) layer: ``decision_auto``, ``scaled_retry_after``,
``anchor_remaining_tokens`` and ``combine_decisions``.
"""

from __future__ import annotations

import random

from gate.config import Threshold
from gate.router import (
    AutoPolicy,
    Decision,
    anchor_remaining_tokens,
    combine_decisions,
    decision,
    decision_auto,
    scaled_retry_after,
)


def tier(kv_pct: float, max_context: int, timeout_s: int) -> Threshold:
    """Small helper for building Threshold fixtures."""
    return Threshold(kv_pct=kv_pct, max_context=max_context, timeout_s=timeout_s)


# The two-tier fixture used throughout: a low, generous tier and a high,
# tight tier. The high tier (80) has the *smaller* max_context and *larger*
# timeout, which makes the highest-tier-only selection observable.
LOW = tier(50.0, 4096, 15)
HIGH = tier(80.0, 1024, 30)
TWO = [LOW, HIGH]


# --- fail-open paths --------------------------------------------------------


def test_usage_none_fails_open_metrics_unavailable():
    """A metrics outage must never block traffic, even with a crossed tier
    and a huge prompt."""
    d = decision(usage_pct=None, ctx_tokens=10_000_000, thresholds=TWO)
    assert d == Decision(
        allow=True, retry_after=None, active_tier=None, reason="metrics_unavailable"
    )


def test_ctx_none_fails_open_prompt_unparseable():
    """An estimation failure must not reject, even at high usage."""
    d = decision(usage_pct=99.0, ctx_tokens=None, thresholds=TWO)
    assert d == Decision(
        allow=True, retry_after=None, active_tier=None, reason="prompt_unparseable"
    )


def test_usage_none_wins_over_ctx_none():
    """Both inputs missing: the metrics check runs first."""
    d = decision(usage_pct=None, ctx_tokens=None, thresholds=TWO)
    assert d.allow is True
    assert d.reason == "metrics_unavailable"


# --- below / above all tiers ------------------------------------------------


def test_below_all_tiers_allows():
    d = decision(usage_pct=40.0, ctx_tokens=999_999, thresholds=TWO)
    assert d == Decision(allow=True, retry_after=None, active_tier=None, reason="below_all_tiers")


def test_above_all_tiers_governed_by_highest():
    """Usage 95 crosses both tiers; the 80 tier (not the 50) governs."""
    d = decision(usage_pct=95.0, ctx_tokens=2000, thresholds=TWO)
    assert d.allow is False
    assert d.active_tier is HIGH
    assert d.reason == "exceeds_tier"


def test_empty_thresholds_allows_without_crashing():
    d = decision(usage_pct=99.0, ctx_tokens=100, thresholds=[])
    assert d == Decision(allow=True, retry_after=None, active_tier=None, reason="below_all_tiers")


# --- inclusive tier-crossing boundary ---------------------------------------


def test_usage_exactly_at_tier_is_active():
    """usage == kv_pct crosses the tier (inclusive <=)."""
    d = decision(usage_pct=50.0, ctx_tokens=100, thresholds=TWO)
    assert d.active_tier is LOW
    assert d.allow is True
    assert d.reason == "within_tier"


def test_usage_just_below_tier_is_not_active():
    """49.999 with a tier at 50 does not cross it."""
    d = decision(usage_pct=49.999, ctx_tokens=100, thresholds=TWO)
    assert d.active_tier is None
    assert d.allow is True
    assert d.reason == "below_all_tiers"


# --- within / over the governing tier (inclusive ctx boundary) --------------


def test_within_tier_below_max_allows():
    d = decision(usage_pct=60.0, ctx_tokens=1000, thresholds=TWO)
    assert d.allow is True
    assert d.active_tier is LOW
    assert d.reason == "within_tier"


def test_within_tier_exactly_at_max_allows():
    """Inclusive boundary: ctx_tokens == max_context is ALLOWED."""
    d = decision(usage_pct=60.0, ctx_tokens=4096, thresholds=TWO)
    assert d.allow is True
    assert d.active_tier is LOW
    assert d.reason == "within_tier"


def test_over_tier_by_one_rejects():
    """ctx_tokens == max_context + 1 rejects."""
    d = decision(usage_pct=60.0, ctx_tokens=4097, thresholds=TWO)
    assert d.allow is False
    assert d.active_tier is LOW
    assert d.retry_after == 15
    assert d.reason == "exceeds_tier"


def test_reject_carries_tier_timeout_and_tier():
    d = decision(usage_pct=95.0, ctx_tokens=5000, thresholds=TWO)
    assert d.allow is False
    assert d.active_tier is HIGH
    assert d.retry_after == HIGH.timeout_s == 30
    assert d.reason == "exceeds_tier"


# --- highest-tier-only regression -------------------------------------------


def test_highest_tier_only_governs():
    """usage 85 crosses both tiers; the 80 tier (max_context 1024, timeout 30)
    governs, NOT the 50 tier (max_context 4096). A ctx of 2000 fits the 50
    tier's 4096 but exceeds the 80 tier's 1024, so it must reject with the 80
    tier's timeout of 30."""
    d = decision(usage_pct=85.0, ctx_tokens=2000, thresholds=TWO)
    assert d.allow is False
    assert d.active_tier is HIGH
    assert d.retry_after == 30
    assert d.reason == "exceeds_tier"


def test_highest_tier_only_within_boundary():
    """Same usage, but ctx 1024 == the 80 tier's max_context: allowed."""
    d = decision(usage_pct=85.0, ctx_tokens=1024, thresholds=TWO)
    assert d.allow is True
    assert d.active_tier is HIGH
    assert d.reason == "within_tier"


# --- clamping ---------------------------------------------------------------


def test_usage_above_100_clamps_to_100():
    """usage 150 behaves like 100: the highest tier governs."""
    d = decision(usage_pct=150.0, ctx_tokens=2000, thresholds=TWO)
    assert d.active_tier is HIGH
    assert d.allow is False
    assert d.retry_after == 30


def test_usage_below_0_clamps_to_0():
    """usage -5 behaves like 0: below all tiers."""
    d = decision(usage_pct=-5.0, ctx_tokens=100, thresholds=TWO)
    assert d.active_tier is None
    assert d.allow is True
    assert d.reason == "below_all_tiers"


# --- randomized invariant ---------------------------------------------------


def test_randomized_governing_tier_invariant():
    """Property: for random (sorted) thresholds and random usage/ctx, the
    governing tier is the one with the max kv_pct <= usage, and the
    allow/reject outcome matches ``ctx <= max_context`` iff a tier is active.
    """
    rng = random.Random(1234)
    for _ in range(1000):
        # Generate a random number of tiers with distinct kv_pct, sorted.
        n = rng.randint(0, 5)
        kvs = rng.sample(range(0, 101), n)  # distinct ints -> distinct floats
        thresholds = [
            tier(float(kv), max_context=rng.randint(1, 8192), timeout_s=rng.randint(1, 60))
            for kv in sorted(kvs)
        ]
        usage = rng.uniform(0.0, 100.0)
        ctx = rng.randint(1, 20_000)

        # Expected governing tier: max kv_pct among those with kv_pct <= usage.
        expected = None
        for t in thresholds:
            if t.kv_pct <= usage and (expected is None or t.kv_pct > expected.kv_pct):
                expected = t

        d = decision(usage_pct=usage, ctx_tokens=ctx, thresholds=thresholds)

        if expected is None:
            assert d.allow is True
            assert d.active_tier is None
            assert d.reason == "below_all_tiers"
            assert d.retry_after is None
        else:
            assert d.active_tier is expected
            if ctx <= expected.max_context:
                assert d.allow is True
                assert d.reason == "within_tier"
                assert d.retry_after is None
            else:
                assert d.allow is False
                assert d.reason == "exceeds_tier"
                assert d.retry_after == expected.timeout_s


# --- autoconfig layer: decision_auto ----------------------------------------


def auto_policy(margin: float = 1.25, min_s: int = 5, max_s: int = 60) -> AutoPolicy:
    """Small helper for building AutoPolicy fixtures."""
    return AutoPolicy(token_margin=margin, retry_min_s=min_s, retry_max_s=max_s)


def test_auto_remaining_none_fails_open_unanchored():
    """A never-anchored counter must not reject (defensive fail-open)."""
    d = decision_auto(remaining_tokens=None, ctx_tokens=10_000_000, policy=auto_policy())
    assert d == Decision(allow=True, retry_after=None, active_tier=None, reason="auto_unanchored")


def test_auto_ctx_none_fails_open_prompt_unparseable():
    """An estimation failure must not reject, even with a tiny counter."""
    d = decision_auto(remaining_tokens=0, ctx_tokens=None, policy=auto_policy())
    assert d == Decision(
        allow=True, retry_after=None, active_tier=None, reason="prompt_unparseable"
    )


def test_auto_remaining_none_wins_over_ctx_none():
    """Both inputs missing: the anchor check runs first."""
    d = decision_auto(remaining_tokens=None, ctx_tokens=None, policy=auto_policy())
    assert d.allow is True
    assert d.reason == "auto_unanchored"


def test_auto_allow_exactly_at_boundary():
    """Inclusive boundary: effective == remaining is ALLOWED.

    ctx 10000, margin 1.25 -> effective ceil(12500.0) = 12500 == remaining.
    """
    d = decision_auto(remaining_tokens=12_500, ctx_tokens=10_000, policy=auto_policy())
    assert d == Decision(
        allow=True, retry_after=None, active_tier=None, reason="auto_within_headroom"
    )


def test_auto_allow_below_boundary():
    d = decision_auto(remaining_tokens=12_500, ctx_tokens=9_000, policy=auto_policy())
    assert d.allow is True
    assert d.reason == "auto_within_headroom"
    assert d.retry_after is None
    assert d.active_tier is None


def test_auto_reject_one_over_with_scaled_retry():
    """effective one over remaining -> reject with the scaled retry.

    ctx 10001, margin 1.25 -> effective ceil(12501.25) = 12502 > 12500.
    retry = ceil(clamp(5 * (1 + 12502/12500), 5, 60)) = ceil(10.0008) = 11.
    """
    d = decision_auto(remaining_tokens=12_500, ctx_tokens=10_001, policy=auto_policy())
    assert d.allow is False
    assert d.retry_after == 11
    assert d.active_tier is None
    assert d.reason == "auto_exceeds_headroom"


def test_auto_reject_on_full_cache_uses_max_retry():
    """remaining == 0 (cache full) -> reject at exactly retry_max_s."""
    d = decision_auto(remaining_tokens=0, ctx_tokens=100, policy=auto_policy())
    assert d.allow is False
    assert d.retry_after == 60
    assert d.reason == "auto_exceeds_headroom"


def test_auto_margin_changes_effective():
    """margin 1.0 vs 1.25 changes the effective size and the outcome.

    ctx 10_000 against remaining 12_000: margin 1.0 -> effective 10_000
    (allow); margin 1.25 -> effective 12_500 (reject, retry
    ceil(clamp(5 * (1 + 12500/12000), 5, 60)) = ceil(10.2083) = 11).
    """
    allowed = decision_auto(remaining_tokens=12_000, ctx_tokens=10_000, policy=auto_policy(1.0))
    assert allowed.allow is True
    assert allowed.reason == "auto_within_headroom"

    # margin 1.25 -> effective ceil(10000 * 1.25) = 12500 > 12000 -> reject.
    # retry = ceil(clamp(5 * (1 + 12500/12000), 5, 60)) = ceil(10.2083) = 11.
    rejected = decision_auto(remaining_tokens=12_000, ctx_tokens=10_000, policy=auto_policy(1.25))
    assert rejected.allow is False
    assert rejected.retry_after == 11
    assert rejected.reason == "auto_exceeds_headroom"


# --- autoconfig layer: scaled_retry_after ------------------------------------


def test_retry_remaining_zero_uses_max():
    assert scaled_retry_after(effective=100, remaining=0, min_s=5, max_s=60) == 60


def test_retry_remaining_negative_uses_max():
    """Subtraction is unclamped: a negative counter is 'over-committed'."""
    assert scaled_retry_after(effective=100, remaining=-1, min_s=5, max_s=60) == 60


def test_retry_zero_ratio_floors_at_min():
    """effective == 0 (no deficit) -> the min_s floor."""
    assert scaled_retry_after(effective=0, remaining=100, min_s=5, max_s=60) == 5


def test_retry_ratio_one_is_double_min():
    """effective == remaining -> min_s * (1 + 1) = 2 * min_s."""
    assert scaled_retry_after(effective=100, remaining=100, min_s=5, max_s=60) == 10


def test_retry_caps_at_max():
    """A huge deficit caps at max_s (never above it)."""
    assert scaled_retry_after(effective=10_000, remaining=1, min_s=5, max_s=60) == 60


def test_retry_monotonic_in_deficit():
    """Larger effective (bigger over-run) never yields a smaller retry."""
    retries = [
        scaled_retry_after(effective=e, remaining=100, min_s=5, max_s=60) for e in range(0, 300)
    ]
    assert all(a <= b for a, b in zip(retries, retries[1:], strict=False))
    assert retries[0] == 5
    # e = 299 -> 5 * (1 + 299/100) = 19.95 -> ceil 20 (not yet at the cap).
    assert retries[-1] == 20
    # A far larger deficit (e = 2000 -> 5 * (1 + 20) = 105) caps at max_s.
    assert scaled_retry_after(effective=2000, remaining=100, min_s=5, max_s=60) == 60


def test_retry_is_integer_and_bounded():
    """The result is an int, always within [min_s, max_s]."""
    for e in range(0, 500, 7):
        for r in range(-5, 500, 37):
            out = scaled_retry_after(effective=e, remaining=r, min_s=5, max_s=60)
            assert isinstance(out, int)
            assert 5 <= out <= 60


def test_retry_worked_example_from_plan():
    """§2 worked example: effective 31250, remaining 10000 -> 21."""
    assert scaled_retry_after(effective=31_250, remaining=10_000, min_s=5, max_s=60) == 21


# --- autoconfig layer: anchor_remaining_tokens --------------------------------


def test_anchor_usage_at_target_is_zero():
    assert anchor_remaining_tokens(100_000, 0.85, 0.85) == 0


def test_anchor_usage_above_target_is_zero():
    assert anchor_remaining_tokens(100_000, 0.85, 0.90) == 0


def test_anchor_usage_clamped_at_zero():
    """A negative usage (flapping metric) clamps to 0 -> full headroom."""
    assert anchor_remaining_tokens(100_000, 0.85, -0.2) == 85_000


def test_anchor_usage_clamped_at_one():
    """usage > 1 clamps to 1; with target 0.85 the headroom is 0."""
    assert anchor_remaining_tokens(100_000, 0.85, 1.5) == 0


def test_anchor_fractional_product_uses_floor():
    """floor(100000 * (0.85 - 0.50)) = 35000 exactly (worked example)."""
    assert anchor_remaining_tokens(100_000, 0.85, 0.50) == 35_000


def test_anchor_floors_fractional_remainder():
    """A genuinely fractional product floors DOWN, not round:

    10 * (0.85 - 0.50) = 3.5 -> floor 3 (round would give 4).
    """
    assert anchor_remaining_tokens(10, 0.85, 0.50) == 3


def test_anchor_target_above_one_gives_full_capacity():
    """target 1.0, usage 0.2 -> floor(100000 * 0.8) = 80000."""
    assert anchor_remaining_tokens(100_000, 1.0, 0.2) == 80_000


# --- autoconfig layer: combine_decisions --------------------------------------


def test_combine_both_allow_normal_is_allowed():
    tier = Decision(True, None, None, "within_tier")
    auto = Decision(True, None, None, "auto_within_headroom")
    assert combine_decisions(tier, auto) == Decision(True, None, None, "allowed")


def test_combine_both_allow_no_tiers_is_allowed():
    tier = Decision(True, None, None, "below_all_tiers")
    auto = Decision(True, None, None, "auto_within_headroom")
    assert combine_decisions(tier, auto) == Decision(True, None, None, "allowed")


def test_combine_both_allow_fail_open_tier_reason_wins():
    """A stale feed (tier fail-open) is the more informative reason."""
    tier = Decision(True, None, None, "metrics_unavailable")
    auto = Decision(True, None, None, "auto_within_headroom")
    assert combine_decisions(tier, auto) == Decision(True, None, None, "metrics_unavailable")


def test_combine_both_allow_fail_open_auto_reason_wins():
    """An unanchored counter (auto fail-open) is the more informative reason."""
    tier = Decision(True, None, None, "within_tier")
    auto = Decision(True, None, None, "auto_unanchored")
    assert combine_decisions(tier, auto) == Decision(True, None, None, "auto_unanchored")


def test_combine_tier_only_rejects_returns_tier_verbatim():
    tier = Decision(False, 15, HIGH, "exceeds_tier")
    auto = Decision(True, None, None, "auto_within_headroom")
    assert combine_decisions(tier, auto) is tier


def test_combine_auto_only_rejects_returns_auto_verbatim():
    tier = Decision(True, None, None, "within_tier")
    auto = Decision(False, 21, None, "auto_exceeds_headroom")
    assert combine_decisions(tier, auto) is auto


def test_combine_both_reject_tier_higher_timeout_is_reported():
    """Tier 30 > auto 21 -> retry 30, rejector = tiered (active_tier set)."""
    tier = Decision(False, 30, HIGH, "exceeds_tier")
    auto = Decision(False, 21, None, "auto_exceeds_headroom")
    assert combine_decisions(tier, auto) == Decision(
        allow=False, retry_after=30, active_tier=HIGH, reason="exceeds_tier"
    )


def test_combine_both_reject_auto_higher_timeout_is_reported():
    """Auto 60 > tier 30 -> retry 60, rejector = autoconfig (active_tier None).

    Mirrors the §2 worked example: usage 90% -> tiered 80-tier retries 30;
    autoconfig anchor 0 -> every request rejects at 60.
    """
    tier = Decision(False, 30, HIGH, "exceeds_tier")
    auto = Decision(False, 60, None, "auto_exceeds_headroom")
    assert combine_decisions(tier, auto) == Decision(
        allow=False, retry_after=60, active_tier=None, reason="auto_exceeds_headroom"
    )


def test_combine_both_reject_tie_reports_tiered():
    """Equal timeouts -> the tiered layer is the reported rejector."""
    tier = Decision(False, 30, HIGH, "exceeds_tier")
    auto = Decision(False, 30, None, "auto_exceeds_headroom")
    assert combine_decisions(tier, auto) == Decision(
        allow=False, retry_after=30, active_tier=HIGH, reason="exceeds_tier"
    )
