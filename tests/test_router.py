"""Tests for gate.router: the pure allow/reject decision function.

Covers the load-bearing invariants: fail-open on missing inputs,
highest-tier-only selection, the inclusive ``ctx_tokens <= max_context``
boundary, the percentage-vs-fraction convention, clamping, and a randomized
property test of the governing-tier invariant.
"""

from __future__ import annotations

import random

from gate.config import Threshold
from gate.router import Decision, decision


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
