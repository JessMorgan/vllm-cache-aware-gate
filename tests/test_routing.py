"""Tests for gate.routing: the pure multi-backend selection function.

Covers the load-bearing semantics of ``select_backend``: the stateless
``round_robin`` / ``primary_fallback`` choice, the inclusive
``large_small`` threshold, the spread-minimization (NOT greedy
least-loaded) objective of ``even`` with unknowns ranked last, the
``fill`` first-fit eligibility (stale/unanchored candidates always
eligible, interleaved in config order), the empty-candidates defensive
``None``, and the ``ValueError`` on an unknown policy.
"""

from __future__ import annotations

import pytest

from gate.routing import (
    POLICIES,
    POLICY_EVEN,
    POLICY_FILL,
    POLICY_LARGE_SMALL,
    POLICY_PRIMARY_FALLBACK,
    POLICY_ROUND_ROBIN,
    CandidateView,
    RoutingSpec,
    select_backend,
)

A = "a"
B = "b"
C = "c"


def view(
    usage_frac: float | None = 0.5,
    remaining_tokens: int | None = 500,
    capacity_tokens: int | None = 1000,
    token_margin: float = 1.25,
) -> CandidateView:
    """Small helper for building CandidateView fixtures."""
    return CandidateView(
        usage_frac=usage_frac,
        remaining_tokens=remaining_tokens,
        capacity_tokens=capacity_tokens,
        token_margin=token_margin,
    )


def spec(policy: str, threshold_tokens: int | None = None) -> RoutingSpec:
    """Small helper for building RoutingSpec fixtures."""
    return RoutingSpec(policy=policy, order=(A, B, C), threshold_tokens=threshold_tokens)


# --- constants ---------------------------------------------------------------


def test_policy_constants_and_set():
    assert POLICY_ROUND_ROBIN == "round_robin"
    assert POLICY_PRIMARY_FALLBACK == "primary_fallback"
    assert POLICY_LARGE_SMALL == "large_small"
    assert POLICY_EVEN == "even"
    assert POLICY_FILL == "fill"
    assert POLICIES == frozenset(
        {
            POLICY_ROUND_ROBIN,
            POLICY_PRIMARY_FALLBACK,
            POLICY_LARGE_SMALL,
            POLICY_EVEN,
            POLICY_FILL,
        }
    )


# --- round_robin / primary_fallback -------------------------------------------


@pytest.mark.parametrize("policy", [POLICY_ROUND_ROBIN, POLICY_PRIMARY_FALLBACK])
@pytest.mark.parametrize("ctx", [None, 100, 10_000_000])
def test_round_robin_and_primary_fallback_return_zero(policy: str, ctx: int | None):
    """Both policies are stateless: index 0 for single and multi candidates,
    with known and stale views."""
    views = {A: view(), B: view(usage_frac=None), C: view(usage_frac=None)}
    assert select_backend(spec(policy), (A,), ctx, views) == 0
    assert select_backend(spec(policy), (A, B, C), ctx, views) == 0
    # A missing view entry is fully unknown — still 0.
    assert select_backend(spec(policy), (A, B), ctx, {A: view()}) == 0


# --- large_small ---------------------------------------------------------------


def test_large_small_at_threshold_goes_large_inclusive():
    """ctx exactly at the threshold is LARGE (inclusive boundary)."""
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A, B), 100, {}) == 0


def test_large_small_below_threshold_goes_small():
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A, B), 99, {}) == 1


def test_large_small_above_threshold_goes_large():
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A, B), 101, {}) == 0


def test_large_small_ctx_none_goes_small():
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A, B), None, {}) == 1


def test_large_small_single_candidate_clamps_to_zero_both_sides():
    """With a 1-candidate order the clamp wins on both sides of the
    threshold."""
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A,), 1_000, {}) == 0
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A,), 1, {}) == 0
    assert select_backend(spec(POLICY_LARGE_SMALL, 100), (A,), None, {}) == 0


def test_large_small_missing_threshold_defensive_goes_small():
    """Defensive: threshold_tokens None is treated as the small side (1),
    or 0 when clamped by a 1-candidate order."""
    assert select_backend(spec(POLICY_LARGE_SMALL, None), (A, B), 10_000, {}) == 1
    assert select_backend(spec(POLICY_LARGE_SMALL, None), (A,), 10_000, {}) == 0


# --- even ----------------------------------------------------------------------


def test_even_single_candidate_is_zero():
    assert select_backend(spec(POLICY_EVEN), (A,), 100, {A: view()}) == 0


def test_even_two_candidates_picks_less_loaded():
    """Hand-computed: charge = ceil(100 * 1.25) = 125.
    p_a = 0.5 + 125/1000 = 0.625 -> spread_a = max(0.625, 0.6) - min(...) = 0.025
    p_b = 0.6 + 125/1000 = 0.725 -> spread_b = max(0.5, 0.725) - min(...) = 0.225
    argmin -> a (index 0)."""
    views = {
        A: view(usage_frac=0.5, capacity_tokens=1000),
        B: view(usage_frac=0.6, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_EVEN), (A, B), 100, views) == 0


def test_even_spread_objective_disagrees_with_greedy_least_loaded():
    """Hand-computed 3-candidate case where the greedy least-loaded answer
    (a, usage 0.2) is NOT the spread-minimizing answer (b).

    charge = ceil(40 * 1.25) = 50.
    p_a = 0.2 + 50/100  = 0.70  -> spread_a = max(0.70, 0.4, 0.42) - min(...) = 0.30
    p_b = 0.4 + 50/1000 = 0.45  -> spread_b = max(0.2, 0.45, 0.42) - min(...) = 0.25
    p_c = 0.42 + 50/1000 = 0.47 -> spread_c = max(0.2, 0.4, 0.47) - min(...) = 0.27
    argmin -> b (index 1), even though a is the least-loaded candidate.
    """
    views = {
        A: view(usage_frac=0.2, capacity_tokens=100),
        B: view(usage_frac=0.4, capacity_tokens=1000),
        C: view(usage_frac=0.42, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_EVEN), (A, B, C), 40, views) == 1


def test_even_unknown_candidate_ranked_last():
    """A candidate with usage_frac None loses to known candidates even when
    its usage would look 'lower' — it cannot be projected."""
    views = {
        A: view(usage_frac=None, capacity_tokens=None),
        B: view(usage_frac=0.9, capacity_tokens=1000),
        C: view(usage_frac=0.91, capacity_tokens=1000),
    }
    # A is index 0 and unknown -> ranked last. B (index 1) vs C (index 2):
    # p_b = 0.9 + 125/1000 = 1.025 -> spread_b = max(1.025, 0.91) - min(...) = 0.115
    # p_c = 0.91 + 125/1000 = 1.035 -> spread_c = max(0.9, 1.035) - min(...) = 0.135
    # argmin among knowns -> B (index 1).
    assert select_backend(spec(POLICY_EVEN), (A, B, C), 100, views) == 1


def test_even_unknown_candidate_ranked_last_among_unknowns_in_index_order():
    """All candidates unknown: the lowest index (config order) wins."""
    views = {A: view(usage_frac=None), B: view(usage_frac=None), C: view(usage_frac=None)}
    assert select_backend(spec(POLICY_EVEN), (A, B, C), 100, views) == 0
    # And a missing view entry counts as unknown too.
    assert select_backend(spec(POLICY_EVEN), (A, B), 100, {}) == 0


def test_even_unknown_capacity_ranked_last():
    """A candidate whose capacity is None (so p_i is undefined) is ranked
    last even with a known usage_frac."""
    views = {
        A: view(usage_frac=0.1, capacity_tokens=None),
        B: view(usage_frac=0.9, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_EVEN), (A, B), 100, views) == 1


def test_even_tie_goes_to_lowest_index():
    """Identical candidates: the spreads tie, the lowest index wins."""
    views = {
        A: view(usage_frac=0.5, capacity_tokens=1000),
        B: view(usage_frac=0.5, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_EVEN), (A, B), 100, views) == 0


# --- fill ----------------------------------------------------------------------


def test_fill_first_fit_skips_full_first_candidate():
    """Candidate 0 is full (charge 125 > remaining 100), candidate 1 has
    room (125 <= 400) -> index 1."""
    views = {
        A: view(usage_frac=0.9, remaining_tokens=100, capacity_tokens=1000),
        B: view(usage_frac=0.1, remaining_tokens=400, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) == 1


def test_fill_all_full_returns_none():
    """No eligible candidate -> None (the walk still runs, without
    proxying, to collect retry_after values)."""
    views = {
        A: view(usage_frac=0.95, remaining_tokens=50, capacity_tokens=1000),
        B: view(usage_frac=0.98, remaining_tokens=10, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) is None


def test_fill_ctx_none_returns_lowest_index():
    """ctx None mirrors decision_auto's prompt_unparseable fail-open: the
    lowest index is eligible regardless of remaining."""
    views = {
        A: view(usage_frac=0.99, remaining_tokens=0, capacity_tokens=1000),
        B: view(usage_frac=0.5, remaining_tokens=500, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), None, views) == 0


def test_fill_stale_candidate_eligible_regardless_of_remaining():
    """A stale candidate (usage_frac None) with remaining 0 is still
    eligible (gotcha #1 — the walk fails it open)."""
    views = {
        A: view(usage_frac=None, remaining_tokens=0, capacity_tokens=None),
        B: view(usage_frac=0.5, remaining_tokens=500, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) == 0


def test_fill_unanchored_candidate_eligible():
    """An unanchored candidate (remaining None) is eligible."""
    views = {
        A: view(usage_frac=0.5, remaining_tokens=None, capacity_tokens=1000),
        B: view(usage_frac=0.9, remaining_tokens=10, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) == 0


def test_fill_unknowns_interleaved_in_config_order_not_ranked_last():
    """A stale candidate at index 0 beats a known-eligible candidate at
    index 1 — proving fill interleaves unknowns in config order (unlike
    even's ranked-last rule)."""
    views = {
        A: view(usage_frac=None, remaining_tokens=None, capacity_tokens=None),
        B: view(usage_frac=0.5, remaining_tokens=500, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) == 0
    # Same with a stale candidate between two known-eligible ones.
    views3 = {
        A: view(usage_frac=0.9, remaining_tokens=100, capacity_tokens=1000),
        B: view(usage_frac=None, remaining_tokens=None, capacity_tokens=None),
        C: view(usage_frac=0.5, remaining_tokens=500, capacity_tokens=1000),
    }
    # A is full (125 > 100), B is stale (eligible) -> index 1, not C.
    assert select_backend(spec(POLICY_FILL), (A, B, C), 100, views3) == 1


def test_fill_charge_exactly_equal_to_remaining_is_eligible():
    """Inclusive boundary: charge == remaining admits (mirrors
    decision_auto)."""
    views = {A: view(usage_frac=0.5, remaining_tokens=125, capacity_tokens=1000)}
    assert select_backend(spec(POLICY_FILL), (A,), 100, views) == 0


def test_fill_charge_above_remaining_is_skipped():
    """A known candidate with charge > remaining is skipped even when its
    usage looks low."""
    views = {
        A: view(usage_frac=0.1, remaining_tokens=124, capacity_tokens=1000),
        B: view(usage_frac=0.8, remaining_tokens=200, capacity_tokens=1000),
    }
    assert select_backend(spec(POLICY_FILL), (A, B), 100, views) == 1


def test_fill_missing_view_entry_is_eligible():
    """A candidate missing from usage_views is fully unknown -> eligible
    (fail-open)."""
    assert select_backend(spec(POLICY_FILL), (A, B), 100, {B: view()}) == 0


# --- defensive paths -----------------------------------------------------------


def test_empty_candidates_returns_none_for_every_policy():
    for policy in POLICIES:
        assert select_backend(spec(policy, 100), (), 100, {}) is None


def test_unknown_policy_raises_value_error():
    with pytest.raises(ValueError, match="unknown routing policy"):
        select_backend(spec("bogus"), (A, B), 100, {})
