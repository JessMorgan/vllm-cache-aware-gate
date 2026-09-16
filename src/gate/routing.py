"""Pure routing-selection logic for the multi-backend gate.

Given a routing spec, the ordered candidate list for the request's model,
the estimated context tokens, and a per-candidate usage view, pick which
candidate the app should attempt **first**. This module is pure and
stdlib-only: no I/O, no state, no side effects — trivially unit-testable.

The returned value is an **index into ``candidates``** (the ordered
candidate list — ``spec.order`` order when a spec exists, else registry
config order), not a backend name. The app's stateful ``round_robin``
offset (per-model index) and the failover walk (``app.py``) build on top of
this stateless choice.

.. warning:: UNIT CONVENTION — fraction vs token space.

    ``CandidateView.usage_frac`` and the projected usage fractions are
    **fractions in 0-1** (the vLLM metric space). ``remaining_tokens`` and
    ``capacity_tokens`` are **token counts**. The two are combined only
    through ``charge / capacity`` (tokens over tokens -> a fraction).

.. warning:: ``remaining_tokens`` is headroom up to target, NOT free tokens.

    The ``KvRemaining`` counter is anchored to
    ``floor(capacity * (target - usage))`` — headroom *up to the target
    usage*, not free tokens. It must **not** be used to reconstruct usage
    (``(capacity - remaining) / capacity`` over-counts by ``1 - target``).
    The ``fill`` policy therefore tests ``charge <= remaining`` **directly**
    (the exact ``decision_auto`` admission test) and never re-derives a
    threshold from usage math.

Load-bearing invariants:

- **Selection is stateless.** ``round_robin`` returns ``0`` here; the
  app's per-model index offsets it later.
- **Fail open on unknown feeds.** A candidate whose feed is stale /
  never-fetched (``usage_frac is None``) or unanchored
  (``remaining_tokens is None``) is **always eligible** under ``fill``
  (gotcha #1 — a monitoring outage must never block traffic).
- **Unknowns are ranked differently per policy.** ``fill`` interleaves
  eligible-but-unknown candidates in global config order (the lowest
  eligible index wins); ``even`` ranks unknowns **last** (a candidate it
  cannot project is not the "most even" choice), among themselves in index
  order.
- **Inclusive boundaries.** ``large_small`` sends ``ctx_tokens ==
  threshold_tokens`` to the large side; ``fill`` admits ``charge ==
  remaining``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# The five routing policy names. These are part of the public contract;
# config validation checks ``RoutingSpec.policy`` against :data:`POLICIES`.
POLICY_ROUND_ROBIN = "round_robin"
POLICY_PRIMARY_FALLBACK = "primary_fallback"
POLICY_LARGE_SMALL = "large_small"
POLICY_EVEN = "even"
POLICY_FILL = "fill"

POLICIES: frozenset[str] = frozenset(
    {
        POLICY_ROUND_ROBIN,
        POLICY_PRIMARY_FALLBACK,
        POLICY_LARGE_SMALL,
        POLICY_EVEN,
        POLICY_FILL,
    }
)


@dataclass(frozen=True)
class RoutingSpec:
    """A parsed ``routing:`` entry for one model.

    ``policy`` is one of the five policy names (see :data:`POLICIES`).
    ``order`` is the ordered candidate list (backend names).
    ``threshold_tokens`` is the large/small size split: required for
    ``large_small`` (config validation guarantees it) and ignored by the
    other policies. Parsed by ``config.py``; this module only consumes it.
    """

    policy: str
    order: tuple[str, ...]
    threshold_tokens: int | None = None


@dataclass(frozen=True)
class CandidateView:
    """Per-candidate usage snapshot consumed by :func:`select_backend`.

    ``usage_frac`` is the backend's current KV usage **fraction (0-1)**, or
    ``None`` when the feed is stale / never-fetched. ``remaining_tokens`` is
    the backend's ``KvRemaining`` counter value — headroom **up to target**,
    not free tokens — or ``None`` when the counter is unanchored.
    ``capacity_tokens`` is the KV capacity in tokens, or ``None`` when
    unknown. ``token_margin`` is the backend's effective token margin
    (>= 1.0).
    """

    usage_frac: float | None
    remaining_tokens: int | None
    capacity_tokens: int | None
    token_margin: float


# A candidate missing from ``usage_views`` is treated as fully unknown:
# fail-open under ``fill``, ranked last under ``even``.
_UNKNOWN_VIEW = CandidateView(
    usage_frac=None, remaining_tokens=None, capacity_tokens=None, token_margin=1.0
)


def _charge(ctx_tokens: int | None, margin: float) -> int:
    """The counter charge for a request: ``ceil(ctx * margin)``, else ``0``.

    Mirrors the app's charge rule: an unparseable body (``ctx_tokens is
    None``) charges ``0``.
    """
    if ctx_tokens is None:
        return 0
    return int(math.ceil(ctx_tokens * margin))


def _select_large_small(
    candidates: tuple[str, ...],
    ctx_tokens: int | None,
    threshold_tokens: int | None,
) -> int:
    """``large_small``: index ``0`` for large requests, index ``1`` for small.

    Index ``0`` iff ``ctx_tokens`` is not None and
    ``ctx_tokens >= threshold_tokens`` (inclusive at the threshold), else
    ``1`` — clamped to ``0`` with a 1-candidate order. When
    ``threshold_tokens`` is None (defensive: config validation guarantees it
    for ``large_small``), the request is treated as small (index ``1``,
    clamped to ``0`` with a 1-candidate order).
    """
    if len(candidates) == 1:
        return 0
    is_large = (
        ctx_tokens is not None and threshold_tokens is not None and ctx_tokens >= threshold_tokens
    )
    return 0 if is_large else 1


def _select_even(
    candidates: tuple[str, ...],
    ctx_tokens: int | None,
    usage_views: Mapping[str, CandidateView],
) -> int:
    """``even``: the index minimizing the projected-usage spread.

    For each candidate ``i`` the vector ``v_i`` has ``v_i[i] = p_i``
    (candidate ``i`` at its own **projected** fraction — the request would
    go there) and ``v_i[j] = usage_frac_j`` for ``j != i`` (every other
    candidate at its current, unprojected usage). ``spread_i = max(v_i) -
    min(v_i)`` over the known entries only. Unknowns (``usage_frac`` or
    ``capacity_tokens`` is None, or a non-positive capacity — so ``p_i`` is
    undefined) are ranked **last**, and among themselves in index order; in
    particular, if candidate ``i`` itself is unknown, ``spread_i`` is
    undefined and ``i`` is ranked last. Ties -> lowest index. This is the
    spread-minimization objective, NOT greedy least-loaded.

    Returns an index into ``candidates`` (the caller guarantees the list is
    non-empty; with all candidates unknown the lowest index wins).
    """
    best: int | None = None
    best_spread: float | None = None
    for i, name in enumerate(candidates):
        view = usage_views.get(name, _UNKNOWN_VIEW)
        if view.usage_frac is None or view.capacity_tokens is None or view.capacity_tokens <= 0:
            # Unknowns are ranked last: skip them and let a known candidate
            # win; if no known candidate qualifies, fall through to the
            # lowest index (config order among the unknowns).
            continue
        values: list[float] = [
            view.usage_frac + _charge(ctx_tokens, view.token_margin) / view.capacity_tokens
        ]
        for j, other in enumerate(candidates):
            if j == i:
                continue
            oview = usage_views.get(other, _UNKNOWN_VIEW)
            if oview.usage_frac is not None:
                values.append(oview.usage_frac)
        spread = max(values) - min(values)
        if best_spread is None or spread < best_spread:
            best = i
            best_spread = spread
    if best is not None:
        return best
    return 0  # all candidates unknown: lowest index (config order)


def _select_fill(
    candidates: tuple[str, ...],
    ctx_tokens: int | None,
    usage_views: Mapping[str, CandidateView],
) -> int | None:
    """``fill``: the lowest index the autoconfig layer would not reject.

    Eligibility per candidate (walked in candidate order):

    1. ``ctx_tokens is None`` -> eligible (mirrors ``decision_auto``'s
       ``prompt_unparseable`` fail-open — an unparseable body is never
       rejected at prefilter time).
    2. ``usage_frac is None`` (stale / never-fetched) -> eligible
       regardless of ``remaining`` (gotcha #1 — the walk fails it open).
    3. ``remaining_tokens is None`` (unanchored) -> eligible.
    4. ``charge <= remaining`` -> eligible (the exact ``decision_auto``
       admission test, inclusive).
    5. Otherwise not eligible.

    Eligible-but-unknown candidates (stale / unanchored) are interleaved in
    global config order with the known ones — the lowest eligible index
    wins; they are NOT ranked last (unlike ``even``). Returns the lowest
    eligible index, or ``None`` when no candidate is eligible (all-full).
    """
    for i, name in enumerate(candidates):
        view = usage_views.get(name, _UNKNOWN_VIEW)
        if ctx_tokens is None or view.usage_frac is None or view.remaining_tokens is None:
            return i
        if _charge(ctx_tokens, view.token_margin) <= view.remaining_tokens:
            return i
    return None


def select_backend(
    spec: RoutingSpec,
    candidates: tuple[str, ...],
    ctx_tokens: int | None,
    usage_views: Mapping[str, CandidateView],
) -> int | None:
    """Pick the candidate index to attempt first.

    Args:
        spec: The routing spec for the request's model.
        candidates: The ordered candidate list (already filtered to
            backends serving the model; ``spec.order`` order when a spec
            exists, else registry config order).
        ctx_tokens: Estimated context tokens for the request, or ``None``
            when the prompt could not be parsed / estimated.
        usage_views: Candidate name -> :class:`CandidateView`. A candidate
            missing from the mapping is treated as fully unknown
            (fail-open under ``fill``, ranked last under ``even``).

    Returns:
        The **index** into ``candidates`` to attempt first (the app's
        failover walk advances from there), or ``None`` when no candidate
        is eligible (``fill`` with every candidate full). An empty
        ``candidates`` tuple returns ``None`` (defensive — the app resolves
        an empty candidate set to the default backend before calling).

    Raises:
        ValueError: If ``spec.policy`` is not one of the five known policy
            names (defensive — config validation guarantees a valid
            policy).
    """
    if not candidates:
        return None
    policy = spec.policy
    if policy == POLICY_ROUND_ROBIN or policy == POLICY_PRIMARY_FALLBACK:
        # Stateless selection: the app's per-model round-robin index offsets
        # this later; primary_fallback always starts at the head and relies
        # on the skip-on-reject failover walk.
        return 0
    if policy == POLICY_LARGE_SMALL:
        return _select_large_small(candidates, ctx_tokens, spec.threshold_tokens)
    if policy == POLICY_EVEN:
        return _select_even(candidates, ctx_tokens, usage_views)
    if policy == POLICY_FILL:
        return _select_fill(candidates, ctx_tokens, usage_views)
    raise ValueError(f"unknown routing policy: {policy!r}")
