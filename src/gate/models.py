"""Model discovery and the model-to-backend routing map.

Pure core: :func:`parse_v1_models` is a dependency-free function over an
OpenAI ``GET /v1/models`` JSON body. :class:`ModelRegistry` owns the
``model -> backend_name`` map used for routing.

Invariants (see docs/plans/multi-backend.md §2.2):

- :func:`parse_v1_models` returns ``None`` on non-JSON text, a missing or
  non-list ``data`` field, or any entry missing a non-empty string ``id`` —
  the caller (poller) then keeps the last known map (discovery is auxiliary
  and never fatal).
- :class:`ModelRegistry` has the same single-asyncio-loop contract as
  :class:`gate.metrics.MetricsCache`: only the poller tasks write
  (:meth:`ModelRegistry.sync`) and request handlers read
  (:meth:`ModelRegistry.resolve_candidates`); every read-modify-write is
  synchronous with no ``await`` between the read and the write, so no
  locking is needed.
- **Duplicate model ids are legal** (decision 11): a model owned by 2+
  backends is the normal multi-candidate case, not a collision.
  :meth:`ModelRegistry.resolve_candidates` is the v2 routing lookup used by
  the failover walk — it returns ALL owners in registration (config) order.
- :meth:`ModelRegistry.resolve` is retained (the current app path) and
  returns the collision winner (the ``default: true`` backend if any
  owner is the default, else the first-registered owner) for multi-owner
  models; the conflict is logged once per (model, backend) pair, not on
  every repeated sync.
- A model no longer owned by any backend stops resolving (``None`` / empty
  tuple); requests for it take the unknown-model path (default backend),
  never a hard 404.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("gate.models")


def parse_v1_models(text: str) -> list[str] | None:
    """Extract the ordered model id list from an OpenAI ``GET /v1/models`` body.

    Pure. Parses ``{"data": [{"id": ...}, ...]}`` into the list of ``id``
    strings in order. Returns ``None`` on non-JSON text, a missing or
    non-list ``data`` field, or any entry that is not an object or lacks a
    non-empty string ``id``. Duplicate ids are preserved as-is (validation is
    the registry's job, not the parser's).
    """
    try:
        body = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    ids: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            return None
        ids.append(model_id)
    return ids


class ModelRegistry:
    """The ``model -> backend_name`` routing map.

    Single-asyncio-loop contract (same as ``MetricsCache``): synchronous
    read-modify-write, no ``await`` inside, no locks.

    Backends are registered once at app build (:meth:`register_backend`, in
    config order — registration order is the candidate order and the
    tie-break for the collision policy). Each backend's owned set is
    reconciled with :meth:`sync` (explicit ``models`` if configured, else the
    discovered served set).

    Duplicate model ids are legal (decision 11): :meth:`resolve_candidates`
    is the routing lookup used by the v2 failover walk and returns ALL
    owners in registration (config) order. :meth:`resolve` is retained until
    the app switches over; for a multi-owner model it returns the collision
    winner (default-wins), and the conflict is logged once per
    (model, backend) pair.
    """

    def __init__(self) -> None:
        self._owned: dict[str, set[str]] = {}  # backend name -> owned model ids
        self._defaults: dict[str, bool] = {}  # backend name -> is_default
        self._order: list[str] = []  # backend names in registration order
        self._logged_conflicts: set[tuple[str, str]] = set()

    def register_backend(self, name: str, *, is_default: bool) -> None:
        """Register a backend (called once at app build, per configured backend).

        Callers must register each backend name exactly once (config
        validation guarantees unique names); a duplicate registration would
        corrupt the registration-order tie-break.
        """
        self._defaults[name] = is_default
        self._owned.setdefault(name, set())
        self._order.append(name)

    def sync(self, name: str, owned_models: set[str]) -> None:
        """Reconcile one backend's owned model set.

        Models newly owned by ``name`` are claimed from any previous owner.
        Models no longer owned by ``name`` are re-resolved against the
        remaining backends (default backend wins, else the existing owner,
        else dropped from the map).

        Raises:
            ValueError: if ``name`` was never registered — ownership for an
                unregistered backend would be invisible to :meth:`resolve`
                (absent from the registration order), so a caller typo must
                fail loudly instead of silently no-opping.
        """
        if name not in self._defaults:
            raise ValueError(f"backend {name!r} is not registered")
        previous = self._owned.get(name, set())
        self._owned[name] = set(owned_models)
        affected = previous | owned_models
        for model in affected:
            self._resolve_model(model)

    def resolve(self, model: str) -> str | None:
        """Resolve a model id to a backend name; ``None`` when unowned.

        Retained as the current app path. For a model owned by 2+
        backends (legal — decision 11) returns the collision winner
        (default-wins); use :meth:`resolve_candidates` for the full
        candidate set.
        """
        owners = self._owners(model)
        if not owners:
            return None
        return self._winner(model, owners)

    def resolve_candidates(self, model: str) -> tuple[str, ...]:
        """Return the ordered backend-name tuple serving ``model``.

        The v2 routing lookup (decision 11): duplicates are the normal
        multi-candidate case, not a collision. A single-owner model returns
        a 1-tuple; a multi-owner model returns ALL owners in registration
        (config) order; an unowned model returns ``()``.
        """
        return tuple(self._owners(model))

    def items(self) -> list[tuple[str, str]]:
        """Enumerate every owned model as ``(model, resolved_backend)``.

        Read-only, for the aggregate ``GET /v1/models`` and ``/healthz``.
        Each model appears exactly once, under its resolved owner (the same
        collision winner as :meth:`resolve`); models owned by no backend do
        not appear. Order follows registration order of the owning backend,
        then the model id.
        """
        result: list[tuple[str, str]] = []
        for backend in self._order:
            for model in sorted(self._owned.get(backend, ())):
                if self._winner(model, self._owners(model)) == backend:
                    result.append((model, backend))
        return result

    def _owners(self, model: str) -> list[str]:
        """Backends owning ``model``, in registration order."""
        return [b for b in self._order if model in self._owned.get(b, ())]

    def _winner(self, model: str, owners: list[str]) -> str:
        """Apply the collision policy and return the owning backend.

        The ``default: true`` backend wins if any owner is the default, else
        the first-registered owner keeps it. Logs the conflict once per
        (model, backend) pair — repeated syncs of the same ownership do not
        re-log.
        """
        if len(owners) <= 1:
            return owners[0]
        default_owners = [b for b in owners if self._defaults.get(b)]
        winner = default_owners[0] if default_owners else owners[0]
        for backend in owners:
            key = (model, backend)
            if key not in self._logged_conflicts:
                self._logged_conflicts.add(key)
                logger.warning(
                    "model %r is claimed by backend %r while also owned by %s; resolved to %r",
                    model,
                    backend,
                    [o for o in owners if o != backend],
                    winner,
                )
        return winner

    def _resolve_model(self, model: str) -> None:
        """Re-resolve one model after an ownership change (no-op when unowned)."""
        owners = self._owners(model)
        if not owners:
            return
        self._winner(model, owners)
