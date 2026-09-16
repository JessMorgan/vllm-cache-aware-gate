"""Entrypoint for the gate: load config, build the app, run uvicorn.

The poller is started by the app lifespan (which uvicorn triggers), so
:func:`main` does not start it separately. On a configuration error the
process exits non-zero so the container does not start with a bad config.

After a successful config load, :func:`main` logs one INFO line per
configured backend — name, host:port, the default flag, the tiered tier
count (or "none — autoconfig only"), and the four autoconfig knobs with
"(override)" marking the per-backend values that are explicitly set (an
unmarked value is the global default) — so the operator can confirm the
active per-backend policy. It then logs one INFO line per ``routing:``
entry (model → policy/order, plus the ``large_small`` threshold when
present); an empty ``routing:`` section logs nothing. There is no CLI:
argv is ignored (autoconfig is always on).
"""

from __future__ import annotations

import logging
import os

import httpx
import uvicorn

from gate.app import create_app
from gate.config import ConfigError, load_config

log = logging.getLogger("gate.main")

# Shared upstream client timeout: connect fast, no read timeout (SSE streams
# can be long-lived), bounded write/pool.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


def main() -> None:
    """Load config, build the app, and run uvicorn until interrupted."""
    level = os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("invalid configuration: %s", e)
        raise SystemExit(1) from e

    # One INFO line per backend (decision 6): name, host:port, default flag,
    # tier count, and the four autoconfig knobs — "(override)" marks a
    # per-backend value; an unmarked value is the global default.
    for b in cfg.backends:
        # Effective tiers: the per-backend override, else the global
        # thresholds (app.py resolves the same way) — so a backend that
        # inherits the global tiers is logged with that tier count, not
        # "none — autoconfig only".
        effective = b.thresholds if b.thresholds is not None else cfg.thresholds
        tiers = f"{len(effective)} tier(s)" if effective else "none — autoconfig only"
        target = (
            f"{b.target_kv_cache_pct}% (override)"
            if b.target_kv_cache_pct is not None
            else f"{cfg.target_kv_cache_pct}%"
        )
        margin = (
            f"{b.token_margin} (override)" if b.token_margin is not None else f"{cfg.token_margin}"
        )
        retry_min = (
            f"{b.retry_min_s} (override)" if b.retry_min_s is not None else f"{cfg.retry_min_s}"
        )
        retry_max = (
            f"{b.retry_max_s} (override)" if b.retry_max_s is not None else f"{cfg.retry_max_s}"
        )
        log.info(
            "backend %r (%s:%d)%s: %s, target %s, margin %s, retry %s-%ss",
            b.name,
            b.host,
            b.port,
            " [default]" if b.default else "",
            tiers,
            target,
            margin,
            retry_min,
            retry_max,
        )

    # One INFO line per routing entry (decision 9/15): model, policy, and
    # candidate order — plus the threshold for large_small — so the operator
    # can confirm the active per-model routing. An empty routing section
    # logs nothing (no noise).
    for entry in cfg.routing:
        threshold = (
            f", threshold {entry.spec.threshold_tokens} tokens"
            if entry.spec.threshold_tokens is not None
            else ""
        )
        log.info(
            "routing %r: policy=%s order=%s%s",
            entry.model,
            entry.spec.policy,
            list(entry.spec.order),
            threshold,
        )

    client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    app = create_app(cfg, upstream=client, start_poller=True)

    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port, log_level=level.lower())


if __name__ == "__main__":
    main()
