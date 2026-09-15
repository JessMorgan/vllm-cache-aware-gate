"""Entrypoint for the gate: load config, build the app, run uvicorn.

The poller is started by the app lifespan (which uvicorn triggers), so
:func:`main` does not start it separately. On a configuration error the
process exits non-zero so the container does not start with a bad config.
"""

from __future__ import annotations

import logging
import os

import httpx
import uvicorn

from gate.app import create_app
from gate.config import ConfigError, load_config
from gate.metrics import MetricsCache

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

    client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT)
    cache = MetricsCache()
    app = create_app(cfg, cache=cache, upstream=client, start_poller=True)

    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port, log_level=level.lower())


if __name__ == "__main__":
    main()
