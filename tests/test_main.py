"""Tests for gate.main: the process entrypoint wiring.

Covers the wiring contract of ``main()``: a ``ConfigError`` exits non-zero,
the happy path builds the app via ``create_app`` with ``start_poller=True``
and hands it to ``uvicorn.run`` with the configured listen host/port, the
``LOG_LEVEL`` env var is forwarded to uvicorn, and the startup INFO log
reports the autoconfig knobs and the tiered tier count (or the none case).
Uvicorn itself is recorded, not run.
"""

from __future__ import annotations

import logging

import pytest

import gate.main as main_mod
from gate.config import ConfigError, GateConfig, Threshold


def make_cfg() -> GateConfig:
    """A realistic config with two thresholds."""
    return GateConfig(
        vllm_host="vllm",
        vllm_port=9000,
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=4.0,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
    )


def test_invalid_config_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ConfigError from load_config must exit with SystemExit(1)."""

    def boom() -> GateConfig:
        raise ConfigError("bad config")

    monkeypatch.setattr(main_mod, "load_config", boom)
    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()
    assert exc_info.value.code == 1


def test_happy_path_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() builds the app via create_app(start_poller=True) and runs
    uvicorn with the configured listen host/port."""
    cfg = make_cfg()
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)

    created: dict = {}
    sentinel_app = object()

    def fake_create_app(
        cfg_arg: GateConfig, *, cache: object, upstream: object, start_poller: bool
    ) -> object:
        created["cfg"] = cfg_arg
        created["cache"] = cache
        created["upstream"] = upstream
        created["start_poller"] = start_poller
        return sentinel_app

    monkeypatch.setattr(main_mod, "create_app", fake_create_app)

    class FakeClient:
        """Stands in for httpx.AsyncClient; records construction kwargs."""

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(main_mod.httpx, "AsyncClient", FakeClient)

    run_calls: list[dict] = []

    def fake_run(app: object, **kwargs: object) -> None:
        run_calls.append({"app": app, **kwargs})

    monkeypatch.setattr(main_mod.uvicorn, "run", fake_run)

    main_mod.main()

    # create_app wiring contract.
    assert created["cfg"] is cfg
    assert created["start_poller"] is True
    assert isinstance(created["cache"], main_mod.MetricsCache)
    assert isinstance(created["upstream"], FakeClient)

    # uvicorn.run received the app create_app returned, with listen host/port.
    assert len(run_calls) == 1
    call = run_calls[0]
    assert call["app"] is sentinel_app
    assert call["host"] == cfg.listen_host
    assert call["port"] == cfg.listen_port


def test_log_level_env_forwarded_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    """LOG_LEVEL=DEBUG is lowercased and passed to uvicorn.run."""
    cfg = make_cfg()
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "create_app", lambda _cfg, **_kw: object())
    monkeypatch.setattr(main_mod.httpx, "AsyncClient", lambda **_kw: object())

    run_calls: list[dict] = []

    def fake_run(app: object, **kwargs: object) -> None:
        run_calls.append(kwargs)

    monkeypatch.setattr(main_mod.uvicorn, "run", fake_run)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    main_mod.main()

    assert run_calls[0]["log_level"] == "debug"


def test_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """With LOG_LEVEL unset, uvicorn.run gets log_level='info'."""
    cfg = make_cfg()
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "create_app", lambda _cfg, **_kw: object())
    monkeypatch.setattr(main_mod.httpx, "AsyncClient", lambda **_kw: object())

    run_calls: list[dict] = []

    def fake_run(app: object, **kwargs: object) -> None:
        run_calls.append(kwargs)

    monkeypatch.setattr(main_mod.uvicorn, "run", fake_run)
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    main_mod.main()

    assert run_calls[0]["log_level"] == "info"


def _stub_uvicorn_and_deps(monkeypatch: pytest.MonkeyPatch, cfg: GateConfig) -> None:
    """Monkeypatch load_config/create_app/AsyncClient/uvicorn.run so main()
    runs to completion without touching the network."""
    monkeypatch.setattr(main_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(main_mod, "create_app", lambda _cfg, **_kw: object())
    monkeypatch.setattr(main_mod.httpx, "AsyncClient", lambda **_kw: object())
    monkeypatch.setattr(main_mod.uvicorn, "run", lambda _app, **_kw: None)


def test_startup_logs_autoconfig_knobs_and_no_tiers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A zero-config (no thresholds) startup logs the autoconfig knobs and
    the 'none — autoconfig only' tier line at INFO."""
    cfg = GateConfig(vllm_host="vllm", vllm_port=9000)
    _stub_uvicorn_and_deps(monkeypatch, cfg)

    with caplog.at_level(logging.INFO, logger="gate.main"):
        main_mod.main()

    messages = [r.message for r in caplog.records if r.name == "gate.main"]
    assert "autoconfig: target 85.0% KV cache, token margin 1.25, retry 5-60s" in messages
    assert "tiered policy: none — autoconfig only" in messages
    assert all(r.levelno == logging.INFO for r in caplog.records if r.name == "gate.main")


def test_startup_logs_tier_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A config with N thresholds logs the N-tier line at INFO."""
    cfg = make_cfg()  # two thresholds
    _stub_uvicorn_and_deps(monkeypatch, cfg)

    with caplog.at_level(logging.INFO, logger="gate.main"):
        main_mod.main()

    messages = [r.message for r in caplog.records if r.name == "gate.main"]
    assert "autoconfig: target 85.0% KV cache, token margin 1.25, retry 5-60s" in messages
    assert "tiered policy: 2 threshold tier(s)" in messages
