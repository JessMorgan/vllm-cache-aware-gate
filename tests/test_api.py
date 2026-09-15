"""End-to-end ASGI tests for the gate app (``gate.app.create_app``).

Drives the full request path against a fake vLLM (``httpx.MockTransport``) and
a directly-controlled :class:`MetricsCache`, with the poller disabled
(``start_poller=False``) so it does not interfere. Covers the load-bearing
invariants: allow/forward, 429 with the governing tier's ``Retry-After``, the
inclusive within-tier boundary, fail-open on stale/never-fetched metrics,
streaming passthrough (not buffered), upstream 5xx propagation (unmasked),
``/healthz`` liveness, 404 for unknown paths, and gating of both generation
endpoints.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from gate.app import create_app
from gate.config import GateConfig, Threshold
from gate.metrics import MetricsCache
from gate.tokens import estimate_context_tokens

SSE_BODY = b'data: {"a":1}\n\ndata: [DONE]\n\n'
JSON_HEADERS = {"content-type": "application/json"}

# A low-usage Prometheus text (10%); the poller is off, so this is only used
# if a test were to fetch /metrics — kept for completeness of the fake.
LOW_METRICS = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.10\n"
)


class FakeVLLM:
    """A controllable fake vLLM served through ``httpx.MockTransport``.

    Records every generation request (method/path/query/headers/body) so tests
    can assert what the gate forwarded. ``/metrics`` returns ``metrics_text``;
    the generation endpoints return a canned non-stream JSON or the SSE body
    when the request body has ``"stream": true``.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.metrics_text = LOW_METRICS
        self.completions_status = 200
        self.completions_json: dict = {"id": "x", "choices": [{"message": {"content": "hi"}}]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/metrics":
            return httpx.Response(200, text=self.metrics_text)
        if path in ("/v1/chat/completions", "/v1/completions"):
            body = request.content
            self.requests.append(
                {
                    "method": request.method,
                    "path": path,
                    "query": request.url.query,
                    "headers": dict(request.headers),
                    "body": body,
                }
            )
            try:
                parsed = json.loads(body)
                is_stream = isinstance(parsed, dict) and parsed.get("stream") is True
            except (json.JSONDecodeError, ValueError):
                is_stream = False
            if is_stream:
                return httpx.Response(
                    200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
                )
            return httpx.Response(self.completions_status, json=self.completions_json)
        return httpx.Response(404)

    @property
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def make_config(stale_after_s: float = 60.0) -> GateConfig:
    """A GateConfig with two tiers (50/4096/15, 80/1024/30) and a large
    ``stale_after_s`` so a freshly-updated cache is not stale."""
    return GateConfig(
        vllm_host="vllm",
        vllm_port=9000,
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=stale_after_s,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
    )


def chat_body(prompt_chars: int, **extra: object) -> bytes:
    """A chat-completions body whose single user message is ``prompt_chars`` long."""
    payload: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "a" * prompt_chars}],
    }
    payload.update(extra)
    return json.dumps(payload).encode()


def completions_body(prompt_chars: int) -> bytes:
    """A completions body with a single string prompt of ``prompt_chars``."""
    return json.dumps({"model": "m", "prompt": "a" * prompt_chars}).encode()


def build_app(cfg: GateConfig, cache: MetricsCache, fake: FakeVLLM) -> TestClient:
    """Build the app with the fake upstream and return a TestClient (lifespan off)."""
    upstream = fake.client
    app = create_app(cfg, cache=cache, upstream=upstream, start_poller=False)
    return TestClient(app)


# --- 1. allow-forward (low usage) -------------------------------------------


def test_allow_forward_low_usage() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)  # 10% -> below all tiers
    with build_app(make_config(), cache, fake) as client:
        body = chat_body(100)
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == fake.completions_json
    assert len(fake.requests) == 1
    req = fake.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["method"] == "POST"
    assert req["body"] == body  # forwarded verbatim


# --- 2. 429 (high usage, big prompt) ----------------------------------------


def test_429_high_usage_big_prompt() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)  # 90% -> 80-tier active, max_context 1024
    cfg = make_config()
    body = chat_body(5000)  # estimate ~1506 > 1024
    assert estimate_context_tokens(body, cfg) > 1024
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"  # governing 80-tier timeout
    assert resp.json()["error"]["code"] == "kv_cache_too_full"
    assert resp.json()["error"]["type"] == "cache_pressure"
    assert len(fake.requests) == 0  # not forwarded


# --- 3. 429 within-tier boundary --------------------------------------------


def test_within_tier_boundary_exactly_at_max_allows() -> None:
    """estimate == max_context (1024) is ALLOWED (inclusive boundary)."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)
    cfg = make_config()
    body = chat_body(3072)  # ceil(3072/4)+256 = 768+256 = 1024
    assert estimate_context_tokens(body, cfg) == 1024
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert len(fake.requests) == 1


def test_within_tier_boundary_just_over_rejects() -> None:
    """estimate == max_context + 1 (1025) is REJECTED."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)
    cfg = make_config()
    body = chat_body(3073)  # ceil(3073/4)+256 = 769+256 = 1025
    assert estimate_context_tokens(body, cfg) == 1025
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
    assert len(fake.requests) == 0


# --- 4. fail-open (stale metrics) -------------------------------------------


def test_fail_open_stale_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """High usage but stale metrics -> fail open (forward), not 429."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)
    monkeypatch.setattr(cache, "is_stale", lambda *a, **k: True)
    cfg = make_config()
    body = chat_body(5000)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200  # forwarded despite high usage
    assert len(fake.requests) == 1


# --- 5. fail-open (never fetched) -------------------------------------------


def test_fail_open_never_fetched() -> None:
    """Fresh cache (no update) -> fail open (forward), not 429."""
    fake = FakeVLLM()
    cache = MetricsCache()  # never updated
    cfg = make_config()
    body = chat_body(5000)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert len(fake.requests) == 1


# --- 6. streaming passthrough ------------------------------------------------


def test_streaming_passthrough() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)
    cfg = make_config()
    body = chat_body(100, stream=True)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream"
    assert resp.content == SSE_BODY  # SSE passed through verbatim, not buffered
    assert len(fake.requests) == 1


# --- 7. upstream 5xx propagated ---------------------------------------------


def test_upstream_5xx_propagated() -> None:
    fake = FakeVLLM()
    fake.completions_status = 503
    fake.completions_json = {"error": "unavailable"}
    cache = MetricsCache()
    cache.update(0.10)
    cfg = make_config()
    body = chat_body(100)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 503  # not masked to 200 or 429
    assert resp.json() == {"error": "unavailable"}
    assert len(fake.requests) == 1


# --- 8. /healthz -------------------------------------------------------------


def test_healthz() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.42)
    with build_app(make_config(), cache, fake) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "metrics_age_s" in data
    assert data["kv_usage"] == pytest.approx(0.42)


def test_healthz_never_fetched() -> None:
    """Liveness is 200 even when metrics were never fetched."""
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["metrics_age_s"] is None
    assert data["kv_usage"] is None


# --- 9. 404 unknown path -----------------------------------------------------


def test_404_unknown_path() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        resp = client.get("/v1/embeddings")
    assert resp.status_code == 404


# --- 10. completions endpoint also gated -------------------------------------


def test_completions_endpoint_gated() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)  # 90% -> 80-tier active, max_context 1024
    cfg = make_config()
    body = completions_body(5000)  # estimate ~1506 > 1024
    assert estimate_context_tokens(body, cfg) > 1024
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
    assert resp.json()["error"]["code"] == "kv_cache_too_full"
    assert len(fake.requests) == 0


def test_completions_endpoint_low_usage_forwards() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)
    cfg = make_config()
    body = completions_body(100)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert len(fake.requests) == 1
    assert fake.requests[0]["path"] == "/v1/completions"
