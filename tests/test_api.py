"""End-to-end ASGI tests for the gate app (``gate.app.create_app``).

Drives the full request path against a fake vLLM (``httpx.MockTransport``) and
a directly-controlled :class:`MetricsCache` / :class:`KvRemaining` /
:class:`CapacityCache`, with the poller disabled (``start_poller=False``) so
it does not interfere. Covers the load-bearing invariants: allow/forward, 429
with the governing tier's ``Retry-After``, the inclusive within-tier boundary,
fail-open on stale/never-fetched metrics (the staleness gate outranks the
counter), the autoconfig counter (charge-on-forward, both-layers-reject with
max-timeout rejector, "small prompts only", zero-config), streaming
passthrough (not buffered), upstream 5xx propagation (unmasked), ``/healthz``
liveness (capacity + remaining, no ``mode``), ``/metrics`` (including
``gate_kv_cache_remaining_tokens``), 404 for unknown paths, gating of both
generation endpoints, and the fatal poller done-callback (``os._exit(1)`` on
task exception, not on clean cancellation).
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from gate.app import _poller_fatal, create_app
from gate.config import GateConfig, Threshold
from gate.metrics import CapacityCache, CapacityUnavailableError, KvRemaining, MetricsCache
from gate.tokens import estimate_context_tokens

SSE_BODY = b'data: {"a":1}\n\ndata: [DONE]\n\n'
JSON_HEADERS = {"content-type": "application/json"}

# The fake vLLM /metrics body carries BOTH gauges the poller needs (usage
# fraction + capacity tokens). The poller is off in these tests, so this is
# only used if a test were to fetch /metrics — kept for completeness of the
# fake.
FULL_METRICS = (
    "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
    "# TYPE vllm:kv_cache_usage_perc gauge\n"
    "vllm:kv_cache_usage_perc 0.5\n"
    "# HELP vllm:kv_cache_size_tokens Total KV cache size in tokens.\n"
    "# TYPE vllm:kv_cache_size_tokens gauge\n"
    "vllm:kv_cache_size_tokens 100000\n"
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
        self.metrics_text = FULL_METRICS
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


def build_app(
    cfg: GateConfig,
    cache: MetricsCache,
    fake: FakeVLLM,
    *,
    counter: KvRemaining | None = None,
    capacity_cache: CapacityCache | None = None,
) -> TestClient:
    """Build the app with the fake upstream and return a TestClient (lifespan off)."""
    upstream = fake.client
    app = create_app(
        cfg,
        cache=cache,
        upstream=upstream,
        start_poller=False,
        counter=counter,
        capacity_cache=capacity_cache,
    )
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


# --- /metrics ---------------------------------------------------------------


def test_metrics_endpoint_200_and_content_type() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


def test_metrics_contains_required_families() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        body = client.get("/metrics").text
    for family in (
        "gate_requests_total",
        "gate_request_ctx_tokens",
        "gate_kv_cache_usage_pct",
        "gate_config_info",
        "gate_metrics_fresh",
        "gate_metrics_age_s",
    ):
        assert family in body


def test_metrics_counts_forwarded_and_rejected() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cfg = make_config()
    with build_app(cfg, cache, fake) as client:
        cache.update(0.10)  # low usage -> forwards
        resp = client.post("/v1/chat/completions", content=chat_body(100), headers=JSON_HEADERS)
        assert resp.status_code == 200
        cache.update(0.90)  # high usage -> 80-tier, max_context 1024
        body = completions_body(5000)  # estimate ~1506 > 1024 -> 429
        assert estimate_context_tokens(body, cfg) > 1024
        resp = client.post("/v1/completions", content=body, headers=JSON_HEADERS)
        assert resp.status_code == 429
        metrics_body = client.get("/metrics").text
    assert 'gate_requests_total{endpoint="chat_completions",model="m",result="forwarded"} 1.0' in (
        metrics_body
    )
    assert 'gate_requests_total{endpoint="completions",model="m",result="rejected"} 1.0' in (
        metrics_body
    )


def test_metrics_per_model_series() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)
    other_body = json.dumps(
        {"model": "other-model", "messages": [{"role": "user", "content": "a" * 100}]}
    ).encode()
    with build_app(make_config(), cache, fake) as client:
        resp1 = client.post("/v1/chat/completions", content=chat_body(100), headers=JSON_HEADERS)
        assert resp1.status_code == 200
        resp2 = client.post("/v1/chat/completions", content=other_body, headers=JSON_HEADERS)
        assert resp2.status_code == 200
        metrics_body = client.get("/metrics").text
    assert 'gate_requests_total{endpoint="chat_completions",model="m",result="forwarded"}' in (
        metrics_body
    )
    assert (
        'gate_requests_total{endpoint="chat_completions",model="other-model",result="forwarded"}'
        in metrics_body
    )


def test_metrics_kv_usage_gauge() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update_by_model({"X": 0.8})
    with build_app(make_config(), cache, fake) as client:
        body = client.get("/metrics").text
    assert 'gate_kv_cache_usage_pct{model_name="X"} 80.0' in body


def test_stats_failure_does_not_break_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stats recording failure must not 500 the request (fail-open invariant)."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stats boom")

    monkeypatch.setattr("gate.stats.GateStats.record_forwarded", _raise)
    monkeypatch.setattr("gate.stats.GateStats.record_rejected", _raise)
    fake = FakeVLLM()
    cache = MetricsCache()
    cfg = make_config()
    with build_app(cfg, cache, fake) as client:
        cache.update(0.10)  # low usage -> forwards
        fwd = client.post("/v1/chat/completions", content=chat_body(100), headers=JSON_HEADERS)
        assert fwd.status_code == 200  # forwarded despite stats failure
        cache.update(0.90)  # high usage -> 429
        body = completions_body(5000)  # estimate ~1506 > 1024
        assert estimate_context_tokens(body, cfg) > 1024
        rej = client.post("/v1/completions", content=body, headers=JSON_HEADERS)
        assert rej.status_code == 429  # rejected despite stats failure


def test_metrics_not_404_and_unknown_still_404() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        assert client.get("/metrics").status_code == 200
        assert client.get("/v1/embeddings").status_code == 404


def test_healthz_unchanged() -> None:
    """Adding /metrics must not disturb /healthz."""
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


def zero_config(stale_after_s: float = 60.0) -> GateConfig:
    """A zero-config GateConfig: no thresholds (autoconfig only)."""
    return replace(make_config(stale_after_s=stale_after_s), thresholds=())


# --- autoconfig: counter decrement ------------------------------------------


def test_counter_decrement_first_forwards_second_rejects() -> None:
    """A request whose effective size fits once but not twice: 200 then 429.

    Effective size = ceil(ctx * margin) = ceil(281 * 1.25) = 352; counter
    anchored at 500 -> first request forwards (counter -> 148), the identical
    second request exceeds the remaining 148 -> 429.
    """
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)  # 10% -> below all tiers
    counter = KvRemaining()
    counter.reanchor(500)
    cfg = make_config()
    body = chat_body(100)  # ctx = ceil(100/4) + 256 = 281
    ctx = estimate_context_tokens(body, cfg)
    assert ctx == 281
    effective = int(math.ceil(ctx * cfg.token_margin))
    assert effective == 352
    with build_app(cfg, cache, fake, counter=counter) as client:
        first = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
        assert first.status_code == 200
        assert counter.value() == 500 - effective  # charged at admission
        second = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
        assert second.status_code == 429
        # scaled retry: ceil(clamp(5 * (1 + 352/148), 5, 60)) = ceil(16.89) = 17
        assert second.headers["Retry-After"] == "17"
        assert counter.value() == 500 - effective  # rejected request charges nothing
        assert len(fake.requests) == 1  # only the first was forwarded


def test_rejected_request_charges_nothing_and_message_names_autoconfig() -> None:
    """An autoconfig reject keeps code/type and names the autoconfig details."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)
    counter = KvRemaining()
    counter.reanchor(100)
    capacity = CapacityCache()
    capacity.update(100000)
    cfg = make_config()
    body = chat_body(100)  # effective ceil(281*1.25) = 352 > 100 -> autoconfig rejects
    with build_app(cfg, cache, fake, counter=counter, capacity_cache=capacity) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    assert counter.value() == 100  # rejected -> no charge
    err = resp.json()["error"]
    assert err["code"] == "kv_cache_too_full"
    assert err["type"] == "cache_pressure"
    assert "~100 of 100000 tokens" in err["message"]  # remaining + capacity
    assert "100000" in err["message"]  # capacity
    assert "85" in err["message"]  # target pct


# --- autoconfig: staleness gate outranks the counter -------------------------


def test_stale_usage_fails_open_even_when_counter_over_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale feed + counter reanchored to 0 -> 200 fail-open; subtraction
    (bookkeeping) still happens."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.90)
    monkeypatch.setattr(cache, "is_stale", lambda *a, **k: True)
    counter = KvRemaining()
    counter.reanchor(0)  # over-committed: every non-trivial request would reject
    cfg = make_config()
    body = chat_body(5000)
    ctx = estimate_context_tokens(body, cfg)
    assert ctx is not None
    with build_app(cfg, cache, fake, counter=counter) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200  # failed open despite counter == 0
    assert len(fake.requests) == 1
    assert counter.value() == -int(math.ceil(ctx * cfg.token_margin))  # unclamped


def test_never_fetched_fails_open_even_when_counter_over_committed() -> None:
    """Never-fetched feed + counter reanchored to 0 -> 200 fail-open."""
    fake = FakeVLLM()
    cache = MetricsCache()  # never updated
    counter = KvRemaining()
    counter.reanchor(0)
    cfg = make_config()
    body = chat_body(5000)
    with build_app(cfg, cache, fake, counter=counter) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 200
    assert len(fake.requests) == 1


# --- autoconfig: both layers reject ------------------------------------------


def test_both_layers_reject_max_retry_and_rejector_message() -> None:
    """Both layers reject -> Retry-After = max(tier, auto); the
    higher-timeout rejector (autoconfig, 60 > 15) is named in the message."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.55)  # 55% -> 50-tier governs (max_context 4096, timeout 15)
    counter = KvRemaining()
    counter.reanchor(100)
    capacity = CapacityCache()
    capacity.update(100000)
    cfg = make_config()
    # 16000 chars -> ceil(16000/4)+256 = 4256 > 4096 -> the 50-tier rejects
    # (retry 15); effective = ceil(4256*1.25) = 5320 > 100 -> autoconfig
    # also rejects (retry 60, remaining <= 100 is far over).
    body = chat_body(16000)
    ctx = estimate_context_tokens(body, cfg)
    assert ctx is not None and ctx > 4096
    effective = int(math.ceil(ctx * cfg.token_margin))
    assert effective > 100  # autoconfig also rejects
    with build_app(cfg, cache, fake, counter=counter, capacity_cache=capacity) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"  # max(15, 60)
    err = resp.json()["error"]
    assert err["code"] == "kv_cache_too_full"
    assert err["type"] == "cache_pressure"
    # The autoconfig rejector (higher timeout) is named, not the tier.
    assert "~100 of 100000 tokens" in err["message"]  # remaining + capacity
    assert "100000" in err["message"]  # capacity
    assert "85" in err["message"]  # target


# --- autoconfig: "small prompts only" scenario --------------------------------


def test_small_prompts_only_scenario() -> None:
    """tiers [{50,4096,15},{80,1024,30}], usage 55%, capacity 100000,
    target 85 -> anchor 30000: small prompt forwards, large prompt 429 with
    the tiered rejector and Retry-After 15."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.55)
    counter = KvRemaining()
    counter.reanchor(30000)  # floor(100000 * (0.85 - 0.55))
    capacity = CapacityCache()
    capacity.update(100000)
    cfg = make_config()

    small = chat_body(3000)  # ctx = 750+256 = 1006; effective 1258 <= 30000
    assert estimate_context_tokens(small, cfg) == 1006
    large = chat_body(16000)  # ctx = 4256 > 4096 -> tiered rejects
    assert estimate_context_tokens(large, cfg) == 4256

    with build_app(cfg, cache, fake, counter=counter, capacity_cache=capacity) as client:
        ok = client.post("/v1/chat/completions", content=small, headers=JSON_HEADERS)
        assert ok.status_code == 200
        assert counter.value() == 30000 - int(math.ceil(1006 * 1.25))
        bad = client.post("/v1/chat/completions", content=large, headers=JSON_HEADERS)
        assert bad.status_code == 429
        assert bad.headers["Retry-After"] == "15"  # tiered rejector's timeout
        assert "4096" in bad.json()["error"]["message"]  # tier text
        assert len(fake.requests) == 1


# --- autoconfig: zero-config app ----------------------------------------------


def test_zero_config_app_serves_both_endpoints() -> None:
    """No thresholds anywhere: both endpoints forward when headroom allows."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.50)
    counter = KvRemaining()
    counter.reanchor(100000)
    capacity = CapacityCache()
    capacity.update(100000)
    cfg = zero_config()
    with build_app(cfg, cache, fake, counter=counter, capacity_cache=capacity) as client:
        chat = client.post("/v1/chat/completions", content=chat_body(100), headers=JSON_HEADERS)
        assert chat.status_code == 200
        comp = client.post("/v1/completions", content=completions_body(100), headers=JSON_HEADERS)
        assert comp.status_code == 200
        assert len(fake.requests) == 2


def test_zero_config_rejects_when_counter_exhausted() -> None:
    """Zero-config: no tiers, but the autoconfig layer still rejects."""
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.50)
    counter = KvRemaining()
    counter.reanchor(100)
    cfg = zero_config()
    body = chat_body(100)  # effective ceil(281*1.25) = 352 > 100
    with build_app(cfg, cache, fake, counter=counter) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 429
    # scaled retry: ceil(clamp(5 * (1 + 352/100), 5, 60)) = ceil(22.6) = 23
    assert resp.headers["Retry-After"] == "23"
    assert len(fake.requests) == 0


# --- /healthz: capacity + remaining, no mode ----------------------------------


def test_healthz_carries_capacity_and_remaining_and_no_mode() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.42)
    counter = KvRemaining()
    counter.reanchor(12345)
    capacity = CapacityCache()
    capacity.update(100000)
    with build_app(make_config(), cache, fake, counter=counter, capacity_cache=capacity) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "metrics_age_s" in data
    assert data["kv_usage"] == pytest.approx(0.42)
    assert data["kv_cache_capacity_tokens"] == 100000
    assert data["kv_cache_remaining_tokens"] == 12345
    assert "mode" not in data


def test_healthz_null_capacity_and_remaining_when_never_set() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        data = client.get("/healthz").json()
    assert data["kv_cache_capacity_tokens"] is None
    assert data["kv_cache_remaining_tokens"] is None
    assert "mode" not in data


# --- /metrics: gate_kv_cache_remaining_tokens ---------------------------------


def test_metrics_remaining_gauge_reflects_decrements() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    cache.update(0.10)
    counter = KvRemaining()
    counter.reanchor(500)
    cfg = make_config()
    body = chat_body(100)  # effective ceil(281*1.25) = 352
    with build_app(cfg, cache, fake, counter=counter) as client:
        pre = client.get("/metrics").text
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
        assert resp.status_code == 200
        post = client.get("/metrics").text
    assert "gate_kv_cache_remaining_tokens 500.0" in pre
    assert "gate_kv_cache_remaining_tokens 148.0" in post


def test_metrics_remaining_gauge_nan_when_never_anchored() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        body = client.get("/metrics").text
    assert "gate_kv_cache_remaining_tokens NaN" in body


# --- fatal done-callback -------------------------------------------------------


async def test_poller_fatal_exits_on_task_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poller task that dies with an exception triggers os._exit(1)."""
    exited: list[int] = []
    monkeypatch.setattr("gate.app.os._exit", exited.append)

    async def boom() -> None:
        raise CapacityUnavailableError("no capacity gauge")

    task = asyncio.create_task(boom())
    task.add_done_callback(_poller_fatal)
    with pytest.raises(CapacityUnavailableError):
        await task
    await asyncio.sleep(0)  # let the done callback run
    assert exited == [1]


async def test_poller_fatal_not_called_on_clean_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cleanly cancelled poller task must NOT trigger os._exit."""
    exited: list[int] = []
    monkeypatch.setattr("gate.app.os._exit", exited.append)

    async def sleeper() -> None:
        await asyncio.sleep(3600)

    task = asyncio.create_task(sleeper())
    task.add_done_callback(_poller_fatal)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)  # let the done callback run
    assert exited == []
