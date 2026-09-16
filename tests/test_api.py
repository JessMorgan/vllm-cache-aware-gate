"""End-to-end ASGI tests for the gate app (``gate.app.create_app``).

Drives the full request path against a fake vLLM (``httpx.MockTransport``)
and a directly-controlled per-backend state (``MetricsCache`` /
``KvRemaining`` / ``CapacityCache`` on ``app.state.backends[0]``), with the
pollers disabled (``start_poller=False``) so they do not interfere. Covers
the load-bearing invariants: allow/forward, 429 with the governing tier's
``Retry-After``, the inclusive within-tier boundary, fail-open on
stale/never-fetched metrics (the staleness gate outranks the counter), the
autoconfig counter (charge-on-forward, both-layers-reject with max-timeout
rejector, "small prompts only", zero-config), streaming passthrough (not
buffered), upstream 5xx propagation (unmasked), ``/healthz`` liveness
(per-backend array + models, no ``mode``), ``/metrics`` (including
per-backend ``gate_kv_cache_remaining_tokens``), 404 for unknown paths,
gating of both generation endpoints, the fatal poller done-callback
(``os._exit(1)`` on task exception, not on clean cancellation), and the
multi-backend routing cases (two fake vLLMs dispatched on host: routing by
model, unknown model -> default backend, per-backend 429, one-stale/
one-fresh, ``GET /v1/models`` aggregate, per-backend ``/healthz`` and
``/metrics``).

The v2 multi-backend routing segment (docs/plans/multi-backend.md §2.4) is
covered by the shared-model section: per-policy routing (round_robin
cycling, primary_fallback, large_small both sides, even/fill less-full
selection), the pre-stream failover walk (transport failure / upstream 5xx /
429 -> next candidate, with the anchor-sequence-guarded charge rollback),
the exhaustion rule (both 429 -> single 429 with max Retry-After; both
transport-failed -> 502 with ``backend="none"``), one-stale/one-fresh
walk reachability, and the duplicate-aware ``GET /v1/models`` /
``/healthz`` / ``/metrics`` (the ``backend`` label on the request counters,
``gate_routing_failovers_total``, the ``routing_json`` config info).
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from gate.app import _poller_fatal, create_app
from gate.config import Backend, GateConfig, RoutingEntry, Threshold
from gate.metrics import CapacityCache, KvRemaining, MetricsCache
from gate.models import ModelRegistry
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
                    self.completions_status,
                    content=SSE_BODY,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(self.completions_status, json=self.completions_json)
        return httpx.Response(404)

    @property
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def make_config(stale_after_s: float = 60.0) -> GateConfig:
    """A single-backend GateConfig (one ``backends`` entry, decision 6) with
    two tiers (50/4096/15, 80/1024/30) and a large ``stale_after_s`` so a
    freshly-updated cache is not stale."""
    return GateConfig(
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=stale_after_s,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
        backends=(Backend(name="vllm", host="vllm", port=9000, default=True),),
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
    cache: MetricsCache | None = None,
    fake: FakeVLLM | None = None,
    *,
    counter: KvRemaining | None = None,
    capacity_cache: CapacityCache | None = None,
) -> TestClient:
    """Build the app with the fake upstream and return a TestClient (lifespan off).

    Single-backend helper: the app builds ONE backend state from the single
    entry in ``cfg.backends`` (decision 6); the test-controlled caches are
    swapped into ``app.state.backends[0]`` before any request is made.
    """
    upstream = (fake if fake is not None else FakeVLLM()).client
    app = create_app(cfg, upstream=upstream, start_poller=False)
    bs = app.state.backends[0]
    if cache is not None:
        bs.cache = cache
    if counter is not None:
        bs.counter = counter
    if capacity_cache is not None:
        bs.capacity_cache = capacity_cache
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
        metrics_body = client.get("/metrics").text
    assert resp.status_code == 503  # not masked to 200 or 429
    assert resp.json() == {"error": "unavailable"}
    assert len(fake.requests) == 1
    # The gate's decision was "forward to this backend": an admitted request
    # that gets a pre-stream 5xx is still recorded as forwarded with that
    # backend (the 5xx is the upstream's answer, not the gate's rejection —
    # baseline stats semantics; pins the last-5xx-propagation path).
    assert (
        'gate_requests_total{backend="vllm",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )
    assert (
        'gate_request_ctx_tokens_count{backend="vllm",model="m",result="forwarded"} 1.0'
        in metrics_body
    )


def test_upstream_streaming_5xx_propagates_body_verbatim() -> None:
    """A single-backend (last-candidate) STREAMING pre-stream 5xx propagates
    the upstream body VERBATIM (not empty): the generator is left intact so
    starlette streams the 5xx body as-is. Pins the regression where an
    unconditional prime-then-close would empty the propagated body."""
    fake = FakeVLLM()
    fake.completions_status = 500  # the backend answers the stream with a 500
    cache = MetricsCache()
    cache.update(0.10)
    cfg = make_config()
    body = chat_body(100, stream=True)
    with build_app(cfg, cache, fake) as client:
        resp = client.post("/v1/chat/completions", content=body, headers=JSON_HEADERS)
    assert resp.status_code == 500  # not masked to 200 or 429
    assert resp.headers["content-type"] == "text/event-stream"
    # The upstream SSE body propagates verbatim — NOT empty (a primed-then-
    # closed generator would re-iterate empty and mask the body to b'').
    assert resp.content == SSE_BODY
    assert resp.content != b""
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
    backends = data["backends"]
    assert len(backends) == 1  # the single backends config entry
    assert backends[0]["name"] == "vllm"
    assert "metrics_age_s" in backends[0]
    assert backends[0]["kv_usage"] == pytest.approx(0.42)


def test_healthz_never_fetched() -> None:
    """Liveness is 200 even when metrics were never fetched."""
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backends"][0]["metrics_age_s"] is None
    assert data["backends"][0]["kv_usage"] is None


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
    # Prometheus renders labels alphabetically (backend, endpoint, model,
    # result) regardless of the definition order.
    assert (
        'gate_requests_total{backend="vllm",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )
    assert (
        'gate_requests_total{backend="vllm",endpoint="completions",model="m",'
        'result="rejected"} 1.0' in metrics_body
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
    # Prometheus renders labels alphabetically (backend, endpoint, model,
    # result) regardless of the definition order.
    assert (
        'gate_requests_total{backend="vllm",endpoint="chat_completions",model="m",'
        'result="forwarded"}' in metrics_body
    )
    assert (
        'gate_requests_total{backend="vllm",endpoint="chat_completions",model="other-model",'
        'result="forwarded"}' in metrics_body
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
    assert "metrics_age_s" in data["backends"][0]
    assert data["backends"][0]["kv_usage"] == pytest.approx(0.42)


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
    entry = data["backends"][0]
    assert "metrics_age_s" in entry
    assert entry["kv_usage"] == pytest.approx(0.42)
    assert entry["kv_cache_capacity_tokens"] == 100000
    assert entry["kv_cache_remaining_tokens"] == 12345
    assert "mode" not in data


def test_healthz_null_capacity_and_remaining_when_never_set() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        data = client.get("/healthz").json()
    entry = data["backends"][0]
    assert entry["kv_cache_capacity_tokens"] is None
    assert entry["kv_cache_remaining_tokens"] is None
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
    assert 'gate_kv_cache_remaining_tokens{backend="vllm"} 500.0' in pre
    assert 'gate_kv_cache_remaining_tokens{backend="vllm"} 148.0' in post


def test_metrics_remaining_gauge_nan_when_never_anchored() -> None:
    fake = FakeVLLM()
    cache = MetricsCache()
    with build_app(make_config(), cache, fake) as client:
        body = client.get("/metrics").text
    assert 'gate_kv_cache_remaining_tokens{backend="vllm"} NaN' in body


# --- fatal done-callback -------------------------------------------------------


async def test_poller_fatal_exits_on_task_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A poller task that dies with an unexpected exception triggers
    os._exit(1). (The capacity-missing path no longer raises — it fails open
    per backend, decision 10 — so the fatal callback now fires only on an
    unexpected task death.)"""
    exited: list[int] = []
    monkeypatch.setattr("gate.app.os._exit", exited.append)

    async def boom() -> None:
        raise RuntimeError("unexpected task death")

    task = asyncio.create_task(boom())
    task.add_done_callback(_poller_fatal)
    with pytest.raises(RuntimeError):
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


# --- multi-backend -------------------------------------------------------------


def _metrics_body(usage: float, capacity: int = 100000) -> str:
    """A fake vLLM /metrics body with the usage fraction + capacity gauge."""
    return (
        "# HELP vllm:kv_cache_usage_perc Fraction of KV cache in use.\n"
        "# TYPE vllm:kv_cache_usage_perc gauge\n"
        f"vllm:kv_cache_usage_perc {usage}\n"
        "# HELP vllm:kv_cache_size_tokens Total KV cache size in tokens.\n"
        "# TYPE vllm:kv_cache_size_tokens gauge\n"
        f"vllm:kv_cache_size_tokens {capacity}\n"
    )


def _models_body(*ids: str) -> str:
    """A fake vLLM /v1/models body for the given model ids."""
    return json.dumps({"object": "list", "data": [{"id": i} for i in ids]})


class MultiFakeVLLM:
    """A fake vLLM bound to one host; records generation requests with the host.

    ``completions_status`` drives the generation response status (for the
    pre-stream 5xx failover cases); ``fail_transport`` makes the generation
    endpoint raise a transport error (for the pre-stream transport-failure
    failover cases).
    """

    def __init__(
        self,
        host: str,
        models: tuple[str, ...],
        metrics_text: str,
        order: list[str] | None = None,
    ) -> None:
        self.host = host
        self.models = models
        self.metrics_text = metrics_text
        self.requests: list[dict] = []
        self.completions_json: dict = {"id": "x", "choices": [{"message": {"content": "hi"}}]}
        self.completions_status = 200
        self.fail_transport = False
        # A shared list (owned by the test) recording the INTERLEAVED order of
        # generation requests across the fakes (per-fake request lists cannot
        # reconstruct it).
        self.order = order

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/metrics":
            return httpx.Response(200, text=self.metrics_text)
        if path == "/v1/models":
            return httpx.Response(200, text=_models_body(*self.models))
        if path in ("/v1/chat/completions", "/v1/completions"):
            body = request.content
            self.requests.append({"host": request.url.host, "path": path, "body": body})
            if self.order is not None:
                self.order.append(request.url.host)
            if self.fail_transport:
                raise httpx.ConnectError(f"simulated transport failure on {self.host}")
            try:
                parsed = json.loads(body)
                is_stream = isinstance(parsed, dict) and parsed.get("stream") is True
            except (json.JSONDecodeError, ValueError):
                is_stream = False
            if is_stream:
                return httpx.Response(
                    self.completions_status,
                    content=SSE_BODY,
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(self.completions_status, json=self.completions_json)
        return httpx.Response(404)


def multi_config() -> GateConfig:
    """A two-backend config: qwen (default, host a) and llama (host b)."""
    return GateConfig(
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=60.0,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
        backends=(
            Backend(name="qwen", host="a", port=8001, models=("qwen3-32b",), default=True),
            Backend(name="llama", host="b", port=8002, models=("llama-70b",)),
        ),
    )


def build_multi_app(
    cfg: GateConfig,
    fakes: dict[str, MultiFakeVLLM],
    *,
    cache_a: MetricsCache | None = None,
    cache_b: MetricsCache | None = None,
    counter_a: KvRemaining | None = None,
    counter_b: KvRemaining | None = None,
    capacity_a: CapacityCache | None = None,
    capacity_b: CapacityCache | None = None,
    registry: ModelRegistry | None = None,
) -> TestClient:
    """Build a two-backend app whose mock upstream dispatches on host.

    The test-controlled caches/counters are swapped into the per-backend
    state (``app.state.backends``) before any request is made. When
    ``registry`` is given it replaces the app's registry (the poller — the
    thing that would populate it via discovery — is disabled in these tests).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return fakes[request.url.host].handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(cfg, upstream=client, start_poller=False)
    bs_a, bs_b = app.state.backends
    assert (bs_a.name, bs_b.name) == ("qwen", "llama")
    if cache_a is not None:
        bs_a.cache = cache_a
    if cache_b is not None:
        bs_b.cache = cache_b
    if counter_a is not None:
        bs_a.counter = counter_a
    if counter_b is not None:
        bs_b.counter = counter_b
    if capacity_a is not None:
        bs_a.capacity_cache = capacity_a
    if capacity_b is not None:
        bs_b.capacity_cache = capacity_b
    if registry is not None:
        app.state.registry = registry
    return TestClient(app)


def _populated_registry() -> ModelRegistry:
    """A registry with both backends' models owned (what discovery would set)."""
    registry = ModelRegistry()
    registry.register_backend("qwen", is_default=True)
    registry.register_backend("llama", is_default=False)
    registry.sync("qwen", {"qwen3-32b"})
    registry.sync("llama", {"llama-70b"})
    return registry


def _multi_fakes(
    usage_a: float = 0.10, usage_b: float = 0.10
) -> tuple[dict[str, MultiFakeVLLM], GateConfig]:
    fakes = {
        "a": MultiFakeVLLM("a", ("qwen3-32b",), _metrics_body(usage_a)),
        "b": MultiFakeVLLM("b", ("llama-70b",), _metrics_body(usage_b)),
    }
    return fakes, multi_config()


def test_multi_routing_by_model() -> None:
    """Each model routes to its owning backend (asserted via the upstream host)."""
    fakes, cfg = _multi_fakes()
    with build_multi_app(
        cfg,
        fakes,
        cache_a=MetricsCache(),
        cache_b=MetricsCache(),
        registry=_populated_registry(),
    ) as client:
        a = client.post(
            "/v1/chat/completions",
            content=json.dumps(
                {"model": "qwen3-32b", "messages": [{"role": "user", "content": "a" * 100}]}
            ).encode(),
            headers=JSON_HEADERS,
        )
        b = client.post(
            "/v1/chat/completions",
            content=json.dumps(
                {"model": "llama-70b", "messages": [{"role": "user", "content": "a" * 100}]}
            ).encode(),
            headers=JSON_HEADERS,
        )
    assert a.status_code == 200
    assert b.status_code == 200
    assert [r["host"] for r in fakes["a"].requests] == ["a"]
    assert [r["host"] for r in fakes["b"].requests] == ["b"]
    assert len(fakes["a"].requests) == 1
    assert len(fakes["b"].requests) == 1


def test_multi_unknown_model_goes_to_default_backend() -> None:
    """A model owned by no backend (and "unknown") routes to the default."""
    fakes, cfg = _multi_fakes()
    with build_multi_app(
        cfg,
        fakes,
        cache_a=MetricsCache(),
        cache_b=MetricsCache(),
        registry=_populated_registry(),
    ) as client:
        for model in ("mystery-9b", "unknown"):
            resp = client.post(
                "/v1/chat/completions",
                content=json.dumps(
                    {"model": model, "messages": [{"role": "user", "content": "a" * 100}]}
                ).encode(),
                headers=JSON_HEADERS,
            )
            assert resp.status_code == 200
    # Both requests landed on the default backend (a); none on b.
    assert len(fakes["a"].requests) == 2
    assert len(fakes["b"].requests) == 0


def test_multi_per_backend_429() -> None:
    """Backend A's pool full -> A's models reject with A's Retry-After; B allows."""
    fakes, cfg = _multi_fakes(usage_a=0.90, usage_b=0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(100)
    cache_a = MetricsCache()
    cache_a.update(0.90)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    with build_multi_app(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        registry=_populated_registry(),
    ) as client:
        a_body = json.dumps(
            {"model": "qwen3-32b", "messages": [{"role": "user", "content": "a" * 5000}]}
        ).encode()
        b_body = json.dumps(
            {"model": "llama-70b", "messages": [{"role": "user", "content": "a" * 5000}]}
        ).encode()
        rej = client.post("/v1/chat/completions", content=a_body, headers=JSON_HEADERS)
        ok = client.post("/v1/chat/completions", content=b_body, headers=JSON_HEADERS)
    assert rej.status_code == 429
    # Both layers reject on A: the 80-tier (timeout 30) and the autoconfig
    # layer (counter anchored at 100 < effective size -> scaled retry at the
    # max of 60). Combined Retry-After = max(30, 60) = 60.
    assert rej.headers["Retry-After"] == "60"
    assert "qwen" in rej.json()["error"]["message"]  # message names backend A
    assert ok.status_code == 200  # B still allows
    assert len(fakes["a"].requests) == 0
    assert len(fakes["b"].requests) == 1


def test_multi_one_stale_one_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale backend A fails open; fresh backend B is still gated."""
    fakes, cfg = _multi_fakes(usage_a=0.90, usage_b=0.90)
    cache_a = MetricsCache()
    cache_a.update(0.90)
    monkeypatch.setattr(cache_a, "is_stale", lambda *a, **k: True)
    cache_b = MetricsCache()
    cache_b.update(0.90)
    with build_multi_app(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        registry=_populated_registry(),
    ) as client:
        a_body = json.dumps(
            {"model": "qwen3-32b", "messages": [{"role": "user", "content": "a" * 5000}]}
        ).encode()
        b_body = json.dumps(
            {"model": "llama-70b", "messages": [{"role": "user", "content": "a" * 5000}]}
        ).encode()
        a_resp = client.post("/v1/chat/completions", content=a_body, headers=JSON_HEADERS)
        b_resp = client.post("/v1/chat/completions", content=b_body, headers=JSON_HEADERS)
    assert a_resp.status_code == 200  # stale -> fail open
    assert b_resp.status_code == 429  # fresh -> gated (80-tier, max_context 1024)
    assert b_resp.headers["Retry-After"] == "30"
    assert len(fakes["a"].requests) == 1
    assert len(fakes["b"].requests) == 0


def test_multi_v1_models_aggregate() -> None:
    """GET /v1/models is gate-local, 200, and aggregates with owned_by."""
    fakes, cfg = _multi_fakes()
    with build_multi_app(
        cfg,
        fakes,
        cache_a=MetricsCache(),
        cache_b=MetricsCache(),
        registry=_populated_registry(),
    ) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    entries = {e["id"]: e for e in data["data"]}
    assert entries == {
        "qwen3-32b": {"id": "qwen3-32b", "object": "model", "owned_by": "qwen"},
        "llama-70b": {"id": "llama-70b", "object": "model", "owned_by": "llama"},
    }
    # Gate-local: nothing was proxied to either upstream.
    assert len(fakes["a"].requests) == 0
    assert len(fakes["b"].requests) == 0


def test_multi_healthz_per_backend_array_and_models() -> None:
    fakes, cfg = _multi_fakes()
    cache_a = MetricsCache()
    cache_a.update(0.42)
    counter_a = KvRemaining()
    counter_a.reanchor(12345)
    capacity_a = CapacityCache()
    capacity_a.update(100000)
    with build_multi_app(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=MetricsCache(),
        counter_a=counter_a,
        capacity_a=capacity_a,
        registry=_populated_registry(),
    ) as client:
        data = client.get("/healthz").json()
    assert data["status"] == "ok"
    by_name = {b["name"]: b for b in data["backends"]}
    assert set(by_name) == {"qwen", "llama"}
    assert by_name["qwen"]["kv_usage"] == pytest.approx(0.42)
    assert by_name["qwen"]["kv_cache_capacity_tokens"] == 100000
    assert by_name["qwen"]["kv_cache_remaining_tokens"] == 12345
    assert by_name["llama"]["kv_usage"] is None  # never fetched
    assert by_name["llama"]["kv_cache_remaining_tokens"] is None
    # No routing: entries -> the policy in effect is round_robin (decision 11).
    assert data["models"] == [
        {"id": "qwen3-32b", "owned_by": "qwen", "routing": "round_robin"},
        {"id": "llama-70b", "owned_by": "llama", "routing": "round_robin"},
    ]


def test_multi_metrics_per_backend_labels() -> None:
    fakes, cfg = _multi_fakes()
    cache_a = MetricsCache()
    cache_a.update(0.50)
    cache_a.update_by_model({"qwen3-32b": 0.50})
    cache_b = MetricsCache()
    cache_b.update(0.25)
    cache_b.update_by_model({"llama-70b": 0.25})
    counter_a = KvRemaining()
    counter_a.reanchor(5000)
    counter_b = KvRemaining()
    counter_b.reanchor(7000)
    with build_multi_app(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        # Flip backend A's capacity-unavailable flag (decision 10 alerting).
        client.app.state.backends[0].capacity_unavailable = True
        body = client.get("/metrics").text
    # Per-backend feed gauges.
    assert 'gate_metrics_fresh{backend="qwen"} 1.0' in body
    assert 'gate_metrics_fresh{backend="llama"} 1.0' in body
    assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 5000.0' in body
    assert 'gate_kv_cache_remaining_tokens{backend="llama"} 7000.0' in body
    # gate_kv_cache_usage_pct is the union across backends (model-keyed).
    assert 'gate_kv_cache_usage_pct{model_name="qwen3-32b"} 50.0' in body
    assert 'gate_kv_cache_usage_pct{model_name="llama-70b"} 25.0' in body
    # Per-backend capacity-unavailable gauge (A flagged, B not).
    assert 'gate_backend_capacity_unavailable{backend="qwen"} 1.0' in body
    assert 'gate_backend_capacity_unavailable{backend="llama"} 0.0' in body
    # gate_config_info serializes the backend structure (no model lists).
    config_labels: dict[str, str] = {}
    for family in text_string_to_metric_families(body):
        if family.name == "gate_config_info":
            for sample in family.samples:
                config_labels = sample.labels
    assert config_labels["backends_json"].startswith("[{")
    assert '"name":"qwen"' in config_labels["backends_json"]
    assert '"name":"llama"' in config_labels["backends_json"]
    # The models: lists are deliberately omitted from the info metric.
    assert "qwen3-32b" not in config_labels["backends_json"]
    assert "llama-70b" not in config_labels["backends_json"]


# --- v2 multi-backend routing: shared model, per-policy routing + failover ----
#
# A model id served by BOTH backends (duplicate model ids are legal, decision
# 11) plus one single-owner model each. The routing: section drives the
# per-model policy; the pre-stream failover walk (decision 12) is exercised
# through the mock transport (per-host dispatch).


def shared_multi_config(routing: tuple[RoutingEntry, ...] = ()) -> GateConfig:
    """A two-backend config where BOTH backends serve the shared model ``m``
    (plus one single-owner model each). ``routing`` is the ``routing:``
    section (RoutingEntry tuples)."""
    return GateConfig(
        listen_host="127.0.0.1",
        listen_port=8080,
        metrics_poll_interval_s=1.5,
        stale_after_s=60.0,
        chars_per_token=4,
        default_max_tokens=256,
        thresholds=(Threshold(50.0, 4096, 15), Threshold(80.0, 1024, 30)),
        backends=(
            Backend(name="qwen", host="a", port=8001, models=("m", "q-only"), default=True),
            Backend(name="llama", host="b", port=8002, models=("m", "l-only")),
        ),
        routing=routing,
    )


def _shared_fakes(
    usage_a: float = 0.10,
    usage_b: float = 0.10,
    capacity_a: int = 100000,
    capacity_b: int = 100000,
) -> tuple[dict[str, MultiFakeVLLM], list[str]]:
    """The two shared-model fakes plus a shared list recording the INTERLEAVED
    order of the generation requests (per-fake request lists cannot
    reconstruct it)."""
    order: list[str] = []
    fakes = {
        "a": MultiFakeVLLM("a", ("m", "q-only"), _metrics_body(usage_a, capacity_a), order),
        "b": MultiFakeVLLM("b", ("m", "l-only"), _metrics_body(usage_b, capacity_b), order),
    }
    return fakes, order


def _shared_registry() -> ModelRegistry:
    """A registry with both backends owning the shared model ``m`` (and each
    its single-owner model) — what discovery would set."""
    registry = ModelRegistry()
    registry.register_backend("qwen", is_default=True)
    registry.register_backend("llama", is_default=False)
    registry.sync("qwen", {"m", "q-only"})
    registry.sync("llama", {"m", "l-only"})
    return registry


def _shared_body(prompt_chars: int = 100, model: str = "m", stream: bool = False) -> bytes:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "a" * prompt_chars}],
    }
    if stream:
        payload["stream"] = True
    return json.dumps(payload).encode()


def _build_shared(
    cfg: GateConfig,
    fakes: dict[str, MultiFakeVLLM],
    *,
    cache_a: MetricsCache | None = None,
    cache_b: MetricsCache | None = None,
    counter_a: KvRemaining | None = None,
    counter_b: KvRemaining | None = None,
    capacity_a: CapacityCache | None = None,
    capacity_b: CapacityCache | None = None,
) -> TestClient:
    """Build the shared-model two-backend app (same swap-in pattern as
    build_multi_app)."""
    return build_multi_app(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
        capacity_a=capacity_a,
        capacity_b=capacity_b,
        registry=_shared_registry(),
    )


def test_shared_round_robin_alternates() -> None:
    """round_robin (the default for a multi-owner model with no entry)
    alternates across successive requests."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    cfg = shared_multi_config(
        (RoutingEntry(model="m", spec=RoutingSpec(policy="round_robin", order=("qwen", "llama"))),)
    )
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        for _ in range(4):
            resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
            assert resp.status_code == 200
    # The RR index advances only on a successful forward: a, b, a, b.
    assert [r["host"] for r in fakes["a"].requests] == ["a", "a"]
    assert [r["host"] for r in fakes["b"].requests] == ["b", "b"]
    assert order == ["a", "b", "a", "b"]


def test_shared_round_robin_default_without_entry() -> None:
    """A multi-owner model with NO routing: entry defaults to round_robin
    over the registry order (decision 11)."""
    fakes, order = _shared_fakes()
    cfg = shared_multi_config()  # no routing section at all
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        for _ in range(3):
            resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
            assert resp.status_code == 200
    assert order == ["a", "b", "a"]


def test_shared_primary_fallback_sticks_to_order_head() -> None:
    """primary_fallback always hits order[0] when it admits."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        for _ in range(3):
            resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
            assert resp.status_code == 200
    assert order == ["a", "a", "a"]


def test_shared_large_small_both_sides() -> None:
    """large_small: ctx >= threshold -> order[0]; ctx < threshold -> order[1]."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m",
                spec=RoutingSpec(
                    policy="large_small", order=("qwen", "llama"), threshold_tokens=400
                ),
            ),
        )
    )
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        # ctx = ceil(100/4) + 256 = 281 < 400 -> small -> order[1] (b).
        small = client.post("/v1/chat/completions", content=_shared_body(100), headers=JSON_HEADERS)
        assert small.status_code == 200
        # ctx = ceil(500/4) + 256 = 381 < 400 -> still small -> b.
        mid = client.post("/v1/chat/completions", content=_shared_body(500), headers=JSON_HEADERS)
        assert mid.status_code == 200
        # ctx = ceil(600/4) + 256 = 406 >= 400 -> large -> order[0] (a).
        large = client.post("/v1/chat/completions", content=_shared_body(600), headers=JSON_HEADERS)
        assert large.status_code == 200
    assert order == ["b", "b", "a"]


def test_shared_even_picks_less_full() -> None:
    """even: candidate 0 driven to high usage and candidate 1 to low usage ->
    candidate 1 is chosen (the spread-minimizing objective, with a large
    enough charge to break the 2-candidate tie)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes(usage_a=0.80, usage_b=0.10)
    cfg = shared_multi_config(
        (RoutingEntry(model="m", spec=RoutingSpec(policy="even", order=("qwen", "llama"))),)
    )
    cache_a = MetricsCache()
    cache_a.update(0.80)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(1000)
    counter_b = KvRemaining()
    counter_b.reanchor(90000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        # A large prompt (ctx = ceil(16000/4)+256 = 4256; charge =
        # ceil(4256*1.25) = 5320): projected A = 0.80 + 5320/100000 = 0.8532,
        # projected B = 0.10 + 5320/100000 = 0.1532. spread_A = 0.8532 - 0.10
        # = 0.7532; spread_B = 0.80 - 0.1532 = 0.6468 -> B wins.
        resp = client.post(
            "/v1/chat/completions", content=_shared_body(16000), headers=JSON_HEADERS
        )
        assert resp.status_code == 200
    assert order == ["b"]


def test_shared_fill_picks_less_full() -> None:
    """fill: candidate 0's counter cannot fit the charge, candidate 1's can
    -> candidate 1 is chosen (the first candidate the autoconfig layer would
    not reject)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    cfg = shared_multi_config(
        (RoutingEntry(model="m", spec=RoutingSpec(policy="fill", order=("qwen", "llama"))),)
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(100)  # effective ceil(281*1.25)=352 > 100 -> A would reject
    counter_b = KvRemaining()
    counter_b.reanchor(5000)  # 352 <= 5000 -> B admits
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
    assert order == ["b"]


def test_shared_unknown_model_goes_to_default() -> None:
    """An unowned model (and "unknown") routes to the default backend."""
    fakes, order = _shared_fakes()
    cfg = shared_multi_config()
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        for model in ("mystery-9b", "unknown"):
            resp = client.post(
                "/v1/chat/completions",
                content=_shared_body(100, model),
                headers=JSON_HEADERS,
            )
            assert resp.status_code == 200
    assert len(fakes["a"].requests) == 2
    assert len(fakes["b"].requests) == 0


def test_shared_single_owner_model_still_routes_directly() -> None:
    """A single-owner model routes to its owner (the length-1 special case)."""
    fakes, order = _shared_fakes()
    cfg = shared_multi_config()
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        q = client.post(
            "/v1/chat/completions",
            content=_shared_body(100, "q-only"),
            headers=JSON_HEADERS,
        )
        llama = client.post(
            "/v1/chat/completions",
            content=_shared_body(100, "l-only"),
            headers=JSON_HEADERS,
        )
    assert q.status_code == 200
    assert llama.status_code == 200
    assert order == ["a", "b"]


# --- v2 pre-stream failover ---------------------------------------------------


def test_shared_failover_transport_failure_rolls_back_charge() -> None:
    """Candidate 0 transport-fails pre-stream -> candidate 1 forwards, AND
    candidate 0's charge is rolled back (KvRemaining.add)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].fail_transport = True
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(5000)
    counter_b = KvRemaining()
    counter_b.reanchor(5000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
        metrics_body = client.get("/metrics").text
    # Candidate 1 forwarded; candidate 0 was attempted (and failed).
    assert order == ["a", "b"]
    # Candidate 0's charge (ceil(281*1.25)=352) was subtracted then re-added:
    # the gauge shows the pre-request anchor value again.
    assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 5000.0' in metrics_body
    # Candidate 1's charge was kept (it forwarded).
    assert 'gate_kv_cache_remaining_tokens{backend="llama"} 4648.0' in metrics_body
    # The failover skip was counted.
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="transport",to="llama"} 1.0'
        in metrics_body
    )
    # The forward was recorded with the final backend (llama).
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )


def test_shared_failover_upstream_5xx_rolls_back_charge() -> None:
    """Candidate 0 returns a pre-stream 5xx -> candidate 1 forwards + charge
    rolled back."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].completions_status = 500
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(5000)
    counter_b = KvRemaining()
    counter_b.reanchor(5000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
        metrics_body = client.get("/metrics").text
    assert order == ["a", "b"]
    assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 5000.0' in metrics_body
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="upstream_5xx",to="llama"} 1.0'
        in metrics_body
    )


def test_shared_failover_streaming_5xx_releases_and_rolls_back() -> None:
    """Candidate 0 returns a STREAMING pre-stream 5xx (a ``stream: true`` body
    answered with a 500 + SSE body) -> candidate 1 forwards. This exercises
    the ``StreamingResponse`` branch of the 5xx path: the proxy's ``_stream()``
    generator is primed then closed (releasing the upstream connection —
    ``aclose()`` on an unstarted generator would be a no-op), the charge is
    rolled back, and the failover is recorded with reason ``upstream_5xx``."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].completions_status = 500  # candidate 0 answers the stream with a 500
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(5000)
    counter_b = KvRemaining()
    counter_b.reanchor(5000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        # A streaming request: the proxy builds a StreamingResponse for the
        # 500, so the 5xx branch takes the StreamingResponse path.
        resp = client.post(
            "/v1/chat/completions",
            content=_shared_body(stream=True),
            headers=JSON_HEADERS,
        )
        assert resp.status_code == 200  # candidate 1 forwarded
        assert resp.headers["content-type"] == "text/event-stream"
        assert resp.content == SSE_BODY  # candidate 1's SSE, not candidate 0's
        metrics_body = client.get("/metrics").text
    # The failover happened (candidate 0 -> candidate 1).
    assert order == ["a", "b"]
    # Candidate 0's charge (ceil(281*1.25)=352) was rolled back.
    assert 'gate_kv_cache_remaining_tokens{backend="qwen"} 5000.0' in metrics_body
    # The failover was recorded with reason upstream_5xx.
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="upstream_5xx",to="llama"} 1.0'
        in metrics_body
    )
    # The forward was recorded with the final backend (llama).
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )


def test_shared_failover_last_candidate_5xx_propagated_and_recorded() -> None:
    """Both candidates return pre-stream 5xx: the LAST candidate's 5xx is
    propagated as-is (gotcha #6, not masked by a 502) and the request is
    recorded as forwarded with THAT backend (the gate's decision was
    "forward"; the 5xx is the upstream's answer)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].completions_status = 500
    fakes["b"].completions_status = 503
    fakes["b"].completions_json = {"error": "unavailable"}
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    with _build_shared(cfg, fakes, cache_a=cache_a, cache_b=cache_b) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        metrics_body = client.get("/metrics").text
    # The last candidate's 5xx is propagated unmasked (not 502, not 200).
    assert resp.status_code == 503
    assert resp.json() == {"error": "unavailable"}
    assert order == ["a", "b"]
    # Recorded as forwarded with the last candidate's backend (llama).
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )
    # The skip from A to B was counted.
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="upstream_5xx",to="llama"} 1.0'
        in metrics_body
    )


def test_shared_failover_last_candidate_streaming_5xx_propagates_verbatim() -> None:
    """Both candidates return a STREAMING pre-stream 5xx: the LAST candidate's
    5xx propagates its SSE body VERBATIM (not empty). This pins the regression
    where an unconditional prime-then-close emptied the last-candidate body —
    the generator must be left intact on the ``next_name is None`` path so
    starlette streams the 5xx body as-is (gotcha #6)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].completions_status = 500  # candidate 0: streaming 500 (discarded)
    fakes["b"].completions_status = 503  # candidate 1 (last): streaming 503 (propagated)
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    with _build_shared(cfg, fakes, cache_a=cache_a, cache_b=cache_b) as client:
        resp = client.post(
            "/v1/chat/completions", content=_shared_body(stream=True), headers=JSON_HEADERS
        )
        metrics_body = client.get("/metrics").text
    # The last candidate's streaming 5xx is propagated unmasked (not 502, not 200).
    assert resp.status_code == 503
    assert resp.headers["content-type"] == "text/event-stream"
    # The upstream SSE body propagates verbatim — NOT empty (a primed-then-
    # closed generator would re-iterate empty and mask the body to b'').
    assert resp.content == SSE_BODY
    assert resp.content != b""
    assert order == ["a", "b"]
    # Recorded as forwarded with the last candidate's backend (llama).
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )
    # The skip from A to B was counted.
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="upstream_5xx",to="llama"} 1.0'
        in metrics_body
    )


def test_shared_failover_skip_on_reject() -> None:
    """Candidate 0 429s (driven high) -> candidate 1 forwards (skip-on-reject)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes(usage_a=0.90, usage_b=0.10)
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.90)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(100)
    counter_b = KvRemaining()
    counter_b.reanchor(5000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(5000), headers=JSON_HEADERS)
        assert resp.status_code == 200
        metrics_body = client.get("/metrics").text
    # A rejected (5000-char prompt: ctx ~1506 > 1024 at 90% usage), B forwarded.
    assert order == ["b"]
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="reject",to="llama"} 1.0'
        in metrics_body
    )
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )


def test_shared_both_reject_gives_max_retry_after() -> None:
    """Both candidates 429 -> a single 429 with Retry-After = max of the two
    candidates' retry_after, naming the max-timeout rejector."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes(usage_a=0.90, usage_b=0.90)
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.90)
    cache_b = MetricsCache()
    cache_b.update(0.90)
    # A: counter anchored at 100 -> autoconfig retry = ceil(5*(1+352/100)) = 18
    #   (effective for a 5000-char prompt: ceil(1506*1.25)=1883 -> remaining
    #   100 -> scaled: 5*(1+1883/100)=94.65 -> capped at 60).
    # B: counter anchored at 50000 -> autoconfig retry = ceil(5*(1+1883/50000))
    #   = ceil(5.19) = 6. Both tiers reject with timeout 30 (80-tier).
    # Combined per candidate: A = max(30, 60) = 60; B = max(30, 6) = 30.
    # Final 429: max(60, 30) = 60, rejector = A (qwen).
    counter_a = KvRemaining()
    counter_a.reanchor(100)
    counter_b = KvRemaining()
    counter_b.reanchor(50000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(5000), headers=JSON_HEADERS)
        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "60"
        assert "qwen" in resp.json()["error"]["message"]  # max-timeout rejector
        metrics_body = client.get("/metrics").text
    # The rejection is recorded with the max-timeout rejector's backend.
    assert (
        'gate_requests_total{backend="qwen",endpoint="chat_completions",model="m",'
        'result="rejected"} 1.0' in metrics_body
    )
    # The skip-on-reject from A to B was counted.
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="reject",to="llama"} 1.0'
        in metrics_body
    )
    assert len(fakes["a"].requests) == 0
    assert len(fakes["b"].requests) == 0


def test_shared_both_transport_fail_gives_502() -> None:
    """Both candidates transport-fail -> 502, and gate_requests_total carries
    backend="none" (and a failover entry per skip)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].fail_transport = True
    fakes["b"].fail_transport = True
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 502
        err = resp.json()["error"]
        assert err["type"] == "upstream_failure"
        assert err["code"] == "all_backends_failed"
        assert "qwen" in err["message"]
        assert "llama" in err["message"]
        metrics_body = client.get("/metrics").text
    # The 502-exhaustion rejection uses the "none" sentinel backend label.
    assert (
        'gate_requests_total{backend="none",endpoint="chat_completions",model="m",'
        'result="rejected"} 1.0' in metrics_body
    )
    # One failover entry per skip (A -> B; B is last so no skip after it).
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="transport",to="llama"} 1.0'
        in metrics_body
    )


def test_shared_anchor_seq_guard_reanchor_drops_charge() -> None:
    """A re-anchor on candidate 0 between charge and transport-failure => the
    charge is NOT re-added (drop-and-fresh-anchor path); no re-anchor =>
    re-added."""
    from gate.routing import RoutingSpec

    # Case 1: a re-anchor on candidate 0 lands BETWEEN the charge and the
    # transport failure (the poller's on_reanchor path) -> the charge is
    # dropped (the fresh anchor is authoritative; NOT re-added).
    fakes, order = _shared_fakes()
    fakes["a"].fail_transport = True
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    counter_a = KvRemaining()
    counter_a.reanchor(5000)
    counter_b = KvRemaining()
    counter_b.reanchor(5000)
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
        counter_b=counter_b,
    ) as client:
        bs_a = client.app.state.backends[0]
        # The fake's handler runs inside the proxy await — i.e. after the
        # charge and before the transport failure propagates. Trigger the
        # re-anchor there (reset the counter AND bump the anchor sequence,
        # exactly as the poller's on_reanchor callback does).
        inner = fakes["a"].handler

        def handler(request: httpx.Request) -> httpx.Response:
            bs_a.counter.reanchor(9000)
            bs_a.anchor_seq += 1
            return inner(request)

        fakes["a"].handler = handler
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
    # The charge (352) was subtracted (5000 -> 4648), then the re-anchor
    # reset the counter to the fresh value (9000), and the guard dropped the
    # rollback (anchor_seq changed). If the guard were broken (re-added), the
    # value would be 9000 + 352 = 9352.
    assert counter_a.value() == 9000

    # Case 2: no re-anchor -> the charge IS re-added.
    fakes2, order2 = _shared_fakes()
    fakes2["a"].fail_transport = True
    cache_a2 = MetricsCache()
    cache_a2.update(0.10)
    cache_b2 = MetricsCache()
    cache_b2.update(0.10)
    counter_a2 = KvRemaining()
    counter_a2.reanchor(5000)
    counter_b2 = KvRemaining()
    counter_b2.reanchor(5000)
    with _build_shared(
        cfg,
        fakes2,
        cache_a=cache_a2,
        cache_b=cache_b2,
        counter_a=counter_a2,
        counter_b=counter_b2,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
    # The charge (352) was subtracted then re-added: back to 5000.
    assert counter_a2.value() == 5000


def test_shared_one_stale_one_fresh_walk_reachability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fresh candidate is gated (can 429); the stale candidate fails open
    and is still reachable through the walk (stale is not skipped)."""
    from gate.routing import RoutingSpec

    # Case 1: fresh A 429s (high usage, big prompt) -> the walk reaches stale
    # B, which fails open -> 200 on B.
    fakes, order = _shared_fakes(usage_a=0.90)
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.90)
    counter_a = KvRemaining()
    counter_a.reanchor(100)
    cache_b = MetricsCache()  # never fetched -> stale -> fails open
    with _build_shared(
        cfg,
        fakes,
        cache_a=cache_a,
        cache_b=cache_b,
        counter_a=counter_a,
    ) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(5000), headers=JSON_HEADERS)
        assert resp.status_code == 200  # stale B failed open
    assert order == ["b"]

    # Case 2: stale A (fails open, reached first) -> 200 on A; fresh B is
    # gated (429) when reached directly via its single-owner model.
    fakes2, order2 = _shared_fakes(usage_a=0.90, usage_b=0.90)
    cache_a2 = MetricsCache()
    cache_a2.update(0.90)
    monkeypatch.setattr(cache_a2, "is_stale", lambda *a, **k: True)
    cache_b2 = MetricsCache()
    cache_b2.update(0.90)
    with _build_shared(cfg, fakes2, cache_a=cache_a2, cache_b=cache_b2) as client:
        # Stale A (monkeypatched) fails open for the shared model.
        resp_a = client.post(
            "/v1/chat/completions", content=_shared_body(5000), headers=JSON_HEADERS
        )
        assert resp_a.status_code == 200  # stale A failed open
        # Fresh B is gated for its single-owner model.
        resp_b = client.post(
            "/v1/chat/completions",
            content=_shared_body(5000, "l-only"),
            headers=JSON_HEADERS,
        )
        assert resp_b.status_code == 429  # fresh B gated (80-tier)
        assert resp_b.headers["Retry-After"] == "30"
    assert order2 == ["a"]


def test_shared_v1_models_owned_by_list_for_shared_model() -> None:
    """GET /v1/models: the shared model's owned_by is a LIST (2+ owners);
    single-owner models keep the scalar form."""
    fakes, order = _shared_fakes()
    cfg = shared_multi_config()
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    entries = {e["id"]: e for e in data["data"]}
    assert entries["m"]["owned_by"] == ["qwen", "llama"]  # list for the shared model
    assert entries["q-only"]["owned_by"] == "qwen"  # scalar for single-owner
    assert entries["l-only"]["owned_by"] == "llama"


def test_shared_healthz_carries_routing_policy() -> None:
    """/healthz: per-model routing names the policy in effect (the routing:
    entry's policy, or round_robin by default)."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    cfg = shared_multi_config(
        (RoutingEntry(model="m", spec=RoutingSpec(policy="fill", order=("qwen", "llama"))),)
    )
    with _build_shared(cfg, fakes, cache_a=MetricsCache(), cache_b=MetricsCache()) as client:
        data = client.get("/healthz").json()
    assert data["status"] == "ok"
    by_id = {m["id"]: m for m in data["models"]}
    assert by_id["m"]["owned_by"] == ["qwen", "llama"]
    assert by_id["m"]["routing"] == "fill"  # the routing: entry's policy
    assert by_id["q-only"]["routing"] == "round_robin"  # default (no entry)
    assert by_id["l-only"]["routing"] == "round_robin"


def test_shared_metrics_backend_label_and_routing_json() -> None:
    """/metrics: the backend label on the request counters, the failover
    counter, and the routing_json config info structure."""
    from gate.routing import RoutingSpec

    fakes, order = _shared_fakes()
    fakes["a"].fail_transport = True
    cfg = shared_multi_config(
        (
            RoutingEntry(
                model="m", spec=RoutingSpec(policy="primary_fallback", order=("qwen", "llama"))
            ),
        )
    )
    cache_a = MetricsCache()
    cache_a.update(0.10)
    cache_b = MetricsCache()
    cache_b.update(0.10)
    with _build_shared(cfg, fakes, cache_a=cache_a, cache_b=cache_b) as client:
        resp = client.post("/v1/chat/completions", content=_shared_body(), headers=JSON_HEADERS)
        assert resp.status_code == 200
        metrics_body = client.get("/metrics").text
    # The backend label on the request counters (forwarded to llama).
    assert (
        'gate_requests_total{backend="llama",endpoint="chat_completions",model="m",'
        'result="forwarded"} 1.0' in metrics_body
    )
    assert (
        'gate_request_ctx_tokens_count{backend="llama",model="m",result="forwarded"} 1.0'
        in metrics_body
    )
    # The failover counter (A transport-failed -> B).
    assert (
        'gate_routing_failovers_total{from="qwen",model="m",reason="transport",to="llama"} 1.0'
        in metrics_body
    )
    # The per-backend gauges.
    assert 'gate_kv_cache_remaining_tokens{backend="qwen"} NaN' in metrics_body
    assert 'gate_kv_cache_remaining_tokens{backend="llama"} NaN' in metrics_body
    # The routing_json config info structure.
    config_labels: dict[str, str] = {}
    for family in text_string_to_metric_families(metrics_body):
        if family.name == "gate_config_info":
            for sample in family.samples:
                config_labels = sample.labels
    assert json.loads(config_labels["routing_json"]) == [
        ["m", "primary_fallback", ["qwen", "llama"]]
    ]
    # backends_json is still present (unchanged).
    assert '"name":"qwen"' in config_labels["backends_json"]
    assert '"name":"llama"' in config_labels["backends_json"]
