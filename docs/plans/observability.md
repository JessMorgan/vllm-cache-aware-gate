# Observability & Stats Collection — Design Plan

Status: **implemented** (2026-09-15)
Branch: `feat/observability`
Scope decisions (user-confirmed):

1. **"Size" = estimated context tokens** — the `ctx_tokens` estimate that drove
   each decision (prompt chars / `chars_per_token` + `max_tokens` headroom).
2. **`/metrics` in Prometheus text exposition format** — a single
   `GET /metrics` endpoint scrapable by Prometheus for Grafana. **No new
   dependency**: `prometheus_client` (already a runtime dep, currently used
   only to *parse* vLLM's text) also provides the client-side
   `CollectorRegistry` / `generate_latest` API (verified in venv, v0.26.0).
3. **Per-model = collect now, single-server decision.** Stats carry per-model
   labels everywhere; the decision logic stays v1 single-server (MAX across
   models). Multi-server routing-by-model is a later feature, but its data
   flows from day one.

---

## 1. Background, Problem, Goal

The gate currently exposes only `GET /healthz` (liveness + a single `kv_usage`
fraction). There is **no way to observe the gate's own behavior**: how many
requests it processed, how many it forwarded vs. rejected (429), how large
those requests were, the current per-model KV-cache fill, or what
configuration it actually loaded. Operators have no Prometheus/Grafana target.

**Goal:** add an in-memory stats layer and a `GET /metrics` endpoint in
**Prometheus exposition format** (same shape vLLM emits) so Prometheus can
scrape it and Grafana can chart:

- total requests processed, and the forwarded/rejected split;
- median (p50) size — in **estimated context tokens** — of forwarded and of
  rejected requests;
- current cache fill % per served model;
- the loaded configuration values;
- freshness of the upstream metrics feed.

**Constraints / decisions (load-bearing):**

- **In-memory only, no persistence.** Stats reset on process restart. This is
  correct for Prometheus: counters are monotonic and `rate()` / `increase()` /
  `histogram_quantile()` handle restarts; we do **not** add disk or external
  state.
- **No new runtime dependency.** `prometheus_client` (already a runtime dep)
  also provides the client-side `CollectorRegistry`/`generate_latest` API.
- **Per-model readiness without changing the decision.** vLLM labels every
  metric — including `vllm:kv_cache_usage_perc` — with `model_name` (confirmed
  against vLLM docs). We parse and expose per-model fill, but the **decision
  logic stays v1 single-server (MAX across models)**. Two distinct "model"
  namespaces must not be conflated (see §2.4).
- **The decision path is untouched except for a side-effecting record call.**
  Fail-open, percentage-vs-fraction, highest-tier-only, and inclusive-boundary
  invariants (AGENTS.md gotchas #1–#5) must be preserved exactly. `/metrics`
  itself always returns 200 with whatever data exists (even when the cache is
  empty/stale) — it never blocks or 429s.

---

## 2. Design Summary

### 2.1 New module `src/gate/stats.py` (the "stats edge")

A `GateStats` class owns a **per-app `CollectorRegistry`** (never the global
default — tests create many apps, and per-app isolation prevents
cross-contamination). It is created in `create_app` (injectable like
`cache`/`upstream` for tests).

- **Constructor `GateStats(cfg, *, registry=None)`**: creates the registry (if
  not given) and registers all `gate_*` metric objects; sets
  `gate_config_info` once from `cfg` (static for the process lifetime).
- **`record_forwarded(endpoint, model, ctx_tokens)` / `record_rejected(endpoint,
  model, ctx_tokens)`**: increment the request counter and observe the histogram
  (only when `ctx_tokens` is not `None`).
- **`set_kv_usage(by_model: dict[str, float])`**: set the per-model fill gauge
  (fraction → percent); **removes** label series for models no longer present
  (avoids stale misleading series).
- **`set_freshness(fresh: bool, age_s: float | None)`**: set the
  freshness/age gauges.
- **`render() -> bytes`**: `generate_latest(registry)`.

`app.py` is the wiring: the `/metrics` handler calls
`set_kv_usage(cache.by_model())`, `set_freshness(...)`, then `render()`.

### 2.2 Metrics exposed (all `gate_`-prefixed to avoid `vllm_` collisions)

| Metric | Type | Labels | Answers |
|---|---|---|---|
| `gate_requests_total` | Counter | `endpoint` (`chat_completions`\|`completions`), `model` (request body `model`, else `unknown`), `result` (`forwarded`\|`rejected`) | total processed = `sum(...)`; forwarded / rejected = filter by `result`. **Scoped to the two generation endpoints only** (not 404s/healthz/metrics). |
| `gate_request_ctx_tokens` | Histogram | `model`, `result` | size distribution; **median via `histogram_quantile(0.5, ...)`** (see §2.5). |
| `gate_kv_cache_usage_pct` | Gauge | `model_name` (vLLM's label; `default` when the series is unlabeled) | current cache fill % per served model. |
| `gate_config_info` | Info | one label per config field + `thresholds_json` | the loaded config values. |
| `gate_metrics_fresh` | Gauge | (none) | `1` if the vLLM feed is fresh, `0` if stale/never-fetched. |
| `gate_metrics_age_s` | Gauge | (none) | seconds since last successful vLLM `/metrics` fetch. |

### 2.3 Request-side recording (in `app._handle_generation`)

After `decision(...)`:

- allow → `stats.record_forwarded(endpoint, model, ctx_tokens)`
- reject → `stats.record_rejected(endpoint, model, ctx_tokens)`

`model` is extracted by a new **pure** helper `extract_request_model(body:
bytes) -> str` added to `tokens.py` (returns the body's `model` string, or
`"unknown"` if absent/non-string/unparseable — same defensive-parse guards as
`estimate_context_tokens`). This is the **future routing key**. `ctx_tokens`
may be `None` (fail-open unparseable) → counter still increments, histogram is
skipped.

### 2.4 Per-model namespaces (do not conflate)

- **Request model** = the OpenAI request body's `model` field (the client's
  model name) → labels request-side stats (`model`). This is what multi-server
  routing will key on.
- **Served model** = vLLM's `model_name` metric label (the model path the
  instance serves) → labels the fill gauge (`model_name`).

In v1 (one server) these are effectively 1:1 but are **different strings**,
so they use **different label names** to stay correct and future-proof. When
multi-server routing lands, request `model` maps to a server (which reports its
`model_name`) via routing config — not via a shared label.

### 2.5 "Median size" semantics (important)

The gate does **not** expose an exact-median series. It exposes a
**histogram** (`gate_request_ctx_tokens`), and the median is derived by
Prometheus/Grafana:

```promql
# median size of forwarded requests (tokens)
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="forwarded"}))
# median size of rejected requests
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="rejected"}))
```

This is the standard Prometheus idiom, is aggregatable across models/instances,
and matches how vLLM exposes its own latencies (histograms). The value is an
approximation at bucket resolution. **Buckets** (fixed at metric creation,
cover typical LLM contexts):
`(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)`.

Note: histogram `_count` for `forwarded` counts only forwarded requests *with
an estimate* (excludes fail-open unparseable), so it can be less than
`gate_requests_total{result="forwarded"}` — expected and documented. (A
`Summary` would expose an exact per-series p50 but is not aggregatable and
non-standard; rejected in favor of histogram.)

### 2.6 `/metrics` endpoint

- `GET /metrics` → `200`, `Content-Type: text/plain; version=0.0.4;
  charset=utf-8`, body = `generate_latest(registry)`.
- Added to the allowed-route set alongside `/healthz`; **everything else still
  404** (invariant preserved).
- **Unauthenticated** (v1, consistent with the no-auth invariant) — README
  notes it should be network-protected like vLLM's own `/metrics`.
- Distinct from the upstream: the poller fetches vLLM's `/metrics` via the
  httpx client to `base_url`; the gate's own `/metrics` is a FastAPI route. No
  collision (called out in README to avoid operator confusion).

### 2.7 `metrics.py` extension (per-model)

- New pure `parse_kv_cache_usage_by_model(text) -> dict[str, float] | None`:
  maps each finite `vllm:kv_cache_usage_perc` sample to its `model_name` label
  (or `default` if unlabeled); `None` when unparseable/absent/no-finite-samples.
  The existing `parse_kv_cache_usage` (MAX) is **kept** (decision path +
  existing tests unchanged).
- `MetricsCache` gains by-model storage (`_by_model`,
  `update_by_model(mapping)`, `by_model() -> dict`) while **preserving** the
  existing `value()/age()/is_stale()/update()` contract (`_value` becomes the
  MAX of the mapping, preserving gotcha #5).
- `fetch_usage_by_model(client, url)` thin edge (the poller switches to it);
  the poller's fail-open behavior (keep last value on error/non-200) is
  unchanged — `update_by_model(None)` is a no-op.

---

## 3. Files to Change/Create

| Path | Action | Description |
|---|---|---|
| `src/gate/stats.py` | **Create** | `GateStats`: per-app `CollectorRegistry`, `gate_*` metric objects, `record_forwarded/rejected`, `set_kv_usage` (with stale-label removal), `set_freshness`, `render()`, `gate_config_info` at construction. |
| `src/gate/tokens.py` | Modify | Add pure `extract_request_model(body: bytes) -> str` (defensive parse; `unknown` fallback). |
| `src/gate/metrics.py` | Modify | Add `parse_kv_cache_usage_by_model`; extend `MetricsCache` with by-model storage (`update_by_model`, `by_model`); add `fetch_usage_by_model`. Keep `parse_kv_cache_usage`/`fetch_usage` and the existing cache contract. |
| `src/gate/poller.py` | Modify | Use `fetch_usage_by_model` + `cache.update_by_model(...)`; fail-open semantics unchanged. |
| `src/gate/app.py` | Modify | Create/inject `GateStats`; `record_*` in `_handle_generation` (endpoint + model + ctx_tokens); add `GET /metrics` handler (wire `cache`→`stats`→`render`); pass `endpoint` label. No decision-path behavior change. |
| `tests/test_stats.py` | **Create** | Unit tests: counter increments, histogram observation (incl. `None` skip), per-model series, kv-gauge set + stale-label removal, `gate_config_info` present, `render()` content. |
| `tests/test_tokens.py` | Modify | `extract_request_model`: present string, missing, non-string, unparseable body, non-object, pathological nesting → `unknown`. |
| `tests/test_metrics.py` | Modify | `parse_kv_cache_usage_by_model` (unlabeled→`default`, multiple models, per-model max, absent/NaN/unparseable→None); `MetricsCache.by_model`/`update_by_model` (incl. `None` no-op keeps last). |
| `tests/test_api.py` | Modify | `/metrics`: 200 + correct content-type; contains `gate_requests_total`, `gate_request_ctx_tokens`, `gate_kv_cache_usage_pct`, `gate_config_info`; per-model series after different-model requests; kv gauge reflects cache by-model; **not 404**; unknown paths still 404; `/healthz` regression (unchanged contract). |
| `README.md` | Modify | New "Observability" section (endpoint, metric table, PromQL for median/reject-rate/fill, scrape config, in-memory-reset note, unauthenticated caveat); update "Proxied endpoints" table (+ `/metrics`); clarify gate vs vLLM `/metrics`. |
| `AGENTS.md` | Modify | Project map: add `src/gate/stats.py`; add invariant/gotcha (in-memory stats, `/metrics` unauthenticated, two model-name namespaces, decision path unaffected). |
| `docker-compose.example.yaml` | Modify (optional) | Comment noting the gate exposes `/metrics` for scraping. |

No change to `pyproject.toml` (no new dep), `Dockerfile` (HEALTHCHECK stays
`/healthz`), `.dockerignore`, or `config.example.yaml` (config shape
unchanged).

---

## 4. Testing Plan

- **`tests/test_stats.py` (new, unit):** Construct `GateStats(cfg,
  registry=fresh)`.
  - `record_forwarded("m", 500)` / `record_rejected("m", 9000)` → rendered
    text has `gate_requests_total{...,result="forwarded"} 1` and
    `...{result="rejected"} 1`; histogram buckets `_count` reflect
    observations.
  - `record_forwarded("m", None)` → counter increments, histogram `_count`
    unchanged.
  - Two different models → distinct `model` label series (no
    cross-contamination).
  - `set_kv_usage({"A": 0.5})` → `gate_kv_cache_usage_pct{model_name="A"}
    50.0`; then `set_kv_usage({"B": 0.2})` → `A` series removed, `B` present.
  - `gate_config_info` present with the cfg's field labels + `thresholds_json`.
  - `render()` returns bytes starting with `# HELP gate_...`.
- **`tests/test_tokens.py`:** `extract_request_model` across present/missing/
  non-string/unparseable/non-object/pathological-nesting → expected (`"m"` /
  `"unknown"`).
- **`tests/test_metrics.py`:** `parse_kv_cache_usage_by_model` (unlabeled→
  `default`, multi-model dict, per-model max, absent/NaN-only/garbage→None);
  `MetricsCache` by-model (update, `by_model` snapshot,
  `update_by_model(None)` no-op keeps last, `_value` = max).
- **`tests/test_api.py`:** Full ASGI (fake vLLM + controlled cache,
  `start_poller=False`).
  - `GET /metrics` → 200, `content-type` == `text/plain; version=0.0.4;
    charset=utf-8`, body contains the four required `gate_` families.
  - After a forwarded chat request (model `m`) and a rejected completions
    request (model `n`) → `gate_requests_total` shows both `result`s and both
    `model`s.
  - kv gauge: `cache.update_by_model({"X": 0.8})` → `/metrics` shows
    `gate_kv_cache_usage_pct{model_name="X"} 80.0`.
  - `/metrics` is **not** 404; `GET /v1/embeddings` still 404.
  - `/healthz` regression: unchanged 200 + `status`/`metrics_age_s`/
    `kv_usage`.
  - Regression: all existing allow/reject/fail-open/streaming/5xx tests still
    pass (decision path untouched).
- **Coverage:** keep `--cov-fail-under=80` (currently 95%); new `stats.py` /
  `metrics.py` / `tokens.py` code is directly unit-tested.

---

## 5. Documentation Updates

- **`README.md`** — new **Observability** section: the `/metrics` endpoint,
  the metric table (names/types/labels/meaning), ready-to-paste **PromQL**
  (total/forwarded/rejected counts, reject rate, median forwarded/rejected
  size, per-model fill, freshness), a Prometheus **scrape config** example,
  the **in-memory-reset** note (counters reset on restart; use `rate` /
  `increase` / `histogram_quantile`), and the **unauthenticated** caveat.
  Update the **Proxied endpoints** table to include `GET /metrics`. Add a line
  disambiguating the gate's `/metrics` from vLLM's.
- **`AGENTS.md`** — project map entry for `src/gate/stats.py`; a new **Known
  gotcha**: (a) stats are in-memory and reset on restart (never persist);
  (b) `/metrics` is unauthenticated — network-protect it; (c) the two model
  namespaces (`model` = request/routing key, `model_name` = vLLM served) must
  not be conflated; (d) recording is a pure side-effect — it must not alter
  the decision or the proxy.
- **`config.example.yaml`** — no change (config shape unchanged); optionally a
  comment pointing at `/metrics`.
- **`docker-compose.example.yaml`** — optional comment: the gate exposes
  `/metrics` for a Prometheus scrape.

Example scrape config (for the README):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm-gate
    static_configs:
      - targets: ["gate:8000"]
    # The gate also exposes vLLM-facing state; vLLM's own /metrics is a
    # separate scrape target.
```

Example PromQL (for the README / dashboards):

```promql
# total requests processed (per minute)
sum(rate(gate_requests_total[5m]))
# forwarded vs rejected
sum(rate(gate_requests_total{result="forwarded"}[5m]))
sum(rate(gate_requests_total{result="rejected"}[5m]))
# reject rate
sum(rate(gate_requests_total{result="rejected"}[5m])) / sum(rate(gate_requests_total[5m]))
# median size (estimated tokens) of forwarded / rejected requests
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="forwarded"}))
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="rejected"}))
# current per-model KV-cache fill %
gate_kv_cache_usage_pct
# is the metrics feed healthy?
gate_metrics_fresh
```

---

## 6. Dev-Ops / Project Structure Updates

- **New module** `src/gate/stats.py` in the existing pure-core / thin-edge
  layout (stateful "stats edge", no I/O).
- **No new dependency** — `prometheus_client>=0.20` is already a runtime dep
  (confirm `pip-audit` / CI still pass; they will).
- **Per-app `CollectorRegistry`** (not global) — required for test isolation
  and correct multi-app behavior.
- **`Dockerfile`** — unchanged; `HEALTHCHECK` remains `/healthz` (liveness).
  `/metrics` is for scraping, not liveness.
- **Prometheus/Grafana** — not checked into the repo; provide the scrape
  config + dashboard PromQL snippets in the README only (avoids doc sprawl).
  No new `.github` workflows needed.
- **No auth/TLS** added (consistent with v1) — the README carries the
  network-protection guidance.

---

## 7. Verification Checklist (for the implementer)

- [ ] `ruff format --check src/ tests/` and `ruff check .` pass.
- [ ] `mypy src/` clean (strict); `python -m compileall -q src/` clean.
- [ ] `pytest --cov=gate --cov-fail-under=80` — all tests pass, coverage ≥
      80% (target ~95%).
- [ ] `bash scripts/local-ci.sh` → **8/8 PASS** (incl. `pip-audit`,
      `pre-commit`, `docker build`).
- [ ] `GET /metrics` returns **200** with `Content-Type: text/plain;
      version=0.0.4; charset=utf-8`.
- [ ] `/metrics` body contains `gate_requests_total`,
      `gate_request_ctx_tokens`, `gate_kv_cache_usage_pct`,
      `gate_config_info` (and the two freshness gauges).
- [ ] `gate_requests_total` increments on both forwarded and rejected;
      `result` and `model` labels correct.
- [ ] `gate_request_ctx_tokens` histogram `_count` reflects observed
      estimates; `None` (fail-open) is skipped.
- [ ] `gate_kv_cache_usage_pct{model_name=...}` reflects the cache's
      per-model values; stale model series are removed.
- [ ] `gate_config_info` labels match the loaded `GateConfig` (incl.
      `thresholds_json`).
- [ ] Per-model: two requests with different `model` → two distinct label
      series (no cross-contamination).
- [ ] `GET /metrics` is **not** 404; unknown paths (e.g. `/v1/embeddings`)
      **still 404**.
- [ ] `/healthz` contract unchanged (200, `status`, `metrics_age_s`,
      `kv_usage`).
- [ ] Decision invariants intact: all existing allow/reject/fail-open/
      streaming/5xx tests pass unmodified.
- [ ] No new runtime dependency added; `pyproject.toml` unchanged.
- [ ] Stats are in-memory (no disk/external writes); a fresh process starts
      with zeroed counters.
- [ ] README + AGENTS.md updated (Observability section, endpoints table,
      gotchas).

---

## 8. Implementation Order (DAG)

```
        ┌──────────────────────────────────┐
        │ A1  tokens.extract_request_model  │  (pure, independent)
        └───────────────┬──────────────────┘
                        │ (parallel with A2)
        ┌───────────────▼──────────────────┐
        │ A2  metrics: parse_by_model +     │  (independent of A1)
        │     MetricsCache.by_model +       │
        │     fetch_usage_by_model + poller │
        └───────────────┬──────────────────┘
                        │ (A3 needs A2: GateStats calls cache.by_model())
        ┌───────────────▼──────────────────┐
        │ A3  stats.GateStats + test_stats  │
        └───────────────┬──────────────────┘
                        │ (B needs A1 + A2 + A3)
        ┌───────────────▼──────────────────┐
        │ B   app.py wiring: record_* in    │
        │     _handle_generation,           │
        │     GET /metrics                  │
        └───────────────┬──────────────────┘
                        │ (C needs B)
        ┌───────────────▼──────────────────┐
        │ C   test_api.py /metrics tests +  │
        │     README/AGENTS docs            │
        └──────────────────────────────────┘
```

- **Parallel:** `A1` ∥ `A2` (independent). `A3` runs after `A2`.
- **Sequential:** `B` after `{A1, A2, A3}`; `C` after `B`.
- **Branching:** one feature branch `feat/observability` in a dedicated
  worktree (main has commits — worktree rule is mandatory). Each segment
  (`A1`, `A2`, `A3`, `B`, `C`) is ~one commit, so **no sub-branches** are
  needed; commit per segment with the standard gates (local CI,
  distinct-model review, attribution block) on each commit.
- **Rebase** the feature branch onto `main` before the final `--no-ff` merge
  so the merge is conflict-free and the code is tested in its post-merge
  state.

---

## Known risks / notes

1. **Median is derived, not exact** — the gate exposes a histogram; Grafana
   computes p50 via `histogram_quantile`. Standard, aggregatable Prometheus
   idiom; matches vLLM's own metrics. An exact per-model p50 series would need
   a `Summary` (non-aggregatable, non-standard) — not chosen.
2. **`/metrics` is unauthenticated** in v1 (consistent with the no-auth
   invariant). It exposes config values and traffic stats; the README directs
   operators to network-protect it (as they already do vLLM's own
   `/metrics`). An opt-in auth flag is a separate follow-up.
3. **Label cardinality** — `model` (request body) is open-ended. In practice
   clients use a small model set; if untrusted clients send arbitrary model
   strings this is a cardinality concern for Prometheus. Mitigation is
   operational (network isolation), not code. Noted for the multi-server
   follow-up to revisit.
