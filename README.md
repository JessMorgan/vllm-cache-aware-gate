# vllm-cache-aware-gate

A small, single-purpose **reverse-proxy "gate"** container that sits in front of
a [vLLM](https://docs.vllm.ai) OpenAI-compatible server and protects its KV
cache from over-subscription. It speaks the OpenAI API and, per incoming
request, decides one of two routes:

- **Forward** — the KV cache has room for this prompt's estimated size, so the
  request is proxied to vLLM verbatim (including `stream: true` SSE).
- **Reject** — not enough room, so the client gets an
  **`HTTP 429` with a `Retry-After` header** telling it how long to wait.

It learns the cache's headroom by polling vLLM's Prometheus `/metrics`
endpoint and reading `vllm:kv_cache_usage_perc` (how full the cache is) and the
KV-cache capacity in tokens — from the `vllm:kv_cache_size_tokens` gauge, or
the `kv_cache_size_tokens` label on `vllm:cache_config_info` (the form current
vLLM emits). It is designed
for **safe, configuration-first, fail-open** operation: a monitoring outage
never blocks inference traffic, and the gate is a transparent pass-through for
the endpoints it proxies.

---

## How it works

```
                ┌────────────────────────────── gate container ─────────────────────────────┐
 client ──8000─►│  FastAPI app                                                             │
 (OpenAI API)   │   • POST /v1/chat/completions ─┐                                           │
                │   • POST /v1/completions   ───┤  RequestRouter ─► (allow)  ─► httpx ───────┼──► vLLM:8000
                │   • GET  /healthz           ───┘            │            (reject) ─► 429 + Retry-After │   (proxied
                │                                             │                                           │    transparently)
                │   Background poller ── every Ns ─► GET vLLM:8000/metrics ─► MetricsCache              │
                │        (fails → fail-open, keep retrying)                                          │
                └──────────────────────────────────────────────────────────────────────────────────────┘
```

### Decision logic (per request)

Every request to the two generation endpoints is evaluated by **two admission
layers, AND-combined**: the request forwards only if **both** allow. The first
is the optional **tiered** policy below; the second is the always-on
**[autoconfig](#autoconfig-always-on)** layer (see the next section). The
combined rule, the 429 `Retry-After`, and the rejector-aware message are
described in [The 429 contract](#the-429-contract).

#### Tiered layer (optional)

1. **Estimate context tokens** `C` from the request body:
   - `prompt_tokens = ceil(prompt_chars / chars_per_token)` (default 4), where
     `prompt_chars` is the total character count of the prompt text
     (`messages[*].content[*]` for chat; `prompt` string/array for completions).
   - `C = prompt_tokens + headroom`, where `headroom = max_tokens` if present in
     the request, else `default_max_tokens` (default 256).
   - This is a **heuristic** — it deliberately over-estimates so the gate is
     conservative. There is no tokenizer dependency in v1.
2. **Read current usage** from the in-memory cache:
   `U_pct = usage_frac * 100` (the vLLM metric is a fraction 0–1; the config
   thresholds are percentages 0–100).
3. **Select the governing tier** (highest-tier-only):
   - `active = the threshold entry with the largest kv_pct such that
     U_pct >= kv_pct`.
   - **No such tier → ALLOW** (forward unconditionally).
4. If an `active` tier exists:
   - `C <= active.max_context` → **ALLOW** (forward).
   - `C > active.max_context` → **REJECT**: `429` with
     `Retry-After: active.timeout_s`.

> **Highest-tier-only.** When usage crosses several thresholds, only the most
> severe (largest `kv_pct`) crossed tier governs. Lower tiers are ignored — they
> are not summed or averaged.

### Autoconfig (always on)

The tiered policy above is **optional**. The gate *always* also runs an
**autoconfig** admission layer that admits each request by comparing its
estimated size against an in-memory estimate of the KV space still available up
to a target usage %. It needs no configuration — only the vLLM host/port — and
uses sane defaults (see the [knob table](#autoconfig-knobs)). The two layers
are **AND-combined**: a request forwards only if the tiered layer *and* the
autoconfig layer both allow (with no tiers configured the tiered layer always
allows, so the autoconfig layer is the sole gate).

#### The remaining-KV counter

The autoconfig layer keeps an in-memory counter of estimated remaining KV
tokens (`KvRemaining`):

- **Reanchor on every good poll.** Each successful poll that carries both the
  usage fraction and the capacity gauge **resets** the counter to
  `floor(capacity × max(target_frac − usage_frac, 0))` — the headroom up to the
  target. Usage at or above the target anchors to `0` (the cache is "full" for
  admission; every non-trivial request rejects until usage drops). The reset is
  not additive: vLLM's real % is ground truth and the counter only accounts for
  requests forwarded since the last anchor, so estimate drift self-corrects
  every poll interval.
- **Subtract on every forwarded request.** On a forward the counter is charged
  `ceil(C × token_margin)` — the same effective size the admission compared
  (the KV the gate reserved) — charged **before** the proxy await. When the
  prompt is unparseable (`C` unknown) the charge is `0`.
- **Unclamped — may go negative.** Subtraction is not clamped; the value may go
  negative (an over-committed signal). The retry formula treats `≤ 0`
  uniformly (see below).
- **Rejected requests charge nothing.** They consumed no KV.
- **Bookkeeping continues while the feed is stale.** Subtraction still happens
  on every forwarded request even while the feed is stale — but the *decision*
  fails open (see the staleness gate below), so the counter cannot reject
  traffic during a metrics outage.

#### Two layers, AND-combined + max-timeout rejector

- **Both allow** → forward.
- **Exactly one rejects** → that layer's `Retry-After` and message are used.
- **Both reject** → `Retry-After = max(tier timeout, autoconfig retry)`, and
  the **reported rejector** is the layer with the **higher** timeout (a tie
  reports the **tiered** layer). The 429 message names the reported rejector.
- **Staleness gate (fail-open override).** If the usage feed is stale or never
  fetched, **both** layers fail open regardless of the counter's value (reason
  `metrics_unavailable`) — a metrics outage never blocks traffic. The counter
  still takes its subtraction (bookkeeping continues) but cannot reject while
  the feed is stale.

> **"Small prompts only" example.** With thresholds
> `[{kv_pct: 50, max_context: 4096, timeout_s: 15}, {kv_pct: 80,
> max_context: 1024, timeout_s: 30}]`, capacity 100 000, target 85%, margin
> 1.25, retry 5/60, and usage at 55%: the tiered layer governs on the 50-tier
> (ctx ≤ 4096, retry 15) and the autoconfig layer anchors to
> `floor(100000 × (0.85 − 0.55)) = 30 000`. A small prompt (ctx 1 000 →
> effective 1 250) passes both → **forward**. A large prompt (ctx 5 000 →
> effective 6 250) is rejected by the tiered layer (5000 > 4096) but allowed by
> the autoconfig layer (6250 ≤ 30 000) → **429, rejector = tiered,
> Retry-After 15**. At usage 90% both reject: the 80-tier retries 30 and the
> autoconfig anchor is `floor(100000 × max(0.85 − 0.90, 0)) = 0` (every request
> retries 60) → combined **429, Retry-After 60, rejector = autoconfig**
> (60 > 30).

#### Autoconfig knobs

| Key | Type | Default | Env override | Meaning |
|---|---|---|---|---|
| `target_kv_cache_pct` | number | `85.0` | `TARGET_KV_CACHE_PCT` | Target KV-cache **percentage (0–100)** the counter anchors headroom up to. Finite, `(0, 100]`. |
| `retry_min_s` | int | `5` | `RETRY_MIN_S` | Minimum `Retry-After` seconds for an autoconfig 429. `>= 1`. |
| `retry_max_s` | int | `60` | `RETRY_MAX_S` | Maximum `Retry-After` seconds for an autoconfig 429. `>= retry_min_s`. |
| `token_margin` | number | `1.25` | `TOKEN_MARGIN` | Conservatism multiplier on the estimated context tokens. Finite, `>= 1.0`. |

#### Worked example

Capacity 100 000, target 85%, margin 1.25, retry 5/60.

- Poll, usage 50% → anchor: `floor(100000 × (0.85 − 0.50)) = 35 000`.
- Request A, ctx 20 000 → effective `ceil(20000 × 1.25) = 25 000` ≤ 35 000 →
  **forward**; counter → 10 000.
- Request B, ctx 25 000 → effective 31 250 > 10 000 → **429**;
  `retry = ceil(clamp(5 × (1 + 31250/10000), 5, 60)) = ceil(20.625) = 21` →
  `Retry-After: 21`.
- Request C, ctx 5 000 → effective 6 250 ≤ 10 000 → **forward**; counter →
  3 750.
- Next poll, real usage 75% (vLLM absorbed A+C) → anchor:
  `floor(100000 × 0.10) = 10 000`. The gate's estimate (3 750) was more
  conservative than reality (25 000 real tokens consumed vs 31 250 charged —
  the margin); the reanchor corrects it.

#### Fail-closed caveat

- **A live vLLM whose `/metrics` lacks a usable KV-cache capacity** (the
  `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
  `vllm:cache_config_info`) cannot be anchored, so the gate **logs an error and
  exits with status 1** (the process dies; an orchestrator restarts it). This
  is the only process-killing failure and is deterministic.
- **An unreachable or stale feed is never fatal** — the gate **fails open**
  (forwards) and keeps retrying. A metrics outage never blocks traffic
  (invariant #1).

#### Zero-config quickstart

The gate runs with **only the vLLM host/port** — no config file, no flags, no
CLI at all:

```sh
docker run -d --name gate \
  --network mynet \
  -p 8000:8000 \
  -e VLLM_HOST=vllm \
  -e VLLM_PORT=8000 \
  vllm-gate
```

There is no `--auto` (or any) CLI flag; argv is ignored. A legacy invocation
that passes `--auto` (or any other argument) still starts normally and behaves
identically — autoconfig is always on.

### Metrics ingestion & fail-open

- A background async task polls `GET http://VLLM_HOST:VLLM_PORT/metrics` every
  `metrics_poll_interval_s` (default 2s) with **one GET per tick** and, from the
  same body, caches the latest `vllm:kv_cache_usage_perc` value (plus its fetch
  timestamp and per-model breakdown) and the KV-cache capacity (the
  `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
  `vllm:cache_config_info`). On a good poll it also re-anchors the remaining-KV counter
  (see [Autoconfig (always on)](#autoconfig-always-on)).
- vLLM may emit one series per `model_name`. **v1 takes the max across all
  series** for both the usage and the capacity gauges (conservative for a
  single instance). v2 will key by `model_name`.
- **Fail-open rules** (a monitoring outage must never take down inference):
  - On any fetch/parse error the last good value is kept and the poller retries.
  - If the cached value is older than `stale_after_s` (default 3× the poll
    interval) it is treated as *unknown* → the request is **allowed** and a
    warning is logged.
- **Fail-closed (unconditional):** an observed (HTTP 200) body that lacks a
  usable KV-cache capacity (the `vllm:kv_cache_size_tokens` gauge or the
  `kv_cache_size_tokens` label on `vllm:cache_config_info`) cannot anchor the
  counter, so the poller logs an error and the process exits with status 1 (see the
  [fail-closed caveat](#fail-closed-caveat)). A merely unreachable vLLM is
  never fatal.

---

## Quickstart

### Build

```sh
docker build -t vllm-gate .
```

### Run (pointing at a vLLM instance)

```sh
docker run -d --name gate \
  --network mynet \
  -p 8000:8000 \
  -e VLLM_HOST=vllm \
  -e VLLM_PORT=8000 \
  -e THRESHOLDS_JSON='[{"kv_pct":50,"max_context":4096,"timeout_s":15},{"kv_pct":80,"max_context":1024,"timeout_s":30}]' \
  vllm-gate
```

`THRESHOLDS_JSON` is **optional** — it adds the optional tiered policy on top of
the always-on autoconfig layer. Omit it (and the config file, if you have one)
for a **zero-config** run: the gate needs only `VLLM_HOST` and `VLLM_PORT`. See
the [zero-config quickstart](#zero-config-quickstart).

Then point your OpenAI client at the gate instead of vLLM:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}]}'
```

### Compose

See [`docker-compose.example.yaml`](docker-compose.example.yaml) for a
reference stack (gate + a vLLM placeholder + a config volume mount).

### Health

```sh
curl http://localhost:8000/healthz
# 200 {"status":"ok","metrics_age_s":1.2,"kv_usage":0.42,
#      "kv_cache_capacity_tokens":100000,"kv_cache_remaining_tokens":35000}
```

`/healthz` always returns 200 for liveness; the body carries readiness info:
`metrics_age_s` (age of the last successful metrics scrape), `kv_usage` (the
last cached usage **fraction**, `null` if never fetched),
`kv_cache_capacity_tokens` (the last cached KV-cache capacity — the
`vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
`vllm:cache_config_info` — `null` if never observed), and
`kv_cache_remaining_tokens` (the autoconfig
counter's current value, `null` if never anchored). Suitable for a Docker
`HEALTHCHECK` or orchestrator liveness probe.

---

## Configuration

Configuration is a single YAML file mounted at `/etc/gate/config.yaml`, with
env-var overrides for the essentials. A reference file is provided at
[`config.example.yaml`](config.example.yaml).

### Schema

```yaml
# env overrides: VLLM_HOST, VLLM_PORT, LISTEN_HOST, LISTEN_PORT
vllm_host: vllm
vllm_port: 8000
listen_host: 0.0.0.0
listen_port: 8000
metrics_poll_interval_s: 2.0
stale_after_s: 6.0
chars_per_token: 4
default_max_tokens: 256

# optional — the tiered policy (env override: THRESHOLDS_JSON, a JSON list).
# Autoconfig runs regardless; with no tiers the tiered layer always allows.
thresholds:
  - kv_pct: 50
    max_context: 4096
    timeout_s: 15
  - kv_pct: 80
    max_context: 1024
    timeout_s: 30

# autoconfig (always on) — all optional, defaults shown
target_kv_cache_pct: 85.0
retry_min_s: 5
retry_max_s: 60
token_margin: 1.25
```

### Reference

| Key | Type | Default | Env override | Meaning |
|---|---|---|---|---|
| `vllm_host` | string | `vllm` | `VLLM_HOST` | Hostname of the vLLM server. |
| `vllm_port` | int | `8000` | `VLLM_PORT` | Port of the vLLM server (API + `/metrics`). |
| `listen_host` | string | `0.0.0.0` | `LISTEN_HOST` | Bind address for the gate. |
| `listen_port` | int | `8000` | `LISTEN_PORT` | Port the gate listens on. |
| `metrics_poll_interval_s` | float | `2.0` | — | How often to poll `/metrics`. |
| `stale_after_s` | float | `6.0` | — | Age after which a cached value is treated as unknown (fail-open). |
| `chars_per_token` | int | `4` | — | Heuristic divisor for prompt token estimation. |
| `default_max_tokens` | int | `256` | — | Output headroom added when the request omits `max_tokens`. |
| `thresholds` | list | *(absent)* | `THRESHOLDS_JSON` | **Optional** tiered policy. With no tiers the tiered layer always allows; autoconfig runs regardless. See below. |
| `target_kv_cache_pct` | number | `85.0` | `TARGET_KV_CACHE_PCT` | Autoconfig: target KV-cache **percentage (0–100)** the remaining-KV counter anchors up to. Finite, `(0, 100]`. |
| `retry_min_s` | int | `5` | `RETRY_MIN_S` | Autoconfig: minimum `Retry-After` seconds. `>= 1`. |
| `retry_max_s` | int | `60` | `RETRY_MAX_S` | Autoconfig: maximum `Retry-After` seconds. `>= retry_min_s`. |
| `token_margin` | number | `1.25` | `TOKEN_MARGIN` | Autoconfig: conservatism multiplier on estimated context tokens. Finite, `>= 1.0`. |

### `thresholds` entry contract (optional)

`thresholds` is **optional** — the gate runs with no tiers at all (autoconfig
is always on). When present, each entry is an object with exactly three fields:

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `kv_pct` | number | `0 <= kv_pct <= 100` | KV-cache **percentage** at which this tier activates. The metric is a fraction (0–1); the gate compares `usage*100 >= kv_pct`. |
| `max_context` | int | `>= 1` | Max estimated context tokens allowed while this tier is active. |
| `timeout_s` | int | `>= 1` | Seconds placed in the `Retry-After` header on a 429 from this tier. |

**Validation (fail-fast at startup):** `kv_pct` within `[0,100]`,
`max_context >= 1`, `timeout_s >= 1`, and `kv_pct` values unique across
entries. Any violation aborts startup with a clear log line.

> **Worked example (tiered layer alone).** With the two tiers above: at 45%
> usage no tier is active → the tiered layer allows everything. At 60% usage the
> 50% tier governs → a 3000-token request passes (≤ 4096) but a 5000-token
> request gets `429 + Retry-After: 15`. At 85% usage the 80% tier governs →
> only ≤ 1024 tokens pass the tiered layer; larger requests get
> `429 + Retry-After: 30`. (In a real deployment the always-on autoconfig layer
> is also evaluated and AND-combined — see the [autoconfig worked
> example](#worked-example) and the ["small prompts only"
> example](#two-layers-and-combined--max-timeout-rejector).)

---

## The 429 contract

When a request is rejected by either admission layer the gate responds. The
`Retry-After` is the **combined** value: when **both** layers reject it is
`max(tier timeout, autoconfig retry)`; when only one rejects it is that layer's
value. An autoconfig `Retry-After` is always an **integer within
`[retry_min_s, retry_max_s]`**, and `remaining <= 0` (cache full /
over-committed) yields exactly `retry_max_s`.

**Tiered rejector** (the tiered layer is the reported rejector):

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
Content-Type: application/json

{
  "error": {
    "message": "vLLM KV cache too full for a ~3800-token request (usage 84.0% >= 80% tier; max 1024). Retry in 30s.",
    "type": "cache_pressure",
    "code": "kv_cache_too_full"
  }
}
```

**Autoconfig rejector** (the autoconfig layer is the reported rejector):

```
HTTP/1.1 429 Too Many Requests
Retry-After: 21
Content-Type: application/json

{
  "error": {
    "message": "vLLM KV cache headroom exhausted for a ~25000-token request (usage 50.0%, ~10000 of 100000 tokens remaining up to the 85% target). Retry in 21s.",
    "type": "cache_pressure",
    "code": "kv_cache_too_full"
  }
}
```

- The **reported rejector** is the layer with the higher timeout when both
  reject (a tie reports the tiered layer); the message names it.
- The body is a small OpenAI-style error envelope so standard SDKs surface it
  cleanly.

---

## Proxied endpoints (v1)

| Method & path | Behavior |
|---|---|
| `POST /v1/chat/completions` | Gated, then proxied to vLLM (SSE streaming supported). |
| `POST /v1/completions` | Gated, then proxied to vLLM. |
| `GET /healthz` | Liveness (always 200; readiness info in body). |
| `GET /metrics` | Gate-local Prometheus stats (see [Observability](#observability)); **not** proxied. |
| anything else | `404`. |

The proxy forwards method, path, query, headers (minus hop-by-hop), and body
verbatim, and propagates upstream status codes — a vLLM 5xx surfaces as a 5xx,
never masked by the gate.

---

## Observability

The gate exposes its own stats at `GET /metrics` — HTTP 200 with
`Content-Type: text/plain; version=0.0.4; charset=utf-8`, body in the
Prometheus text exposition format (the same shape vLLM emits). It is a
gate-local endpoint like `/healthz`, **not proxied** — do not confuse the two:
the background poller scrapes **vLLM's** `/metrics` (on vLLM's port), while the
gate's `GET /metrics` (on the gate's port) serves the gate's own `gate_*`
metrics. They are different endpoints and different Prometheus scrape targets.

- **In-memory only, resets on restart.** All `gate_*` metrics live in memory
  and reset when the process restarts. That is by design: Prometheus
  `rate()`, `increase()`, and `histogram_quantile()` all handle counter resets
  correctly, so no persistence is added.
- **Unauthenticated (v1).** Consistent with the no-auth invariant. The
  endpoint exposes the loaded config values and traffic stats, so
  network-protect it the same way you already protect vLLM's own `/metrics`.
- **Decision path unchanged.** Recording is a pure side effect of each
  allow/reject decision — it never alters the decision, the 429 response, or
  the proxy.

### Metric reference

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `gate_requests_total` | Counter | `endpoint` (`chat_completions` \| `completions`), `model` (the request body's `model` field, else `unknown`), `result` (`forwarded` \| `rejected`) | Generation requests the gate decided, scoped to the two generation endpoints (404s, `/healthz`, and `/metrics` are not counted). |
| `gate_request_ctx_tokens` | Histogram | `model`, `result` | Estimated context tokens (prompt + headroom) of each request. Median via `histogram_quantile(0.5, ...)`. Buckets: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144. |
| `gate_kv_cache_usage_pct` | Gauge | `model_name` (vLLM's `model_name` label; `default` when the series is unlabeled) | Current KV-cache fill **percentage (0–100)** per served model (vLLM's real metric, converted from the fraction). |
| `gate_kv_cache_remaining_tokens` | Gauge | (none) | The gate's **own live estimate** of remaining KV-cache tokens up to the target (post-subtraction, pre-reanchor); `NaN` until the first anchor. Distinct from `gate_kv_cache_usage_pct` — this is the gate's estimate, not vLLM's real usage. |
| `gate_config_info` | Info | one label per config field + `thresholds_json` | The loaded configuration, set once at startup. |
| `gate_metrics_fresh` | Gauge | (none) | `1` if the vLLM metrics feed is fresh, `0` if stale or never fetched. |
| `gate_metrics_age_s` | Gauge | (none) | Seconds since the last successful vLLM `/metrics` fetch; `NaN` if never fetched. |

**Two model namespaces, kept distinct.** `model` is the OpenAI request body's
`model` field (the client's model name — the future multi-server routing key).
`model_name` is vLLM's served-model label. They are different strings on
different metrics with different label names; do not conflate them in queries
or dashboards.

### PromQL examples

```promql
# total requests processed (rate over 5m)
sum(rate(gate_requests_total[5m]))
# forwarded vs rejected rates
sum(rate(gate_requests_total{result="forwarded"}[5m]))
sum(rate(gate_requests_total{result="rejected"}[5m]))
# reject rate (fraction of requests the gate rejected)
sum(rate(gate_requests_total{result="rejected"}[5m])) / sum(rate(gate_requests_total[5m]))
# median size (estimated tokens) of forwarded / rejected requests
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="forwarded"}))
histogram_quantile(0.5, sum by (model) (gate_request_ctx_tokens_bucket{result="rejected"}))
# current per-model KV-cache fill %
gate_kv_cache_usage_pct
# the gate's own live estimate of remaining KV tokens (NaN until anchored)
gate_kv_cache_remaining_tokens
# is the upstream metrics feed healthy?
gate_metrics_fresh
```

> **Median is derived, not exposed.** The gate exposes a histogram;
> `histogram_quantile` interpolates at bucket resolution. A histogram `_count`
> counts only requests that had a token estimate (fail-open unparseable
> requests are skipped), so it can be lower than
> `gate_requests_total{result=...}` for the same labels — expected.

### Prometheus scrape config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm-gate
    static_configs:
      - targets: ["gate:8000"]   # the gate's own /metrics
  # vLLM's own /metrics is a separate scrape target (e.g. job_name: vllm).
```

---

## Operational notes

- **Fail-open is intentional.** If you stop or lose the vLLM `/metrics`
  endpoint, the gate keeps forwarding (it cannot measure headroom, so it does
  not block). Watch the warning logs and the `metrics_age_s` in `/healthz`.
- **Fail-closed is the one exception.** A *live* vLLM whose `/metrics` body
  lacks a usable KV-cache capacity (the `vllm:kv_cache_size_tokens` gauge or
  the `kv_cache_size_tokens` label on `vllm:cache_config_info`) cannot anchor
  the remaining-KV counter, so the gate logs an error and exits with status 1
  (an orchestrator restarts it).
  An unreachable or stale feed is never fatal — that path fails open. See the
  [fail-closed caveat](#fail-closed-caveat).
- **Startup log.** On a successful config load the gate logs the active
  autoconfig policy — `autoconfig: target 85.0% KV cache, token margin 1.25,
  retry 5-60s` — and the tiered policy: `tiered policy: 2 threshold tier(s)`
  (or `tiered policy: none — autoconfig only`).
- **Estimation is conservative.** The gate over-estimates token counts, so it
  may reject a request that vLLM would actually have fit. Tune
  `chars_per_token` and `default_max_tokens` to your workload if you see
  over-rejection.
- **No auth, no TLS termination (v1).** The gate trusts the deployment network
  and is a drop-in front of vLLM. Put it behind your existing ingress/auth if
  you need it.
- **Logging.** Structured `logging` at INFO; one line per decision
  (`usage_pct`, `ctx_tokens`, `active_tier`, `allow/reject`, `retry_after`).
  Prompt text is never logged.
- **Observability.** The gate exposes its own stats at `GET /metrics`
  (Prometheus text format) — forwarded/rejected counts, request-size
  histograms, per-model KV-cache fill, loaded config, and feed freshness. See
  the [Observability](#observability) section for the metric reference and
  ready-to-paste PromQL.

---

## Limitations & v2 roadmap

- **Single instance only.** v1 assumes one vLLM backend and takes the max
  across `model_name` series. Multi-instance monitoring keyed by `model_name`
  (per-model thresholds and routing) is the flagship v2 item.
- **Heuristic sizing.** No exact tokenization in v1 (no tokenizer dependency).
- **No auth / TLS / load-balancing / request queuing** in v1.

---

## Development

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check .                       # lint
mypy src/                          # typecheck
pytest -q                          # unit + ASGI integration tests
pytest --cov=gate --cov-fail-under=80   # 80% coverage gate
docker build -t vllm-gate .        # container build (no push)
```

The test suite (`tests/test_api.py`) drives the gate end-to-end against a fake
vLLM (`httpx.ASGITransport` + a mock upstream transport), so the allow / 429 /
fail-open / streaming paths are all exercised without a real model or GPU.

### CI / local parity

- **Local CI**: `bash scripts/local-ci.sh` (or `make ci`) runs the exact same
  checks as the GitHub Actions CI (`.github/workflows/tests.yml`) —
  source-tracking, `ruff format --check`, `ruff check`, `mypy src/` +
  `python -m compileall -q src/`, pytest with the 80% coverage gate,
  `pip-audit` dependency audit, `pre-commit run --all-files`, and the
  container build — in the same order. A green local run should mean a green
  CI run (the container step is skipped with a warning if docker is absent).
- **CI jobs** (`.github/workflows/tests.yml`): on push to `main` and on
  pull requests, CI runs the `format`, `lint`, `typecheck`, and `test`
  (pytest + 80% coverage on Python 3.11/3.12/3.13) jobs, plus
  `source-tracking`, `security` (pip-audit dependency audit), and
  `pre-commit` (pre-commit hooks). The `container` job (docker build, no
  push) is gated on all of the others, so a code failure blocks the image
  build.
- **Releases** (`.github/workflows/release.yml`): after the checks pass, the
  release job builds multi-platform images
  (`linux/amd64,linux/arm64`) and pushes them to GHCR — tagged with the
   `vX.Y.Z` version on `v*` tag pushes, and `:latest` on `main` pushes and
   non-prerelease `v*` tags (prerelease tags like `vX.Y.Z-rcN` get only their
   version tag). Prerelease tags (e.g. `vX.Y.Z-rc1`) produce a draft +
   prerelease GitHub Release.
- **Pre-commit hooks**: `.pre-commit-config.yaml` installs pre-commit hooks
  (ruff + ruff-format plus basic hygiene checks). They run in CI (the
  `pre-commit` job) and are also available locally before committing.
- **Dependency updates**: Dependabot (`.github/dependabot.yml`) opens weekly
  update PRs for `pip` dependencies and GitHub Actions.

See [`AGENTS.md`](AGENTS.md) for the architecture map, the load-bearing
invariants (fail-open, percentage-vs-fraction, highest-tier-only, inclusive
boundary, transparent proxy), and the git workflow.
