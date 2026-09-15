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
endpoint and reading `vllm:kv_cache_usage_perc`. It is designed for
**safe, configuration-first, fail-open** operation: a monitoring outage never
blocks inference traffic, and the gate is a transparent pass-through for the
endpoints it proxies.

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

### Metrics ingestion & fail-open

- A background async task polls `GET http://VLLM_HOST:VLLM_PORT/metrics` every
  `metrics_poll_interval_s` (default 2s) and caches the latest
  `vllm:kv_cache_usage_perc` value plus its fetch timestamp.
- vLLM may emit one series per `model_name`. **v1 takes the max across all
  series** (conservative for a single instance). v2 will key by `model_name`.
- **Fail-open rules** (a monitoring outage must never take down inference):
  - On any fetch/parse error the last good value is kept and the poller retries.
  - If the cached value is older than `stale_after_s` (default 3× the poll
    interval) it is treated as *unknown* → the request is **allowed** and a
    warning is logged.

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
# 200 {"status":"ok","metrics_age_s":1.2}
```

`/healthz` always returns 200 for liveness; the body carries readiness info
(age of the last successful metrics scrape). Suitable for a Docker
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

# required, >= 1 entry (env override: THRESHOLDS_JSON, a JSON list)
thresholds:
  - kv_pct: 50
    max_context: 4096
    timeout_s: 15
  - kv_pct: 80
    max_context: 1024
    timeout_s: 30
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
| `thresholds` | list | — | `THRESHOLDS_JSON` | Tiered policy. See below. |

### `thresholds` entry contract

Each entry is an object with exactly three fields:

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `kv_pct` | number | `0 <= kv_pct <= 100` | KV-cache **percentage** at which this tier activates. The metric is a fraction (0–1); the gate compares `usage*100 >= kv_pct`. |
| `max_context` | int | `>= 1` | Max estimated context tokens allowed while this tier is active. |
| `timeout_s` | int | `>= 1` | Seconds placed in the `Retry-After` header on a 429 from this tier. |

**Validation (fail-fast at startup):** `kv_pct` within `[0,100]`,
`max_context >= 1`, `timeout_s >= 1`, and `kv_pct` values unique across
entries. Any violation aborts startup with a clear log line.

> **Worked example.** With the two tiers above: at 45% usage no tier is active
> → everything forwards. At 60% usage the 50% tier governs → a 3000-token
> request forwards (≤ 4096) but a 5000-token request gets
> `429 + Retry-After: 15`. At 85% usage the 80% tier governs → only ≤ 1024
> tokens forward; larger requests get `429 + Retry-After: 30`.

---

## The 429 contract

When a request is rejected the gate responds:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
Content-Type: application/json

{
  "error": {
    "message": "vLLM KV cache too full for a ~3800-token request (usage 84% >= 80% tier; max 1024). Retry in 30s.",
    "type": "cache_pressure",
    "code": "kv_cache_too_full"
  }
}
```

- `Retry-After` is the governing tier's `timeout_s`.
- The body is a small OpenAI-style error envelope so standard SDKs surface it
  cleanly.

---

## Proxied endpoints (v1)

| Method & path | Behavior |
|---|---|
| `POST /v1/chat/completions` | Gated, then proxied to vLLM (SSE streaming supported). |
| `POST /v1/completions` | Gated, then proxied to vLLM. |
| `GET /healthz` | Liveness (always 200; readiness info in body). |
| anything else | `404`. |

The proxy forwards method, path, query, headers (minus hop-by-hop), and body
verbatim, and propagates upstream status codes — a vLLM 5xx surfaces as a 5xx,
never masked by the gate.

---

## Operational notes

- **Fail-open is intentional.** If you stop or lose the vLLM `/metrics`
  endpoint, the gate keeps forwarding (it cannot measure headroom, so it does
  not block). Watch the warning logs and the `metrics_age_s` in `/healthz`.
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
