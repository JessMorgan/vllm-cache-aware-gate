# Multi-backend / multi-model routing — Design Plan

Status: **proposed** (2026-09-16; design Q&A complete, not yet implemented)
Base: `main` at `2e8c9c2` (post-autoconfig merge)

This is the flagship v2 item from the README roadmap ("per-model thresholds and
routing"), delivered as **per-backend policy + per-model routing**: the request
body's `model` field selects which configured vLLM backend the request is
proxied to and which backend's KV-cache state (usage feed, capacity,
remaining-KV counter) the two admission layers evaluate.

---

## Decision log

**Round 1 (2026-09-16 — design Q&A, user-confirmed):**

1. **A backend serves a list of models (1:N); the whole backend is treated
   the same.** All admission state (usage cache, capacity, remaining-KV
   counter, poller, policy) is scoped **per backend**, not per model. Models on
   one backend share its KV pool (single-engine vLLM), so per-backend is the
   correct accounting granularity. A strict 1:1 deployment is the length-1
   special case. (Original Q&A started from a 1:1 option; the final
   confirmation was "multi-models on one backend is fine as long as they share
   the same KV cache and the whole backend is treated the same.")
2. **Tiered `thresholds`: per-backend, optional global default.** Top-level
   `thresholds` remain as the default; each backend entry may override with its
   own list.
3. **Autoconfig knobs: per-backend overrides, global defaults.** The four
   always-on knobs (`target_kv_cache_pct`, `token_margin`, `retry_min_s`,
   `retry_max_s`) may each be overridden per backend; the global value (with
   its existing env override) is the default when a backend omits it.
4. **Unknown model → default backend.** A request whose `model` matches no
   backend is forwarded to the backend flagged `default: true` (fallback: the
   first entry); vLLM returns its own canonical "model not found" error. The
   admission layers still apply on that backend. Same fail-open spirit as the
   unparseable-body rule (gotcha #7).
5. **Model discovery: auto-adopt from `GET /v1/models`.** Each backend's owned
   model set is learned from its `/v1/models` endpoint at startup and refreshed
   on an interval (piggybacked on the poller). An optional explicit `models:`
   list per backend acts as a **mnemonic** (operators see which model id they
   are writing config for) and as a validation anchor; when present it defines
   the owned set, otherwise the discovered served set is adopted.
6. **`backends:` list replaces `vllm_host`/`vllm_port` entirely.** The scalar
   keys and the `VLLM_HOST`/`VLLM_PORT` env vars are **removed** (a hard
   breaking change; legacy configs must migrate to `backends:`). `listen_host`/
   `listen_port` are unaffected. **This supersedes the zero-config property
   (AGENTS.md gotcha #13):** with the host/port shorthand gone, **at least one
   backend (host + port) must always be configured** (file or
   `BACKENDS_JSON`) — the gate has no other way to learn its upstream. The
   gate can no longer start from nothing; this is user-approved and
   unavoidable.
7. **`backends:` is file- *and* env-configurable: `BACKENDS_JSON` (a JSON list)
   overrides the file, mirroring `THRESHOLDS_JSON`.** The other env overrides
   (`THRESHOLDS_JSON`, `TARGET_KV_CACHE_PCT`, `TOKEN_MARGIN`, `RETRY_MIN_S`,
   `RETRY_MAX_S`) continue to set the **global default** values only.
8. **New gate-local `GET /v1/models` (aggregate endpoint).** The gate adds an
   OpenAI-shaped `GET /v1/models` that aggregates the known models of all
   backends (with an `owned_by` extension naming the backend). It is
   gate-local, always 200, and never proxied.

**Derived invariants (not separately asked; follow from 1 + 8):**

9. **Model ids are globally unique across backends**, enforced at config
   validation. This makes the `model → backend` map total and the
   unknown-model fallback unambiguous, and lets existing model-labeled
   metrics (`gate_requests_total`, `gate_request_ctx_tokens`) stay
   model-keyed. A deployment that legitimately serves the same model id on two
   backends (A/B, canary) is out of scope — it would require a routing rule
   instead of a unique map.
10. **Capacity-missing ⇒ per-backend fail-open (deliberate deviation from
    gotcha #10).** Today a live vLLM whose observed `/metrics` body lacks a
    usable KV-cache capacity raises `CapacityUnavailableError` and the process
    exits 1. With N backends, killing the whole proxy because **one** backend is
    misconfigured would take down the healthy ones — contradicting "a
    monitoring problem never blocks inference traffic" (gotcha #1). So: a
    backend whose observed body lacks usable capacity gets a **repeated error
    log**, its remaining-KV counter stays unanchored (its autoconfig layer
    fails open, reason `auto_unanchored`), and a new
    `gate_backend_capacity_unavailable{backend}` gauge is set to 1 for alerting.
    Transport errors and non-200 responses still keep all state (unchanged);
    the stale feed still fails open via the staleness gate (unchanged).

---

## 1. Background, Problem, Goal

**Background.** The gate today fronts exactly **one** vLLM instance
(`vllm_host`/`vllm_port`). One `MetricsCache`, one `CapacityCache`, one
`KvRemaining` counter, one poller task, one proxy `base_url`. The request
body's `model` field is already extracted (`extract_request_model`) but used
**only** as a stats label — it is documented as the "future routing key."
vLLM labels `vllm:kv_cache_usage_perc` per `model_name`, but the KV-cache
capacity (`vllm:kv_cache_size_tokens` / the `kv_cache_size_tokens` label on
`vllm:cache_config_info`) is **per-engine, not per-model** — which is why
admission state must be per-backend (decision 1), not per-model.

**Problem.** Operators run several vLLM backends, each serving one model (or a
set of models sharing one KV pool). They need the gate to (a) pick the correct
backend per request from the `model` field, and (b) make admission decisions
using **that backend's** KV state rather than one global pool.

**Goal.**

1. Route each generation request to exactly one configured backend by its
   `model` field; unmatched/unparseable models go to the **default** backend.
2. Scope all admission state **per backend** (its own usage feed, capacity,
   remaining-KV counter, poller), so a shared-KV backend is treated the same
   across its models.
3. Provide a sane, mnemonic config schema (`backends:` list; per-backend
   overrides of the tiered `thresholds` and the four autoconfig knobs; optional
   explicit `models:` mnemonic/anchor).
4. Learn each backend's served model ids from `GET /v1/models` at startup and
   on a refresh interval, for correct routing matches.
5. Expose an aggregated `GET /v1/models` and per-backend observability.

**Non-goals.** No per-model policy (policy is per-backend — decision 1). No
real tokenizer. No auth/TLS. No change to `router.py`/`tokens.py`/`proxy.py`
decision math. No multi-backend routing for duplicate model ids (decision 9).

## 2. Design Summary

### 2.1 Config (`config.py`)

Replace scalar `vllm_host`/`vllm_port` with a **required** `backends:` list.
New `Backend` dataclass:

```
name: str                    # operator-chosen mnemonic; unique, non-empty
host: str
port: int                    # 1..65535
models: tuple[str, ...]      # optional mnemonic + validation anchor;
                             # empty => auto-adopt from /v1/models (decision 5)
default: bool                # at most one True across all backends
thresholds: tuple[Threshold, ...] | None   # None => global default
target_kv_cache_pct: float | None          # None => global default
token_margin: float | None
retry_min_s: int | None
retry_max_s: int | None
```

`GateConfig` keeps the **global default** values of the five overridable knobs
(the four autoconfig knobs + `thresholds`), plus `chars_per_token`,
`default_max_tokens`, `metrics_poll_interval_s`, `stale_after_s`, and a new
`model_refresh_interval_s` (default `30.0`).

Env overrides: `BACKENDS_JSON` (JSON list of backend objects; **env wins over
file**), `THRESHOLDS_JSON` (global default tiers), `TARGET_KV_CACHE_PCT`,
`TOKEN_MARGIN`, `RETRY_MIN_S`, `RETRY_MAX_S` (global default knobs).
`VLLM_HOST`/`VLLM_PORT` are removed — a config containing `vllm_host` fails
with the existing unknown-key `ConfigError`.

Validation (all `ConfigError`):

- ≥ 1 backend; unique non-empty `name`s; `port` in 1–65535.
- **Model ids globally unique across backends** (decision 9): two backends
  listing the same id in their explicit `models:` is a startup error. (Runtime
  collisions from auto-adopt are handled by the registry, §2.2 — logged, not
  fatal, since the operator did not type them.)
- ≤ 1 backend with `default: true`.
- Per-backend knob values, when present, satisfy the same ranges as the
  globals (target finite and in (0, 100]; margin finite and ≥ 1.0;
  `retry_min_s` ≥ 1; `retry_max_s` ≥ `retry_min_s`; tier fields as today).

### 2.2 Model discovery (`models.py` — new module)

- `parse_v1_models(text) -> list[str] | None` — **pure.** Parses an OpenAI
  `GET /v1/models` body (`{"data": [{"id": ...}, ...]}`) into the ordered id
  list. `None` on non-JSON text, missing/non-list `data`, or missing `id`
  fields. No network.
- `ModelRegistry` — stateful, single-asyncio-loop (same contract as
  `MetricsCache`: synchronous read-modify-write, no `await` inside, no locks).
  Owns the `model → backend_name` map.
  - `register_backend(name: str, *, is_default: bool)` — called once at app
    build, per configured backend.
  - `sync(name: str, owned_models: set[str])` — reconciles one backend's owned
    set (explicit `models` if present, else the discovered served set). Models
    no longer owned by `name` are re-resolved against the remaining backends
    (default wins, else existing owner). If a model ends up owned by **no**
    backend, it simply stops resolving: `resolve` returns `None` and requests
    for it take the unknown-model path (default backend, decision 4) — the
    model is never hard-404'd by the gate.
  - `resolve(model: str) -> str | None` — the routing lookup.
  - **Collision policy:** a model owned by two backends resolves to the
    `default: true` backend if any owner is the default, else the first
    registered owner; the conflict is logged once. (Only reachable via
    auto-adopt; explicit duplicates are a config error.)

### 2.3 Poller (`poller.py`)

`run_poller` gains per-backend parameters and is invoked **once per backend**:
its own `cache`/`capacity_cache`/`counter`, its own `target_frac`, its
`/metrics` URL, plus its `/v1/models` URL, `model_refresh_interval_s`, and the
shared `ModelRegistry`. Per tick, in order:

1. **Metrics work** — the existing `fetch_metrics` → update usage + by-model
   cache, reanchor `counter` (when usage present), update `capacity_cache`.
   Unchanged.
2. **Model discovery** — when due (first tick immediately, then every
   `model_refresh_interval_s`, tracked with `time.monotonic`): `GET
   /v1/models`, `parse_v1_models`, `registry.sync(name, ...)`. A failed fetch
   or `None` parse **keeps the last known map** (fail-open; discovery is
   auxiliary). Never fatal.
3. **Capacity-missing (observed 200, no usable capacity)** — **no longer
   raises `CapacityUnavailableError` / exits** (decision 10): log an error,
   leave that backend's counter unanchored (its autoconfig layer fails open),
   and set the per-backend unavailable flag (surfaced by the stats gauge,
   §2.5). **The poller task keeps looping** — the fatal
   done-callback (`_poller_fatal` → `os._exit(1)`) only fires when the task
   *dies* with an exception, so with the capacity path no longer raising it is
   never triggered by a capacity condition. Transport errors and non-200
   responses still keep all state (unchanged, fail-open).

`CapacityUnavailableError` becomes **dead code** (it is raised only in the
poller; the class is defined in `metrics.py` and referenced from `app.py`
docstrings, `tests/test_poller.py`, and `tests/test_api.py`) and is
**deleted** from `metrics.py`, with those references cleaned up as part of
this change (see the `metrics.py` row in §3).

### 2.4 App (`app.py`)

- `create_app` builds a `BackendState` bundle per configured backend: its
  `MetricsCache`, `CapacityCache`, `KvRemaining`, resolved
  `thresholds` (per-backend or global default), resolved `AutoPolicy`
  (per-backend overrides or global defaults), resolved `target_frac`
  (single percentage→fraction conversion at wiring, gotcha #14 preserved),
  `base_url`, `is_default`, and explicit `models`. Plus one `ModelRegistry`
  with all backends registered.
- **Lifespan** starts **one poller task per backend** (each with its own fatal
  done-callback — note the capacity path no longer fires it).
- `_handle_generation(request, endpoint)`:
  1. `model = extract_request_model(body)` (unchanged).
  2. `backend = registry.resolve(model) or default_backend` (unknown/
     unparseable `"unknown"` models fall to the default backend, decision 4).
  3. **Staleness gate per backend**: if `backend.cache` is stale/never-fetched
     → both layers fail open for that backend (gotcha #1), counter subtraction
     still happens (bookkeeping).
  4. Else: `usage_pct = backend.cache.value() * 100.0` → `decision(...)` with
     **that backend's** thresholds; `decision_auto(backend.counter.value(),
     ...)` with **that backend's** `AutoPolicy`. Pool-level max usage
     (`cache.value()`, the max across the backend's `model_name` series) is
     correct for shared-KV backends and matches gotcha #5.
  5. `combine_decisions` (unchanged), then on allow charge
     `backend.counter.subtract(ceil(ctx × backend.token_margin))` **before**
     the proxy await and proxy to `backend.base_url`; on reject, 429 with
     `Retry-After` and the rejector-aware message built from **that
     backend's** values.
- **`GET /v1/models`** (new, gate-local, always 200): aggregate the registry's
  known models into one OpenAI-shaped list, one entry per model with an
  `owned_by` extension naming the backend (default backend's models listed
  once; a collided model appears under its resolved owner).
- **`GET /healthz`**: `status: "ok"` unchanged; the single
  `kv_usage`/`kv_cache_capacity_tokens`/`kv_cache_remaining_tokens` fields
  become a per-backend array:
  `{"backends": [{"name", "metrics_age_s", "kv_usage",
  "kv_cache_capacity_tokens", "kv_cache_remaining_tokens"}]}` plus
  `models: [{"id", "owned_by"}]`.
- **`GET /metrics`**: as today, but `set_kv_usage` unions all backends'
  `by_model()` maps, `set_freshness`/`set_remaining` run per backend
  (labeled, §2.5).

### 2.5 Stats (`stats.py`)

- **`backend` label** added to `gate_kv_cache_remaining_tokens`,
  `gate_metrics_fresh`, `gate_metrics_age_s`; `set_remaining(backend, value)`
  / `set_freshness(backend, fresh, age_s)` gain the backend argument.
- `gate_kv_cache_usage_pct{model_name}` and `gate_requests_total` /
  `gate_request_ctx_tokens{model,...}` are **unchanged** (model ids are
  globally unique — decision 9; the backend is recoverable from the model).
- **New** `gate_backend_capacity_unavailable{backend}` gauge (0/1), set by the
  app from the poller's per-backend flag (decision 10, alerting hook).
- `gate_config_info` serializes the backend structure: per backend
  `name`/`host`/`port`/`default` + which knobs are overridden (global vs
  backend-specific), but **not** the `models:` lists (discovered data would
  go stale; keep the info metric small and startup-static).

### 2.6 No change

`router.py` (pure decision math), `tokens.py` (token estimation stays
**global** — `chars_per_token`/`default_max_tokens` are not per-backend in
v1; the per-backend override of these is a follow-up if needed), `proxy.py`
(`proxy_request` already takes `base_url`), `metrics.py` caches
(`MetricsCache`/`CapacityCache`/`KvRemaining` reused once per backend).

## 3. Files to Change/Create

| Path | Action | Description |
|---|---|---|
| `src/gate/models.py` | Create | Pure `parse_v1_models`; stateful `ModelRegistry` (register/sync/resolve, collision policy, fail-open on discovery failure). |
| `src/gate/config.py` | Modify | `Backend` dataclass; required `backends:` list replaces `vllm_host`/`vllm_port`; global knobs + `model_refresh_interval_s`; YAML + `BACKENDS_JSON` (env wins) parsing; validation (≥1 backend, unique names, globally-unique explicit models, ≤1 default, port/knob ranges); remove `VLLM_HOST`/`VLLM_PORT`. |
| `src/gate/poller.py` | Modify | Per-backend invocation (own caches/counter/target); `/v1/models` discovery at `model_refresh_interval_s` (first tick immediate), failure keeps last map; capacity-missing ⇒ per-backend fail-open (no exit) + unavailable flag; keep transport fail-open. |
| `src/gate/app.py` | Modify | `BackendState` per backend; `ModelRegistry` wiring; per-backend two-layer decision in `_handle_generation`; route by model (unknown ⇒ default); per-backend `base_url`; new `GET /v1/models`; per-backend `/healthz`; per-backend `/metrics`; one poller per backend; module docstring updated (the exit-1 half of the capacity story is gone — decision 10). |
| `src/gate/stats.py` | Modify | `backend` label on remaining/fresh/age gauges; `set_remaining`/`set_freshness` take backend; new `gate_backend_capacity_unavailable{backend}`; `gate_config_info` serializes backends (no model lists). |
| `src/gate/main.py` | Modify | Log backend count/names/default/overrides; remove host/port logging. |
| `src/gate/router.py` | No change | Reused as-is. |
| `src/gate/tokens.py` | No change | Global token estimation unchanged. |
| `src/gate/proxy.py` | No change | Already takes `base_url`. |
| `src/gate/metrics.py` | Modify (small) | Caches reused per backend, unchanged; **delete** the now-dead `CapacityUnavailableError` (decision 10 — raised only in the poller, which no longer raises it). |
| `tests/test_models.py` | Create | `parse_v1_models` + `ModelRegistry` (register/sync/resolve, collision default-wins, explicit vs auto-adopt, removal, discovery-failure no-op). |
| `tests/test_config.py` | Extend | `backends` YAML + `BACKENDS_JSON` (env wins); per-backend override resolution; all validation failures; global defaults; legacy `vllm_host`/`VLLM_HOST` rejected. |
| `tests/test_poller.py` | Extend | Per-backend poller + discovery cadence; discovery failure keeps map; capacity-missing ⇒ fail-open, no exception, no task end; per-backend reanchor. |
| `tests/test_api.py` | Extend | Two fake vLLMs; routing by model (assert proxy target); unknown ⇒ default; per-backend 429; one-stale/one-fresh fail-open; `/v1/models` aggregate; per-backend `/healthz` + `/metrics` labels. |
| `tests/test_stats.py` | Extend | Per-backend labels; stale-label removal; `gate_config_info` backend structure; `gate_backend_capacity_unavailable`. |
| `tests/test_main.py` | Extend | Backend logging. |
| `config.example.yaml` | Modify | Multi-backend reference with mnemonic + per-backend overrides + auto-adopt note (sketch in §6). |
| `docker-compose.example.yaml` | Modify | `BACKENDS_JSON` example; remove `VLLM_HOST`/`VLLM_PORT`; mounted-config comment. |
| `README.md` | Modify | Multi-backend section, config reference, diagram (N backends), `/v1/models`, per-backend metrics/healthz, capacity fail-open deviation (decision 10), v2 roadmap. |
| `AGENTS.md` | Modify | Project map (new `models.py`, per-backend state); new gotchas (decision 9 + 10 + routing + `/v1/models` + shorthand removal); test list adds `test_models.py`. |

## 4. Testing Plan

- **Unit, pure/edge (`test_models.py`).** `parse_v1_models`: valid list,
  empty `data`, missing `data`, non-JSON, duplicate ids, non-dict entries,
  missing `id`. `ModelRegistry`: register + resolve; `sync` add/remove;
  **collision ⇒ default wins, else first-registered owner wins** (logged);
  explicit-list vs auto-adopt owned set; a model removed on refresh stops
  resolving (or re-resolves to another owner); discovery-failure path is a
  no-op (last map kept).
- **Config (`test_config.py`).** Parse `backends` from YAML; `BACKENDS_JSON`
  **overrides** the file; per-backend override resolution (present vs
  `None` → global default, incl. per-backend `target_frac` derivation);
  validation failures (zero backends, dup names, **model duplicated across
  backends**, two `default: true`, bad port, bad knob ranges, legacy
  `vllm_host` key / `VLLM_HOST` env no longer honored).
- **Poller (`test_poller.py`).** Per-backend: good poll reanchors **only**
  its own counter; transport error keeps **only its own** state; discovery
  fires first tick + on cadence (fake clock), failure keeps last map;
  **capacity-missing ⇒ no exception, counter stays unanchored, unavailable
  flag set, task still running** (assert the poller does not end).
- **ASGI end-to-end (`test_api.py`).** Two fake vLLMs (distinct
  `/v1/models` + `/metrics`). Routing: `model=A` → backend A, `model=B` →
  backend B, unknown/`"unknown"` → default backend (assert the upstream
  `base_url` per case via the mock transport). Per-backend 429 (fill A's
  pool; A's models reject with A's `Retry-After`; B still allows). **One
  stale / one fresh:** request to the fresh backend is gated; request to the
  stale backend fails open. `GET /v1/models` returns the union with
  `owned_by`. `/healthz` per-backend array. `/metrics` carries per-backend
  labels + `gate_kv_cache_usage_pct` union + `gate_config_info` backend
  structure.
- **Stats (`test_stats.py`).** Per-backend label series + stale-label
  removal; `gate_backend_capacity_unavailable` set/clear;
  `gate_config_info` includes backend structure and omits model lists.
- **Gates.** `ruff format --check`, `ruff check`, `mypy src/`,
  `pytest --cov=gate --cov-fail-under=80`, `bash scripts/local-ci.sh`.
  Coverage must stay ≥ 80% — new `models.py` and the `app`/`config` branches
  are the risk; the ASGI multi-backend cases are the main coverage driver.

## 5. Documentation Updates

- **`README.md`:** new "Multi-backend & model routing" section (config schema
  + mnemonic, auto-adopt vs explicit `models:`, default-backend rule,
  globally-unique model ids, `GET /v1/models`, per-backend metrics/healthz,
  and the **capacity fail-open deviation**); updated architecture diagram (N
  backends, N pollers, `ModelRegistry`); config reference table; **v2
  roadmap** — strike "per-model thresholds and routing" (delivered as
  per-backend policy + per-model routing).
- **`config.example.yaml`:** full multi-backend reference (see §6) with inline
  comments on the mnemonic, auto-adopt, and the global-vs-override split.
- **`AGENTS.md`:** project map adds `models.py` and per-backend state;
  **new known gotchas** — (g) model ids are globally unique across backends
  (decision 9), (h) unmatched/unknown models route to the default backend,
  (i) a backend with a live-but-capacity-less `/metrics` fails open for
  itself (no process exit) + `gate_backend_capacity_unavailable` gauge
  (decision 10, supersedes the exit-1 half of gotcha #10), (j) `/v1/models`
  is gate-local/aggregated (always 200), (k) `vllm_host`/`vllm_port`
  shorthand and `VLLM_HOST`/`VLLM_PORT` are removed, and **gotcha #13
  (zero-config) is superseded**: at least one backend must be configured
  (file or `BACKENDS_JSON`) because the gate has no other way to learn its
  upstreams (decision 6). Test list adds `test_models.py`.
- **`docker-compose.example.yaml`:** `BACKENDS_JSON` example; remove
  `VLLM_HOST`/`VLLM_PORT`; comment for the mounted-config alternative.
- **`docs/plans/multi-backend.md`:** this file.

## 6. Dev-Ops / Project Structure Updates

- **No new runtime dependencies** — `/v1/models` parsing is stdlib `json`;
  `httpx` is already present. Dependency policy respected.
- **CI/release:** unchanged; `local-ci.sh` / `tests.yml` / `release.yml` run
  as-is. The 80% coverage gate is the only new pressure point (addressed by
  §4).
- **`pyproject.toml` / `Dockerfile`:** no changes.
- **Breaking-change note for operators:** configs using `vllm_host`/
  `vllm_port` (or `VLLM_HOST`/`VLLM_PORT`) must migrate to `backends:` /
  `BACKENDS_JSON` — a hard break (unknown-key `ConfigError`), documented in
  README + release notes.

Reference config sketch (for `config.example.yaml`):

```yaml
# Backends — REQUIRED (env override: BACKENDS_JSON, a JSON list; env wins).
# name is your mnemonic: use it to know which model you are configuring.
# models: is OPTIONAL — the served model ids (as exposed by the backend's
# /v1/models, which also labels its vllm:kv_cache_usage_perc series). Omit to
# auto-adopt whatever the backend serves. Model ids must be unique across
# all backends. Exactly one backend may be default: true (fallback for
# unknown models; the first backend when none is flagged). All per-backend
# knobs are optional and default to the global values below.
backends:
  - name: qwen
    host: vllm-qwen
    port: 8000
    models: [qwen3-32b]        # mnemonic + validation anchor (optional)
    default: true
    # Per-backend overrides (all optional — globals below are the defaults):
    # thresholds:
    #   - kv_pct: 80
    #     max_context: 8192
    #     timeout_s: 30
    # target_kv_cache_pct: 80.0
    # token_margin: 1.5
    # retry_min_s: 10
    # retry_max_s: 120
  - name: llama
    host: vllm-llama
    port: 8001
    # no models: => auto-adopt from /v1/models; no overrides => global policy

listen_host: 0.0.0.0
listen_port: 8000
chars_per_token: 4
default_max_tokens: 256
metrics_poll_interval_s: 2.0
stale_after_s: 6.0
model_refresh_interval_s: 30.0   # how often /v1/models is re-fetched per backend

# Global defaults (env overrides as before):
thresholds:
  - kv_pct: 50
    max_context: 4096
    timeout_s: 15
  - kv_pct: 80
    max_context: 1024
    timeout_s: 30
target_kv_cache_pct: 85.0
retry_min_s: 5
retry_max_s: 60
token_margin: 1.25
```

## 7. Verification Checklist

- [ ] `load_config`: `backends` from YAML parses; `BACKENDS_JSON` overrides the file; per-backend `None` fields resolve to globals.
- [ ] Validation rejects: zero backends; duplicate `name`; **model id present in two backends**; two `default: true`; `port` ∉ 1–65535; out-of-range knob; legacy `vllm_host` key / `VLLM_HOST` env.
- [ ] `parse_v1_models`: valid / empty / non-JSON / missing `data` all behave per spec.
- [ ] `ModelRegistry.resolve` returns the right backend; default-wins on collision (logged once); removed models stop resolving.
- [ ] Per-backend poller reanchors **only** its own counter; discovery runs first tick + on cadence; discovery failure keeps the map; discovery never fatal.
- [ ] Capacity-missing on one backend ⇒ **no** process exit, that backend's autoconfig fails open (`auto_unanchored`), `gate_backend_capacity_unavailable{backend}=1`, other backends unaffected; the poller task keeps running (fatal callback not triggered by a capacity condition).
- [ ] `CapacityUnavailableError` deleted from `metrics.py`; all references cleaned up (`app.py` docstrings, `tests/test_poller.py`, `tests/test_api.py`).
- [ ] ASGI: `model` routes to the correct `base_url` (asserted per case); unknown/`"unknown"` ⇒ default backend.
- [ ] Per-backend 429 + `Retry-After` use the right backend's tier/autoconfig values; one stale backend fails open while a fresh one gates.
- [ ] `GET /v1/models` returns the union with `owned_by`, always 200, gate-local.
- [ ] `/healthz` per-backend array (plus `status: "ok"`); `/metrics` per-backend labels + `gate_kv_cache_usage_pct` union + `gate_config_info` backend structure (no model lists).
- [ ] `ruff format --check`, `ruff check`, `mypy src/`, `pytest --cov=gate --cov-fail-under=80` all green; `bash scripts/local-ci.sh` green.
- [ ] README/AGENTS/config.example/docker-compose updated and consistent with the code (esp. the capacity fail-open deviation + the shorthand removal).

## 8. Implementation Order (DAG)

```
[1] models.py ─┐                    [2] config.py ─┐
(parse+registry)│                               │
                └──────────┬─────────────────────┘
                           ▼
       ┌────────────────────┴────────────────────┐
       ▼                                        ▼
[5] poller.py (per-backend + discovery)   [6] stats.py (backend labels)
       └────────────────────┬────────────────────┘
                           ▼
                   [7] app.py (BackendState, routing,
                      /v1/models, healthz, N pollers)
                           ▼
                   [8] main.py (logging)
```

- **Parallel (independent):** `[1] models.py` ∥ `[2] config.py` — no shared
  symbols; build concurrently (`test_models.py` ∥ `test_config.py` with them).
- **Parallel (after 1+2):** `[5] poller.py` ∥ `[6] stats.py` — both depend
  only on 1+2.
- **Sequential:** `[7] app.py` needs 1, 2, 5, 6. `[8] main.py` needs 2, 7.
- **Tests** track their module (interleave); `test_api.py` last (needs app).
  **Docs** (README/AGENTS/config.example/compose) after code is green.
- **Branching/commits (AGENTS.md dev-style):** feature branch
  `feat/multi-backend`; the per-segment pieces are individually small enough
  that sub-branches are optional, but if split, `config` and `models` are the
  parallelizable sub-branches. One commit per segment:
  `models` → `config` → `poller` → `stats` → `app` → `main` → tests →
  docs/packaging. Rebase along the way; per-commit distinct-model review
  (qwen + gemma) before merge/PR.
