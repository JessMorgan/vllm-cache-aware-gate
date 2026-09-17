# vllm-cache-aware-gate

A small, single-purpose **reverse-proxy "gate"** container that sits in front
of one or more OpenAI-compatible inference backends ([vLLM](https://docs.vllm.ai)
or [SGLang](https://docs.sglang.ai)) and protects their KV caches from
over-subscription. It speaks the OpenAI API and,
per request, resolves the body's `model` to the configured backends that serve
it (a **candidate set** — a model id may be served by more than one backend),
picks one of them to proxy to (by a per-model **routing policy**, or the
default backend when the model is unknown/unparseable), and then decides one of
two routes **using that backend's KV state**:

- **Forward** — that backend's KV cache has room for this prompt's estimated
  size, so the request is proxied to it verbatim (including `stream: true`
  SSE).
- **Reject** — not enough room, so the client gets an
  **`HTTP 429` with a `Retry-After` header** telling it how long to wait.

It learns each backend's headroom by polling that backend's Prometheus
`/metrics` endpoint (one background poller per backend) and reading the
backend's KV-cache **usage fraction** and **capacity in tokens** (vLLM:
`vllm:kv_cache_usage_perc` + the `vllm:kv_cache_size_tokens` gauge or the
`kv_cache_size_tokens` label on `vllm:cache_config_info`; SGLang: its own
`sglang:` gauges, new-name with a legacy fallback — see
[SGLang backends](#sglang-backends)); it also polls the backend's
`GET /v1/models` to learn which model ids it serves. It is designed for
**safe, configuration-first, fail-open** operation: a monitoring outage never
blocks inference traffic, and the gate is
a transparent pass-through for the endpoints it proxies.

> **Multi-backend / multi-model routing.** The gate now fronts **multiple**
> inference backends (vLLM or SGLang — the engine is auto-detected, see
> [SGLang backends](#sglang-backends)), configured via a **required**
> `backends:` list. Admission
> state (usage feed, capacity, remaining-KV counter, poller, policy) is
> **per backend**: models on one backend share its KV pool (one inference
> engine), so a shared-KV backend is treated the same across its models. The
> request body's `model` field is the **routing key** — it selects the
> candidate backends for the request and which backend's KV state the
> admission layers evaluate. **Model ids may be duplicated across backends**
> (A/B or canary serving of the same id is the normal multi-candidate case);
> a model owned by 2+ backends is routed per its `routing:` policy, defaulting
> to `round_robin`. The legacy `vllm_host`/`vllm_port` config keys and
> `VLLM_HOST`/`VLLM_PORT` env vars are **removed** (see
> [Configuration](#configuration)).

---

## How it works

```
                  ┌────────────────────────────── gate container ─────────────────────────────┐
 client ──8000─►│  FastAPI app                                                             │
 (OpenAI API)   │   • POST /v1/chat/completions ─┐                                           │
                │   • POST /v1/completions   ───┤  route by the body's `model` →            │
                │   • GET  /v1/models        ───┤  candidate set (2+ backends may          │
                │   • GET  /healthz           ───┘  serve one model) → routing policy       │
                  │                               picks the first candidate                 │
                  │                               (unknown → default backend)               │
                  │                                 │ per-candidate admission               │
                  │                                 │ + pre-stream failover walk            │
                  │                          (allow) ─► httpx ──────┼──► vLLM backend A:8000 │
                  │                          (reject) ─► 429 +      └──► vLLM backend B:8001 │
                  │                                 Retry-After            (proxied           │
                  │                                                transparently)             │
                  │  one background poller per backend ─ every Ns ─► GET <backend>/metrics   │
                  │    (fails → fail-open, keep retrying)                                    │
                  │  every M s ─► GET <backend>/v1/models ─► ModelRegistry                   │
                  │    (discovery fails → keep last known map; never fatal)                   │
                  └───────────────────────────────────────────────────────────────────────────┘
```

### Model routing (per request)

Every request to the two generation endpoints carries a `model` field. The gate
resolves it to a **candidate set** — the backends that serve the model — via
the model→candidate-set map the pollers learn (see
[Model discovery](#metrics-ingestion--model-discovery)):

- **Owned by exactly one backend** → the request is proxied to that backend,
  and **that backend's** admission state (usage feed, capacity, remaining-KV
  counter, policy) is evaluated.
- **Owned by 2+ backends** (duplicate model ids are **legal** — the normal
  multi-candidate case, e.g. A/B or canary serving) → a per-model
  **routing policy** picks which candidate to attempt first (see
  [Routing policies](#routing-policies-per-model)); the candidate `order` is also the
  **failover order** (see [Failover](#pre-stream-failover)). A model with
  multiple owners and no `routing:` entry defaults to `round_robin` over its
  candidates in config order.
- **Unknown or unparseable `model`** (the gate reads an unparseable body as
  `"unknown"`) → the request falls to the **default backend** — the one flagged
  `default: true` in config, or the first `backends:` entry when none is
  flagged — and vLLM returns its own canonical "model not found" error. The
  admission layers still apply on that backend.

Routing is evaluated *before* admission: a 429 from one candidate never
affects the other candidates' decisions.

### Routing policies (per model)

A model owned by 2+ backends is routed by a **per-model routing policy**,
configured in the optional top-level `routing:` section (file-only — there is
no env override):

```yaml
routing:
  <model-id>:
    policy: round_robin   # round_robin | primary_fallback | large_small | even | fill
    order: [a, b]         # candidate backend names — also the failover order
    threshold_tokens: 8000  # required iff policy is large_small; otherwise absent
```

- The section is **optional** — omitted or `{}` is valid (no multi-candidate
  routing configured). Only models served by 2+ backends need an entry; a
  model with multiple owners and **no** entry defaults to `round_robin` over
  its candidates in config order.
- `order` is **both** the candidate order and the failover order. A backend
  named in `order` that does not actually serve the model is a **dead
  candidate** the failover walk skips; each poller logs a WARNING once per
  (model, backend) pair about it (logged, not fatal).
- Every malformed entry is a startup `ConfigError` (`policy` from the
  five-value set; non-empty `order` of configured backend names;
  `threshold_tokens`, an integer ≥ 1, present **iff** `policy` is
  `large_small`).

**Policy reference** (all selection is pure and stateless except where noted;
`ctx` is the estimated context tokens, `None` when the prompt is
unparseable):

| Policy | Selection rule | State | Usage-data sensitivity |
|---|---|---|---|
| `round_robin` | Cycle the candidates evenly — each successful forward advances a per-model index. | Stateful per-model index (advances only on a successful forward). | None — ignores usage data. |
| `primary_fallback` | Start at `order[0]`; the first candidate in `order` that **admits** wins (a reject, transport failure, or 5xx walks to the next — see failover). | None. | None at selection — admission is still evaluated per candidate in the walk. |
| `large_small` | `order[0]` is the "large" backend: `ctx >= threshold_tokens` (inclusive) → `order[0]`, else `order[1]` (clamped to `order[0]` with a 1-candidate order); `ctx` unknown → small side. `threshold_tokens` is **required** for this policy. | None. | None — the size split comes from the estimated `ctx` tokens, not the usage feed. |
| `even` | Route to the candidate that **minimizes the spread** (max − min) of the *projected* KV-usage fractions across **all** candidates: the chosen candidate is projected to `usage_i + ceil(ctx × margin_i) / capacity_i`, every other candidate stays at its current usage. | None. | High — consumes each candidate's usage fraction and capacity. **This is the spread-minimization objective, NOT greedy least-loaded**: a request may go to a fuller candidate if that keeps the pool more even. Unknown candidates (stale/never-fetched feed, or unknown capacity) are ranked **last** (a candidate it cannot project is not the "most even" choice), in config order among themselves. |
| `fill` | Fill in `order`: the first candidate the autoconfig layer would **not** reject wins (the exact admission test: `ctx` unknown → eligible; else `ceil(ctx × margin_i) <= remaining_i`); unknown candidates (stale/never-fetched or unanchored counter) are always eligible and interleaved in config order; all full → 429 (the walk still runs, without proxying, to collect the retries). | None. | High — consumes each candidate's remaining-KV counter. |

> **`fill` ≈ `primary_fallback` on outcomes.** Both run the same failover
> walk (skip-on-reject, skip-on-transport-failure, pre-stream only), and
> `fill`'s prefilter is the *exact* autoconfig admission test, so the two
> reach the same surviving candidate for every request. They differ in
> selection cost (`fill` shortcuts known-rejecting candidates at selection)
> and in one corner — a candidate whose autoconfig layer admits but whose
> **tiered** layer rejects: `fill` keeps it (it passed the prefilter) and the
> walk rejects it on the tiered layer, while `primary_fallback` evaluates it
> in the walk; the final outcome matches either way. The operator intent
> differs ("fill my pools in order" vs "hot primary, backups"), so both are
> kept as distinct policies — the behavioral difference is small.

### Pre-stream failover

For every policy, the candidate `order` is also the **failover order**: the
gate attempts the selected candidate first and walks the rest in order.

- **Pre-stream transport error or upstream 5xx** (before the first response
  byte) → the candidate is skipped and the walk re-runs admission on the next
  one. The charge taken from the failed candidate's remaining-KV counter is
  **rolled back** (re-added via `KvRemaining.add`) — guarded by a per-backend
  anchor sequence: if the candidate's poller re-anchored its counter in
  between, the fresh anchor is authoritative and the charge is dropped rather
  than re-added.
- **A 429 from a candidate also triggers the next** (skip-on-reject).
- **The walk never short-circuits:** every remaining candidate is attempted
  before exhaustion is declared.
- **Once a streamed response has started, errors surface as-is** (no
  failover) — the gate is a transparent proxy (see the
  [proxied endpoints](#proxied-endpoints-v1) note on status propagation).
- Each skip is counted by the
  `gate_routing_failovers_total{model, from, to, reason}` counter
  (`reason` ∈ `transport` | `upstream_5xx` | `reject`).

**Exhaustion rule** (all candidates attempted, none forwarded):

- **Any candidate 429'd** → a single `429` with `Retry-After` = the **max** of
  the candidates' retries; the reported rejector is the max-timeout backend
  (a tie → the first in walk order), and the 429 message names it.
- **Every candidate transport-failed** (no HTTP answer at all) → **502**
  (never 200, never 429 — the request never got a capacity answer); the body
  names the backends tried.
- **A pre-stream 5xx on the last candidate** (no next candidate to skip to) is
  **propagated as-is** — status and body unmasked, exactly like the
  single-backend case — and the request is recorded as *forwarded* with that
  backend (the gate's decision was "forward"; the 5xx is the upstream's
  answer, not the gate's rejection).

  **Precedence:** the 429-exhaustion check runs *before* last-5xx
  propagation, so if an earlier candidate 429'd **and** the last candidate
  returned a pre-stream 5xx, the **429 wins** (the gate's capacity decision is
  more actionable than a raw 5xx) and the 5xx is *not* propagated — last-5xx
  propagation applies only when *no* candidate 429'd.

### Decision logic (per request, per candidate)

Every request is then evaluated by **two admission layers, AND-combined, on the
candidate's state**: the request forwards only if **both** allow. The first is
the optional **tiered** policy below (evaluated with **that candidate
backend's** thresholds); the second is the always-on
**[autoconfig](#autoconfig-always-on)** layer (evaluated with **that candidate
backend's** remaining-KV counter and policy). The combined rule, the 429
`Retry-After`, and the rejector-aware message are described in [The 429
contract](#the-429-contract). The other candidates are unaffected — a stale
feed or full pool on one candidate does not block traffic to the others (and a
candidate reject is a *skip* to the next one, see
[Pre-stream failover](#pre-stream-failover)).

#### Tiered layer (optional)

1. **Estimate context tokens** `C` from the request body:
   - `prompt_tokens = ceil(prompt_chars / chars_per_token)` (default 4), where
     `prompt_chars` is the total character count of the prompt text
     (`messages[*].content[*]` for chat; `prompt` string/array for completions).
   - `C = prompt_tokens + headroom`, where `headroom = max_tokens` if present in
     the request, else `default_max_tokens` (default 256).
   - This is a **heuristic** — it deliberately over-estimates so the gate is
     conservative. There is no tokenizer dependency in v1.
2. **Read current usage** from the candidate backend's in-memory cache:
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
to a target usage %. It needs no per-request configuration (a `backends` list
is the only hard requirement, and the knobs have sane defaults — see the
[knob table](#autoconfig-knobs)). The two layers
are **AND-combined**: a request forwards only if the tiered layer *and* the
autoconfig layer both allow (with no tiers configured the tiered layer always
allows, so the autoconfig layer is the sole gate).

#### The remaining-KV counter

The autoconfig layer keeps an in-memory counter of estimated remaining KV
tokens (`KvRemaining`) **per backend** — models on one backend share its KV
pool, so a shared-KV backend is treated the same across its models:

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
- **Rolled back on a pre-stream failover.** If a forwarded request's candidate
  fails pre-stream (transport error or upstream 5xx before the first response
  byte), the charge is re-added to that candidate's counter (via
  `KvRemaining.add`) when the walk skips to the next candidate — unless the
  poller re-anchored the counter in the meantime (the fresh anchor is
  authoritative; the charge is dropped, see
  [Pre-stream failover](#pre-stream-failover)).
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
- **Staleness gate (fail-open override, per candidate).** If a candidate's
  usage feed is stale or never fetched, **both** layers fail open for it
  regardless of the counter's value (reason `metrics_unavailable`) — a metrics
  outage never blocks traffic, and a stale candidate is **not** skipped (it
  remains reachable through the failover walk). The counter still takes its
  subtraction (bookkeeping continues) but cannot reject while the feed is
  stale.
- **A reject is a skip, not a stop.** In a multi-candidate walk a candidate
  reject moves to the next candidate (skip-on-reject); the recorded rejects are
  what the exhaustion rule (429-max / 502) builds its final answer from (see
  [Pre-stream failover](#pre-stream-failover)).

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

These are the **global defaults**. Each backend may override any of them (and
the tiered `thresholds`) per-backend — a `None`/omitted per-backend value
falls back to the global default (env overrides still set only the global
defaults). The effective per-backend values are logged at startup with an
`(override)` mark (see [Operational notes](#operational-notes)).

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

#### Capacity-missing caveat (per-backend fail-open)

- **A live backend whose `/metrics` lacks a usable KV-cache capacity**
  (the *detected engine's* capacity gauges — `vllm:kv_cache_size_tokens` / the
  `kv_cache_size_tokens` label on `vllm:cache_config_info` for vLLM;
  `sglang:kv_cache_total_tokens` / `sglang:max_total_num_tokens` for SGLang —
  see [SGLang backends](#sglang-backends)) cannot anchor **that
  backend's** counter. This is
  **not fatal**: the poller logs an error, the counter stays unanchored (so
  that backend's autoconfig layer fails open), and the
  `gate_backend_capacity_unavailable{backend}` gauge is set to `1` for
  alerting. The poller keeps looping and **all other backends are unaffected**
  — killing the whole proxy because one backend is misconfigured would take
  down the healthy ones. Alert on the gauge rather than expecting a crash.
- **An unreachable or stale feed is never fatal** — the gate **fails open**
  (forwards) and keeps retrying. A metrics outage never blocks traffic
  (invariant #1).

#### Minimal run

The gate needs **at least one configured backend** — there is no zero-config
mode (the gate has no other way to learn its upstream):

```sh
docker run -d --name gate \
  --network mynet \
  -p 8000:8000 \
  -e BACKENDS_JSON='[{"name":"vllm","host":"vllm","port":8000,"default":true}]' \
  vllm-gate
```

`backends` (file key) or `BACKENDS_JSON` (env) is **required** — an
empty/absent list fails at startup with `at least one backend is required`.
There is no `--auto` (or any) CLI flag; argv is ignored. A legacy invocation
that passes `--auto` (or any other argument) still starts normally and behaves
identically — autoconfig is always on.

### Metrics ingestion & model discovery

- **One background poller per configured backend.** Each poller polls
  `GET http://<host>:<port>/metrics` every `metrics_poll_interval_s` (default
  2s) with **one GET per tick** and, from the same body, updates
  **that backend's** caches: the latest `vllm:kv_cache_usage_perc` value (plus
  its fetch timestamp and per-model breakdown) and the KV-cache capacity (the
  `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
  `vllm:cache_config_info`). On a good poll it also re-anchors **that
  backend's** remaining-KV counter (see
  [Autoconfig (always on)](#autoconfig-always-on)).
- vLLM may emit one series per `model_name`. **Per backend, the gate takes the
  max across all series** for both the usage and the capacity gauges —
  conservative, and correct because all models on one backend share its KV
  pool.
- **Model discovery.** When model discovery is wired (always, in the app),
  each poller also fetches that backend's `GET /v1/models` — on the first tick
  and then every `model_refresh_interval_s` (default 30s) — and reconciles the
  backend's owned model set in the shared model→candidate-set registry. The
  **explicit `models:` list (if present) always wins** over discovery; a
  failed fetch or unparseable body **keeps the last known map** (discovery is
  auxiliary and never fatal), logging a warning only on a state change.
- **Fail-open rules** (a monitoring outage must never take down inference),
  per backend:
  - On any fetch/parse error the last good value is kept and the poller
    retries.
  - If the cached value is older than `stale_after_s` (default 3× the poll
    interval) it is treated as *unknown* → the request is **allowed** (for
    that backend) and a warning is logged.
- **Capacity-missing (per-backend fail-open):** an observed (HTTP 200) body
  that lacks a usable KV-cache capacity cannot anchor that backend's counter —
  the poller logs an error, sets the
  `gate_backend_capacity_unavailable{backend}` gauge, and keeps looping (see
  the [capacity-missing caveat](#capacity-missing-caveat-per-backend-fail-open)).
  A merely unreachable backend is never fatal.

### SGLang backends

A `backends:` entry's `host`/`port` may point at a vLLM **or** an SGLang
server — **no config change** is needed: the gate **auto-detects the engine
from each backend's `/metrics` body** (fixed precedence, first match wins —
by the usage gauge's *family-name presence*, not by sample values):
`vllm:kv_cache_usage_perc` → **vLLM**; else `sglang:kv_cache_usage_perc` →
**SGLang**; else `sglang:token_usage` → **SGLang**; else **unknown** (a 200
body with no recognizable KV gauge, or an unparseable body). There is no
config knob for the engine, and model discovery (`GET /v1/models`) is
unchanged — SGLang's is OpenAI-schema.

What the gate reads per engine (usage + capacity, new→legacy):

| | vLLM | SGLang (newer) | SGLang (older / legacy) |
|---|---|---|---|
| usage fraction | `vllm:kv_cache_usage_perc` | `sglang:kv_cache_usage_perc` | `sglang:token_usage` |
| capacity tokens | `vllm:kv_cache_size_tokens` (or the `kv_cache_size_tokens` label on `vllm:cache_config_info`) | `sglang:kv_cache_total_tokens` | `sglang:max_total_num_tokens` |
| per-model label | `model_name` | `model_name` | `model_name` |

When both a newer and a legacy SGLang gauge are present, the newer one is
read.

> **Legacy `token_usage` caveat.** SGLang's legacy `sglang:token_usage` is a
> bottleneck across **all** token pools (`max(full, swa, mamba)`) and
> **over-reports KV pressure on hybrid-SSM models**. The newer
> `sglang:kv_cache_usage_perc` is KV-pools-only and is preferred — it is read
> first whenever present.

> **SGLang must be launched with `--enable-metrics`.** SGLang only serves
> `/metrics` when started with `--enable-metrics`. Without it the backend
> **fails open** (traffic is never blocked — the feed is simply stale) and
> `gate_metrics_endpoint_unavailable{backend}` is set to `1`; the gate logs a
> `--enable-metrics` warning **once per transition** into the unavailable
> state (a purely unreachable server is a transport failure, not this — watch
> `gate_metrics_fresh`).

Two gauges surface the per-backend engine state (see
[Metric reference](#metric-reference)):

- `gate_backend_engine{backend,engine}` — `1` for the currently detected
  engine (`engine` ∈ `vllm` \| `sglang` \| `unknown`), `0` for the other
  engines; `unknown=1` is the initial state before the first detection.
- `gate_metrics_endpoint_unavailable{backend}` — `0/1`; `1` while the
  backend's `/metrics` is reachable-but-unrecognizable (non-200, or a 200
  body with no recognizable KV gauge), `0` once a real engine is detected.

The admission logic itself is engine-agnostic: both admission layers and the
failover walk consume the same per-backend usage/capacity/remaining-KV state
regardless of which engine produced it.

---

## Quickstart

### Build

```sh
docker build -t vllm-gate .
```

### Run (pointing at one or more vLLM backends)

```sh
docker run -d --name gate \
  --network mynet \
  -p 8000:8000 \
  -e BACKENDS_JSON='[{"name":"qwen","host":"vllm-qwen","port":8000,"default":true},{"name":"llama","host":"vllm-llama","port":8001}]' \
  -e THRESHOLDS_JSON='[{"kv_pct":50,"max_context":4096,"timeout_s":15},{"kv_pct":80,"max_context":1024,"timeout_s":30}]' \
  vllm-gate
```

`BACKENDS_JSON` is **required** (a JSON list of backend objects — it replaces
any file `backends:` entirely; see [the `backends` entry
contract](#backends-entry-contract-required)). `THRESHOLDS_JSON` is
**optional** — it adds the optional tiered policy on top of the always-on
autoconfig layer. The gate needs at least one backend and no CLI — see the
[minimal run](#minimal-run).

Then point your OpenAI client at the gate instead of vLLM — the `model` field
selects the backend the request is routed to:

```sh
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}]}'

# the gate's own aggregate model list (model → owning backend(s))
curl http://localhost:8000/v1/models
# 200 {"object":"list","data":[{"id":"qwen3-32b","object":"model","owned_by":"qwen"},
#                               {"id":"llama-70b","object":"model","owned_by":"llama"}]}
# A model served by 2+ backends appears once with owned_by as a LIST of backend
# names (single-owner models keep the scalar form).
```

### Compose

See [`docker-compose.example.yaml`](docker-compose.example.yaml) for a
reference stack (gate + two vLLM placeholder backends + a `BACKENDS_JSON`
example).

### Health

```sh
curl http://localhost:8000/healthz
# 200 {"status":"ok",
#      "backends":[{"name":"qwen","metrics_age_s":1.2,"kv_usage":0.42,
#                   "kv_cache_capacity_tokens":100000,"kv_cache_remaining_tokens":35000},
#                  {"name":"llama","metrics_age_s":1.8,"kv_usage":0.31,
#                   "kv_cache_capacity_tokens":200000,"kv_cache_remaining_tokens":90000}],
#      "models":[{"id":"qwen3-32b","owned_by":"qwen","routing":"round_robin"},
#                {"id":"llama-70b","owned_by":"llama","routing":"round_robin"}]}
# A multi-owner model would show "owned_by":["qwen","qwen-fast"] and the
# policy in effect for it (its routing: entry, or round_robin by default).
```

`/healthz` always returns 200 for liveness; the body carries readiness info:
`status`, a per-backend `backends` array (each entry with `name`,
`metrics_age_s` — age of that backend's last successful metrics scrape,
`kv_usage` — that backend's last cached usage **fraction**, `null` if never
fetched, `kv_cache_capacity_tokens` — that backend's last cached KV-cache
capacity (the detected engine's capacity gauges — see
[SGLang backends](#sglang-backends)), `null` if never observed, and
`kv_cache_remaining_tokens` — that backend's autoconfig counter's current
value, `null` if never anchored), plus `models` (the known model ids with
`owned_by` — a scalar name when single-owner, a **list** of names when the
model is served by 2+ backends — and `routing` — the routing policy in effect
for the model, its `routing:` entry or `round_robin` by default). Suitable for
a Docker `HEALTHCHECK` or orchestrator liveness probe.

---

## Configuration

Configuration is a single YAML file mounted at `/etc/gate/config.yaml`, with
env-var overrides for the essentials. A reference file is provided at
[`config.example.yaml`](config.example.yaml).

### Schema

```yaml
# env overrides: BACKENDS_JSON, LISTEN_HOST, LISTEN_PORT, THRESHOLDS_JSON,
# TARGET_KV_CACHE_PCT, RETRY_MIN_S, RETRY_MAX_S, TOKEN_MARGIN,
# MODEL_REFRESH_INTERVAL_S, LOG_LEVEL

# REQUIRED — at least one backend (name + host + port). The legacy
# vllm_host/vllm_port keys and VLLM_HOST/VLLM_PORT env vars are REMOVED.
# A model id MAY be listed by more than one backend (duplicate model ids are
# legal — the routing: section below decides which backend gets a request).
backends:
  - name: qwen          # unique, non-empty; your mnemonic
    host: vllm-qwen
    port: 8000
    default: true       # at most one backend may be default (fallback for
                        # unknown models; the first entry if none is flagged)
    # models: [qwen3-32b]   # optional mnemonic + validation anchor; omit to
                            # auto-adopt from the backend's GET /v1/models
    # Per-backend overrides (all optional — the global values below are the
    # defaults):
    # thresholds:
    #   - kv_pct: 80
    #     max_context: 8192
    #     timeout_s: 30
    # target_kv_cache_pct: 80.0
    # token_margin: 1.5
    # retry_min_s: 10
    # retry_max_s: 120
  - name: qwen-fast
    host: vllm-qwen-fast
    port: 8001
    models: [qwen3-32b] # same model id as 'qwen' above (legal — duplicate)
    # no overrides => global policy
  - name: llama
    host: vllm-llama
    port: 8002
    # no models: => auto-adopt from /v1/models; no overrides => global policy

# OPTIONAL — per-model routing for model ids served by 2+ backends (file-only;
# no env override). Omitted or {} is valid. order = candidate AND failover
# order; policy is one of the five (see Routing policies). threshold_tokens is
# required iff policy is large_small. A multi-owner model with no entry
# defaults to round_robin.
# routing:
#   qwen3-32b:
#     policy: fill
#     order: [qwen, qwen-fast]
#   some-other-model:
#     policy: large_small
#     order: [big-box, small-box]
#     threshold_tokens: 8000

listen_host: 0.0.0.0
listen_port: 8000
log_level: INFO             # debug | info | warning | error | critical (env: LOG_LEVEL)
metrics_poll_interval_s: 2.0
stale_after_s: 6.0
model_refresh_interval_s: 30.0  # per-backend /v1/models discovery cadence
chars_per_token: 4
default_max_tokens: 256

# optional global default — the tiered policy (env override: THRESHOLDS_JSON,
# a JSON list). Autoconfig runs regardless; with no tiers the tiered layer
# always allows. A backend may override this per-backend.
thresholds:
  - kv_pct: 50
    max_context: 4096
    timeout_s: 15
  - kv_pct: 80
    max_context: 1024
    timeout_s: 30

# global defaults for the autoconfig (always on) knobs — all optional,
# defaults shown. A backend may override any of them per-backend.
target_kv_cache_pct: 85.0
retry_min_s: 5
retry_max_s: 60
token_margin: 1.25
```

### Reference

| Key | Type | Default | Env override | Meaning |
|---|---|---|---|---|
| `backends` | list | *(required — no default)* | `BACKENDS_JSON` | **REQUIRED** list of inference-engine backends (vLLM or SGLang — the engine is auto-detected, see [SGLang backends](#sglang-backends)). An empty/absent list fails with `at least one backend is required`. `BACKENDS_JSON` (a JSON list) **replaces** the file's `backends:` entirely (env wins). Model ids **may be duplicated** across backends (the normal multi-candidate case). See the [`backends` entry contract](#backends-entry-contract-required) below. |
| `routing` | mapping | *(absent)* | — | **Optional, file-only** (no env override) per-model routing for model ids served by 2+ backends: `model → {policy, order[, threshold_tokens]}`. Omitted or `{}` is valid; every malformed entry is a `ConfigError`. A multi-owner model with no entry defaults to `round_robin` over its candidates in config order. See [Routing policies](#routing-policies-per-model) and the [`routing` entry contract](#routing-entry-contract-optional). |
| `listen_host` | string | `0.0.0.0` | `LISTEN_HOST` | Bind address for the gate. |
| `listen_port` | int | `8000` | `LISTEN_PORT` | Port the gate listens on. |
| `log_level` | string | `INFO` | `LOG_LEVEL` | Logging level for the gate's own loggers **and** uvicorn: one of `debug`, `info`, `warning`, `error`, `critical` (case-insensitive; default `INFO`). Anything else is a `ConfigError`. |
| `metrics_poll_interval_s` | float | `2.0` | — | How often **each backend's** `/metrics` is polled (one poller per backend, one GET per tick). |
| `stale_after_s` | float | `6.0` | — | Age after which a backend's cached value is treated as unknown (fail-open for that backend). |
| `model_refresh_interval_s` | float | `30.0` | `MODEL_REFRESH_INTERVAL_S` | How often **each backend's** `GET /v1/models` is re-fetched (model discovery; the first fetch is immediate). |
| `chars_per_token` | int | `4` | — | Heuristic divisor for prompt token estimation (global — not per-backend). |
| `default_max_tokens` | int | `256` | — | Output headroom added when the request omits `max_tokens` (global — not per-backend). |
| `thresholds` | list | *(absent)* | `THRESHOLDS_JSON` | **Optional** global-default tiered policy. With no tiers the tiered layer always allows; autoconfig runs regardless. A backend may override it per-backend. See below. |
| `target_kv_cache_pct` | number | `85.0` | `TARGET_KV_CACHE_PCT` | Autoconfig global default: target KV-cache **percentage (0–100)** the remaining-KV counter anchors up to. Finite, `(0, 100]`. Per-backend overridable. |
| `retry_min_s` | int | `5` | `RETRY_MIN_S` | Autoconfig global default: minimum `Retry-After` seconds. `>= 1`. Per-backend overridable. |
| `retry_max_s` | int | `60` | `RETRY_MAX_S` | Autoconfig global default: maximum `Retry-After` seconds. `>= retry_min_s`. Per-backend overridable. |
| `token_margin` | number | `1.25` | `TOKEN_MARGIN` | Autoconfig global default: conservatism multiplier on estimated context tokens. Finite, `>= 1.0`. Per-backend overridable. |

> **Removed:** the legacy `vllm_host`/`vllm_port` config keys and the
> `VLLM_HOST`/`VLLM_PORT` env vars (the single-backend shorthand) are **gone**.
> A config file containing `vllm_host` or `vllm_port` fails at startup with a
> migration-hint error naming `backends`/`BACKENDS_JSON`. A legacy `VLLM_HOST`
> / `VLLM_PORT` env var is simply **ignored** (it is no longer read).

### `backends` entry contract (required)

`backends` is a **required, non-empty** list. Each entry is an object with:

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `name` | string | non-empty; **unique across backends** | Operator-chosen mnemonic (used in logs, `/healthz`, `/metrics` labels, the 429 message). |
| `host` | string | non-empty | Hostname of the inference-engine server — vLLM **or** SGLang (API + `/metrics` + `/v1/models`; the engine is auto-detected, see [SGLang backends](#sglang-backends)). |
| `port` | int | `1 <= port <= 65535` | Port of the inference-engine server. |
| `models` | list of strings | *(optional)*; **no duplicates within a backend**, but **duplicates across backends are legal** | The backend's owned model ids (mnemonic + validation anchor). **Omit (or leave empty) to auto-adopt** the served set from the backend's `GET /v1/models`. When present and non-empty it **wins** over discovery. A model id listed by 2+ backends is the normal multi-candidate case — the `routing:` section decides which backend gets a request. |
| `default` | bool | *(optional, default `false`)*; **at most one `true` across backends** | The fallback backend for unknown/unparseable models. If none is flagged, the **first** entry is the default. |
| `thresholds` | list | *(optional)*; same entry contract as the global `thresholds` | Per-backend tiered policy. `null`/omitted ⇒ the global `thresholds`. |
| `target_kv_cache_pct` | number | *(optional)*; finite, `(0, 100]` | Per-backend override of the global target. Omitted ⇒ global. |
| `token_margin` | number | *(optional)*; finite, `>= 1.0` | Per-backend override. Omitted ⇒ global. |
| `retry_min_s` | int | *(optional)*; `>= 1` | Per-backend override. Omitted ⇒ global. |
| `retry_max_s` | int | *(optional)*; `>= retry_min_s` | Per-backend override. Omitted ⇒ global. (The **effective** pair — override falling back to global — must satisfy `>=`.) |

**Validation (fail-fast at startup):** `backends` non-empty; `name`s unique and
non-empty; `port` in `[1, 65535]`; no **duplicate model ids within one
backend's** `models:` list (**duplicate model ids *across* backends are
**legal** — a model listed in 2+ backends is the normal multi-candidate case,
routed per its `routing:` policy, defaulting to `round_robin`); at most one
`default: true`; and each per-backend knob, when present, satisfies the same
ranges as the global default (the effective `retry_max_s >= retry_min_s` rule
is checked after the per-backend fallback is applied). Any violation aborts
startup with a clear log line.

> **Model ids may be duplicated across backends.** (Supersedes the earlier
> rule that model ids were globally unique across backends.) A model id served
> by 2+ backends is the normal multi-candidate case: the request is routed by
> its `routing:` policy (default `round_robin`), and because the backend is no
> longer recoverable from the model alone, the traffic metrics
> (`gate_requests_total`, `gate_request_ctx_tokens`) carry a `backend` label
> alongside `model` (see [Observability](#observability)).

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

### `routing` entry contract (optional)

`routing` is a top-level **mapping of model id → entry**, and is **optional**:
omitted or `{}` is valid (no multi-candidate routing configured). It is
**file-only** — there is no env override for it (the `BACKENDS_JSON`,
`THRESHOLDS_JSON`, and knob env vars are unchanged). Each entry is an object
with:

| Field | Type | Constraint | Meaning |
|---|---|---|---|
| `policy` | string | one of `round_robin`, `primary_fallback`, `large_small`, `even`, `fill` | The routing policy for the model (see [Routing policies](#routing-policies-per-model)). |
| `order` | list of strings | non-empty; every name is a configured backend; no duplicates | The ordered candidate list — **also the failover order** (first entry tried first). A backend named here that does not serve the model is a dead candidate the walk skips (each poller warns once per (model, backend) pair). |
| `threshold_tokens` | int | `>= 1`; **present iff `policy` is `large_small`** | The large/small size split: `ctx >= threshold_tokens` (inclusive) → `order[0]` (the "large" backend), else the remaining candidates in `order`. |

**Validation (fail-fast at startup):** every malformed entry is a `ConfigError`
— a `policy` outside the five-value set, a missing/empty `order`, an `order`
naming a backend that is not configured, or `threshold_tokens` present for a
non-`large_small` policy / absent for `large_small`. Whether the `order`
backends *serve* the model is **not** a config-time check: it is verified
against the discovered model sets at startup (after the first discovery tick)
and violations are **logged, not fatal** (a dead candidate is simply skipped
by the walk). A model with multiple owners and no entry defaults to
`round_robin` over its candidates in config order.

> **At most one `routing:` entry per model — via last-wins, not a
> `ConfigError`.** `yaml.safe_load` cannot detect duplicate mapping keys (it
> silently keeps the last), so a file with two entries for the same model id
> loads as the last entry's spec — no error, no visibility. The "at most one
> entry per model" rule therefore holds in practice via last-wins. This is
> safe in outcome (last-wins still yields a single well-formed entry, never a
> fail-open); detecting duplicates would require a custom loader / raw-text
> pre-parse, which is out of scope.

---

## The 429 contract

When a request is rejected by either admission layer of the **routed
candidate**, the gate responds. The `Retry-After` is the **combined** value: when
**both** layers reject it is `max(tier timeout, autoconfig retry)`; when only
one rejects it is that layer's value. An autoconfig `Retry-After` is always an
**integer within `[retry_min_s, retry_max_s]`** (the candidate's effective
values — per-backend override, else the global default), and `remaining <= 0`
(cache full / over-committed) yields exactly `retry_max_s`. Only the routed
candidate rejects at selection: a request to another, less-full candidate is
unaffected.

> **Multi-candidate exhaustion.** In a multi-candidate walk, a candidate reject
> is a **skip** (the next candidate is tried — see
> [Pre-stream failover](#pre-stream-failover)). Only when **every** candidate
> has been attempted does the gate produce the final answer:
> - **Any candidate 429'd** → a single `429` with `Retry-After` = the **max**
>   of the candidates' retries; the reported rejector is the max-timeout
>   backend (a tie → the first in walk order).
> - **Every candidate transport-failed** (no HTTP answer) → **502** (never
>   200/429).
> - **A pre-stream 5xx on the last candidate** is **propagated as-is**
>   (unmasked) and recorded as forwarded with that backend — but only when
>   *no* candidate 429'd (the 429-exhaustion check runs first, so an earlier
>   429 supersedes a last-candidate 5xx).

**Tiered rejector** (the tiered layer is the reported rejector):

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
Content-Type: application/json

{
  "error": {
    "message": "backend qwen (model qwen3-32b): KV cache too full for a ~3800-token request (usage 84.0% >= 80% tier; max 1024). Retry in 30s.",
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
    "message": "backend qwen (model qwen3-32b): KV cache headroom exhausted for a ~25000-token request (usage 50.0%, ~10000 of 100000 tokens remaining up to the 85% target). Retry in 21s.",
    "type": "cache_pressure",
    "code": "kv_cache_too_full"
  }
}
```

- The **reported rejector** is the layer with the higher timeout when both
  reject (a tie reports the tiered layer); the message names the **routed
  backend and the request's `model`** (`backend <name> (model <id>): ...`),
  followed by the rejector-specific text. On a multi-candidate 429 the
  reported rejector is the **max-timeout backend** (a tie → the first in walk
  order) and its `Retry-After` is the max over the candidates'.
- The body is a small OpenAI-style error envelope so standard SDKs surface it
  cleanly.

---

## Proxied endpoints (v1)

| Method & path | Behavior |
|---|---|
| `POST /v1/chat/completions` | Gated (per the routed candidate), then proxied to that candidate's backend (SSE streaming supported); pre-stream failover to the next candidate on a candidate's transport error, 5xx, or 429. |
| `POST /v1/completions` | Same as above. |
| `GET /v1/models` | Gate-local **aggregate** of the known models of all backends — one entry per model with an `owned_by` field: a **list** of backend names when the model is served by 2+ backends, a **scalar** name when single-owner (always 200, never 429, **not** proxied). |
| `GET /healthz` | Liveness (always 200; per-backend readiness info + the known models — `owned_by` and the `routing` policy in effect — in body). |
| `GET /metrics` | Gate-local Prometheus stats (see [Observability](#observability)); **not** proxied. |
| anything else | `404`. |

The proxy forwards method, path, query, headers (minus hop-by-hop), and body
verbatim to the **routed candidate's** `base_url`, and propagates upstream
status codes — a vLLM 5xx surfaces as a 5xx, never masked by the gate (a
pre-stream 5xx *before* the first response byte is a failover skip to the next
candidate, and the last candidate's pre-stream 5xx is propagated as-is; once a
streamed response has started, errors surface as-is — see
[Pre-stream failover](#pre-stream-failover)). The backends' own `/v1/models`
endpoints are **not** reachable through the gate — the gate's own
`GET /v1/models` is the only model-listing endpoint on the gate.

---

## Observability

The gate exposes its own stats at `GET /metrics` — HTTP 200 with
`Content-Type: text/plain; version=0.0.4; charset=utf-8`, body in the
Prometheus text exposition format (the same shape vLLM emits). It is a
gate-local endpoint like `/healthz`, **not proxied** — do not confuse the two:
the per-backend pollers scrape **each backend's** `/metrics` (on the
backend's port — vLLM or SGLang), while the gate's `GET /metrics` (on the gate's
port) serves
the gate's own `gate_*` metrics. They are different endpoints and different
Prometheus scrape targets.

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
| `gate_requests_total` | Counter | `endpoint` (`chat_completions` \| `completions`), `model` (the request body's `model` field, else `unknown`), `result` (`forwarded` \| `rejected`), `backend` (the final backend: the forwarder, the max-timeout rejector on 429 exhaustion, or the sentinel `none` on 502 exhaustion) | Generation requests the gate decided, scoped to the two generation endpoints (404s, `/healthz`, and `/metrics` are not counted). With duplicate model ids legal, the `backend` label is what disambiguates which backend the request landed on. |
| `gate_request_ctx_tokens` | Histogram | `model`, `result`, `backend` | Estimated context tokens (prompt + headroom) of each request. Median via `histogram_quantile(0.5, ...)`. Buckets: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144. |
| `gate_routing_failovers_total` | Counter | `model`, `from`, `to`, `reason` (`transport` \| `upstream_5xx` \| `reject`) | Pre-stream failover skips: a candidate rejected (429), transport-failed, or returned a pre-stream upstream 5xx, so the walk moved to the next candidate. |
| `gate_kv_cache_usage_pct` | Gauge | `model_name` (vLLM's `model_name` label; `default` when the series is unlabeled) | Current KV-cache fill **percentage (0–100)** per served model (vLLM's real metric, converted from the fraction). The gate **unions all backends' per-model breakdowns** on each render; a key can repeat for a model id served by 2+ backends or for the synthetic `default` label — the max on a repeated key wins (conservative). |
| `gate_kv_cache_remaining_tokens` | Gauge | `backend` | The gate's **own live estimate** of remaining KV-cache tokens up to that backend's target (post-subtraction, pre-reanchor); `NaN` until the first anchor. Distinct from `gate_kv_cache_usage_pct` — this is the gate's estimate, not vLLM's real usage. |
| `gate_backend_capacity_unavailable` | Gauge | `backend` | `1` if that backend's observed (HTTP 200) `/metrics` body lacks a usable KV-cache capacity (its autoconfig layer fails open — see the [capacity-missing caveat](#capacity-missing-caveat-per-backend-fail-open)), `0` otherwise. Alert on this. |
| `gate_backend_engine` | Gauge | `backend`, `engine` (`vllm` \| `sglang` \| `unknown`) | The **currently detected inference engine** per backend: `1` for the detected engine, `0` for the other two engines (`unknown=1` is the initial state before the first detection — the gauge is always fully populated and scrape-able). The engine is auto-detected from the `/metrics` body; see [SGLang backends](#sglang-backends). |
| `gate_metrics_endpoint_unavailable` | Gauge | `backend` | `1` while that backend's `/metrics` endpoint is reachable-but-unrecognizable (non-200, or a 200 body with no recognizable KV-cache gauge — e.g. SGLang launched without `--enable-metrics`), `0` once a real engine is detected. A purely unreachable server is *not* this (watch `gate_metrics_fresh`). Alert on this. |
| `gate_config_info` | Info | one label per config field + `thresholds_json` + `backends_json` + `routing_json` | The loaded configuration, set once at startup. `backends_json` serializes the backend structure (per backend: `name`, `host`, `port`, `default`, and which knobs are overridden per-backend) but **not** the `models` lists (discovered data would go stale). `routing_json` serializes the `routing:` section as `[[model, policy, [order...]], ...]` (`[]` when absent — startup-static, for operator audit). |
| `gate_metrics_fresh` | Gauge | `backend` | `1` if that backend's metrics feed is fresh, `0` if stale or never fetched. |
| `gate_metrics_age_s` | Gauge | `backend` | Seconds since the last successful `/metrics` fetch for that backend; `NaN` if never fetched. |

**Two model namespaces, kept distinct.** `model` is the OpenAI request body's
`model` field (the client's model name — **the routing key** that selects the
candidate backends). `model_name` is vLLM's served-model label. They are
different strings on different metrics with different label names; do not
conflate them in queries or dashboards. (Because a model id may be served by
2+ backends, the traffic metrics additionally carry a `backend` label — the
backend is no longer recoverable from `model` alone.)

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
# the gate's own live estimate of remaining KV tokens per backend (NaN until anchored)
gate_kv_cache_remaining_tokens
# per-backend feed health / age
gate_metrics_fresh
gate_metrics_age_s
# alert: which backend cannot anchor its counter (capacity missing)?
gate_backend_capacity_unavailable == 1
# per-backend detected engine (1 for the detected engine: vllm | sglang | unknown)
gate_backend_engine
# alert: which backend's /metrics is reachable but unrecognized (e.g. SGLang without --enable-metrics)?
gate_metrics_endpoint_unavailable == 1
# forwarded requests per model per backend
sum by (model, backend) (rate(gate_requests_total{result="forwarded"}[5m]))
# failover skips (why did requests bounce between candidates?)
sum by (model, reason) (rate(gate_routing_failovers_total[5m]))
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

- **Fail-open is intentional, per backend.** If you stop or lose one backend's
  `/metrics` endpoint, the gate keeps forwarding requests routed to that
  backend (it cannot measure headroom, so it does not block); the other
  backends are unaffected. Watch the warning logs and the per-backend
  `metrics_age_s` / `kv_usage` in `/healthz`.
- **Capacity-missing is per-backend fail-open (not fatal).** A *live* backend
  whose `/metrics` body lacks a usable KV-cache capacity (the detected
  engine's capacity gauges — see [SGLang backends](#sglang-backends)) cannot
  anchor **that backend's** remaining-KV counter: the gate logs an error, that
  backend's autoconfig layer fails open,
  and the `gate_backend_capacity_unavailable{backend}` gauge is set to `1`.
  The process does **not** exit and the other backends keep gating normally.
  An unreachable or stale feed is likewise never fatal — that path fails open.
  See the [capacity-missing
  caveat](#capacity-missing-caveat-per-backend-fail-open).
- **Startup log.** On a successful config load the gate logs **one INFO line
  per backend** — `backend 'qwen' (vllm-qwen:8000) [default]: 2 tier(s), target
  85.0%, margin 1.25, retry 5-60s` — where the tier count is the effective
  tiered policy for that backend (`N tier(s)` or `none — autoconfig only`) and
  the four autoconfig knobs carry an `(override)` mark when the value is
  per-backend (an unmarked value is the global default). It then logs **one
  INFO line per `routing:` entry** — `routing 'qwen3-32b': policy=fill
  order=['qwen', 'qwen-fast']` (plus the `threshold_tokens` for
  `large_small`); an empty `routing:` section logs nothing.
- **Estimation is conservative.** The gate over-estimates token counts, so it
  may reject a request that vLLM would actually have fit. Tune
  `chars_per_token` and `default_max_tokens` to your workload if you see
  over-rejection.
- **No auth, no TLS termination (v1).** The gate trusts the deployment network
  and is a drop-in front of vLLM. Put it behind your existing ingress/auth if
  you need it.
- **Logging.** Level is configurable via the `log_level` config key /
  `LOG_LEVEL` env var (one of `debug`, `info`, `warning`, `error`,
  `critical`, case-insensitive; default `INFO`), applied to the gate's own
  loggers and uvicorn. Structured `logging` at the configured level: one line
  per reject decision
  (`reject backend=… reason=… usage_pct=… ctx_tokens=… retry_after=…` — the
  backend is the routed candidate), one line per pre-stream failover skip
  (`routing_failover model=… from=… to=… reason=…`, `reason` ∈ `transport` /
  `upstream_5xx` / `reject`), poller warnings (failed fetches, failed model
  discovery, the dead-candidate `routing:` warning once per (model, backend)
  pair, the capacity-missing error), and the per-backend and per-`routing:`-
  entry startup lines. Prompt text is never logged.
- **Observability.** The gate exposes its own stats at `GET /metrics`
  (Prometheus text format) — forwarded/rejected counts, request-size
  histograms, per-model KV-cache fill, per-backend remaining/freshness/age
  gauges, the per-backend capacity-unavailable alert gauge, and the loaded
  config. See the [Observability](#observability) section for the metric
  reference and ready-to-paste PromQL.

---

## Limitations & v2 roadmap

- **Engines are detected, not configured.** vLLM and SGLang are supported
  today (see [SGLang backends](#sglang-backends)); further
  OpenAI-compatible engines are additive via the parser interface (`detect_engine`
  + a `MetricsParser` implementation).
- **Policy is per backend, not per model.** Admission state and policy
  (thresholds, the four autoconfig knobs) are scoped per backend: models on
  one backend share its KV pool (one inference engine) and are treated the
  same.
  Per-model *thresholds* (an independent admission policy per model on a
  shared-KV backend) are not supported. Routing *by* model **is** implemented
  (the per-model `routing:` policies in
  [Routing policies](#routing-policies-per-model)) — the former v2 roadmap item
  "per-model thresholds and routing" was delivered as per-backend admission
  policy + per-model routing policies.
- **Duplicate model ids across backends are legal and routed.** A model id
  served by 2+ backends (A/B or canary serving of the same id) is the normal
  multi-candidate case, routed per its `routing:` policy (default
  `round_robin`) with a pre-stream failover walk across the `order`.
- **Token estimation is global.** `chars_per_token` / `default_max_tokens` are
  not per-backend knobs in this version.
- **Heuristic sizing.** No exact tokenization (no tokenizer dependency).
- **No auth / TLS / request queuing.** (The routing policies are per-model
  selection rules, not a full load balancer: no health scoring beyond the
  pre-stream failover walk.)

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

The test suite (`tests/test_api.py`) drives the gate end-to-end against fake
vLLMs (`httpx.ASGITransport` + a mock upstream transport — including
multi-backend and multi-candidate routing cases), so the allow / 429 /
fail-open / streaming / per-policy routing / failover paths are all exercised
without a real model or GPU; `tests/test_routing.py` unit-tests the pure
per-candidate selection (`select_backend`) for all five policies.

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
