# Multi-backend / multi-model routing — Design Plan

Status: **proposed** (2026-09-16; design Q&A complete — round 1 base routing +
round 2 duplicate-model routing policies; not yet implemented)
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

**Round 2 (2026-09-16 — routing-policy Q&A, user-confirmed; decisions 9–17
extend the decision log: decision 11 supersedes Round 1 derived invariant 9
(globally-unique model ids); Round 1 derived invariant 10 (capacity fail-open)
is **retained** and renumbered to 18; invariant 19 is new):**

**This round supersedes derived invariant 9 (below): model ids are no longer
required to be globally unique.** The `model → backend` map becomes
`model → candidate set`, and a **routing policy** picks one candidate per
request. The five policies and the cross-cutting rules are user-confirmed:

9. **Routing is per-model, not per-backend.** The policy belongs to the
    *request's model*; the candidate set is the set of backends that serve that
    model (explicit `models:` or auto-adopted discovery, decision 5). A new
    top-level `routing:` section maps `model → {policy, order[, threshold_tokens]}`.
    The `Backend` dataclass gains **no** routing knobs.
10. **Policy set (five, all in scope).** All selection rules are defined
    precisely in §2.2b (single source of truth). `round_robin` (per-model
    stateful index, cycles candidates in `order`), `primary_fallback`
    (per-request, first candidate in `order` that admits), `large_small`
    (config `order[0]` = "large" backend; `ctx_tokens >= threshold_tokens` →
    `order[0]`, else `order[1]`, then remaining in `order`), `even` (route to
    the candidate whose admission **minimizes the spread** — max minus min —
    of the *projected* KV usage fractions across **all** candidates, where
    candidate i's projected fraction is `usage_frac_i + charge_i /
    capacity_i` with `charge_i = ceil(ctx × margin_i)`; ties → config order),
    `fill` (config order; first candidate the autoconfig layer would **not**
    reject wins; all would-reject → 429). **`primary_fallback` and `fill` are
    kept as distinct policies although they pick the same backend in nearly
    every case** (see the honesty note in derived invariant 19); `fill`
    shortcuts known-rejecting candidates at selection, `primary_fallback`
    always starts at `order[0]` and relies on the skip-on-reject walk
    (decision 12).
11. **Duplicate model ids are legal** (supersedes Round 1 derived invariant 9). A model
    id owned by **exactly one** backend routes to it (today's behavior). A
    model owned by **2+** backends routes via its `routing:` entry when one
    exists; otherwise the **default policy is `round_robin`** over the
    candidates in config order. `ModelRegistry.resolve` is replaced by
    `resolve_candidates(model) -> tuple[str, ...]` (ordered: config order;
    single-owner models return a 1-tuple).
12. **Failover on proxy failure — pre-stream only (all policies).** On a
    transport error or upstream 5xx **before the first response byte**, the
    gate skips that candidate, walks the remainder of the candidate order, and
    re-runs admission on the next (rolling back the counter charge on the
    failed backend — re-anchoring is the poller's job; the immediate
    bookkeeping fix is re-adding `ceil(ctx × margin)` to the failed
    backend's `KvRemaining`). Once a streamed response has started, errors
    surface as-is (gotcha #6 preserved for the in-flight case). A 429 from a
    candidate also triggers the next candidate (skip-on-reject) — this is
    what makes `primary_fallback` and `fill` outcome-equivalent. All
    candidates exhausted → the final 429 (max `Retry-After` among the
    rejects) or 502 if every candidate failed at transport level.
13. **A `backend` label joins the traffic metrics.** `gate_requests_total`
    and `gate_request_ctx_tokens` gain `backend` alongside `model` — decision
    9's rationale (backend "recoverable from the model" via a unique id map)
    is dead once duplicates are legal. The `gate_kv_cache_usage_pct` series is
    unchanged (it is keyed by vLLM's `model_name`, a different namespace —
    gotcha #9's two-namespace rule still holds).
14. **`GET /v1/models` with duplicates.** A model id served by N backends
    appears **once**, with `owned_by` listing all N backend names (list
    field, not string); single-owner models keep the scalar form for
    backward compatibility of the aggregate response shape.
15. **Validation updates.** Duplicate explicit `models:` across backends is
    **no longer an error** (it is what `routing:` is for). The `routing:`
    section itself is **optional** — omitted or an empty mapping (`{}`) is
    valid (no multi-candidate routing configured). When present and
    non-empty, **every malformed entry is a `ConfigError`** (no silent
    fail-open for typed config; the fail-open default only applies to a
    multi-owner model with *no* entry): a `routing:` entry must
    - name a policy from the five-value set;
    - have a non-empty `order` of configured backend names;
    - have `threshold_tokens` (integer ≥ 1) present **iff** policy is
      `large_small`;
    - appear **at most once per model** (two entries for the same model ⇒
      `ConfigError`); and
    - have an `order` containing **only** backends that serve that model
      (checked at config time against explicit `models:` lists; for
      auto-adopt backends the check is startup-time — after the first
      discovery tick — **logged, not fatal**, mirroring the discovery-
      failure fail-open spirit).
16. **Routing selection is pure.** `select_backend(policy, candidates,
    ctx_tokens, usage_views) -> index | None` lives in a new
    `src/gate/routing.py` next to the other pure decision math; the
    stateful per-model RR index and the failover walk are thin `app.py`
    wrappers (same single-asyncio-loop contract as `KvRemaining`).
17. **`even` and `fill` are best-effort under stale feeds.** They consume
    per-backend usage (fraction) and capacity/remaining (tokens); a
    candidate with a stale/never-fetched feed is treated as "unknown" and is
    **always eligible** (fail-open, gotcha #1) — the prefilter can never
    reject it (that is what keeps `fill` ≡ `primary_fallback` on stale
    feeds, invariant 19 corner (b)). Ranking unknowns: `fill` interleaves
    them in **global** config order (so the degenerate case walks the
    order exactly as `primary_fallback` does); `even` ranks them **last**
    (a candidate it cannot project is not the "most even" choice).
    `round_robin`, `primary_fallback`, `large_small` are unaffected by
    usage data (they ignore it by construction).

**Derived invariants (not separately asked): 18 is Round 1's derived
invariant 10, renumbered and retained as-is; Round 1's derived invariant 9
(globally-unique model ids) is **superseded** by decision 11 and no longer
exists; 19 is new to Round 2:**

18. **Capacity-missing ⇒ per-backend fail-open (deliberate deviation from
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

19. **`fill` ≈ `primary_fallback` on routing outcomes** (design honesty note,
    2026-09-16 round 2; re-derived after the formula correction of the review
    round). Both policies run the same failover walk (decision 12:
    skip-on-reject, skip-on-transport-failure, pre-stream only); the only
    difference is **selection**: `fill`'s prefilter test is *the exact*
    `decision_auto` admission condition
    (including the ctx-None `prompt_unparseable` fail-open), so a
    `fill`-prefiltered candidate is one the autoconfig layer **will**
    reject; `primary_fallback` has no prefilter (always starts at
    `order[0]`). Because the prefilter and the walk's per-candidate
    admission evaluate the **same** test, they can disagree on no
    candidate — the two policies therefore reach the same surviving
    candidate for every request. They can differ in selection cost and
    in only one corner:
    (a) **tiered-layer rejects**: a candidate whose autoconfig layer
    admits but whose tiered layer rejects — `fill` keeps it (it passed
    the prefilter) and the walk rejects it on the tiered layer; the
    final outcome still matches `primary_fallback`, but `fill` evaluated
    it. (The v1-draft "±1 token boundary" corner is **gone**: it came
    from the retired reconstructed-usage formula, which the review
    round replaced with the direct admission test.)
    (b) **stale/unanchored feed**: `fill`'s prefilter passes **every**
    unknown candidate (decision 17), so on a fully stale/unanchored
    candidate set the selection degenerates to the config-order walk —
    i.e. exactly `primary_fallback` (on a mixed stale/fresh set, `fill`
    still prefilter-rejects *fresh* candidates but ranks the unknowns in
    the same global config order, so the walk reaches the same
    candidate).
    The operator intent also differs ("fill my pools in order" vs "hot
    primary, backups"), which is why both are kept as distinct policies
    (user-confirmed): the behavioral difference is small, and the README
    states it plainly rather than pretending one is a duplicate of the
    other.

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

1. Route each generation request to exactly one configured backend: the
   request's `model` field yields a **candidate set** (the backends that
   serve that model), and a **routing policy** (per model, decision 9) picks
   one; unmatched/unparseable models go to the **default** backend.
2. Scope all admission state **per backend** (its own usage feed, capacity,
   remaining-KV counter, poller), so a shared-KV backend is treated the same
   across its models.
3. Provide a sane, mnemonic config schema (`backends:` list; per-backend
   overrides of the tiered `thresholds` and the four autoconfig knobs; optional
   explicit `models:` mnemonic/anchor; a `routing:` section for models served
   by more than one backend, decision 15).
4. Learn each backend's served model ids from `GET /v1/models` at startup and
   on a refresh interval, for correct candidate-set resolution.
5. Expose an aggregated `GET /v1/models` and per-backend observability.
6. Fail over pre-stream to the next candidate on transport failure / 5xx
   (decision 12), so a candidate set is also a health-ordered list.

**Non-goals.** No per-model *admission* policy (admission is per-backend —
decision 1; routing, not admission, is per-model). No real tokenizer. No
auth/TLS. No change to `router.py`/`tokens.py` decision math (new
`routing.py` adds the pure selection layer alongside it). No failover once a
streamed response has begun (decision 12, gotcha #6 preserved in-flight).

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

A new top-level **`routing:`** section (decision 9) maps model id → routing
spec:

```
routing:
  <model-id>:
    policy: str               # round_robin | primary_fallback | large_small | even | fill
    order: list[str]          # candidate backend names; non-empty; all must serve the model
    threshold_tokens: int     # REQUIRED iff policy == large_small (>= 1); absent otherwise
```

- Omitted `routing:` (or a model not present in it) ⇒ single-owner models
  route directly; multi-owner models default to **`round_robin`** over their
  candidates in config order (decision 11).
- `order` defines the candidate order **and** the failover order (decision 12);
  a model not listed in `order` but served by a backend is not a candidate for
  that request (the operator explicitly scoped the candidate set).
- `large_small`: `order[0]` is the "large" backend; `ctx_tokens >=
  threshold_tokens` → `order[0]`, else the remainder in `order` (decision 10).

Env overrides: `BACKENDS_JSON` (JSON list of backend objects; **env wins over
file**), `THRESHOLDS_JSON` (global default tiers), `TARGET_KV_CACHE_PCT`,
`TOKEN_MARGIN`, `RETRY_MIN_S`, `RETRY_MAX_S` (global default knobs).
`VLLM_HOST`/`VLLM_PORT` are removed — a config containing `vllm_host` fails
with the existing unknown-key `ConfigError`.

Validation (all `ConfigError`):

- ≥ 1 backend; unique non-empty `name`s; `port` in 1–65535.
- **Model ids may be duplicated across backends** (decision 11 — the
  `routing:` section decides which backend gets a request). Explicit
  duplicates are accepted; runtime duplicates from auto-adopt are accepted
  the same way (the operator did not type them, but the multi-candidate path
  handles them identically).
- ≤ 1 backend with `default: true`.
- Per-backend knob values, when present, satisfy the same ranges as the
  globals (target finite and in (0, 100]; margin finite and ≥ 1.0;
  `retry_min_s` ≥ 1; `retry_max_s` ≥ `retry_min_s`; tier fields as today).
- **`routing:`** (decision 15): optional — omitted or `{}` is valid. When
  present and non-empty, every malformed entry is a `ConfigError`: `policy`
  ∈ {`round_robin`, `primary_fallback`, `large_small`, `even`, `fill`};
  `order` non-empty and every name is a configured backend;
  `threshold_tokens` (integer ≥ 1) present **iff** `policy == large_small`; a
  model may appear in at most one entry (duplicate entries for the same
  model ⇒ `ConfigError`).
- **`order` vs. serving (startup-time, logged, not fatal):** every backend in
  an entry's `order` should serve that model — checkable at startup against
  explicit `models:` lists; for auto-adopt backends the check runs after the
  first discovery tick and logs a warning (a mis-scoped candidate is simply
  never served the request; the walk fails open).

### 2.2 Model discovery (`models.py` — new module)

- `parse_v1_models(text) -> list[str] | None` — **pure.** Parses an OpenAI
  `GET /v1/models` body (`{"data": [{"id": ...}, ...]}`) into the ordered id
  list. `None` on non-JSON text, missing/non-list `data`, or missing `id`
  fields. No network.
- `ModelRegistry` — stateful, single-asyncio-loop (same contract as
  `MetricsCache`: synchronous read-modify-write, no `await` inside, no locks).
  Owns the `model → ordered candidate list` map (supersedes Round 1's
  single-owner map, decision 11).
  - `register_backend(name: str, *, is_default: bool)` — called once at app
    build, per configured backend. Registration order defines the default
    candidate order for models owned by 2+ backends (config order).
  - `sync(name: str, owned_models: set[str])` — reconciles one backend's owned
    set (explicit `models` if present, else the discovered served set). A model
    left owned by **no** backend stops resolving: `resolve_candidates` returns
    `()` and requests for it take the unknown-model path (default backend,
    decision 4) — the model is never hard-404'd by the gate.
  - `resolve_candidates(model: str) -> tuple[str, ...]` — the routing lookup.
    Returns the ordered backend-name tuple that serves `model`:
    single-owner ⇒ 1-tuple; multi-owner ⇒ all owners in registration (config)
    order. `()` ⇒ unknown model.
  - **No collision policy** — duplicates are the normal multi-candidate case
    (decision 11); the routing policy (§2.2b) chooses among the candidates.
    (Round 1's "default wins" collision rule is deleted; it is superseded.)

### 2.2b Routing selection (`routing.py` — new module)

- **`RoutingPolicy` dataclass + `RoutingSpec`** — parsed from the `routing:`
  config (decision 9): `policy` (one of `round_robin`, `primary_fallback`,
  `large_small`, `even`, `fill`), `order: tuple[str, ...]`,
  `threshold_tokens: int | None`.
- **`select_backend(spec, candidates, ctx_tokens, usage_views) -> int |
  None` — pure.** `candidates: tuple[str, ...]` (already filtered to
  backends that serve the model, in `spec.order` order when a spec exists,
  else registry config order), `usage_views: Mapping[str, CandidateView]`
  with `CandidateView(usage_frac: float | None, remaining_tokens: int |
  None, capacity_tokens: int | None, token_margin: float)`. Returns the
  **index** into `candidates` to attempt first (the failover walk, §2.4,
  advances from there), or `None` when no candidate is eligible (`fill`
  all-full).
  - **Shared definitions.** `charge_i = ceil(ctx_tokens × margin_i)` when
    `ctx_tokens` is known, else `0` (mirrors the app's charge rule). The
    **projected usage fraction** of candidate i is `p_i =
    usage_frac_i + charge_i / capacity_i` — i.e. the current usage plus what
    this request would add, in fraction space. Note `remaining_i` (the
    `KvRemaining` counter) is **not** free tokens: it is headroom *up to
    target* (`anchor_remaining_tokens = floor(capacity × (target − usage))`),
    so it must **not** be used to reconstruct usage (`(capacity − remaining)
    / capacity` over-counts by `1 − target` — this was a v1-draft error,
    corrected 2026-09-16 review round).
  - `round_robin` → always `0` at selection; the app's stateful per-model
    index (decision 16) offsets it — see the app's stateful wrapper.
  - `primary_fallback` → `0` (walk the whole order, reject per candidate).
  - `large_small` → `0` iff `ctx_tokens` is not None and
    `ctx_tokens >= threshold_tokens` (inclusive), else `1` (clamped: with a
    1-candidate order, always `0`).
  - `even` → argmin over i of `spread_i`: form the vector `v_i` with
    `v_i[i] = p_i` (candidate i at its own *projected* fraction — the
    request would go there) and `v_i[j] = usage_frac_j` (every other
    candidate at its **current, unprojected** usage — the request would
    *not* go there); `spread_i = max(v_i) − min(v_i)` over the known
    entries. Unknowns (a candidate's `usage_frac`/`capacity` is None) are
    ranked **last** and among themselves in config order (decision 17) —
    in particular, if candidate i itself is unknown, `spread_i` is
    undefined and i is ranked last. Ties → lowest index. (This matches
    decision 10's "minimizes the spread" objective; the greedy
    least-loaded variant is a known simplification and is **not** what
    this policy specifies.)
  - `fill` → the lowest index `i` the autoconfig layer would **not**
    reject, tested directly: candidate i is eligible when
    `ctx_tokens is None` (mirroring `decision_auto`'s
    `prompt_unparseable` fail-open — an unparseable body is never
    rejected at prefilter time, gotcha #7; without this, a `0 <=
    remaining_i` test would wrongly 429 an over-committed pool for an
    unparseable body) or when `charge_i <= remaining_i` (the exact
    `decision_auto` admission test — `fill` never re-derives the
    threshold comparison from usage math, so there is no unit-conversion
     surface to get wrong). A candidate whose feed is **stale or never-
     fetched is eligible regardless of `remaining_i`** (decision 17,
     gotcha #1 — the walk fails it open; pre-filtering it here could
     make `fill` skip it while `primary_fallback` would fail it open,
     breaking invariant 19's degeneracy claim), and so is a candidate
     whose `remaining_i` is None (unanchored). Eligible-but-unknown
     candidates (stale or unanchored) are ranked by **global** config
     order interleaved with the known ones — unlike `even`'s ranked-
     last rule, this is what makes corner (b) below ("exactly
     `primary_fallback`") true. No eligible candidate → `None`
    (⇒ **no candidate is proxied**; the walk still runs every
    candidate's admission evaluation, without proxying, to collect the
    `Decision.retry_after` values for the max-`Retry-After` 429, §2.4
    f). Because the prefilter *is* the autoconfig admission test
    itself (including the ctx-None fail-open), `fill` and the walk's
    per-candidate admission can disagree on **no** candidate — the
    ±1/boundary corner of the v1 draft (which came from the retired
    reconstructed-usage formula) no longer exists.
- **Stateful wrapper (`app.py`)** — a `RoutingState` object (per model,
  single-asyncio-loop) holds the `round_robin` index (increments on each
  *successful* forward; wraps modulo `len(candidates)`); the failover walk is
  the loop in `_handle_generation` (§2.4).
- **No `proxy.py` change** for selection; `proxy.py` already takes `base_url`.
  The failover wrap around `proxy_request` lives in `app.py`.

### 2.3 Poller (`poller.py`)

`run_poller` gains per-backend parameters and is invoked **once per backend**:
its own `cache`/`capacity_cache`/`counter`, its own `target_frac`, its
`/metrics` URL, plus its `/v1/models` URL, `model_refresh_interval_s`, and the
shared `ModelRegistry`. Per tick, in order:

1. **Metrics work** — the existing `fetch_metrics` → update usage + by-model
   cache, reanchor `counter` (when usage present), update `capacity_cache`.
   Unchanged, **plus**: the poller increments the backend's **anchor
   sequence** (a monotonic `int` owned by `BackendState` in `app.py`, passed
   to the poller for the increment) on every re-anchor — the failover
   charge-rollback guard (decision 12, §2.4 e) compares sequences to decide
   whether a rollback is still valid. The increment is **synchronous with
   the re-anchor** (no `await` between the two; same single-asyncio-loop
   contract as the re-anchor itself), so the guard's read-sequence +
   `add` critical section in §2.4 e is atomic with respect to re-anchoring.
2. **Model discovery** — when due (first tick immediately, then every
   `model_refresh_interval_s`, tracked with `time.monotonic`): `GET
   /v1/models`, `parse_v1_models`, `registry.sync(name, ...)`. A failed fetch
   or `None` parse **keeps the last known map** (fail-open; discovery is
   auxiliary). Never fatal.
3. **Capacity-missing (observed 200, no usable capacity)** — **no longer
    raises `CapacityUnavailableError` / exits** (decision 18): log an error,
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
  1. `model = extract_request_model(body)` (unchanged); `ctx_tokens =
     estimate_context_tokens(body, cfg)` (unchanged — computed **before**
     routing; size-driven policies need it, decision 10).
  2. **Candidate resolution**: `candidates =
     registry.resolve_candidates(model)`; empty tuple (or unparseable
     `"unknown"`) ⇒ single candidate = the default backend (decision 4).
     A multi-candidate model uses its `routing:` spec when present, else
     `round_robin` over the registry order (decision 11). `i0 =
     select_backend(spec, candidates, ctx_tokens, usage_views)` (§2.2b).
   3. **Failover walk** (decision 12), `i = i0, i0+1, …` wrapping through the
      candidate order (when `select_backend` returned `None` — `fill`
      all-full — the walk starts at `0`; it still runs, without proxying
      any candidate, to collect the per-candidate `Decision.retry_after`
      values for the final 429, step f):
     a. **Staleness gate for candidate i**: if its cache is stale/never-
        fetched → both layers fail open for it (gotcha #1) — a stale backend
        is **not** skipped (failing open is the point).
     b. Else: `usage_pct = cache_i.value() * 100.0` → `decision(...)` with
        **that candidate's** thresholds; `decision_auto(counter_i.value(),
        ...)` with **that candidate's** `AutoPolicy` (pool-level max usage
        per candidate, gotcha #5).
     c. `combine_decisions` (unchanged). **Reject ⇒ next candidate** (the
        skip-on-reject that makes `primary_fallback` ≈ `fill`, invariant 19);
        keep each reject's `Decision` for the final 429.
      d. **Allow** ⇒ charge `counter_i.subtract(charge_i)` **before** the
         proxy await (rejected/failed candidates charge nothing; a charge on
         a candidate that later fails at transport is rolled back — step e)
         — then `proxy_request(..., base_url_i)`. Remember the sequence: the
         app records the candidate's **anchor sequence** (a per-backend
         monotonic counter incremented by the poller on each re-anchor)
         alongside the charge, at charge time.
      e. **Transport failure or upstream 5xx before the first response byte**
         ⇒ roll back the charge on candidate i, record the failure, **next
         candidate**. The walk **never short-circuits**: every remaining
         candidate is attempted before exhaustion is declared (a first
         transport failure is a skip, not a stop). Rollback is **guarded by
         the anchor sequence**: the charge is re-added via
         `KvRemaining.add(charge_i)` **only if** candidate i's anchor
         sequence is unchanged since the charge (no re-anchor happened in
         between); if a re-anchor occurred, the old charge is dropped and the
         fresh anchor is authoritative — the worst case is one request's
         worth of tokens un-counted against the new anchor, which the next
         poll re-anchors anyway (bounded, self-correcting; never
         double-counted against a fresh anchor). Once a streamed response has
         started, errors surface as-is (gotcha #6 preserved in-flight) — no
         walk.
      f. **All candidates exhausted** (i.e. after step e has skipped the
         last candidate): if **any** candidate 429'd → a single 429 with
         `Retry-After = max` over the candidates' `Decision.retry_after`
         values (each is that candidate's tier `timeout_s` or
         `scaled_retry_after` result — the same values a single-candidate
         429 would have carried), rejector-aware message naming the
         max-timeout rejector, mirroring `combine_decisions`' rule extended
         across candidates (tie → first in walk order); else (every candidate
         failed at transport, none 429'd) → **502** (never 200, never 429 —
         the request never got a capacity answer; the 502 body names the
         candidate backends tried).
      g. **Recording**: `stats.record_forwarded(endpoint, model, ctx_tokens,
         backend_i)` / `record_rejected(..., backend_i)` carry the final
         backend (decision 13) — on 429-exhaustion that is the max-timeout
         rejector's backend; on 502-exhaustion (no rejector exists) the label
         is the sentinel `backend="none"`. A candidate tried-then-failed is
         logged (`routing_failover` log line with model, from, to, reason)
         and counted by the new `gate_routing_failovers_total{model, from,
         to, reason}` (reason ∈ `transport` | `upstream_5xx` | `reject`).
- **`GET /v1/models`** (new, gate-local, always 200): aggregate the registry's
  known models into one OpenAI-shaped list, one entry per model with an
  `owned_by` extension — a **list** of backend names when the model is
  served by 2+ backends, a **scalar** name when single-owner (decision 14).
- **`GET /healthz`**: `status: "ok"` unchanged; the single
  `kv_usage`/`kv_cache_capacity_tokens`/`kv_cache_remaining_tokens` fields
  become a per-backend array:
   `{"backends": [{"name", "metrics_age_s", "kv_usage",
   "kv_cache_capacity_tokens", "kv_cache_remaining_tokens"}]}` plus
   `models: [{"id", "owned_by": [names] | name, "routing": <policy> }]`
   (`owned_by` list when 2+ owners, decision 14; `routing` names the policy
   in effect — the `routing:` entry or `round_robin` by default).
- **`GET /metrics`**: as today, but `set_kv_usage` unions all backends'
  `by_model()` maps, `set_freshness`/`set_remaining` run per backend
  (labeled, §2.5).

### 2.5 Stats (`stats.py`)

- **`backend` label** added to `gate_kv_cache_remaining_tokens`,
  `gate_metrics_fresh`, `gate_metrics_age_s`; `set_remaining(backend, value)`
  / `set_freshness(backend, fresh, age_s)` gain the backend argument.
- `gate_kv_cache_usage_pct{model_name}` is **unchanged** (keyed by vLLM's
  `model_name` label — the two-namespace rule of gotcha #9 still holds).
- **`backend` label added** to `gate_requests_total` and
  `gate_request_ctx_tokens` (decision 13 — with duplicate model ids legal,
  the backend is no longer recoverable from the model):
  `gate_requests_total{endpoint, model, result, backend}` and
  `gate_request_ctx_tokens{model, result, backend}` — the existing label
  sets (`endpoint`/`model`/`result`) are **kept as-is** (see
  `stats.py:74,81`); `result` is not renamed. Recording uses the **final**
  backend (the one that forwarded, the max-timeout rejector on 429-
  exhaustion, or the sentinel `"none"` on 502-exhaustion, §2.4 g).
- **New** `gate_routing_failovers_total{model, from, to, reason}` counter
  (decision 12, reason ∈ `transport` | `upstream_5xx` | `reject`).
- **New** `gate_backend_capacity_unavailable{backend}` gauge (0/1), set by the
  app from the poller's per-backend flag (decision 18, alerting hook).
- `gate_config_info` serializes the backend structure: per backend
  `name`/`host`/`port`/`default` + which knobs are overridden (global vs
  backend-specific), but **not** the `models:` lists (discovered data would
  go stale; keep the info metric small and startup-static). The `routing:`
  section **is** serialized (policy + order per model — startup-static,
  useful for operator audit).

### 2.6 No change

`router.py` (pure decision math — `decision`, `decision_auto`,
`scaled_retry_after`, `combine_decisions` are reused per-candidate, unchanged),
`tokens.py` (token estimation stays **global** — `chars_per_token`/
`default_max_tokens` are not per-backend in v1; the per-backend override of
these is a follow-up if needed), `proxy.py` (`proxy_request` already takes
`base_url`; the failover wrap lives in `app.py`, §2.4 e), `metrics.py` caches
(`MetricsCache`/`CapacityCache`/`KvRemaining` reused once per backend).

## 3. Files to Change/Create

| Path | Action | Description |
|---|---|---|
| `src/gate/models.py` | Create | Pure `parse_v1_models`; stateful `ModelRegistry` (register/sync/`resolve_candidates` — ordered candidate tuples, no collision policy (duplicates are normal, decision 11); fail-open on discovery failure). |
| `src/gate/routing.py` | Create | `RoutingSpec` dataclass; pure `select_backend(spec, candidates, ctx_tokens, usage_views) -> int \| None` (five policies, decision 10); `CandidateView` dataclass; policy constants. No I/O. |
| `src/gate/config.py` | Modify | `Backend` dataclass; required `backends:` list replaces `vllm_host`/`vllm_port`; **`routing:` section** (per-model `policy`/`order`/`threshold_tokens`, decision 9; optional — omitted/`{}` valid); global knobs + `model_refresh_interval_s`; YAML + `BACKENDS_JSON` (env wins) parsing; validation (≥1 backend, unique names, **duplicates legal** (decision 15), ≤1 default, port/knob ranges, routing entries well-formed: valid policy, non-empty order of configured names, `threshold_tokens` iff `large_small`, **at most one entry per model**, order ⊆ model's serving backends (startup-logged for auto-adopt)); remove `VLLM_HOST`/`VLLM_PORT`. |
| `src/gate/poller.py` | Modify | Per-backend invocation (own caches/counter/target); `/v1/models` discovery at `model_refresh_interval_s` (first tick immediate), failure keeps last map; capacity-missing ⇒ per-backend fail-open (no exit, decision 18) + unavailable flag; keep transport fail-open. |
| `src/gate/app.py` | Modify | `BackendState` per backend (**incl. the anchor-sequence counter**, decision 12); `ModelRegistry` wiring; **`RoutingState`** (per-model RR index, decision 16); `_handle_generation` rewritten: candidate resolution → failover walk (selection, staleness gate, per-candidate admission, skip-on-reject, pre-stream transport/5xx failover with **anchor-sequence-guarded** charge rollback, full-walk exhaustion → 429-max or 502, `backend="none"` on 502, decision 12); per-candidate `base_url`; new `GET /v1/models` (duplicate-aware `owned_by`, decision 14); per-backend `/healthz`; per-backend `/metrics`; one poller per backend; module docstring updated (exit-1 capacity story gone — decision 18; failover wrap documented). |
| `src/gate/stats.py` | Modify | `backend` label on remaining/fresh/age gauges **and** `gate_requests_total`/`gate_request_ctx_tokens` — existing label sets kept (`{endpoint, model, result, backend}` / `{model, result, backend}`), `backend="none"` sentinel on 502-exhaustion (decision 13); `set_remaining`/`set_freshness` take backend; new `gate_routing_failovers_total{model,from,to,reason}`; new `gate_backend_capacity_unavailable{backend}`; `gate_config_info` serializes backends **and** the `routing:` section (no `models:` lists). |
| `src/gate/main.py` | Modify | Log backend count/names/default/overrides; log routing entries (model → policy/order); remove host/port logging. |
| `src/gate/router.py` | No change | Reused as-is (per-candidate admission math). |
| `src/gate/tokens.py` | No change | Global token estimation unchanged. |
| `src/gate/proxy.py` | No change | Already takes `base_url`; failover wrap lives in `app.py`. |
| `src/gate/metrics.py` | Modify (small) | Caches reused per backend, unchanged; `KvRemaining` gains a public `add(tokens)` (inverse of `subtract`, used by the failover charge rollback — no-op when unanchored, same as `subtract`); **delete** the now-dead `CapacityUnavailableError` (decision 18 — raised only in the poller, which no longer raises it). |
| `tests/test_models.py` | Create | `parse_v1_models` + `ModelRegistry` (register/sync/`resolve_candidates`: single-owner 1-tuple, multi-owner ordered tuple, removal, discovery-failure no-op). |
| `tests/test_routing.py` | Create | Pure `select_backend` per policy: RR=0-at-selection; PF=0; large_small at/below/above threshold + 1-candidate clamp; even ranking incl. unknowns-last and ties; fill first-fit, all-full → None, unknowns eligible; spec/order edge cases (empty candidates, ctx None for large_small). |
| `tests/test_config.py` | Extend | `backends` YAML + `BACKENDS_JSON` (env wins); per-backend override resolution; **`routing:` parsing + validation** (bad policy, empty order, unknown backend in order, `threshold_tokens` iff `large_small`, duplicate model ids now legal); all legacy validation failures; legacy `vllm_host`/`VLLM_HOST` rejected. |
| `tests/test_poller.py` | Extend | Per-backend poller + discovery cadence; discovery failure keeps map; capacity-missing ⇒ fail-open, no exception, no task end; per-backend reanchor. |
| `tests/test_api.py` | Extend | Two fake vLLMs sharing a model id; **per-policy routing** (assert proxy target for RR cycling, PF, large_small both sides of the threshold, even, fill); unknown ⇒ default; **pre-stream failover** (transport error on candidate 1 ⇒ candidate 2 forwards; 5xx same; **charge rollback** asserted via `gate_kv_cache_remaining_tokens`); skip-on-reject (candidate 1 429 ⇒ candidate 2); all-full ⇒ 429 with max `Retry-After`; all-transport-failed ⇒ 502; one-stale/one-fresh fail-open; `/v1/models` aggregate with duplicate `owned_by`; per-backend `/healthz` + `/metrics` labels incl. the new `backend` label on request counters + failover counter. |
| `tests/test_stats.py` | Extend | Per-backend labels; `backend` label on `gate_requests_total`/`gate_request_ctx_tokens`; stale-label removal; `gate_routing_failovers_total`; `gate_config_info` backend + routing structure; `gate_backend_capacity_unavailable`. |
| `tests/test_main.py` | Extend | Backend + routing logging. |
| `config.example.yaml` | Modify | Multi-backend reference with mnemonic + per-backend overrides + auto-adopt note + a `routing:` section example (two policies commented) — sketch in §6. |
| `docker-compose.example.yaml` | Modify | `BACKENDS_JSON` example; remove `VLLM_HOST`/`VLLM_PORT`; mounted-config comment. |
| `README.md` | Modify | Multi-backend section, config reference (incl. `routing:`), diagram (N backends, N pollers, ModelRegistry + routing walk), `/v1/models` (duplicates), per-backend metrics/healthz, **routing policies reference** (five policies, the `fill` ≈ `primary_fallback` note (invariant 19), failover semantics, charge rollback, 429-max/502 exhaustion rule), capacity fail-open deviation (decision 18), v2 roadmap. |
| `AGENTS.md` | Modify | Project map (new `models.py` + `routing.py`, per-backend state, `KvRemaining.add`); new gotchas (duplicates-legal + per-model routing policy, pre-stream failover + charge rollback, 429-max/502 exhaustion, `fill` ≈ PF equivalence, `/v1/models` duplicates, shorthand removal, zero-config superseded); test list adds `test_models.py` + `test_routing.py`. |

## 4. Testing Plan

- **Unit, pure/edge (`test_models.py`).** `parse_v1_models`: valid list,
  empty `data`, missing `data`, non-JSON, duplicate ids, non-dict entries,
  missing `id`. `ModelRegistry`: register + `resolve_candidates` (single-owner
  1-tuple; multi-owner ordered tuple in registration order); `sync` add/remove;
  a model removed on refresh stops resolving (empty tuple → unknown-model
  path); discovery-failure path is a no-op (last map kept). **No collision
  tests** — duplicates are normal (decision 11).
- **Routing (`test_routing.py`).** `select_backend` is pure — table-drive it:
  each policy × (single/multi candidates, known/stale usage views, ctx
  None/known). `large_small` at exactly the threshold (inclusive → large);
  `even`/`fill` with unknowns (ranked last / eligible-by-order, decision 17),
  ties → lowest index; `fill` all-full → `None`; 1-candidate order clamps.
- **Config (`test_config.py`).** Parse `backends` from YAML; `BACKENDS_JSON`
  **overrides** the file; per-backend override resolution (present vs
  `None` → global default, incl. per-backend `target_frac` derivation);
  **`routing:` parsing** (policy/order/threshold; omitted and `{}` both
  valid) and validation failures (unknown policy, empty order, order naming
  a non-configured backend, `threshold_tokens` present on non-`large_small`
  / absent on `large_small`, **two entries for the same model**);
  **duplicate model ids are legal** (parse succeeds); other validation
  failures (zero backends, dup names, two `default: true`, bad port, bad
  knob ranges, legacy `vllm_host` key / `VLLM_HOST` env no longer honored).
- **Poller (`test_poller.py`).** Per-backend: good poll reanchors **only**
  its own counter; transport error keeps **only its own** state; discovery
  fires first tick + on cadence (fake clock), failure keeps last map;
  **capacity-missing ⇒ no exception, counter stays unanchored, unavailable
  flag set, task still running** (assert the poller does not end).
- **ASGI end-to-end (`test_api.py`).** Two fake vLLMs, one model id served by
  **both** (plus one single-owner model each; distinct `/metrics`).
  **Per-policy routing** (assert the upstream `base_url` via the mock
  transport): `round_robin` alternates across requests; `primary_fallback`
  always hits candidate 1 when it admits; `large_small` routes ≥ threshold
  to `order[0]`, < threshold to `order[1]`; `even`/`fill` pick the
  less-full candidate (drive each fake's `/metrics` to different
  usage/capacity). Unknown/`"unknown"` model ⇒ default backend.   **Failover:**
  candidate 1 transport-fails (mock transport raises) ⇒ candidate 2 forwards
  **and** candidate 1's `gate_kv_cache_remaining_tokens` shows the rolled-back
  charge (`KvRemaining.add`); candidate 1 returns 5xx pre-stream ⇒ same;
  candidate 1 429s ⇒ candidate 2 forwards (skip-on-reject); **both** 429 ⇒
  final 429 with `Retry-After = max` of the two; **both** transport-fail ⇒
  502 with `backend="none"` on `gate_requests_total` (and a
  `gate_routing_failovers_total` entry per skip). **Anchor-sequence guard:** a re-anchor on
  candidate 1 between charge and transport-failure ⇒ the charge is *not*
  re-added (drop-and-fresh-anchor path, §2.4 e); no re-anchor ⇒ re-added.
  (Drive the re-anchor by calling the poller's re-anchor path directly or by
  faking a good poll before the transport failure.) **One stale / one fresh:** the fresh candidate is gated; the stale
  candidate fails open (and is still reachable through the walk). `GET
  /v1/models` returns the union with `owned_by` (list for the shared model).
  `/healthz` per-backend array + per-model `routing`. `/metrics` carries
  per-backend labels, the `backend` label on `gate_requests_total`/
  `gate_request_ctx_tokens`, `gate_routing_failovers_total`, the
  `gate_kv_cache_usage_pct` union, and the `gate_config_info` backend +
  routing structure.
- **Stats (`test_stats.py`).** Per-backend label series + stale-label
  removal; `backend` label on `gate_requests_total`/`gate_request_ctx_tokens`
  (existing `endpoint`/`model`/`result` labels preserved; `backend="none"`
  on 502-exhaustion); `gate_routing_failovers_total` increments;
  `gate_backend_capacity_unavailable` set/clear; `gate_config_info` includes
  backend + routing structure and omits model lists.
- **Gates.** `ruff format --check`, `ruff check`, `mypy src/`,
  `pytest --cov=gate --cov-fail-under=80`, `bash scripts/local-ci.sh`.
  Coverage must stay ≥ 80% — new `models.py`/`routing.py` and the `app`
  failover-walk branches are the risk; the ASGI multi-backend/failover cases
  are the main coverage driver.

## 5. Documentation Updates

- **`README.md`:** new "Multi-backend & model routing" section (config schema
  + mnemonic, auto-adopt vs explicit `models:`, default-backend rule,
  **per-model routing policies** — a reference table of the five policies
  (selection rule, state, usage-data sensitivity) plus the
  `fill` ≈ `primary_fallback` note (invariant 19), the pre-stream failover
  rule + charge rollback (decision 12), the exhaustion rule (429-max / 502),
  duplicate model ids (now legal), `GET /v1/models` (duplicates),
  per-backend metrics/healthz, and the **capacity fail-open deviation**);
  updated architecture diagram (N backends, N pollers, `ModelRegistry` +
  the routing walk); config reference table (incl. `routing:`); **v2 roadmap**
  — strike "per-model thresholds and routing" (delivered as per-backend
  admission policy + per-model routing policies).
- **`config.example.yaml`:** full multi-backend reference (see §6) with inline
  comments on the mnemonic, auto-adopt, the global-vs-override split, and the
  `routing:` section (two worked examples: `fill` and `large_small`).
- **`AGENTS.md`:** project map adds `models.py` + `routing.py` and
  per-backend state; `KvRemaining` gains `add` (charge rollback);
  **new/updated known gotchas** — (g) **duplicate model ids are legal**; a
  model on 2+ backends routes per its `routing:` policy, defaulting to
  `round_robin` (decision 11, supersedes the Round 1 uniqueness rule),
  (h) unmatched/unknown models route to the default backend,
  (i) a backend with a live-but-capacity-less `/metrics` fails open for
  itself (no process exit) + `gate_backend_capacity_unavailable` gauge
  (decision 18, supersedes the exit-1 half of gotcha #10),
  (j) **pre-stream failover**: transport error / 5xx before the first byte
  ⇒ next candidate, charge rolled back via `KvRemaining.add`; in-flight
  stream errors surface as-is (gotcha #6 preserved) (decision 12),
  (k) **exhaustion**: all candidates 429 ⇒ one 429 with `Retry-After = max`
  (rejector = max-timeout, tie → walk order); all transport-failed ⇒ 502
  (decision 12), (l) **`fill` ≈ `primary_fallback`** on outcomes (invariant
  19) — both kept as distinct policies on operator intent,
  (m) `/v1/models` is gate-local/aggregated (always 200); duplicate ids
  appear once with a list `owned_by` (decision 14),
  (n) `vllm_host`/`vllm_port` shorthand and `VLLM_HOST`/`VLLM_PORT` are
  removed, and **gotcha #13 (zero-config) is superseded**: at least one
  backend must be configured (file or `BACKENDS_JSON`) because the gate has
  no other way to learn its upstreams (decision 6). Test list adds
  `test_models.py` + `test_routing.py`.
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
# auto-adopt whatever the backend serves. A model id MAY be served by more
# than one backend (the routing: section below decides which one gets a
# request; without a routing entry, duplicates default to round_robin).
# Exactly one backend may be default: true (fallback for unknown models;
# the first backend when none is flagged). All per-backend knobs are
# optional and default to the global values below.
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
  - name: qwen-fast
    host: vllm-qwen-fast
    port: 8001
    models: [qwen3-32b]        # same model id, second backend (legal)
    target_kv_cache_pct: 70.0  # fill it less: smaller effective KV headroom
  - name: llama
    host: vllm-llama
    port: 8002
    # no models: => auto-adopt from /v1/models; no overrides => global policy

# Routing — OPTIONAL. Only needed for model ids served by 2+ backends.
# order: candidate AND failover order (first entry tried first; on
# pre-stream transport failure / 5xx / 429 the next candidate is tried).
# policy: round_robin | primary_fallback | large_small | even | fill
#   round_robin      — cycle candidates evenly (stateful, per model).
#   primary_fallback — always prefer order[0]; fall back in order.
#   large_small      — ctx >= threshold_tokens -> order[0] ("large"),
#                      otherwise the remaining candidates in order.
#   even             — route to the candidate that keeps all candidates'
#                      projected KV-cache % closest together.
#   fill             — fill order[0] up to its target_kv_cache_pct, then
#                      order[1], ...; 429 when every candidate is full.
#                      (≈ primary_fallback on outcomes; see README.)
# A model with multiple owners and NO entry here defaults to round_robin.
routing:
  qwen3-32b:
    policy: fill
    order: [qwen, qwen-fast]
  # Alternative shapes (commented):
  # some-model:
  #   policy: large_small
  #   order: [big-box, small-box]
  #   threshold_tokens: 8000   # required for large_small only

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

- [ ] `load_config`: `backends` from YAML parses; `BACKENDS_JSON` overrides the file; per-backend `None` fields resolve to globals; `routing:` parses (policy/order/threshold); omitted `routing:` and `routing: {}` are both valid.
- [ ] Validation rejects: zero backends; duplicate `name`; two `default: true`; `port` ∉ 1–65535; out-of-range knob; legacy `vllm_host` key / `VLLM_HOST` env; bad routing entry (unknown policy, empty order, unknown backend in order, `threshold_tokens` iff `large_small`, **two entries for the same model**). Validation **accepts** a model id listed in two backends.
- [ ] `parse_v1_models`: valid / empty / non-JSON / missing `data` all behave per spec.
- [ ] `ModelRegistry.resolve_candidates` returns the right tuple: single-owner 1-tuple; multi-owner ordered (registration order); removed models ⇒ empty tuple (unknown-model path).
- [ ] `select_backend` (pure): each policy returns the right index; `large_small` inclusive at threshold; `even` = spread objective (not greedy; other candidates at their *current* usage, chosen candidate at its projection; unknowns ranked last); `fill` prefilter is the exact `decision_auto` admission test — `ctx None` ⇒ eligible, else `charge_i <= remaining_i` (never re-derives the threshold from usage math), stale/never-fetched or unanchored ⇒ always eligible, ranked in global config order (interleaved, not last); `fill` all-full ⇒ `None` (no proxy, walk still collects `retry_after`); 1-candidate clamps.
- [ ] Per-backend poller reanchors **only** its own counter; discovery runs first tick + on cadence; discovery failure keeps the map; discovery never fatal.
- [ ] Capacity-missing on one backend ⇒ **no** process exit, that backend's autoconfig fails open (`auto_unanchored`), `gate_backend_capacity_unavailable{backend}=1`, other backends unaffected; the poller task keeps running (fatal callback not triggered by a capacity condition).
- [ ] `CapacityUnavailableError` deleted from `metrics.py`; all references cleaned up (`app.py` docstrings, `tests/test_poller.py`, `tests/test_api.py`).
- [ ] `KvRemaining.add` behaves as the inverse of `subtract` (no-op when unanchored).
- [ ] ASGI: per-policy routing asserts the right `base_url` (RR alternates; PF sticks to candidate 1; large_small both sides of the threshold; even/fill pick the less-full); unknown/`"unknown"` ⇒ default backend.
- [ ] ASGI failover: transport error / 5xx pre-stream on candidate 1 ⇒ candidate 2 forwards **and** candidate 1's remaining-tokens gauge shows the rolled-back charge (rollback guarded by the anchor sequence: re-anchored ⇒ charge dropped, not re-added); the walk never short-circuits; 429 on candidate 1 ⇒ candidate 2 forwards; both 429 ⇒ 429 with `Retry-After = max` of the candidates' `Decision.retry_after`; both transport-fail ⇒ 502 with `backend="none"`; `gate_routing_failovers_total` increments per case.
- [ ] Poller increments the backend's anchor sequence on every re-anchor (the rollback guard's source of truth).
- [ ] Per-backend 429 + `Retry-After` use the right backend's tier/autoconfig values; one stale backend fails open while a fresh one gates (and a stale candidate is still reachable through the walk).
- [ ] `GET /v1/models` returns the union with `owned_by` (list for multi-owner ids), always 200, gate-local.
- [ ] `/healthz` per-backend array + per-model `routing` (plus `status: "ok"`); `/metrics` per-backend labels + `backend` label on request counters + `gate_kv_cache_usage_pct` union + `gate_config_info` backend + routing structure (no `models:` lists).
- [ ] `ruff format --check`, `ruff check`, `mypy src/`, `pytest --cov=gate --cov-fail-under=80` all green; `bash scripts/local-ci.sh` green.
- [ ] README/AGENTS/config.example/docker-compose updated and consistent with the code (esp. the five-policy reference + `fill` ≈ PF note, failover/rollback semantics, exhaustion rule, capacity fail-open deviation, shorthand removal).

## 8. Implementation Order (DAG)

```
[1] routing.py ──────► [3] config.py (imports RoutingSpec for routing:
                       validation)
[2] models.py ──────┐
                    ├──► [5] poller.py (takes ModelRegistry + per-backend
                    │    knobs for discovery & re-anchor sequencing)
                    │         │
                    │         ▼
                    └──► [7] app.py (BackendState incl. anchor sequence,
                          RoutingState, failover walk, /v1/models,
                          healthz, N pollers)
[3] ──────────────────┤
[6] stats.py ─────────┤ (independent of 1–5; backend labels + failover
     │                │  counter; can be built any time after 3)
     └────────────────┘
                          [7] ──► [8] main.py (logging)
```

- **Parallel (independent):** `[1] routing.py` ∥ `[2] models.py` ∥
  `[6] stats.py` — none imports project code except the existing `stats`
  surface; build concurrently (`test_routing.py` ∥ `test_models.py` ∥
  `test_stats.py` with them).
- **Sequential:** `[3] config.py` needs `[1]` (imports `RoutingSpec` for
  `routing:` validation).
- **Sequential:** `[5] poller.py` needs `[2]` (`ModelRegistry` parameter,
  anchor-sequence increment).
- **Sequential:** `[7] app.py` needs 1, 2, 3, 5, 6 (selection + registry +
  config + poller wiring + stats labels). `[8] main.py` needs 3, 7.
- **Tests** track their module (interleave); `test_api.py` last (needs app).
  **Docs** (README/AGENTS/config.example/compose) after code is green.
- **Branching/commits (AGENTS.md dev-style):** feature branch
  `feat/multi-backend`; the per-segment pieces are individually small enough
  that sub-branches are optional, but if split, `routing` + `models` (parallel)
  and `config` are the natural sub-branches. One commit per segment:
  `routing` → `models` → `config` → `poller` → `stats` → `app` → `main` →
  tests → docs/packaging. Rebase along the way; per-commit distinct-model
  review (qwen + gemma) before merge/PR.
