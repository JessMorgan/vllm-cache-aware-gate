# AGENTS.md — project context for fresh agent sessions

A small, single-purpose **reverse-proxy "gate"** container that sits in front of
one or more vLLM OpenAI-compatible backends and protects their KV caches from
over-subscription. It speaks the OpenAI API, resolves each incoming request's
body `model` to the configured backends that serve it (a **candidate set** — a
model id may be served by 2+ backends; unknown/unparseable models fall to the
default backend) and picks a candidate via a per-model **routing policy**
(default `round_robin`), and per request decides: **forward** (that
candidate's KV cache has room for this prompt's estimated size) or **reject
with HTTP 429 + `Retry-After`** (not enough room — retry in N seconds; in a
multi-candidate walk a reject/transport-failure/5xx is a *skip* to the next
candidate — pre-stream only). It learns
each backend's headroom by polling that backend's Prometheus `/metrics` endpoint
(one background poller per backend, reading `vllm:kv_cache_usage_perc` and the
KV-cache capacity) and its `GET /v1/models` (model discovery). Admission state
(usage cache, capacity, remaining-KV counter) is **per backend**: models on one
backend share its KV pool (single-engine vLLM), so a shared-KV backend is
treated the same across its models. Designed for safe, configuration-first,
fail-open operation: a monitoring outage never blocks inference traffic, and the
gate is a transparent pass-through for the endpoints it proxies. The full design
lives in `docs/plans/multi-backend.md`; the operator-facing reference is
`README.md` (links at the bottom). Read this file to know what to grep first and
which invariants are load-bearing.

## Honesty and integrity

Honesty and integrity outrank every other consideration in this project —
speed, convenience, and looking competent included. Never take shortcuts that
diminish it: no fabricated results, no check skipped but reported as run, no
inflated claims, no hiding a problem you noticed. Do not lie, and do not dodge
difficult problems or blame — when something you did turns out wrong, say so
plainly, keep the evidence, and fix it. The same duty runs toward the user:
do not stay silent to appease them. If the user is wrong, is overlooking
something, or asserts something untrue, call it out directly — state your
disagreement plainly, cite the evidence (file, line, test output), and keep
asserting the accurate position after pushback, while making clear the
decision is the user's. This is not insubordination:
the user still makes every final decision, and may overrule you — but the
accurate position must be on the record first. (The Git-management
`qwen` + `gemma` review and attribution gates are concrete applications of
this section.)

## Project map

Single Python package (`gate`) under a src-layout; the deployable artifact is a
Docker image that runs `python -m gate.main`. The codebase follows a
**pure-core / thin-edges** structure: the decision logic and token estimation are
pure, dependency-free functions (trivially unit-testable); the I/O lives in thin
edge modules (HTTP proxy, metrics poller, config loader) that only the
FastAPI app wires together.

- **`src/gate/config.py`** — `Threshold`, `Backend`, `RoutingEntry`, and
  `GateConfig` dataclasses; `load_config(path)` reads a **YAML** file (if any)
  and applies env-var overrides, then validates strictly. `kv_pct` is a
  **percentage (0–100)**; the vLLM metric is a **fraction (0–1)**. The
  `THRESHOLDS_JSON` and `BACKENDS_JSON` env overrides are parsed as JSON (JSON
  lists), even though the file is YAML; `BACKENDS_JSON` **replaces** the file's
  `backends:` entirely (env wins). `backends` is a **REQUIRED** non-empty list
  of `Backend(name, host, port, models=(), default=False,
  thresholds=None, target_kv_cache_pct=None, token_margin=None, retry_min_s=None,
  retry_max_s=None)` entries — each `None` knob means "use the global default";
  at most one `default: true`; **model ids may be duplicated across backends**
  (the normal multi-candidate case — the `routing:` section decides which
  backend gets a request); only **duplicate ids within one backend's `models:`**
  list are a `ConfigError`; the legacy `vllm_host`/`vllm_port` keys and
  `VLLM_HOST`/`VLLM_PORT` env vars are **removed** (a config using them fails
  with a migration-hint `ConfigError`). `thresholds` is **optional** (an
  empty/absent tiered policy just means the tiered layer always allows). A new
  optional top-level **`routing:`** section (file-only — no env override) maps
  model id → `RoutingSpec` (`policy` ∈ {`round_robin`, `primary_fallback`,
  `large_small`, `even`, `fill`}, non-empty `order` of configured backend
  names — also the failover order — and `threshold_tokens` (int ≥ 1) present
  **iff** `policy` is `large_small`); omitted or `{}` is valid; every
  malformed entry is a `ConfigError`; the "at most one entry per model" rule
  holds via **last-wins** (yaml.safe_load cannot detect duplicate mapping
  keys — safe in outcome, see the known-deviation note in `_parse_routing`).
  The `order ⊆ serving` check is deliberately **not** config-time: it is a
  startup-time, logged-not-fatal check (the poller's dead-candidate warning).
  Also carries the global defaults of the four
  autoconfig knobs with env overrides: `target_kv_cache_pct` (`85.0`,
  `TARGET_KV_CACHE_PCT`), `retry_min_s` (`5`, `RETRY_MIN_S`), `retry_max_s`
  (`60`, `RETRY_MAX_S`), `token_margin` (`1.25`, `TOKEN_MARGIN`), plus
  `model_refresh_interval_s` (`30.0`, `MODEL_REFRESH_INTERVAL_S`) — the
  per-backend `/v1/models` discovery cadence — and `log_level` (`INFO`,
  `LOG_LEVEL`), the logging level (case-insensitive, must be one of
  `debug`/`info`/`warning`/`error`/`critical`; anything else is a
  `ConfigError`) applied at startup to the gate's loggers and uvicorn.
- **`src/gate/tokens.py`** — `estimate_context_tokens(body, cfg) -> int |
  None`. Pure. Estimates prompt tokens as `ceil(prompt_chars / chars_per_token)`
  plus `max_tokens` headroom (default 256); returns `None` on an
  unparseable body (caller fails open). Chat vs completions prompt-text
  extraction. Intentionally a heuristic (no tokenizer dependency). Also
  `extract_request_model(body) -> str` (pure, never raises): the request
   body's `model` string, or `"unknown"` — the **routing key** (selects the
   candidate backends via the `ModelRegistry`'s
   `resolve_candidates`; `"unknown"` falls to the default backend) and the
   stats `model` label.
- **`src/gate/metrics.py`** — engine-agnostic `/metrics` parsing (vLLM or
  SGLang, detected per body — `docs/plans/sglang-backend.md`). Carries the
  engine constants (`ENGINE_VLLM` / `ENGINE_SGLANG` / `ENGINE_UNKNOWN` +
  `ENGINES`), `detect_engine(text) -> str` (pure; probes the **usage-gauge
  families by name presence** in the fixed precedence
  `vllm:kv_cache_usage_perc` → `sglang:kv_cache_usage_perc` →
  `sglang:token_usage` → `unknown`), the `MetricsParser` protocol (three pure
  methods: `usage` / `usage_by_model` / `capacity`) with three implementations
  — `VllmMetricsParser` (the original vLLM logic verbatim),
  `SglangMetricsParser` (newer gauge with a legacy fallback:
  `sglang:kv_cache_usage_perc` over `sglang:token_usage`,
  `sglang:kv_cache_total_tokens` over `sglang:max_total_num_tokens`; the newer
  one wins when present), and `UnknownMetricsParser` (all-`None` no-op) — plus
  the `_PARSERS` registry (engine → parser). The three
  `parse_kv_cache_*` free functions are retained as **thin vLLM facades** over
  `VllmMetricsParser` (max across all `vllm:kv_cache_usage_perc` series;
  per-model fractions keyed by `model_name`, `"default"` when unlabeled; max
  positive `vllm:kv_cache_size_tokens` gauge sample falling back to the
  `kv_cache_size_tokens` label on `vllm:cache_config_info` — the form current
  vLLM emits); `fetch_usage` / `fetch_usage_by_model` are **gone** (folded into
  `fetch_metrics`). `fetch_metrics(client, url) ->
  MetricsSample` — a single GET that detects the engine and parses usage +
  by-model + capacity from one body; `MetricsSample` carries
  **`engine: str`** (`"vllm"` / `"sglang"` / `"unknown"`; non-200 →
  `engine="unknown"`). `MetricsCache` (last value + `fetched_at` + `age()`;
  `by_model()`
  stores the per-model breakdown while `_value` stays the MAX of it),
   `CapacityCache` (last good capacity token count; never stale, only unknown),
   and `KvRemaining` — the in-memory estimated-remaining-KV counter
    (`reanchor(tokens)` resets, `subtract(tokens)` is unclamped and a no-op when
    never anchored, `add(tokens)` re-adds — the **inverse of `subtract`**, the
    failover charge-rollback primitive, same no-op-when-unanchored contract,
    `value()` returns `int | None`). All three are **reused once
    per backend** by the app (`BackendState`); the parsers are shared across
    backends.
- **`src/gate/poller.py`** — `run_poller(client, cache, url, interval_s, *,
  counter, target_frac, capacity_cache, model_url, model_refresh_s, registry,
  backend_name, explicit_models, capacity_unavailable, on_reanchor,
  on_engine, on_metrics_unavailable,
  routing_order)`: **one instance per
  backend** (the app's lifespan starts one task per configured backend). Each
  tick makes ONE `GET` via `fetch_metrics` and, on an observed (HTTP 200) body,
  updates **that backend's** usage + per-model caches, **re-anchors its
  `KvRemaining` counter** to `anchor_remaining_tokens(capacity, target_frac,
  usage_frac)`, and updates its `CapacityCache`. On every re-anchor it invokes
  the `on_reanchor` callback **synchronously** (no `await` between) — the app
  wires it to increment the backend's monotonic `anchor_seq` (the failover
  charge-rollback guard). **Fails open** on any
  transport error / non-200 (keeps ALL state — caches and counter — no crash).
  **Engine callbacks (state-change only, synchronous, logged-not-fatal):**
  `on_engine(engine)` fires when the detected engine **changes** (first
  detection included, and a change to or from `unknown`); the poller keeps a
  local `last_engine` that only observed (HTTP 200) bodies update, so
  transport errors and non-200 responses never change it.
  `on_metrics_unavailable(state)` fires with the **new state** whenever the
  *unavailable* state changes (either direction): `True` for a non-200
  response or a 200 body detected `unknown`, `False` for a 200 body detected
  `vllm`/`sglang`; a **pure transport error leaves the state unchanged**
  (the outage is already visible via `gate_metrics_fresh`). On a transition
  **into** unavailable the poller additionally logs the `--enable-metrics`
  WARNING **once** (SGLang only serves `/metrics` when launched with
  `--enable-metrics`); on recovery nothing is logged. **Capacity-missing is
  per-backend fail-open (no exit):** an observed body missing a usable
  KV-cache capacity (the *detected engine's* capacity gauges — the
  `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
  `vllm:cache_config_info` for vLLM;
  `sglang:kv_cache_total_tokens` / `sglang:max_total_num_tokens` for SGLang;
  "no recognizable KV-cache capacity" for `unknown` — `_capacity_gauge_names`
  makes the error message **engine-aware**) logs an error,
  leaves that backend's counter unanchored (its autoconfig layer fails open),
  and invokes the `capacity_unavailable` callback (surfaced by the
  `gate_backend_capacity_unavailable{backend}` gauge); the loop keeps running.
  Also does **model discovery** when `model_url`/`registry`/`backend_name`
  are wired (first tick immediately, then every `model_refresh_s`): `fetch_models`
  (a `GET /v1/models` + `parse_v1_models`) and `registry.sync(backend_name,
  owned)` where `owned` is the explicit `models` set when a non-empty
  `explicit_models` tuple is given, else the discovered served set; a failed
  fetch or unparseable body keeps the last known map (auxiliary, never fatal).
  Also runs the **`order` ⊆ serving dead-candidate check** when `routing_order`
  is wired: on each successful discovery tick, for every model whose `routing:`
  `order` names this backend, if the model is NOT in this backend's owned set
  the backend is a **dead candidate** for that model and a WARNING is logged
  **once per (model, backend) pair** — logged, not fatal (the walk fails open).
- **`src/gate/models.py`** — `parse_v1_models(text) -> list[str] | None`
  (pure; parses an OpenAI `GET /v1/models` body `{"data": [{"id": ...}]}` into
  the ordered id list; `None` on non-JSON, missing/non-list `data`, or a
   missing/empty `id`). `ModelRegistry` — the stateful `model → ordered
   candidate list` routing map, single-asyncio-loop contract (pollers write via
   `sync`, request handlers read via `resolve_candidates`; no locks):
   `register_backend(name, *, is_default)` (once per backend at app build;
   config/registration order is the candidate order), `sync(name, owned_models)`
   (reconcile one backend's owned set; a model left owned by no backend stops
   resolving — the unknown-model path), **`resolve_candidates(model) ->
   tuple[str, ...]`** — the v2 routing lookup: the ordered backend-name tuple
   serving `model` (single-owner ⇒ 1-tuple; multi-owner ⇒ **all owners** in
   registration order — **duplicate model ids are legal**, the normal
   multi-candidate case; unowned ⇒ `()`), `resolve(model) -> str | None`
   (retained — returns the single resolved owner for the `owned_by` scalar
   form; for a multi-owner model it is the collision winner: the `default:
   true` backend if any owner is the default, else the first-registered owner;
   the conflict is logged once per (model, backend) pair), `items()` (every
   owned model as `(model, resolved_backend)`, for the aggregate
   `GET /v1/models` and `/healthz`). The routing policy (not a collision
   policy) chooses among a model's candidates.
- **`src/gate/router.py`** — `decision(usage_pct, ctx_tokens, thresholds) ->
  Decision`. Pure. The heart of the tiered gate: highest-tier-only selection +
  inclusive `ctx_tokens <= max_context` comparison. Also the always-on
  autoconfig layer (all pure, fraction/token space): `AutoPolicy(token_margin,
  retry_min_s, retry_max_s)`, `decision_auto(remaining_tokens, ctx_tokens,
  policy) -> Decision` (admits when `ceil(ctx × token_margin) <= remaining`,
  inclusive), `anchor_remaining_tokens(capacity, target_frac, usage_frac) ->
  int` (`floor(capacity × max(target_frac − clamp(usage,0,1), 0))`),
  `scaled_retry_after(effective, remaining, min_s, max_s) -> int` (integer in
  `[min_s, max_s]`; `remaining <= 0` → `max_s`), and
  `combine_decisions(tier, auto) -> Decision` (AND; both-reject → max timeout,
  tie → tiered).
- **`src/gate/routing.py`** — the pure **routing-selection** layer.
  `RoutingSpec(policy, order, threshold_tokens=None)` and `CandidateView(
  usage_frac, remaining_tokens, capacity_tokens, token_margin)` (fraction/token
  space; `None` = stale/never-fetched feed or unanchored counter); the five
  policy-name constants + `POLICIES`; and `select_backend(spec, candidates,
  ctx_tokens, usage_views) -> int | None` (pure; returns the **index** into
  `candidates` to attempt first — the app's failover walk advances from there —
  or `None` when `fill` finds no eligible candidate / `candidates` is empty).
  Rules: `round_robin`/`primary_fallback` → `0` (the app's stateful per-model
  RR index offsets the RR selection; PF relies on the skip-on-reject walk);
  `large_small` → `0` iff `ctx_tokens` is known and `ctx_tokens >=
  threshold_tokens` (inclusive), else `1` (clamped to `0` with a 1-candidate
  order); `even` → the index **minimizing the spread** (max−min) of the
  *projected* usage fractions — candidate `i` at `usage_i + ceil(ctx × margin_i)
  / capacity_i`, every other candidate at its current usage — unknowns ranked
  **last** (in index order), ties → lowest index (**not** greedy
  least-loaded); `fill` → the lowest index whose feed is stale/unanchored
  (always eligible — fail-open) or whose `ctx_tokens is None` (fail-open) or
  where `ceil(ctx × margin) <= remaining` (the exact `decision_auto` admission
  test); eligible-but-unknowns are interleaved in **global config order** (not
  ranked last — this is what keeps `fill` ≡ `primary_fallback` on stale
  feeds). The stateful per-model RR index and the failover walk live in
  `app.py` (thin wrappers, same single-asyncio-loop contract as `KvRemaining`).
- **`src/gate/proxy.py`** — `proxy_request(httpx, request, base_url, *,
  stream) -> Response`. Transparent forward of method/path/query/headers/body
  to the given `base_url` (the routed backend's `http://host:port`); streams
  SSE verbatim; strips hop-by-hop headers; propagates upstream status (does
  not mask 5xx).
- **`src/gate/stats.py`** — `GateStats`: the stats edge. Owns a **per-app
  `prometheus_client.CollectorRegistry`** (never the global default), the
  `gate_*` metric objects, and `render()` (Prometheus text exposition for
  `GET /metrics`). In-memory only; resets on restart by design.
    `record_forwarded/rejected` (carry `endpoint`, `model`, `result`, and a
    **`backend` label** — the final backend: the forwarder, the max-timeout
    rejector on 429 exhaustion, or the sentinel `"none"` on 502 exhaustion —
    because with duplicate model ids legal the backend is no longer
    recoverable from the model), `record_failover(model, from, to, reason)`
    (the `gate_routing_failovers_total{model, from, to, reason}` counter —
    pre-stream failover skips, reason ∈ `transport` | `upstream_5xx` |
    `reject`), `set_kv_usage` (fraction→percent, removes stale model series;
    the app unions all backends' by-model maps, max on key collisions),
    `set_freshness(backend, fresh, age_s)`, `set_remaining(backend, int | None)`
    (the `gate_kv_cache_remaining_tokens{backend}` gauge — the gate's own live
    estimate of remaining KV tokens; `None` renders `NaN`),
    `set_backend_capacity_unavailable(backend, bool)` (the
    `gate_backend_capacity_unavailable{backend}` 0/1 alert gauge),
    `set_backend_engine(backend, engine)` (the
    `gate_backend_engine{backend,engine}` gauge — `1` for the currently
    detected engine (`engine` ∈ `vllm` | `sglang` | `unknown`), `0` for the
    other two, `unknown=1` initially — set for every backend on each
    `/metrics` render), and `set_backend_metrics_unavailable(backend, bool)`
    (the `gate_metrics_endpoint_unavailable{backend}` 0/1 alert gauge — `1`
    while the backend's `/metrics` is reachable-but-unrecognizable, `0` once
    a real engine is detected).
    `gate_config_info` serializes the backend structure (`backends_json`:
    name/host/port/default + which knobs are overridden) **and** the `routing:`
    section (`routing_json`: `[[model, policy, [order...]], ...]`, `[]` when
    absent — startup-static for operator audit) but **not** the `models` lists
    (discovered data would go stale).
- **`src/gate/app.py`** — Builds the FastAPI app: routes
  (`POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`,
  `GET /healthz`, `GET /metrics`), the 429/502 builders, and the wiring of the
  per-backend pollers + caches + proxy + stats (each decision is recorded as a
  pure side effect; `/metrics` renders the stats — including the two lines
  `stats.set_backend_engine(bs.name, bs.engine)` and
  `stats.set_backend_metrics_unavailable(bs.name, bs.metrics_unavailable)`
  per backend — and always returns 200). Owns
  one `BackendState` per configured backend (`MetricsCache`, `CapacityCache`,
  `KvRemaining`, resolved thresholds/autoconfig policy/target_frac, `base_url`,
  the per-backend `capacity_unavailable` flag, **`engine`** (currently
  detected engine, `ENGINE_UNKNOWN` until first detection) and
  **`metrics_unavailable`** (the per-backend reachable-but-unrecognizable
  flag, `False` initially) — the latter two set by the app's
  `_on_engine_setter` / `_metrics_unavailable_setter` closures, which the
  lifespan wires to the poller's `on_engine` / `on_metrics_unavailable`
  callbacks; the flag's `False` recovery transition rides the
  `on_metrics_unavailable` callback, **not** the engine setter — and the
  per-backend monotonic
  `anchor_seq` — incremented by the poller's `on_reanchor` callback; the
  failover charge-rollback guard), the per-model `RoutingState` (stateful
  per-model `round_robin` index — advances only on a successful forward), the
  per-model `RoutingSpec` map (from `cfg.routing`), plus the shared
  `ModelRegistry` and the default-backend name. The lifespan starts **one
  poller task per
  backend** (each with its own fatal done-callback, which now fires only on an
  unexpected task death — the capacity path no longer raises). Per generation
  request: resolves the body's `model` to a **candidate set**
  (`registry.resolve_candidates`; empty ⇒ single candidate = the default
  backend, the `round_robin` spec), filters the `routing:` spec's `order` to
  serving backends only (a backend named in `order` that does not serve the
  model is a **dead candidate** the walk skips — the poller separately warns
  about it; if the filter drops every candidate, the default backend is used),
  selects the first candidate with the pure `select_backend` (a `fill`
  all-full `None` ⇒ no candidate is proxied, but the walk still runs — without
  proxying — to collect the per-candidate `retry_after` values for the
  max-Retry-After 429), then runs the **failover walk** (per candidate: the
  **staleness
  gate** (stale/never-fetched feed → both layers fail open for it regardless of
  the counter — a stale candidate is *not* skipped), the two AND-combined
  admission layers via `combine_decisions` on **that candidate's**
  thresholds/counter/policy; a reject is a **skip** (skip-on-reject); an allow
  charges **that candidate's** counter
  (`counter.subtract(ceil(ctx × margin))`) **before** the proxy await, then
  proxies to its `base_url`; a **pre-stream transport failure or upstream 5xx**
  is also a skip — the charge is rolled back via `KvRemaining.add` **only if
  the candidate's `anchor_seq` is unchanged since the charge** (a re-anchor in
  between makes the fresh anchor authoritative; the charge is dropped); once a
  streamed response has started, errors surface as-is (no walk); the walk never
  short-circuits). **Exhaustion:** any candidate 429'd → one 429 with
  `Retry-After = max` of the candidates' retries (reported rejector = the
  max-timeout backend, tie → first in walk order; the message names it and the
  model); every candidate transport-failed (no HTTP answer) → **502** (never
  200/429; the body names the backends tried; stats record the sentinel
  `backend="none"`); a pre-stream 5xx on the **last** candidate is
  **propagated as-is** (status and body unmasked) and recorded as forwarded
  with that backend. Each skip is counted by
  `gate_routing_failovers_total` (and logged `routing_failover model=… from=…
  to=… reason=…`). `GET /v1/models` is gate-local (aggregate of the registry's
  known models with an `owned_by` extension — a **list** of backend names when
  the model is served by 2+ backends, a **scalar** when single-owner; always
  200, never proxied).
  `GET /healthz` carries a per-backend array (`name`, `metrics_age_s`,
  `kv_usage`, `kv_cache_capacity_tokens`, `kv_cache_remaining_tokens`) plus the
  known models (each with `owned_by` — scalar or list — and `routing`, the
  policy in effect: the `routing:` entry's policy or `round_robin`).
- **`src/gate/main.py`** — `main()` entrypoint: `logging.basicConfig` at the
  `INFO` default (so a `ConfigError` still logs before the config exists),
  load config, then apply the config's `log_level` to the root logger
  (`logging.root.setLevel`), log **one INFO line per backend** (name,
  host:port, `[default]`, effective tier count — `N tier(s)` or
  `none — autoconfig only` — and the four autoconfig knobs with `(override)`
  marking the per-backend values), then **one INFO line per `routing:`
  entry** (model → policy + order, plus `threshold_tokens` for `large_small`;
  an empty `routing:` section logs nothing), build the app (the pollers are
  started by the app lifespan, not here), and run uvicorn (the config's
  `log_level`, lowercased, is passed as `log_level=`). There is **no CLI** —
  argv is ignored (autoconfig is always on).
- **`tests/`** — `test_tokens.py`, `test_router.py`, `test_metrics.py`
  (incl. `detect_engine` per the fixed precedence by family-name presence —
  vllm / sglang-new / sglang-legacy / both-sglang → sglang /
  capacity-without-usage → unknown / unparseable → unknown / **all-NaN vLLM
  usage → vllm** / vLLM-wins-when-all-three-families-present; the per-engine
  `usage` / `usage_by_model` / `capacity` parsers incl. the new→legacy SGLang
  fallbacks, `None`-not-`{}` by-model, the byte-identity edge — all-NaN vLLM
  usage + valid vLLM capacity → `engine="vllm"` with `capacity` parsed; and
  `fetch_metrics` end-to-end per engine),
  `test_models.py` (parse + registry, incl. `resolve_candidates`: single-owner
  1-tuple, multi-owner ordered tuple, removal), `test_routing.py` (pure
  `select_backend` per policy — RR/PF = 0 at selection, `large_small`
  at/below/above the inclusive threshold + 1-candidate clamp, `even` ranking
  incl. unknowns-last and ties, `fill` first-fit / all-full → `None` /
  unknowns eligible, spec/order edge cases), `test_config.py` (backends +
  `BACKENDS_JSON` + validation + `routing:` parsing/validation — bad policy,
  empty order, unknown backend in order, `threshold_tokens` iff
  `large_small`; `log_level` file/env parse + validation; duplicate model
  ids across backends **now legal**),
  `test_stats.py`,
  `test_api.py` (ASGI end-to-end with fake vLLMs **and a fake SGLang** —
  multi-backend routing, per-policy routing, pre-stream failover + charge
  rollback, exhaustion 429-max / 502 / last-5xx-propagated, `/v1/models`
  duplicate `owned_by`, per-backend + per-model `routing` in `/healthz`, the
  `backend` label and `gate_routing_failovers_total`; plus SGLang coverage:
  a legacy-SGLang backend end-to-end (engine gauge
  `gate_backend_engine{...,sglang}=1`, `gate_metrics_endpoint_unavailable=0`),
  a mixed vLLM+SGLang fleet with independent per-backend admission, and a
  SGLang-without-metrics backend (gauge `=1`, engine `unknown`, the
  `--enable-metrics` warning logged once)), `test_poller.py` (incl. the
  `on_engine` / `on_metrics_unavailable` state-change semantics and the
  engine-aware capacity-missing message), `test_stats.py` (incl. the two new
  per-backend engine gauges), `test_main.py`.
- **`Dockerfile`**, **`docker-compose.example.yaml`**, **`config.example.yaml`**,
  **`Makefile`**, **`README.md`** — packaging, operator reference, and docs.

**Proxied endpoints (v1):** `POST /v1/chat/completions` and
`POST /v1/completions` (both consume KV cache; each is routed to the candidate
backends owning the request body's `model` — via the `routing:` policy, or the
default backend when the model is unowned/unparseable — with a pre-stream
failover walk across the candidate `order`). `GET /v1/models`
is gate-local (the aggregate model list with an `owned_by` extension — a list
of backend names for a model served by 2+ backends — never
proxied, always 200). `GET /healthz` for liveness (per-backend state + the
known models with their `routing` policy in effect),
`GET /metrics` for the gate's own Prometheus stats (gate-local, never proxied,
never 429s — always 200). Everything else → 404. No auth, no TLS
termination (v1).

## Smoke / test commands (no external services required)

Quick validation while iterating — no vLLM instance or API keys needed:

```sh
python -m venv .venv && source .venv/bin/activate   # or: uv venv
pip install -e ".[dev]"

ruff check .                       # lint
mypy src/                          # typecheck
pytest -q                          # unit + ASGI integration tests
pytest --cov=gate --cov-fail-under=80   # 80% coverage gate
docker build -t vllm-gate .        # container build (no push)

bash scripts/local-ci.sh           # full local CI — mirrors .github/workflows/tests.yml
make ci                            # same as the above (Makefile target)
make format                        # ruff format src/ tests/
```

`scripts/local-ci.sh` (or `make ci`) is the **exact local equivalent of
`.github/workflows/tests.yml`**: it runs the same checks, in the same order
(source-tracking, `ruff format --check`, `ruff check`, `mypy` +
`compileall`, pytest with the 80% coverage gate, `pip-audit` dependency
audit, `pre-commit run --all-files`, container build), with pass/fail
tracking instead of aborting on the first failure. A green local run should
mean a green CI run; the container job is skipped (with a warning) when
docker is absent.

The `tests/test_api.py` suite drives the gate end-to-end against fake vLLMs
**and a fake SGLang** (`httpx.ASGITransport` + a mock upstream transport —
including multi-backend routing, per-policy routing, pre-stream failover, and
the SGLang cases), so the allow / 429 / fail-open / streaming / routing /
failover paths are all exercised without a real model;
`tests/test_routing.py` unit-tests the pure `select_backend` selection for
all five policies.

## Working copies: always work in a worktree, never in the main checkout

**Work must NEVER be done directly in the main working copy** (the repository's
primary checkout). The main checkout is the user's and any other session's
living workspace: it may hold uncommitted work that belongs to neither the task
at hand nor the current session. Editing, staging, or committing in it can
destroy that work or silently fold it into your commit.

**ALWAYS create a dedicated worktree for any agent work**, no matter how small
the change — a single-line doc fix included:

```sh
git worktree add --detach <dir> <base-ref>   # e.g. <dir>=/tmp/opencode/<task>, <base-ref>=main
```

- Do all edits, CI runs, reviews, and commits inside that worktree. To land
  work on a branch, advance the branch ref after committing (e.g.
  `git update-ref refs/heads/<branch> <sha>`); never edit files in the main
  checkout.
- If the main checkout appears to need updating to a branch you advanced, sync
  it with `git reset` (mixed) + `git checkout HEAD -- .` — never by hand-editing
  its files — and treat any residual working-tree differences as other
  sessions' work to leave untouched.
- When the work is merged, remove the worktree (`git worktree remove <dir>`).

**Greenfield exception (bounded).** This repository's `main` has **no commits
yet**, so `git worktree add <dir> main` cannot resolve a base ref. Until the
first commit exists, the main checkout *is* the only checkout and is empty, so
the initial bootstrap commit(s) are made directly in it. The moment `main` has
a commit, the worktree rule above is mandatory again with no further exception.
Do not let this exception become a habit: after the first commit, always use a
worktree.

## Development style (non-trivial features and fixes)

For any non-trivial feature or fix — one spanning multiple commits, multiple
architecture sections, or with non-obvious interdependencies:

1. **One branch per feature.** Cut a feature branch off `main`
   (e.g. `feat/<name>` or `fix/<name>`). Develop the whole feature on that
   branch; `main` only receives finished, merged work.
2. **Sub-branch each non-trivial subset.** When the feature splits into
   subsets that are individually non-trivial (i.e. each would reasonably take
   more than one commit), give each subset its own sub-branch cut from the
   feature branch (e.g. `feat/<name>/<subset>`). Implement each sub-branch in
   its own dedicated worktree so its in-flight work stays isolated from the
   working trees of other sub-branches that may be developed in parallel
   (rule 3) — a shared checkout cannot hold two subsets' uncommitted work at
   once.
3. **Develop in parallel where the dependency DAG allows.** Map the feature's
   pieces into a DAG: which pieces can be built concurrently and which need
   other pieces complete before they can begin. Work independent branches in
   parallel to the greatest reasonable extent; start a branch only when its
   prerequisites are already merged into the feature branch. (For this project
   the natural DAG is: `config` → {`tokens`, `metrics`, `router`} → `proxy` →
   `app` → `main` → packaging/CI/docs; `tokens`, `metrics`, `router` are pure
   and parallelizable.)
4. **One commit per segment or sub-segment.** Each segment of the feature — or
   of a subset — gets its own commit. Even when a feature focuses on one
   thing, if the plan splits it by architecture section (e.g. config →
    decision → proxy), the commits are split the same way. The Git-management
    gates (CI checks, per-commit two-model review, attribution block)
    apply to each commit individually at merge time, not to the feature as a
    whole.
5. **Keep the review pipeline honest.** Commits may be made on the branch
   before they are reviewed, but the review pipeline must not lag: review
   findings are addressed inline via history rewriting *within the unmerged
   branch only* (see rule 10), and no branch is merged to `main` or pushed
   as a PR until **every** commit on it has been reviewed by **both** `gemma`
   and `qwen` (each commit, each reviewer — see rule 10 for running them in
   parallel) and carries proper attribution (see Git-management, steps 2–3
   and 7).
6. **Merge sub-branches into the feature branch.** When a non-trivial
   sub-branch (more than one commit) is complete, merge it back with
   `git merge --no-ff` so each subset stays visible as its own merge in
   history. A sub-feature that requires only one commit has no merge commit to
   keep it visible: merge it with a fast-forward merge (`git merge --ff-only`)
   so the single commit lands linearly on the feature branch.
7. **Merge the feature to `main` with `--no-ff`.** When the feature is
   complete (all sub-branches merged, all gates green), merge the feature
   branch into `main` with `git merge --no-ff`.
8. **Rebase along the way to keep merges conflict-free.** Rebase sub-branches
   (and the feature branch) as work progresses so no merge ever produces
   conflicts. This also lets the code be tested in the state it will have after
   merging — before the merge is actually made: rebase onto the merge target,
   run the CI gates against that state, then merge.
9. **Keep the todo list current at all times.** When working from a todo list,
   update it in real time, not in batches: the moment an item is started, mark
   it `in_progress`; the moment an item is finished (and verified), mark it
   `completed`. A stale todo list hides progress and misrepresents state — an
   item that is done but still shown as pending, or work that is underway but
   still shown as pending, is a reporting defect the same way a skipped check
   reported as run is one.
10. **Treat a pre-merge branch as a PR stack.** A feature or sub-branch that
    is not yet merged can consider its work as if it were a stack of PRs:
    **before** merging, a branch may run all of the necessary reviews for
    all of its commits **in parallel** (running multiple reviewers
    simultaneously against different commits) rather than serially, and
    feedback is addressed inline via history rewriting — amends and
    rebases — **WITHIN THE UNMERGED BRANCH ONLY**. Never rewrite history on
    `main` or on any branch already merged; those commits are final.
    Running reviewers in parallel is a scheduling optimization only: it does
    **not** relax the per-commit, per-reviewer gate (Git-management steps
    2–3 and 7) — a single review of multiple commits **never** counts as a
    sufficient review for any of those commits. Each commit still needs its
    **own** review **by both** `gemma` and `qwen` and its own attribution
    before merge or PR creation.

## Git management

Before merging any complete change to `main`, or pushing it as a PR:

1. Run the full CI checks locally with the canonical local entry point —
   `bash scripts/local-ci.sh` (or `make ci`). That script is the exact local
   equivalent of `.github/workflows/tests.yml` and runs, in order: source
   tracking (`scripts/check-ignored-sources.sh`), `ruff format --check`,
   `ruff check .`, `mypy src/`, `pytest --cov=gate --cov-fail-under=80`, and
   `docker build` (skipped with a warning if docker is absent). Fix every
   issue reported by these checks, then rerun `scripts/local-ci.sh` until it
   passes, before merging the change or pushing it as a PR.
2. **Require a code review by `qwen` and `gemma` as a mandatory gate before
   merging to `main` or pushing a PR.** Before any branch is merged to
   `main`, or pushed as a PR, every commit on it must be reviewed —
   individually, per commit — by **both** the `qwen` and `gemma` families.
   A multi-commit branch may run all of those reviews **in parallel**
   (multiple reviewers simultaneously against different commits; see
   Dev-style rule 10: a pre-merge branch is a PR stack), with feedback
   addressed inline via history rewriting within the unmerged branch only.
   Running reviews in parallel is purely a scheduling optimization: a single
   review of multiple commits **never** satisfies the per-commit requirement
   for any of those commits. Each commit **MUST** have its own review by both
   families (or, where a family is unavailable, its own review by the
   remaining family — below) and proper attribution before merge or PR
   creation (step 7). Do **not** merge to `main` or push a PR until the
   required reviews have been produced.

    **Required reviewers: both `qwen` and `gemma`.** Every commit requires a
   review from **each** of the `qwen` and `gemma` families. Prefer running
   each required review from a model **different from the authoring
   model** (e.g. a different provider, or a distinct qwen/gemma model when
   the author is of that family). If the preferred distinct-family review
   fails **twice** (the reviewer flakes, times out, or produces no usable
   verdict on both attempts), a review from the **authoring model's own
   family** is acceptable in its place — including the authoring model
   itself. When even that fallback is unavailable, the remaining family's
   review alone satisfies the requirement for that commit; a single family's
   repeated failure is not a blocker. The commit's attribution block records
   exactly which model(s) actually produced each review (step 7), and
   records `flake: <family>` for any required slot that ultimately had no
   review.

   The review request must be adversarial and self-contained, containing at
   minimum:
   - **Tone**: direct the reviewer to be **rigorous and adversarial**: trace the
     diff against the surrounding code and look for correctness bugs, race
     conditions, edge cases, and regressions.
   - **Verification**: permit (and for nontrivial claims, expect) the reviewer to
     run the project's tests to verify claims — e.g. the affected test files via
     `pytest -q <affected-test-file>`, or the full suite when needed
     (`pytest -q`).
   - **Change context**: what the code did before, what it does now, and why
     (including the relevant non-obvious invariants the change must preserve —
     see "Known gotchas" below).
   - **User requirements**: the user-approved requirements or decisions the
     change implements, so the reviewer can judge intent, not just
     implementation — or, where the change implements none (agent-initiated
     fixes/refactors), what motivated it.
   - **Review scope**: an explicit check-at-minimum list tailored to the change
     (boundary conditions, the percentage-vs-fraction comparison,
     highest-tier-only selection, inclusive `<=` comparison, fail-open paths,
     SSE streaming integrity, consumer consistency, off-by-one risks,
     population/semantics of new config parameters, test meaningfulness, stale
     docs/comments — as applicable).
   - **Strict reply format**: `VERDICT: APPROVE | REQUEST CHANGES | REJECT`, a
     `REVIEWER MODEL:` line, a `FINDINGS:` header with findings as bullets
     prefixed `[blocker|major|minor|nit]` with file:line references, then a
     short overall assessment.

    The review must cover correctness, adherence to existing codebase patterns,
    the CI/coverage results, and any security or regression risks.

    **Validate review feedback before acting on it.** Reviewer findings are
    claims, not commands: before fixing anything, verify the claim against
    the actual code (trace the cited path, run the suggested repro or the
    affected tests, e.g. `pytest -q <affected-test-file>`). Address the
    finding only when it checks out; invalid findings are recorded as
    false positives and dropped (or pushed back on), never blindly applied.
    Valid `major`-or-above findings must be addressed before the commit
    proceeds; valid `minor`/`nit` findings are worth addressing when
    reasonable. Reviewers differ in reliability — `gemma` in particular is
    prone to false positives — so extra skepticism toward its findings is
    expected, and a `REQUEST CHANGES` verdict never by itself blocks a merge
    (step 3 governs the outcome).
3. **Present the review results to the user and ask how to proceed.**
   Surface **each** reviewer's findings and recommendation (both `gemma` and
   `qwen` where both were run) to the user in exactly this shape: a
   `VERDICT: APPROVE | REQUEST CHANGES | REJECT` line and a `REVIEWER MODEL:`
   line naming the reviewing model, followed by a bulleted findings list in
   which every bullet is prefixed with its severity (`[blocker]`, `[major]`,
   `[minor]`, or `[nit]`); then ask the user how to proceed. Before
   presenting, validate the findings against the code (see the validation
   note in step 2) and mark which check out; valid `major`-or-above findings
   must be addressed (and the commit amended) before presenting. When a
   reviewer's `REQUEST CHANGES` or other findings turned out to be false
   positives, say so explicitly — the verdict is disclosed accurately as
   given, and the commit message clarifies that the `REQUEST CHANGES` was
   due to false positives (step 7) rather than masking or softening the
   verdict. A `REQUEST CHANGES` does not by itself prevent merging when its
   findings are invalid; it does block merging while valid `major`+ findings
   remain unaddressed. Options include merging or pushing as-is, revising
   per the review, or abandoning. Do not treat a clean or critical review as
   an automatic decision — the user's final call is definitive and cannot be
   overridden.
4. Update all relevant documentation to reflect the new reality, including
   `AGENTS.md`, `README.md`, `config.example.yaml`, and any other checked-in
   documentation affected by the change.
5. Confirm the documentation and runtime metadata agree, then commit the
   complete change to git only after the user has decided how to proceed from
   the review.
6. When adding a footer or co-author attribution, include the agent name (e.g.
   OpenCode, FreeBuff, CodeBuff, Hermes Agent, etc.) and model name (e.g. GPT-5.6
   Luna, Big Pickle, DeepSeek v4 Flash 0731, etc.).
7. **Attribute every commit message immediately after its detail body —
   truthfully.** Every commit message must include, immediately after the commit
   message detail (the explanatory body) and prior to any footer (such as a
   `Co-Authored-By:` block), an attribution block of exactly this form:

    ```
    Written by: [model] ([agent])
    Reviewed by: [review-model-1] ([review agent]) — VERDICT   # one line per reviewer that actually ran
    Reviewed by: [review-model-2] ([review agent]) — VERDICT
    ```

    One blank line separates the block from the message detail above (the
    explanatory body when one exists, otherwise the subject line) and from any
    footer below. `[model]` / `[review-model]` are human-readable display names,
    e.g. `GLM-5.3 Flash` / `Beast Qwen3.8 27B`; the parenthesized agent is the
    tool that produced the change or ran the review (e.g. `CodeBuff`, `OpenCode`,
    `FreeBuff`) and is included on all lines whenever the agent is known.
    The `Reviewed by:` lines — **one per reviewer, in the order the reviewers
    ran** — name the reviews of **this commit's diff** produced for it
    (normally one per family: `qwen` and `gemma`), and each `VERDICT` is
    that reviewer's actual verdict surfaced in step 3 from the step-2 review
    (`APPROVE`, `REQUEST CHANGES`, or `REJECT`) — the final verdict for that
    reviewer when the change went through multiple review rounds. A
    same-family review (the author's own family reviewing its own commit,
    allowed only after step 2's distinct-family attempts failed twice) is
    recorded exactly like any other. When a family's review ultimately could
    not be produced (step 2's fallback), the block carries no `Reviewed by:`
    line for that family and instead notes `flake: <family>` so the record
    shows which required slot went unreviewed. A `REQUEST CHANGES` verdict
    whose findings were validated as false positives is still recorded
    verbatim; the commit message detail (or a short `Note:` line after the
    block) clarifies that the `REQUEST CHANGES` was due to false positives,
    so the record shows both the true verdict and the accurate outcome.

    Each `Reviewed by:` line is an evidence-bearing assertion, not a
    formality: it must name a step-2 reviewer of **this commit's diff**,
    obtained **before** the commit is merged to `main` or pushed as a PR.
    Same-family reviews are permitted (step 2), but only where the distinct-
    family preference genuinely could not be met; the line must still name a
    real model that produced a real review of this diff. A commit created
    before its review is legitimate (see step 2 and Dev-style rules 5 and
    10), but a `Reviewed by:` line may only ever describe a review that has
    actually happened: never include it on a change that has not been
    reviewed, never reuse another diff's review, and never fill it in
    prospectively — if the reviews have not happened yet, the merge or PR
    does not happen yet (step 2), and the lines are added or corrected via
    history rewriting within the unmerged branch (Dev-style rule 10).
    Formatting-only or trivially mechanical changes are not exempt. This is
    an honesty-based gate: the message alone does not let a reader
    mechanically verify the claim, so hardening beyond wording (e.g. a
    pre-commit hook validating a review artifact) remains an option if
    violations recur.

## CI / release / local parity

- **CI** (`.github/workflows/tests.yml`): runs on push to `main`, PRs, and
  manual dispatch, as eight jobs — `source-tracking` (fails if any `*.py`
  under `src/` is git-ignored), `format` (`ruff format --check src/ tests/`),
  `lint` (`ruff check .`), `typecheck` (`mypy src/` +
  `python -m compileall -q src/`), `test` (pytest with the 80% coverage gate
  on a Python 3.11/3.12/3.13 matrix, coverage XML uploaded as an artifact),
  `security` (`pip-audit .` — audits the declared dependencies against the
  PyPI advisory DB), `pre-commit` (`pre-commit run --all-files
  --show-diff-on-failure`), and `container` (docker build, no push). The
  `container` job is gated by `needs:` on all the others, so a code failure
  blocks the image build.
- **Release** (`.github/workflows/release.yml`): a `ci` gate job runs the four
  code checks (`ruff format --check`, `ruff check`, `mypy src/`,
  `pytest --cov=gate --cov-fail-under=80`) — not the source-tracking or
  container-build jobs; the `release` job (`needs: ci`) then builds and pushes a
  multi-platform image (`linux/amd64,linux/arm64`) to GHCR on `v*` tag pushes
  and `main` pushes. `:latest` is tagged on `main` and on non-prerelease tags;
  `vX.Y.Z-<prerelease>` tags (e.g. `-rc1`) produce a draft + prerelease GitHub
  Release (generated notes) on tag pushes.
- **Dependabot** (`.github/dependabot.yml`): weekly (Monday 09:00 UTC) updates
  for `pip` (`deps` prefix) and `github-actions` (`ci` prefix).
- **Pre-commit hooks** (`.pre-commit-config.yaml`): run in CI by the
  `pre-commit` job (`pre-commit run --all-files --show-diff-on-failure`) and
  available as dev-time hooks before committing. ruff + ruff-format (pinned to
  the project's ruff version) plus hygiene hooks (check-yaml,
  end-of-file-fixer, trailing-whitespace, check-added-large-files,
  check-merge-conflict).
- **Local parity**: `bash scripts/local-ci.sh` / `make ci` runs the exact
  checks from `tests.yml` in the same order (see "Smoke / test commands").
  `scripts/pre-push-tests.sh` runs the full suite with the 80% coverage gate
  against the committed state (stashing uncommitted work for the run);
  `scripts/check-ignored-sources.sh` backs the `source-tracking` job.

## Dependency policy

Prefer adding a well-maintained dependency over rolling your own, especially
for complex problems and for areas where a hand-rolled implementation would
only approximate the solution.

When considering a dependency:

- Verify it is actively maintained, has a compatible license, and fits the
  problem.
- Prefer small, focused libraries over pulling in a large framework for one
  function.
- Keep roll-your-own code only where a dependency genuinely doesn't cover the
  need.

For this project the runtime footprint is deliberately tiny (`fastapi`,
`uvicorn[standard]`, `httpx`, `prometheus_client` for both its text parser and
its client-side `CollectorRegistry` (gate's own `/metrics`), `PyYAML` for
config parsing). Do not
pull in a tokenizer, an LLM SDK, or a large framework in v1 — the gate is a
proxy, not an inference engine.

## Known gotchas (each is a real invariant; read before touching these areas)

1. **Fail open on any metrics outage.** When vLLM's `/metrics` is unreachable,
   times out, or returns a value older than `stale_after_s`, the gate MUST
   forward (allow) the request — never block. A monitoring outage must never
   take down inference traffic. Do not "helpfully" fail-closed: that converts a
   harmless metrics blip into a total outage of the model. (`router.py`,
   `poller.py`, `metrics.py`)

2. **`kv_pct` is a percentage (0–100); the metric is a fraction (0–1).** The
   comparison is `usage_frac * 100 >= kv_pct`. Comparing the raw fraction to a
   percentage value is off by 100× and silently disables (or always trips) the
   gate. This is the single most likely correctness bug in the project.
   (`config.py`, `router.py`)

3. **Highest-tier-only selection.** When usage crosses multiple thresholds, the
   governing tier is the one with the **largest `kv_pct`** that is crossed;
   lower tiers are ignored. Do not sum, average, or use the lowest crossed
   tier. If no tier is crossed, the request is allowed unconditionally.
   (`router.py`)

4. **Inclusive boundary: `ctx_tokens <= max_context` allows.** Exactly
   `ctx_tokens == max_context` is allowed; only `ctx_tokens > max_context`
   rejects. Off-by-one here changes behavior precisely at the boundary the
   operator configured. (`router.py`)

5. **Per backend, the gate takes the MAX across all
   `vllm:kv_cache_usage_perc` series.** vLLM may emit one series per
    `model_name`; models on one backend share its KV pool (single-engine vLLM),
    so the per-backend max (conservative) is the correct admission input for
    that backend. Do not take the first or the mean. Across backends the gate's
    `gate_kv_cache_usage_pct` gauge unions all backends' per-model breakdowns
    (max on a repeated key — a model id served by 2+ backends, or the synthetic
    `default` label). (`metrics.py`, `app.py`)

6. **The gate is a transparent proxy for the two generation endpoints only.**
   `/v1/chat/completions` and `/v1/completions` are forwarded verbatim to the
   **routed backend's** `base_url`, including `stream: true` SSE — do not
   buffer the response body in a way that breaks streaming, and do not mask
   upstream status codes (a vLLM 5xx must surface as a 5xx, not a 200 or a
   429). `GET /v1/models`, `GET /healthz`, and `GET /metrics` are gate-local
   (never proxied, always 200); the backends' own `/v1/models` endpoints are
   not reachable through the gate. Everything else is 404. (`proxy.py`,
   `app.py`)

7. **Token estimation is a heuristic, biased to over-estimate.** Context is
   `ceil(prompt_chars / chars_per_token) + max_tokens` (headroom default 256).
   It is intentionally approximate and errs high so the gate is conservative.
   Do not introduce a real tokenizer in v1 (it needs model access and a heavy
   dependency). If the body is unparseable JSON, fail open (forward) and let
   vLLM produce the real error — do not reject on a parse error. (`tokens.py`)

8. **No auth, no TLS termination in v1.** The gate trusts the deployment
   network and is a drop-in front of vLLM. Do not add authentication or TLS
   that would break that contract without an explicit, default-off config flag.
   (`app.py`, `config.py`)

9. **Stats are in-memory, reset on restart, and recording never alters the
   decision.** `GateStats` keeps everything in a per-app
   `prometheus_client.CollectorRegistry` — never persist them (add no disk or
   external state): Prometheus `rate()`/`increase()`/`histogram_quantile()`
   handle restarts by design. `GET /metrics` is **unauthenticated** (v1) and
   exposes config values plus traffic stats, so it must be network-protected
   like vLLM's own `/metrics`. Two model namespaces must not be conflated:
   `model` is the request body's `model` field (**the routing key** — it
   selects the backend) while `model_name` is vLLM's served-model label —
   different strings, different label names, different metrics. And recording
   is a **pure side effect**:
   `record_*` must never change the decision, the 429, or the proxy path, and
   `/metrics` always returns 200 (never 429/404). (`stats.py`, `app.py`)

10. **Capacity-missing is per-backend fail-open (no process exit) — but a
    metrics outage is also fail-open.** A backend whose *observed* (HTTP 200)
    `/metrics` body lacks a usable KV-cache capacity (the
    `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
    `vllm:cache_config_info`) cannot anchor its counter: the poller logs an
    error, leaves **that backend's** counter unanchored (its autoconfig layer
    fails open), flips its `capacity_unavailable` flag (surfaced by the
    `gate_backend_capacity_unavailable{backend}` gauge, set to 1), and keeps
    looping — the process does **not** exit (the old
    `CapacityUnavailableError`/exit-1 behavior is gone; killing the whole
    proxy because one backend is misconfigured would take down the healthy
    ones). A merely unreachable, erroring, or stale feed is likewise never
    fatal: it keeps all state and the app fails open via staleness (gotcha #1
    outranks the counter — a metrics outage never blocks traffic). Alert on
    the gauge instead of relying on a crash; the only process exit left is an
    unexpected poller task death (`_poller_fatal` → `os._exit(1)`).
    (`poller.py`, `app.py`, `stats.py`)

11. **Remaining-KV counter semantics (per backend).** Each backend has its own
    `KvRemaining`; it is **re-anchored (reset,
    not additive)** by that backend's poller on every good poll to
    `anchor_remaining_tokens(capacity, target_frac, usage_frac)`; it is
    **subtracted only on forwarded requests**, charging `ceil(ctx ×
    token_margin)` (`0` when `ctx` is unknown); subtraction is **unclamped** (the
    value may go negative — an over-committed signal); and **rejected requests
    charge nothing**. Subtraction happens on every forward even while the feed
    is stale (bookkeeping continues), but the *decision* fails open during
    staleness, so the counter cannot reject then. (`metrics.py`, `app.py`,
    `router.py`)

12. **Two admission layers are AND-combined per backend; the rejector is the
    max-timeout one.** A request forwards only if the routed backend's tiered
    layer **and** its autoconfig layer both allow (no tiers ⇒ tiered always
    allows). When both reject, `Retry-After = max(tier timeout, autoconfig
    retry)` and the **reported rejector** is the layer with the higher timeout
    (**tie → tiered**); the 429 message names the **routed backend and the
    request's `model`** (`backend <name> (model <id>): ...`) followed by the
    rejector-specific text. Do not OR the layers, pick the lower timeout, or
    let one backend's decision influence another. (`router.py`, `app.py`)

13. **`backends` is REQUIRED — there is no zero-config mode anymore.**
    (Supersedes the old zero-config gotcha.) The gate has no other way to
    learn its upstreams: the `backends` list (file key `backends` or the
    `BACKENDS_JSON` env var, which replaces the file entirely) must contain
    at least one backend — an empty/absent list fails with
    `at least one backend is required (file key 'backends' or the BACKENDS_JSON env var)`.
    The legacy `vllm_host`/`vllm_port`
    keys and `VLLM_HOST`/`VLLM_PORT` env vars are **removed**: a config using
    them fails with a migration-hint `ConfigError`. `thresholds` remains
    **optional** (an empty/absent tiered policy just means the tiered layer
    always allows), the four autoconfig knobs still have sane global defaults
    (with per-backend overrides), and there is still **no CLI** (argv is
    ignored; a legacy `--auto` is a harmless no-op). Do not reintroduce a
    zero-config default upstream or a mode/enablement switch. (`config.py`)

14. **The single percentage→fraction conversion is `target_kv_cache_pct /
    100.0` at wiring time, per backend.** The autoconfig path
    (`anchor_remaining_tokens`, `decision_auto`, `scaled_retry_after`) works
    in **fraction (0–1) and token space** and never sees a percentage. Gotcha
    #2 still applies to the *tiered* path (`usage_frac * 100.0 >= kv_pct`); do
    not conflate the two conversions. (`app.py`, `router.py`)

15. **Superseded by gotcha #19 — model ids are no longer globally unique
    across backends.** (Kept in place so the old "uniqueness enforced at
    config load" rule is explicitly retired.) The `routing:` section
    (decision 11) decides which backend gets a request, and duplicate ids are
    the normal multi-candidate case. (Only **duplicate ids within one
    backend's `models:`** list remain a `ConfigError`.) The `model → backend`
    map became a `model → candidate set` map, and the traffic metrics carry a
    `backend` label alongside `model` (gotcha #20).
    (`config.py`, `models.py`, `routing.py`, `stats.py`)

16. **Unknown/unparseable models route to the default backend.** A request
    whose `model` is owned by no backend (or is the unparseable `"unknown"`)
    is forwarded to the backend flagged `default: true` (fallback: the first
    entry), and vLLM returns its own canonical "model not found" error; the
    admission layers still apply on that backend. A model that stops being
    owned (dropped on a discovery refresh) resolves to an **empty candidate
    tuple** (`resolve_candidates` → `()`) and takes the same path — the gate
    never hard-404s a model. (`app.py`, `models.py`)

17. **`GET /v1/models` is gate-local and aggregated (always 200, never 429,
    never proxied) — duplicate-aware.** It lists the registry's known models
    — one entry per model with an `owned_by` extension: a **list** of backend
    names when the model is served by 2+ backends (duplicate ids are legal —
    gotcha #19), a **scalar** name when single-owner — so the backends' own
    `/v1/models` endpoints are *not* reachable through the gate. It is the
    third always-200 gate-local endpoint next to `/healthz` and
    `/metrics` (`/healthz` also carries a per-model `routing` field — the
    policy in effect, the `routing:` entry's policy or `round_robin`).
    (`app.py`)

18. **Model discovery is auxiliary and never fatal.** Each backend's owned
    model set is the explicit `models:` list when present (it wins), else the
    discovered served set from that backend's `GET /v1/models` (first tick
    immediately, then every `model_refresh_interval_s`, default 30.0; env
    `MODEL_REFRESH_INTERVAL_S`). A failed fetch or unparseable body keeps the
    last known map (the poller logs a warning only on a state change). Do not
    make discovery failures reject requests or clear the registry.
    (`poller.py`, `models.py`)

19. **Duplicate model ids are LEGAL (supersedes the v1 uniqueness rule —
    gotcha #15).** A model id owned by 2+ backends (explicit `models:` or
    auto-adopt) is the **normal multi-candidate case**, not a collision:
    `ModelRegistry.resolve_candidates` returns ALL owners in registration
    (config) order, and the request is routed per its **`routing:` policy** —
    the entry's `policy` when one exists, else **`round_robin`** over the
    candidates in config order (decision 11). The `routing:` section is
    **optional** (omitted or `{}` valid) and **file-only** (no env override);
    every malformed entry is a `ConfigError` (policy from the five-value set;
    non-empty `order` of configured backend names; `threshold_tokens` int ≥ 1
    present **iff** `large_small`). The "at most one `routing:` entry per
    model" rule holds via **last-wins**, not a `ConfigError` (`yaml.safe_load`
    silently keeps the last of duplicate mapping keys — safe in outcome).
    `order` is **both** the candidate order and the **failover order**; a
    backend named in `order` that does not serve the model is a **dead
    candidate** the walk skips (each poller logs a WARNING once per
    (model, backend) pair — logged, not fatal). (`config.py`, `models.py`,
    `routing.py`, `app.py`, `poller.py`)

20. **Pre-stream failover + anchor-guarded charge rollback (all policies,
    decision 12).** The candidate `order` is also the failover order: on a
    **transport error or upstream 5xx before the first response byte**, the
    gate skips that candidate and walks the remainder, re-running admission on
    the next; **a 429 from a candidate also triggers the next** (skip-on-reject
    — what makes `fill` ≈ `primary_fallback`, gotcha #21). On a skip, the
    failed candidate's counter charge is **rolled back via
    `KvRemaining.add(ceil(ctx × margin))`**, but **only if that backend's
    `anchor_seq` is unchanged since the charge** (a re-anchor in between makes
    the fresh anchor authoritative — the old charge is dropped, never
    double-counted; the poller increments `anchor_seq` synchronously with each
    re-anchor). The **walk never short-circuits** — every remaining candidate
    is attempted before exhaustion. **Once a streamed response has started,
    errors surface as-is** (no failover — gotcha #6 preserved in-flight).
    Each skip is counted by `gate_routing_failovers_total{model, from, to,
    reason}` (reason ∈ `transport` | `upstream_5xx` | `reject`). (`app.py`,
    `metrics.py`, `poller.py`)

21. **Exhaustion rule + `fill` ≈ `primary_fallback` (decision 12, invariant
    19).** All candidates 429'd (none forwarded) → a single **429** with
    `Retry-After = max` of the candidates' retries (reported rejector = the
    max-timeout backend, **tie → first in walk order**; the message names it
    and the model). All candidates transport-failed (no HTTP answer) →
    **502** — never 200, never 429 (the request never got a capacity answer;
    the body names the backends tried; stats record the sentinel
    `backend="none"`). A pre-stream 5xx on the **last** candidate is
    **propagated as-is** (status AND body unmasked — gotcha #6; the
    single-backend case is exactly this) and recorded as **forwarded** with
    that backend. `fill` and `primary_fallback` **reach the same surviving
    candidate for every request** (both run the same walk; `fill`'s prefilter
    is the exact `decision_auto` admission test, so it can disagree with the
    walk on no candidate): they differ only in selection cost and in one
    corner — a candidate whose autoconfig layer admits but whose **tiered**
    layer rejects (and on stale/unanchored feeds `fill`'s prefilter passes
    every unknown, degenerating to the config-order walk). Both are kept as
    **distinct policies on operator intent** ("fill my pools in order" vs
    "hot primary, backups"); do not collapse them. (`app.py`, `routing.py`,
    `stats.py`)

22. **Engine-specific metric families are detected per body — there is NO
    config knob.** A backend may be a **vLLM or an SGLang** server; the engine
    is **auto-detected from each observed `/metrics` body** by
    `detect_engine` (the single source of truth) in the fixed precedence
    **vLLM → newer SGLang → older SGLang → unknown** (`vllm:kv_cache_usage_perc`
    → `sglang:kv_cache_usage_perc` → `sglang:token_usage` → `unknown`).
    Detection is by **usage-gauge FAMILY-NAME PRESENCE, not sample
    finiteness** — a body whose `vllm:kv_cache_usage_perc` is **all-NaN** (but
    still parses) must still detect as **`vllm`** (the all-NaN case is a real
    input — `tests/test_metrics.py` has a `NAN_ONLY` vector); finiteness is
    the parser's concern (`_finite_samples`), not detection's. `config.py` is
    unchanged — there is no `engine`/`kind` key and no env var. **`unknown`
    and transport failures fail open:** a body with no recognizable KV usage
    gauge (e.g. SGLang launched without `--enable-metrics`, or some other
    OpenAI-compatible server) detects `unknown` — its `UnknownMetricsParser`
    returns all `None`, so the backend never anchors and fails open via
    staleness (gotcha #1); it is surfaced (not silent) by
    `gate_metrics_endpoint_unavailable{backend}` (set to 1) +
    `gate_backend_engine{...,engine="unknown"}` (= 1) and a logged
    `--enable-metrics` WARNING once per transition into the unavailable state.
    The per-model **`model_name` label is shared across engines** (same label
    name and `"default"`-when-unlabeled semantics in both), so the per-model
    breakdown and the `gate_kv_cache_usage_pct` union are engine-agnostic.
    **vLLM behavior is byte-identical** (decision 10): `VllmMetricsParser` is
    the original logic verbatim and the existing `test_metrics.py` vectors are
    the contract — a vLLM body detects `vllm` and yields exactly the values it
    yields today (the all-NaN usage + valid capacity body must NOT take the
    capacity-missing path). Future engines are additive: a new
    `MetricsParser` implementation + a `detect_engine` probe + a `_PARSERS`
    registry entry (touches no shared control flow; only adds a label value to
    `gate_backend_engine`). (`metrics.py`, `poller.py`, `app.py`, `stats.py`)

## Authoritative docs (read on demand)

- `README.md` — quickstart, config reference, decision logic, 429 contract,
  observability (`/metrics` reference, PromQL, scrape config), operational
  notes, v2 roadmap.
- `docs/plans/observability.md` — the observability/stats design plan (metric
  table, two-model-namespaces rationale, median-as-histogram semantics).
- `docs/plans/autoconfig.md` — the autoconfig (always-on KV admission) design
  plan: the two AND-combined admission layers, the remaining-KV counter, the
  staleness gate, retry scaling, the combination rule, and the worked example.
- `docs/plans/multi-backend.md` — the multi-backend / multi-model routing
  design plan: the decision log (per-backend admission state, per-model
  routing, `backends:` replacing `vllm_host`/`vllm_port`, capacity-missing
  per-backend fail-open), the config schema, model discovery, and the
  per-backend observability changes.
- `docs/plans/sglang-backend.md` — the SGLang backend support design plan:
  engine auto-detection from the `/metrics` body (fixed precedence, by
  family-name presence), the `MetricsParser` interface + per-engine
  implementations, the new→legacy SGLang gauge fallbacks, the two new
  per-backend gauges (`gate_backend_engine`,
  `gate_metrics_endpoint_unavailable`) and the `--enable-metrics` warning,
  and the byte-identical-vLLM invariant.
- `config.example.yaml` — the reference configuration with the `backends`
  entry contract (global defaults + per-backend overrides) and the
  `thresholds` entry contract.
- `AGENTS.md` (this file) — architecture map, test commands, git workflow, and
  the load-bearing invariants above.
- `.github/workflows/tests.yml` / `release.yml` — the CI and release
  pipelines; `scripts/local-ci.sh` is their local equivalent (see "CI /
  release / local parity").
