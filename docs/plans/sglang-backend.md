# SGLang backend support — Design Plan

Status: **proposed** (2026-09-16; design Q&A complete — engine auto-detection,
parser interface, per-backend engine + metrics-availability gauges; not yet
implemented)
Base: `main` at `799d24b`

This adds support for **SGLang** servers as gate backends, alongside vLLM.
The gate today reads KV-cache state only from vLLM's Prometheus gauges
(`vllm:kv_cache_usage_perc`, `vllm:kv_cache_size_tokens` / the
`kv_cache_size_tokens` label on `vllm:cache_config_info`). SGLang emits a
different set of gauges under the `sglang:` prefix — and its names have
**changed across releases** — so a vLLM-only parser sees nothing and the
backend silently fails open forever. This feature makes the gate **detect the
engine from the metrics body itself** (no config change, per the user's
direction) and parse each engine's gauges, while keeping vLLM behavior
byte-identical and the whole thing **extensible to further engines** (per the
user's direction that more backends are likely).

---

## Decision log

**2026-09-16 — design Q&A, user-confirmed:**

1. **No config knob. The engine is detected from the `/metrics` body.** The
   user explicitly does not want the operator to report the engine ("Can't we
   easily determine what server it is from the metrics it emits? I don't see
   why we need to have the user report that."). If a knob existed at all it
   would be an `auto` default that tries vLLM first, then newer SGLang, then
   older SGLang — but since detection is unambiguous and free, **there is no
   knob at all**. `Backend`, `BACKENDS_JSON`, and config validation are
   unchanged. Every existing config is valid and behaves identically.
 2. **Detection order is a fixed, documented precedence.** For each observed
    (HTTP 200) `/metrics` body, the engine is the first of these **usage-gauge
    families present** in the body (by **family-name presence**, not by sample
    value/finiteness):
    1. `vllm:kv_cache_usage_perc` present → **`vllm`**
    2. else `sglang:kv_cache_usage_perc` present → **`sglang`**
    3. else `sglang:token_usage` present → **`sglang`**
    4. else → **`unknown`** (a 200 body with no recognizable KV gauge — e.g.
       some other OpenAI-compatible server, or SGLang launched without
       `--enable-metrics` on a release that answers 200 with a sparse body).

    Detection probes the **usage gauge only** (never capacity): the usage gauge
    is the load-bearing signal — a body with no usable usage gauge fails open
    via staleness regardless, so classifying on it is both sufficient and the
    most stable of the three signals (capacity names churn more than usage
    names). The order encodes "vLLM first, then newer SGLang, then older
    SGLang" exactly as the user specified.

    **Present means the family name appears in the parsed metric families —
    regardless of how many samples it has or whether they are finite.** This is
    a load-bearing detail for decision 10 (byte-identical vLLM): a vLLM body
    whose `vllm:kv_cache_usage_perc` series are **all-NaN** but that still
    carries a valid `vllm:kv_cache_size_tokens` must detect as **`vllm`** and
    let the vLLM parser run (it will return `usage=None` but a real
    `capacity`), exactly as today. If detection instead required a *finite*
    usage sample, that body would mis-detect as `unknown`, the
    `UnknownMetricsParser` would return `capacity=None`, and the poller would
    take the spurious capacity-missing path (unanchor + error + gauge) every
    tick — a real behavior change for vLLM. Engine *identity* is a name-based
    signal; sample *finiteness* belongs to the parser's value extraction
    (`_finite_samples`), not to detection. (All-NaN usage is a real input —
    `tests/test_metrics.py` has a `NAN_ONLY` vector.)
3. **The SGLang version split is an *internal* parser choice, not a public
   value.** Newer SGLang emits both `sglang:kv_cache_usage_perc` and the
   legacy `sglang:token_usage`; older SGLang emits only
   `sglang:token_usage`. The gate treats them as **one operator-facing
   engine, `sglang`**, and the internal distinction (which gauge to read) is
   resolved by detection order: `sglang:kv_cache_usage_perc` wins when
   present because it is the **KV-pools-only** signal (`max(full, swa)`),
   whereas legacy `sglang:token_usage` is the **bottleneck across all pools**
   (`max(full, swa, mamba)`) and over-reports KV pressure on hybrid-SSM
   models. So new-name-first is simultaneously the correct detection and the
   correct value. Capacity is derived the same way: newer
   `sglang:kv_cache_total_tokens`, falling back to
   `sglang:max_total_num_tokens` (older, and also present on transition
   releases that emit both).
4. **A clean parser interface with per-engine implementations.** The user
   asked for a parser interface + different implementations. `metrics.py`
   gains a `Protocol MetricsParser` (three pure methods: `usage`,
   `usage_by_model`, `capacity`) with three implementations —
   `VllmMetricsParser` (the **current** vLLM logic verbatim),
   `SglangMetricsParser` (new-name gauge with legacy fallback), and the
    `unknown` case (a no-op parser returning `None`/`None`/`None` —
    `usage_by_model` returns `None`, never `{}`, matching the vLLM contract).
   A `detect_engine(text) -> str` function maps a body to an engine name and a
   `_PARSERS: dict[str, MetricsParser]` registry maps
   engine → parser. **Adding a future engine = one new parser class + one
   detection probe + one registry entry** — no changes to the poller, app,
   config, or stats (the stats engine label is the one surface that gains a
   new label value, documented in §2.5).
5. **No `kind` field, no config validation work, fully backward compatible.**
   `config.py` is **not modified** (the only change to `Backend`-adjacent
   surface is a docstring note that `host`/`port` may point at a vLLM *or*
   SGLang backend). `fetch_metrics` keeps its signature `fetch_metrics(client,
   url) -> MetricsSample` (the engine is a property of the body, not a
   parameter).
6. **`MetricsSample` gains `engine: str`** — the detected engine for that
   body (`"vllm"` / `"sglang"` / `"unknown"`). The non-200 path sets
   `engine="unknown"`. This is the single channel through which the poller
   learns the engine.
7. **Two new per-backend stateful facts, surfaced as gauges.** (a) the
   **detected engine** and (b) whether the backend's `/metrics` endpoint is
   **reachable and recognizable** (i.e. has ever produced a body with a
   recognized engine). These live in `BackendState` and are rendered on each
   `GET /metrics` as `gate_backend_engine{backend,engine}` and
   `gate_metrics_endpoint_unavailable{backend}`. See §2.4/§2.5.
8. **The SGLang-specific `--enable-metrics` hint is a state-change warning +
   a stat gauge, not a config error, and not repeated every tick.** SGLang
   only serves `/metrics` when launched with `--enable-metrics`. Without it
   the poller gets a connection-refused / 404 (or a 200 with no KV gauge on
   some releases) and the backend fails open forever. The gate logs the
   flag-naming warning **once per transition into the unavailable state**
   (the exact "warn on state change only, not every tick" pattern already
   used for model-discovery failures) when a backend's `/metrics` is
   HTTP-reachable-but-unrecognizable (non-200, or a 200 with
   `engine="unknown"`), and the `gate_metrics_endpoint_unavailable` gauge is
   `1` for the whole unavailable period (reset when a real engine is
   detected, so a later re-degradation is warned and gauged again). A pure
   **transport** failure (connection refused) is *not* treated as "metrics
   disabled" — it keeps the existing `metrics fetch failed` warning (server
   down, not a flag problem) and leaves the gauge at its last value (the
   outage is already covered by `gate_metrics_fresh`). This mirrors the
   existing per-backend, logged-not-fatal style (capacity-missing,
   dead-candidate, discovery failures).
9. **Engine changes are allowed and tracked.** Detection re-runs every tick,
   so if the operator upgrades SGLang (legacy→new gauges) or swaps the server
   behind a backend, the gate converges to the new engine on the next good
   body. The engine gauge reflects the *currently* detected engine. There is
   no hysteresis (no "require N consecutive bodies to switch") in v1 — a
   single good body is authoritative; this matches how a single good body is
   already authoritative for usage/capacity.
10. **vLLM behavior is byte-identical.** `VllmMetricsParser` is the existing
    vLLM code moved verbatim (same families, same fallback order, same
    max-across-series, same `None` conditions). A vLLM body detects as `vllm`
    and yields exactly the values it yields today. This is a load-bearing
    invariant and the review's first check.
11. **The two test-only fetch helpers are deleted.** `fetch_usage` and
    `fetch_usage_by_model` are referenced **only** from `tests/test_metrics.py`
    (not from `src/`). They are folded into `fetch_metrics` (the single GET
    that returns the full `MetricsSample`) and deleted, so there is exactly
    one public fetch surface. (User confirmed this is fine.)
12. **Docs: README + AGENTS.md + this plan doc.** A README "SGLang backends"
    section (auto-detection, metric-name fallbacks + the legacy
    `token_usage` semantics caveat, the `--enable-metrics` requirement, the
    two new gauges) and AGENTS.md project-map + a new gotcha entry. This file
    is committed as the design record (consistent with
    `multi-backend.md` / `autoconfig.md`).
13. **Design for expansion.** More engines are expected. The parser
    interface + detection + registry (decision 4) is the seam: each new
    engine is additive (new parser + probe + registry entry), touches no
    shared control flow, and only adds a label value to
    `gate_backend_engine`. The poller/app/config remain engine-agnostic.

---

## 1. Background, Problem, Goal

**Background.** The gate fronts one or more **vLLM** backends. Each backend's
per-backend poller issues ONE `GET /metrics` per tick
(`fetch_metrics`, `src/gate/metrics.py:367`) and parses exactly three things
from the body: the KV-cache **usage fraction**
(`parse_kv_cache_usage` → max across all `vllm:kv_cache_usage_perc` series),
the **per-model usage** breakdown (`parse_kv_cache_usage_by_model` → keyed by
the `model_name` label), and the **capacity in tokens**
(`parse_kv_cache_capacity` → `vllm:kv_cache_size_tokens` gauge, else the
`kv_cache_size_tokens` label on `vllm:cache_config_info`). Those three feed
the per-backend `MetricsCache`, `CapacityCache`, and the `KvRemaining`
counter, which drive the two AND-combined admission layers. Model discovery
(`fetch_models` → `parse_v1_models`) reads `GET /v1/models`, which is
**OpenAI-schema in both vLLM and SGLang** — so it already works for SGLang
unchanged.

**Problem.** SGLang exposes its KV state under the `sglang:` prefix, and the
names have churned:

| What the gate needs | vLLM (current) | SGLang (newer) | SGLang (older) |
|---|---|---|---|
| usage fraction | `vllm:kv_cache_usage_perc` | `sglang:kv_cache_usage_perc` | `sglang:token_usage` |
| capacity tokens | `vllm:kv_cache_size_tokens` / `cache_config_info` label | `sglang:kv_cache_total_tokens` | `sglang:max_total_num_tokens` |
| per-model label | `model_name` | `model_name` | `model_name` |

The per-model **label is identical** (`model_name`) in both engines, so the
per-model breakdown and the `gate_kv_cache_usage_pct` union are engine-agnostic.
Only the **family names** differ. A vLLM-only parser returns `None` for every
SGLang body → the poller never updates the cache (staleness) and never
re-anchors the counter → the backend fails open forever, invisibly.

**Goal.**

1. Let a configured backend point at an **SGLang** server and have the gate
   learn its usage/capacity/per-model state correctly, **with no config
   change** (the engine is detected from the body).
2. Keep **vLLM** behavior **byte-identical** (decision 10).
3. Introduce a **clean, extensible parser interface** so a future engine is
   additive (decision 4, 13).
4. Make the detection **observable and operable**: a per-backend **engine
   gauge** and a per-backend **metrics-endpoint-availability gauge**, plus a
   state-change **`--enable-metrics`** warning (once per transition into the
   unavailable state; decisions 7, 8).
5. Preserve every load-bearing fail-open invariant (gotchas #1, #5, #10, #14)
   and the proxy-transparency contract (gotcha #6).

**Non-goals.** No per-engine *admission* math (the tiered + autoconfig layers
are engine-agnostic; only the *measurement* is engine-specific). No
per-engine config knob (decision 1). No change to `router.py` / `tokens.py` /
`proxy.py` decision math. No auth/TLS. No change to the SGLang requirement
that `--enable-metrics` be set (we *detect and hint* it; we cannot enable it
remotely). No hysteresis on engine switching (decision 9).

---

## 2. Design Summary

### 2.1 Engine model and detection (`metrics.py`)

**New types (all pure):**

- `Engine` — a string union / constants module: `"vllm"`, `"sglang"`,
  `"unknown"`. Exposed as three module-level constants
  (`ENGINE_VLLM`, `ENGINE_SGLANG`, `ENGINE_UNKNOWN`) plus
  `ENGINES: frozenset[str]`. (A `TypedDict`/enum is deliberately *not* used
  to keep the label value a plain string that drops straight into a Prometheus
  label and a log line.)
- `detect_engine(text: str) -> str` — **pure**, the single source of truth for
  "what engine does this body come from". Returns the first match in the
  decision-2 precedence, or `ENGINE_UNKNOWN`. It probes the **usage-gauge
  families by name presence** — a family is "present" iff its name appears in
  the parsed metric families, **regardless of sample count or finiteness**
  (decision 2; the load-bearing reason is spelled out there — an all-NaN
  vLLM usage gauge must still detect as `vllm`). Implementation: parse the
  families once (the existing `text_string_to_metric_families` call the
  parsers already make) and test `family.name` membership; it does **not**
  reuse `_finite_samples` (that helper is for the parser's value extraction,
  where finiteness *does* matter). Unparseable text → `ENGINE_UNKNOWN`. The
  probes, in order: `vllm:kv_cache_usage_perc` → `vllm` ;
  `sglang:kv_cache_usage_perc` → `sglang` ; `sglang:token_usage` → `sglang` ;
  else `unknown`.

  *Why usage-gauge-only detection:* the usage gauge is what the admission
  layers actually consume. A body that has a capacity gauge but no usage gauge
  still cannot drive admission (the re-anchor requires `usage_frac is not
  None`), so classifying on usage is sufficient; and usage names are more
  stable than capacity names, so it is the most reliable discriminator.

### 2.2 Parser interface and implementations (`metrics.py`)

**`Protocol MetricsParser`** (structural, so the registry values need no
inheritance):

```python
class MetricsParser(Protocol):
    """Pure parsers for one engine's /metrics body."""

    def usage(self, text: str) -> float | None: ...
    def usage_by_model(self, text: str) -> dict[str, float] | None: ...
    def capacity(self, text: str) -> int | None: ...
```

All three methods are **pure** and return **`None` on unparseable / absent /
all-non-finite** — matching the current free functions *exactly*. (Note:
`usage_by_model` returns `None`, **never `{}`**, on the empty/absent case —
today's `parse_kv_cache_usage_by_model` returns `None`, and the existing tests
assert `is None`, so a junior implementing `{}` would break them.) `usage`
returns the max finite sample or `None`; `capacity` returns the max finite
positive value or `None`. `usage_by_model` keeps the existing behavior — keyed
by the `model_name` label, `"default"` when unlabeled — which is shared across
both engines, so a small shared `_by_model_from(text, family)` helper
implements it once and each parser passes its own usage family name.

**Implementations:**

- **`VllmMetricsParser`** — the **current** vLLM logic, verbatim:
  - `usage` = max finite sample of `vllm:kv_cache_usage_perc`
    (the body of today's `parse_kv_cache_usage`).
  - `usage_by_model` = today's `parse_kv_cache_usage_by_model`.
  - `capacity` = today's `parse_kv_cache_capacity`
    (`vllm:kv_cache_size_tokens` max positive sample, else the
    `kv_cache_size_tokens` label on `vllm:cache_config_info`).
- **`SglangMetricsParser`** —
  - `usage` = max finite sample of `sglang:kv_cache_usage_perc` **if that
    family has ≥1 finite sample, else** max finite sample of
    `sglang:token_usage` (else `None`). New-name-first is the KV-only
    signal (decision 3).
  - `usage_by_model` = `_by_model_from` over the **same family selection**
    as `usage` (i.e. `kv_cache_usage_perc` if present, else `token_usage`) so
    the per-model map and the overall max agree on which gauge they read.
  - `capacity` = max finite **positive** sample of `sglang:kv_cache_total_tokens`
    **if present, else** max finite positive sample of
    `sglang:max_total_num_tokens` (else `None`).
- **`UnknownMetricsParser`** — all three return `None` / `None` (by_model) /
  `None`. (A body that detected as `unknown` has no usable gauge; returning
  `None` keeps the fail-open path identical to "metric absent".)

**The legacy free functions are refactored, not duplicated.** The three
existing `parse_kv_cache_usage` / `parse_kv_cache_usage_by_model` /
`parse_kv_cache_capacity` are retained as **thin vLLM facades** that delegate
to the (module-level, shared) `VllmMetricsParser` instance — so any existing
import or test that references them keeps working, but the *logic* lives in
one place (the parser). This is the smallest-diff way to satisfy decision 10
(byte-identical vLLM) while introducing the interface.

**`_PARSERS: dict[str, MetricsParser]`** = `{ENGINE_VLLM: VLLM_PARSER,
ENGINE_SGLANG: SGLANG_PARSER, ENGINE_UNKNOWN: UNKNOWN_PARSER}`. (Module-level
singletons — parsers are stateless.)

### 2.3 `fetch_metrics` and `MetricsSample` (`metrics.py`)

- `MetricsSample` **gains `engine: str`** (the detected engine). The dataclass
  becomes `(observed, usage_frac, by_model, capacity_tokens, engine)`.
- `fetch_metrics(client, url) -> MetricsSample` — **signature unchanged**. On
  a **non-200** response it returns
  `MetricsSample(False, None, None, None, ENGINE_UNKNOWN)`. On a **200** it
  runs `engine = detect_engine(text)`, selects `parser = _PARSERS[engine]`,
  and returns
  `MetricsSample(True, parser.usage(text), parser.usage_by_model(text),
  parser.capacity(text), engine)`.
- **`fetch_usage` and `fetch_usage_by_model` are deleted** (decision 11).
  Their `test_metrics.py` cases are folded into `fetch_metrics` (asserting
  `.usage_frac` / `.by_model` on the returned `MetricsSample`).

### 2.4 Poller and app wiring (`poller.py`, `app.py`)

**`run_poller`** — two new **optional** keyword callbacks, same style as the
existing `capacity_unavailable` / `on_reanchor` (invoked synchronously with
the tick that observed the change, state-change only; the poller owns no
other new state):

- `on_engine: Callable[[str], None] | None` — invoked when the **detected
  engine differs from the last detected** (first detection, and any
  change — including a change *to* or *from* `unknown`). The poller keeps a
  local `last_engine: str | None`; for every **observed** body it compares
  `sample.engine` against `last_engine` and updates `last_engine` (so
  `unknown` is tracked like any engine value). Transport errors and
  non-200 responses do not change `last_engine`. (Note: this is *not* the
  mechanism that clears the unavailable flag on recovery — a
  `404 → healthy vllm` recovery does not change the engine, so it relies on
  the state-transition callback below.)
- `on_metrics_unavailable: Callable[[bool], None] | None` — invoked with the
  **new state** whenever the *unavailable* state **changes** (either
  direction), where the state is computed per tick from the fetch outcome:
  - **non-200 status** (404/405 — the SGLang-without-`--enable-metrics`
    signature; `fetch_metrics` returns `observed=False`, which the poller
    distinguishes from a transport error, since transport errors never
    produce a `MetricsSample`) → state **`True`**
  - **200 body detected `unknown`** (no recognizable KV gauge) → state
    **`True`**
  - **200 body detected `vllm`/`sglang`** → state **`False`**
  - **pure transport error** (`httpx.HTTPError`) → state **unchanged** (the
    existing `metrics fetch failed` warning covers a downed server; the
    outage is already visible via `gate_metrics_fresh`).

  The poller keeps a local `unavailable: bool | None` (last computed state)
  and invokes the callback only when the newly computed state differs (state
  change, either direction — so a recovery from an earlier 404 clears the
  app's flag even when the engine itself never changed). When the new state
  is **`True`**, the poller additionally logs the WARNING: *"metrics endpoint
  at {url} returned no recognizable KV-cache gauge — if this is an SGLang
  server, launch it with --enable-metrics; the tiered layer fails open
  (staleness) and the autoconfig layer fails open (no capacity)."* (On the
  `False` transition it logs nothing — recovery is visible via the
  `gate_metrics_fresh`/`gate_backend_engine` gauges, consistent with the
  discovery path, which does log a recovery INFO; an INFO here is a
  permissible variant if preferred.) This is what keeps the once-per-
  transition warning and the `gate_metrics_endpoint_unavailable` gauge
  consistent for the whole unavailable period, with re-degradation after
  recovery warned again (decision 8).

  *No new process exit:* this is logged-not-fatal (decision 8), mirroring the
  capacity-missing and dead-candidate paths (gotcha #10, decision 15).

**Engine-aware capacity-missing message.** The existing `log.error` for a
capacity-less observed body (`poller.py:240-248`) currently hard-codes the
vLLM gauge names. It becomes **engine-aware**: it names the capacity gauges
of the *detected* engine — for `vllm`, `vllm:kv_cache_size_tokens` / the
`cache_config_info` label (unchanged); for `sglang`,
`sglang:kv_cache_total_tokens` / `sglang:max_total_num_tokens`. (A helper
`_capacity_gauge_names(engine) -> str` produces the string; for `unknown` it
says "no recognizable KV-cache capacity".)

**`BackendState` (app.py)** — **gains two fields:**
- `engine: str = ENGINE_UNKNOWN` — set by the app's `on_engine` closure.
- `metrics_unavailable: bool = False` — set to whatever the app's
  `on_metrics_unavailable` closure is invoked with (the new state, either
  direction). The app owns both fields; the poller only signals. (The
  `False` recovery transition comes through the state callback, not
  `on_engine` — see the note in the poller bullet above.)

**Two new per-backend setter closures** (app.py, next to the existing
`_capacity_flag_setter` / `_on_reanchor_setter`):
- `_on_engine_setter(bs) -> Callable[[str], None]` — `bs.engine = engine`.
- `_metrics_unavailable_setter(bs) -> Callable[[bool], None]` —
  `bs.metrics_unavailable = unavailable`.

The `lifespan` wiring passes `on_engine=_on_engine_setter(bs)` and
`on_metrics_unavailable=_metrics_unavailable_setter(bs)` to `run_poller` for
each backend (alongside the existing callbacks).

### 2.5 Stats (`stats.py`)

Two new labeled gauges, following the existing per-backend gauge pattern
(`set_freshness` / `set_remaining` / `set_backend_capacity_unavailable`),
rendered on each `GET /metrics`:

- **`gate_backend_engine{backend, engine}`** — `1` for the **currently
  detected** engine, labels `engine ∈ {vllm, sglang, unknown}`. The app sets
  exactly one series to `1.0` per backend (its `BackendState.engine`) and
  the other two to `0.0` on each render, so the gauge is always populated and
  scrape-able even before the first detection (`unknown=1` is the initial
  state). A new `set_backend_engine(backend, engine)` method does the set.
  (The `sglang` series deliberately does **not** expose new/legacy —
  decision 3; the internal distinction is not operator-facing.)
- **`gate_metrics_endpoint_unavailable{backend}`** — `0/1` alert flag, `1`
  while the backend's `/metrics` is reachable-but-unrecognizable (per §2.4),
  `0` once a real engine is detected. New
  `set_backend_metrics_unavailable(backend, bool)`.

`gate_config_info` is **not** changed (engine is runtime-detected, not
configured — the new `gate_backend_engine` gauge is the audit surface).

The `GET /metrics` render loop (`app.py:884-888`) gains two lines per backend:
`stats.set_backend_engine(bs.name, bs.engine)` and
`stats.set_backend_metrics_unavailable(bs.name, bs.metrics_unavailable)`.

### 2.6 No change

- `config.py` — **no change** to `Backend`/`BACKENDS_JSON`/validation
  (decision 1, 5); only a docstring note on `Backend` that `host`/`port` may
  point at vLLM **or** SGLang.
- `router.py`, `tokens.py`, `proxy.py` — no change (decision math, token
  estimation, and proxying are engine-agnostic; `proxy_request` already takes
  `base_url`).
- `models.py` (`parse_v1_models`) — no change (SGLang's `/v1/models` is
  OpenAI-schema, so discovery works unchanged).
- The two admission layers and the failover walk — no change; they consume
  the same per-backend `MetricsCache`/`KvRemaining` regardless of engine.

---

## 3. Files to Change/Create

| Path | Action | Description |
|---|---|---|
| `src/gate/metrics.py` | Modify | Add `Engine` constants + `ENGINES`; `detect_engine(text)` (pure); `MetricsParser` Protocol; `VllmMetricsParser` (current vLLM logic verbatim), `SglangMetricsParser` (new-gauge-with-legacy-fallback), `UnknownMetricsParser`; `_PARSERS` registry; `MetricsSample` gains `engine`; `fetch_metrics` runs detect→parse and sets `engine`; **delete** `fetch_usage` / `fetch_usage_by_model`; the three existing `parse_*` vLLM facades delegate to `VllmMetricsParser` (logic in one place); **module docstring rewritten** — it currently says "Parsing and caching of vLLM's KV-cache metrics" and names the deleted `fetch_usage`/`fetch_usage_by_model` as the thin edges; it becomes engine-agnostic ("vLLM or SGLang, detected per body") and documents the new detection/parser structure (the gotcha-invariants list in the docstring is preserved and extended with the detection order). |
| `src/gate/poller.py` | Modify | `run_poller` gains `on_engine` and `on_metrics_unavailable` callbacks (state-change only, synchronous, logged-not-fatal); the capacity-missing `log.error` becomes engine-aware (`_capacity_gauge_names(engine)`); `--enable-metrics` warning once per transition into the unavailable state (HTTP-reachable-but-unrecognizable: non-200 or gauge-less 200; not on transport errors; re-arms only on a real-engine recovery). |
| `src/gate/app.py` | Modify | `BackendState` gains `engine` + `metrics_unavailable`; two new setter closures (`_on_engine_setter`, `_metrics_unavailable_setter`); `lifespan` wires the two new callbacks per backend; `GET /metrics` render sets the two new gauges. |
| `src/gate/stats.py` | Modify | New gauges `gate_backend_engine{backend,engine}` and `gate_metrics_endpoint_unavailable{backend}`; new `set_backend_engine` / `set_backend_metrics_unavailable` methods. |
| `src/gate/config.py` | Modify (docstring only) | Note on `Backend` that `host`/`port` may target vLLM or SGLang (no schema/validation change). |
| `tests/test_metrics.py` | Extend | `detect_engine` **by family-name presence** (vllm / sglang-new / sglang-legacy / no-KV-gauge→unknown / unparseable→unknown / both-sglang-gauges→sglang / **all-NaN vllm usage→vllm** / vllm-NaN+finite-sglang→vllm); per-engine `usage`/`usage_by_model`/`capacity` incl. the new→legacy capacity fallback, multi-series max, all-non-finite→None, by-model returns `None` not `{}`; the byte-identity edge (all-NaN vllm usage + valid vllm capacity → `engine="vllm"`, `capacity` parsed); `fetch_metrics` end-to-end per engine (folds the deleted `fetch_usage*` cases); `MetricsSample.engine` populated. |
| `tests/test_poller.py` | Extend | `on_engine` fires on first detection and on engine change only (not every tick; legacy→new SGLang does **not** fire — still `sglang`); `on_metrics_unavailable` fires **once per transition** on the 404 case and on the gauge-less-200 case, **not** on a pure transport error, not repeatedly, and re-arms after a real-engine recovery; engine-aware capacity-missing message. |
| `tests/test_api.py` | Extend | A fake **SGLang** upstream (legacy metrics body) end-to-end: anchoring works, `gate_kv_cache_usage_pct` populated per `model_name`, `/v1/models` discovery unchanged, both new gauges correct (`gate_backend_engine{backend,sglang}=1`, `gate_metrics_endpoint_unavailable=0`); a **mixed fleet** (one vLLM + one SGLang backend) — per-backend engine gauges and independent admission state; an SGLang-without-metrics backend → `gate_metrics_endpoint_unavailable=1`, `gate_backend_engine{...,unknown}=1`, and the `--enable-metrics` warning logged once. |
| `tests/test_stats.py` | Extend | The two new gauges render / initial-`unknown` / reset-on-recovery behavior. |
| `README.md` | Modify | New "SGLang backends" section (auto-detection — no config change; the metric-name fallback table and the legacy `token_usage` semantics caveat; the `--enable-metrics` requirement; the two new gauges); the two new gauges added to the `/metrics` reference; v2 roadmap note. |
| `AGENTS.md` | Modify | Project-map updates (`metrics.py` parser interface + detection; `poller.py`/`app.py` new callbacks; `stats.py` new gauges); a new gotcha entry — **engine-specific metric families, detected per body** (detection order, fallback semantics, fail-open on `unknown`, no config knob); test list notes the SGLang end-to-end coverage. |
| `docs/plans/sglang-backend.md` | Create | This file — the design record. |

---

## 4. Testing Plan

**Unit — `test_metrics.py` (the pure layer, no I/O):**
- `detect_engine`: vLLM body → `vllm`; SGLang-new body (has
  `sglang:kv_cache_usage_perc`) → `sglang`; SGLang-legacy body (only
  `sglang:token_usage`) → `sglang`; a body with **both**
  `sglang:kv_cache_usage_perc` and `sglang:token_usage` → `sglang` (new wins,
  and is read); a body with a capacity gauge but **no** usage gauge →
  `unknown`; unparseable text → `unknown`. **Family presence, not
  finiteness:** a `vllm:kv_cache_usage_perc` body whose samples are **all
  NaN** → `vllm` (detection is name-based — the `NAN_ONLY` vector in
  `test_metrics.py` is exactly this shape); a body whose `vllm` usage gauge is
  all-NaN while an `sglang` usage gauge is finite → `vllm` (vLLM-first
  precedence holds regardless of values). Precedence: a body that has all
  three usage families → `vllm` (first match).
- `VllmMetricsParser`: identical assertions to the current `test_metrics.py`
  for the three vLLM functions (this is the byte-identical invariant — the
  *same* test vectors must pass against the parser). Since the `parse_*`
  free functions are now thin facades over `VllmMetricsParser`, **the existing
  `test_metrics.py` vectors passing unchanged against the facades is the
  strongest proof of decision 10** — do not rewrite them.
- **Byte-identity edge (today's behavior that detection must preserve):** a
  200 vLLM body with all-NaN `vllm:kv_cache_usage_perc` **and** a valid
  `vllm:kv_cache_size_tokens` → `fetch_metrics` returns
  `usage_frac=None`, `capacity_tokens=<int>` (not None), `engine="vllm"`; the
  poller stores the capacity, does **not** unanchor, and logs **no**
  capacity-missing error (contrast: an `unknown` body *would* hit the
  capacity-missing path — see the poller tests below).
- `SglangMetricsParser`: usage from `kv_cache_usage_perc` when present; usage
  from `token_usage` when the new gauge is absent; **new wins when both
  present** (different values asserted so the test is meaningful);
  by-model keyed by `model_name` from the *selected* family; capacity from
  `kv_cache_total_tokens` when present, else `max_total_num_tokens`;
  all-absent / all-non-finite → `None`.
- `fetch_metrics`: per-engine `MetricsSample` (observed, all three values,
  `engine` field); non-200 → `MetricsSample(False, None, None, None,
  "unknown")`; transport error propagates (unchanged contract).

**Unit — `test_poller.py` (wiring + state-change semantics):**
- A poller against a fake SGLang (legacy) body: the counter re-anchors,
  `on_engine` is called with `"sglang"` exactly once (then not, on repeat
  ticks).
- A poller whose body changes **engine** (vllm→sglang): `on_engine` called
  with the new value once. Note the inverse case: a legacy→**new SGLang**
  upgrade stays `sglang` at engine level (only the internal gauge selection
  changes), so `on_engine` does **not** fire — assert that.
- `on_metrics_unavailable` fires with the new state **once per transition**,
  in either direction: `True` for (a) a 404 `/metrics`, (b) a 200 body
  detecting `unknown`; `False` on recovery (a real-engine body) — including
  the `404 → healthy-vllm` case where the engine never changed. It does
  **not** fire on a pure transport error (state unchanged). It does **not**
  re-fire on the next failing tick (non-200 after non-200; `unknown` after
  `unknown`). After a recovery it **re-arms and fires again** on a new
  degradation.
- Engine-aware capacity-missing: a SGLang body with usage but no capacity logs
  the message naming `sglang:kv_cache_total_tokens` /
  `sglang:max_total_num_tokens`.

**Integration — `test_api.py` (ASGI, fake upstreams):**
- **SGLang end-to-end:** a fake SGLang backend (legacy metrics +
  `/v1/models`). Assert: a request forwards; `gate_kv_cache_usage_pct`
  reflects the `sglang` body's `model_name`; `/healthz` + `/metrics` are
  correct; `gate_backend_engine{backend,sglang}=1`;
  `gate_metrics_endpoint_unavailable{backend}=0`.
- **Mixed fleet:** one vLLM + one SGLang backend, distinct models. Assert
  per-backend engine gauges and that each backend's admission uses *its own*
  state (a full SGLang pool 429s while the vLLM pool is empty and forwards).
- **SGLang without metrics:** a backend whose `/metrics` 404s. Assert
  `gate_metrics_endpoint_unavailable{backend}=1`,
  `gate_backend_engine{backend,unknown}=1`, and the `--enable-metrics`
  warning is in the log (exactly once, not repeated per tick); the backend
  still forwards (fail-open).
- **vLLM regression:** the existing vLLM end-to-end tests are untouched and
  must pass unchanged (byte-identical behavior, decision 10).

**Coverage gate:** `pytest --cov=gate --cov-fail-under=80` stays green; the
new code is fully covered by the above (the parsers and detection are pure and
directly asserted).

---

## 5. Documentation Updates

- **`README.md`** — a **"SGLang backends"** section under the existing
  multi-backend/observability material, stating: (a) no config change — the
  engine is auto-detected from `/metrics`; (b) the metric-name fallback table
  (what the gate reads per engine, new→legacy) and the **legacy
  `token_usage` caveat** (it is `max(full, swa, mamba)` — a bottleneck that
  over-reports KV on hybrid-SSM models; the newer `kv_cache_usage_perc` is
  KV-only and preferred); (c) **SGLang must be launched with
  `--enable-metrics`** or the backend fails open and
  `gate_metrics_endpoint_unavailable` is set; (d) the two new gauges. Both
  gauges added to the `/metrics` reference table. The "Limitations & v2
  roadmap" section gains a bullet: *"Engines are detected, not configured —
  vLLM and SGLang are supported today; further OpenAI-compatible engines are
  additive via the parser interface (`detect_engine` + a `MetricsParser`
  implementation)."* (That section currently says "single-engine vLLM" in the
  per-backend-policy bullet — update that phrasing to "one inference engine"
  since a backend may now be vLLM *or* SGLang.)
- **`AGENTS.md`** — project map: `metrics.py` now carries the engine
  constants, `detect_engine`, the `MetricsParser` interface + three
  implementations, the `_PARSERS` registry, and `MetricsSample.engine`;
  `fetch_usage`/`fetch_usage_by_model` are gone. `poller.py`/`app.py` note the
  two new callbacks + `BackendState.engine`/`.metrics_unavailable`. `stats.py`
  notes the two new gauges. **New gotcha:** *engine-specific metric families
  are detected per body (vLLM → newer SGLang → older SGLang → unknown); no
  config knob; unknown/transport fail open; the per-model `model_name` label
  is shared across engines; vLLM behavior is byte-identical.*
- **This file** is committed as the design record (consistent with
  `docs/plans/multi-backend.md` and `docs/plans/autoconfig.md`).

---

## 6. Dev-Ops / Project Structure Updates

- **`config.example.yaml`** — no schema change; add a short comment on the
  `backends:` entry that `host`/`port` may point at a vLLM **or** an SGLang
  server (engine auto-detected), and that SGLang must be run with
  `--enable-metrics`.
- **`docker-compose.example.yaml`** — optionally add a commented example
  `BACKENDS_JSON` entry pointing at an SGLang backend with a
  `# run sglang with --enable-metrics` note.
- **No new dependencies** — `prometheus_client`'s text parser is already
  used; SGLang's body is plain Prometheus text exposition.
- **No new CI jobs** — the existing pipeline (ruff / mypy / pytest 80% /
  pip-audit / pre-commit / container) covers the change.

---

## 7. Verification Checklist (for the implementer)

- [ ] `detect_engine` returns `vllm` / `sglang` / `unknown` per the
      decision-2 precedence, by **family-name presence** (not finiteness),
      including: both-SGLang-gauges→`sglang`, capacity-without-usage→
      `unknown`, unparseable→`unknown`, an all-NaN `vllm:kv_cache_usage_perc`
      body→`vllm`, vLLM-usage-all-NaN-with-finite-sglang→`vllm` (precedence
      holds on names, not values), and vLLM-wins when all three families are
      present.
- [ ] `VllmMetricsParser` produces **identical** results to the pre-change
      `parse_kv_cache_usage` / `parse_kv_cache_usage_by_model` /
      `parse_kv_cache_capacity` on every existing test vector (decision 10).
- [ ] `SglangMetricsParser`: `kv_cache_usage_perc` wins over `token_usage`
      when both present; `kv_cache_total_tokens` wins over
      `max_total_num_tokens`; by-model is keyed by `model_name` from the
      *selected* family; all-absent → `None`.
- [ ] `fetch_metrics` sets `MetricsSample.engine` correctly; non-200 →
      `(False, None, None, None, "unknown")`; `fetch_usage` /
      `fetch_usage_by_model` are **deleted** and no `src/` code references
      them; their test cases are folded into `fetch_metrics`.
- [ ] `run_poller` `on_engine` fires on first detection and on change only;
      `on_metrics_unavailable` fires with the new state **once per
      transition** on 404 and on gauge-less-200 (`True`), fires `False` on
      recovery (incl. `404 → healthy-vllm`, engine unchanged),
      **not** on a pure transport error, and clears on recovery.
- [ ] The capacity-missing `log.error` names the detected engine's capacity
      gauges (`sglang:kv_cache_total_tokens` /
      `sglang:max_total_num_tokens` for SGLang; unchanged vLLM names for
      vLLM).
- [ ] `BackendState.engine` / `.metrics_unavailable` update via the app's
      setter closures; the `metrics_unavailable` clear on recovery rides the
      state callback (`_metrics_unavailable_setter(False)`) — **not** the
      engine setter (which only sets `bs.engine`); a `404 → healthy-vllm`
      recovery (engine unchanged) still clears the flag.
- [ ] `gate_backend_engine{backend,engine}` renders `unknown=1` initially and
      the detected engine `=1` thereafter; `gate_metrics_endpoint_unavailable`
      is `1` while reachable-but-unrecognizable and `0` after a real engine.
- [ ] ASGI: an SGLang (legacy) backend forwards, anchors, and populates
      `gate_kv_cache_usage_pct` per `model_name`; a mixed vLLM+SGLang fleet
      gates each backend independently; an SGLang-without-metrics backend
      fails open with the two gauges set and the `--enable-metrics` warning
      logged once.
- [ ] The **existing** vLLM end-to-end tests pass **unchanged** (decision 10).
- [ ] `ruff format --check`, `ruff check`, `mypy src/`,
      `pytest --cov=gate --cov-fail-under=80` all green;
      `bash scripts/local-ci.sh` green.
- [ ] README / AGENTS.md / config.example.yaml / docker-compose.example.yaml
      updated and consistent with the code (detection order, the fallback
      table + legacy caveat, the `--enable-metrics` requirement, the two new
      gauges).

---

## 8. Implementation Order (DAG)

```
[1] metrics.py (Engine + detect_engine + MetricsParser + 3 parsers +
     _PARSERS + MetricsSample.engine + fetch_metrics rewrite + delete
     fetch_usage*) ────────────────────────────────┐
                                                   │
[2] stats.py (gate_backend_engine +                │ (independent of 1;
     gate_metrics_endpoint_unavailable + 2        │  new gauges need no
     setters) ────────────────────────────────────┤  project imports)
                                                   │
                                                   ▼
[3] poller.py (on_engine + on_metrics_unavailable +
     engine-aware capacity message) ── needs [1]
                                                   │
                                                   ▼
[4] app.py (BackendState.engine/.metrics_unavailable +
     2 setter closures + lifespan wiring + /metrics render) ── needs [1],[2],[3]
                                                   │
                                                   ▼
[5] config.py (docstring note only) ── needs nothing (can be any time)
                                                   │
                                                   ▼
[6] docs (README / AGENTS.md / config.example / docker-compose)
```

- **Parallel (independent):** `[1] metrics.py` ∥ `[2] stats.py` — neither
  imports the other (`test_metrics.py` ∥ `test_stats.py` with them).
- **Sequential:** `[3] poller.py` needs `[1]` (`MetricsSample.engine`, the
  detection result). `[4] app.py` needs `[1]`, `[2]`, `[3]` (the sample's
  engine, the new stats setters, the new poller callbacks). `[5]` is trivial
  and independent. `[6]` after code is green.
- **Tests** track their module (interleaved); `test_api.py` last (needs the
  app). **Docs** after code is green.
- **Branching/commits (AGENTS.md dev-style):** feature branch
  `feat/sglang-backend`. The pieces are individually small; if split into
  sub-branches the natural seams are `metrics` (parsers + detection) and
  `wiring` (poller + app + stats). One commit per segment: `metrics` →
  `stats` → `poller` → `app` → `config` → `tests` → `docs`. Rebase along the
  way; per-commit distinct-model review (qwen + gemma) before merge/PR.

---

## Known risks / notes

- **Metric-name churn.** SGLang has renamed these gauges before
  (`token_usage` → `kv_cache_usage_perc`, `max_total_num_tokens` →
  `kv_cache_total_tokens`, with a PR that *reverted* a rename). The
  new→legacy fallback is the mitigation, but a future SGLang that renames
  again (or drops the legacy gauges without adding the new ones) would be
  detected as `unknown` and fail open — surfaced (not silent) by
  `gate_metrics_endpoint_unavailable` and the `gate_backend_engine{...,
  unknown}` label, so an operator can see it and add a parser.
- **Detection is usage-gauge-only.** A body that exposes capacity but no usage
  gauge is classified `unknown` and fails open. This is correct (no usage →
  no admission possible) but means such a backend never anchors; the gauge
  makes this visible.
- **No hysteresis.** A single good body switches the engine (decision 9). If
  a flapping backend alternates engines tick-to-tick, the gauge flaps. This
  is acceptable in v1 (it would be far more alarming than a silent misparse)
  and is trivially hardenable later if it proves noisy.
- **`unknown` is a real state, not an error.** A non-vLLM/non-SGLang
  OpenAI-compatible server (or SGLang with metrics disabled on a
  200-with-sparse-body release) is legitimately `unknown`; the gate must keep
  forwarding (fail-open) and only *hint*.
- **Byte-identical vLLM is the top review priority.** The refactor
  (free functions → parser) must not alter any vLLM value; the existing
  `test_metrics.py` vectors are the contract.
