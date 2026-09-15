# Autoconfig (always-on KV admission) — Design Plan

Status: **implemented** (2026-09-15; see Round 3 for the capacity-source
correction)
Branch: `feat/auto-config` (worktree `/tmp/opencode/autoconfig`, base `main` =
`ddb166a`, post-observability rebase)

**Supersedes v1** — the same-day v1 design (mode toggle: `--auto` flag /
implicit auto mode, per-request projected-usage formula, `auto_*` config
keys). V1 is committed on this branch (8 commits, `6d5a578..48ea8da`) and will
be reworked per §8. The v1↔observability rebase remains valid groundwork: the
unified `fetch_metrics` fetch layer (`MetricsSample` with usage + by-model +
capacity) and the observability stats wiring (`GateStats`, `GET /metrics`,
`extract_request_model`) are unchanged in v2.

---

## Decision log

**Round 3 (2026-09-15 — capacity source correction):**

12. **Capacity is read from two sources, not one.** vLLM (PR #42206, merged
    2026-06-12) exposes the KV-cache capacity as the `kv_cache_size_tokens`
    **label** on the `vllm:cache_config_info` info gauge (sample value `1.0`),
    not as a standalone `vllm:kv_cache_size_tokens` gauge. `parse_kv_cache_capacity`
    therefore tries the standalone gauge first (it wins if it yields a positive
    value) and falls back to the `kv_cache_size_tokens` label on
    `vllm:cache_config_info` (string→int, skipping `"None"`/non-numeric/
    non-positive, max across samples). This supersedes the single-source
    wording in decision 7 and §1/§2/§3 below: "missing `vllm:kv_cache_size_tokens`"
    now means "missing a *usable* capacity — neither the standalone gauge nor
    the label." The unconditional fail-closed behavior (decision 7) is
    unchanged; only the source it reads is broadened.

**Round 2 (2026-09-15 — current):**

1. **Autoconfig is always on.** No mode, no `--auto` CLI flag, no config/CLI
   switch to enable or disable it.
2. **No CLI params, no enablement config.** The gate runs with only the vLLM
   host/port; autoconfig uses sane defaults (zero-config, see §2 Config).
3. **Config switches:** (a) target KV cache max %, (b) reject timeouts
   (min/max, scaled by incoming context size vs free KV tokens).
4. **In-memory counter of estimated remaining KV tokens**: refreshed (reset to
   the real-derived value) when the real % is polled; decremented by the
   estimated size of each **forwarded** request; used for admission.
5. **Tiered thresholds coexist side-by-side** (optional, unchanged): AND
   combination, enabling e.g. "small prompts only" when usage is above a tier
   but below the target (config-driven).

**Round 2 clarifications (2026-09-15 Q&A):**

6. **Counter refresh = reset to the real-derived value.** Each successful poll
   sets `remaining = floor(capacity × (target_frac − usage_frac))` (clamped ≥
   0). vLLM's real % is ground truth; the counter only accounts for requests
   forwarded since the last anchor. Estimate drift self-corrects every poll
   interval.
7. **Missing `vllm:kv_cache_size_tokens` on a live vLLM → fail closed**: exit 1
   (logged) once an HTTP-200 `/metrics` body is observed without a usable
   capacity gauge. A merely unreachable vLLM is never fatal (fail open,
   invariant #1).
8. **Token margin stays a knob**: `token_margin` (default 1.25, ≥ 1).
9. **Layer interplay = AND; report the active rejector.** Forward only if
   autoconfig allows AND tiered allows (no tiers ⇒ tiered allows
   unconditionally). Rejected requests never decrement the counter. When a
   tier **and** autoconfig both reject: `Retry-After = max(tier timeout_s,
   autoconfig retry)` and the reported rejector is the layer with the **higher
   timeout** (tie → tiered).
10. **Config keys + defaults** (all optional; zero-config is valid):
    `target_kv_cache_pct: 85.0`, `retry_min_s: 5`, `retry_max_s: 60`,
    `token_margin: 1.25`; env overrides `TARGET_KV_CACHE_PCT`, `RETRY_MIN_S`,
    `RETRY_MAX_S`, `TOKEN_MARGIN`.
11. **New Prometheus metric** `gate_kv_cache_remaining_tokens` (gauge; the
    gate's live estimated remaining tokens; NaN until anchored), set at
    `/metrics` scrape time.

Round 1 decisions (`--auto` precedence, four `auto_*` knobs, the
projected-utilization formula) are superseded; their surviving ideas —
fail-closed on missing capacity, dynamic capped retry, margin-based
conservatism — carry over in the forms above.

---

## 1. Background, Problem, Goal

**Problem.** The gate *cannot start without a policy* (`load_config()` raises
`ConfigError` unless ≥1 `thresholds` entry is configured) and tiered
`kv_pct`/`max_context` pairs are the wrong shape for most deployments:
operators must guess thresholds relative to a cache size that vLLM already
publishes as `vllm:kv_cache_size_tokens`. A pure %-check ("reject above 85%")
is coarse — it cannot admit individual requests by size.

**Goal.** An **always-on autoconfig layer** that admits each request by
comparing its estimated size against an in-memory estimate of the KV space
still available up to a target usage %, anchored continuously to vLLM's real
metrics — **plus** the existing optional tiered policy, applied
simultaneously:

1. **Always-on, zero-config.** Only the vLLM host/port is required; defaults
   are sane (`target 85%`, margin 1.25, retry 5–60s). No `--auto`, no mode,
   no enablement key.
2. **Remaining-KV counter.** An in-memory counter of estimated remaining KV
   tokens: re-anchored to `capacity × (target − usage)` on every successful
   poll (decision 6) and decremented by the estimated size of each forwarded
   request; the admission comparison is `ceil(ctx × margin) ≤ remaining`.
3. **Coexisting tiered policy (optional, AND).** `thresholds` is optional and
   unchanged; a request forwards only if **both** layers allow (decision 9).
   This enables operator policies like "reject large contexts while usage is
   above 50%, even though we're under the target" — a temporary
   small-prompts-only mode, fully config-driven.
4. **Fail closed on missing capacity**: a live vLLM whose observed `/metrics`
   lacks `vllm:kv_cache_size_tokens` cannot be anchored → the gate exits 1
   (decision 7). **Fail open on any metrics outage** (invariant #1) — a stale
   or unreachable feed never blocks traffic.

**Non-goals.** No per-model/per-engine keying (v2), no tokenizer, no request
queuing, no new runtime dependencies.

> **v1 formula note — resolved.** v1's open question (projected-utilization
> vs `free × (1−reserve)`) is moot: the target now lives at *anchor time*
> (`remaining = capacity × (target − usage)` — exactly the headroom up to the
> target), and the counter decrements it. There is no per-request utilization
> projection anymore.

---

## 2. Design Summary

### Two admission layers, AND-combined

Both layers are evaluated on every request to the two generation endpoints;
the request forwards iff **both allow**.

**Tiered layer (optional — unchanged).** `decision(usage_pct, ctx_tokens,
thresholds)` exactly as today (highest-tier-only, inclusive
`ctx_tokens <= max_context`, no tiers ⇒ allow). On stale/never-fetched usage
it receives `usage_pct = None` and allows (`metrics_unavailable`) — existing
fail-open.

**Autoconfig layer (always on).**

```python
def decision_auto(
    remaining_tokens: int | None,   # KvRemaining.value(); None = never anchored
    ctx_tokens: int | None,
    policy: AutoPolicy,             # (token_margin, retry_min_s, retry_max_s)
) -> Decision:
```

- Stale/never-fetched usage → **bypassed in the app layer** (allow,
  `metrics_unavailable`) — the staleness gate outranks the counter
  (invariant #1; see below).
- `remaining_tokens is None` → allow, `reason="auto_unanchored"` (defensive:
  fresh usage without an anchor is only a cold-start sliver — an observed
  body with usage but no capacity is fatal per decision 7).
- `ctx_tokens is None` → allow, `reason="prompt_unparseable"` (invariant #7).
- Else `effective = ceil(ctx_tokens × token_margin)`:
  - `effective ≤ remaining_tokens` → **allow**,
    `reason="auto_within_headroom"` (**inclusive** boundary).
  - `effective > remaining_tokens` → **reject**,
    `reason="auto_exceeds_headroom"`,
    `retry_after = scaled_retry_after(effective, remaining_tokens,
    retry_min_s, retry_max_s)`.

**Staleness gate (fail-open override).** In `_handle_generation`: if the
usage cache is stale or never fetched, the tiered layer gets
`usage_pct = None` (allow) **and the autoconfig layer is bypassed** (allow,
`metrics_unavailable`) — regardless of the counter's value. The counter keeps
its state (subtractions continue on forwarded requests) but cannot cause a
reject while the feed is stale. This preserves invariant #1: a metrics
outage never blocks traffic.

### Remaining-KV counter (`KvRemaining`, in metrics.py)

```python
class KvRemaining:
    def reanchor(self, tokens: int) -> None   # poller: value = tokens (each good poll)
    def subtract(self, tokens: int) -> None   # app:    value -= tokens (no-op if None)
    def value(self) -> int | None
```

- **Reanchor** (poller, every successful observed poll with usage + capacity):
  `anchor_remaining_tokens(capacity, target_frac, usage_frac) =
  floor(capacity × max(clamp(usage_frac, 0, 1) subtracted from target_frac,
  0))` — a pure helper in `router.py`. Usage ≥ target ⇒ anchor 0 (cache is
  "full" for admission; every non-trivial request rejects until usage drops).
- **Subtract** (app, at admission, on every forwarded request): charge
  `effective = ceil(ctx_tokens × token_margin)` — the same amount the
  admission compared, i.e. the KV we reserved. `ctx_tokens is None`
  (unparseable, failed open) ⇒ charge 0 (the next anchor re-corrects).
  Rejected requests charge **nothing** (they consumed no KV). Subtraction is
  **not clamped** — the value may go negative (over-committed signal; the
  retry formula treats `≤ 0` uniformly). If a forwarded request later 5xxs
  upstream, the charge stands until the next anchor (self-correcting).
- **Concurrency:** single asyncio loop; read-modify-write is synchronous (no
  await between read and write), so no locking — same contract as
  `MetricsCache`. Subtraction happens **before** the proxy await (charge at
  admission time).
- **Charge vs staleness:** subtraction happens on every forwarded request
  even while the feed is stale (bookkeeping continues; the decision is the
  part that fails open).

### Retry scaling (pure, in router.py)

```python
def scaled_retry_after(effective: int, remaining: int, min_s: int, max_s: int) -> int:
    if remaining <= 0:
        return max_s
    return int(ceil(clamp(min_s * (1 + effective / remaining), min_s, max_s)))
```

Integer seconds, always in `[min_s, max_s]`; slightly over ⇒ ≈ 2×min; far over
or cache full ⇒ `max_s`. `ceil` (never round a backoff down).

### Combination (pure, in router.py)

```python
def combine_decisions(tier: Decision, auto: Decision) -> Decision:
```

- Both allow ⇒ `allow=True`, `retry_after=None`, `active_tier=None`;
  `reason` is the more informative layer reason (a fail-open reason wins, for
  logging; otherwise `"allowed"`).
- Exactly one rejects ⇒ that layer's `Decision` is returned verbatim.
- Both reject ⇒ `retry_after = max(tier.retry_after, auto.retry_after)`; the
  **reported rejector** is the tier if `tier.retry_after >= auto.retry_after`
  (tie ⇒ tiered) else autoconfig; `reason` = reported layer's reason;
  `active_tier` = the tier iff it is the reported rejector (drives the 429
  message text).

### Poller

`run_poller(client, cache, url, interval_s, *, counter, target_frac,
capacity_cache)` — one GET per tick via `fetch_metrics` (unchanged):

- `sample.observed` (HTTP 200):
  - `capacity_tokens is None` → log error + **raise
    `CapacityUnavailableError`** (unconditional — the v1 `require_capacity`
    flag is gone). The app's fatal callback exits the process (`os._exit(1)`).
  - `usage_frac is not None` → `counter.reanchor(anchor_remaining_tokens(
    capacity_tokens, target_frac, usage_frac))`; also
    `cache.update(usage_frac)` / `cache.update_by_model(by_model)` as today.
  - `capacity_tokens is not None` → `capacity_cache.update(...)`.
  - Usage absent but capacity present → no reanchor (nothing to anchor
    against), not fatal; last anchor persists.
- Non-observed / `httpx.HTTPError` → keep **all** state (caches and counter);
  never fatal (fail open via staleness).

### App & main (wiring)

- `create_app` owns a `KvRemaining` (plus the existing `MetricsCache`,
  `CapacityCache`, `GateStats`, upstream client). The poller starts with
  `counter` + `target_frac = cfg.target_kv_cache_pct / 100.0` (the single
  percentage→fraction conversion for autoconfig; tiered keeps its existing
  `frac * 100.0` — gotcha #2).
- `_handle_generation`: compute `tier_dec` and `auto_dec` per the staleness
  gate above; `dec = combine_decisions(tier_dec, auto_dec)`. On allow:
  `counter.subtract(ceil(ctx × margin) if ctx is not None else 0)`, stats
  recording (try/except, unchanged), proxy. On reject: stats recording, 429
  with `Retry-After: dec.retry_after` and the rejector-aware message
  (`active_tier` present → tier text; `auto_exceeds_headroom` → autoconfig
  text: usage %, remaining, capacity, target). Body keeps
  `code: "kv_cache_too_full"`, `type: "cache_pressure"`.
- Fatal propagation: unchanged shape — `poller_task.add_done_callback` logs
  and `os._exit(1)` on a pre-shutdown task exception.
- `GET /healthz`: existing `status`/`metrics_age_s`/`kv_usage` +
  `kv_cache_capacity_tokens` + **`kv_cache_remaining_tokens`** (counter
  value, `null` if never anchored). The v1 `"mode"` field is **removed**
  (there is no mode).
- `GET /metrics`: existing behavior + `stats.set_remaining(counter.value())`.
- `main()`: **no CLI at all** (v1's `argparse --auto` is removed); argv is
  ignored, so legacy invocations passing `--auto` keep working (behavior
  identical — autoconfig is always on). *Open (minor):* if we would rather
  hard-error on unknown flags (catch typos in compose `command:`), that's a
  one-argparse-call change — default assumption: ignore argv.
  `load_config()` (no `auto` param). Startup log: the autoconfig knobs
  (`target 85.0%, margin 1.25, retry 5–60s`) and the tiered tier count
  (`N threshold tier(s)` or `none — autoconfig only`).

### Stats

`GateStats` gains a gauge `gate_kv_cache_remaining_tokens` (no labels; the
gate's own live estimate, as opposed to `gate_kv_cache_usage_pct` which is
vLLM's real per-model %) and `set_remaining(value: int | None)` (None ⇒ NaN,
mirroring the `gate_metrics_age_s` convention).

### Config surface

All optional; validated like the rest of the config (finite-checked):

| Key | Default | Constraint | Env override |
|---|---|---|---|
| `target_kv_cache_pct` | `85.0` | `(0, 100]` | `TARGET_KV_CACHE_PCT` |
| `retry_min_s` | `5` | int ≥ 1 | `RETRY_MIN_S` |
| `retry_max_s` | `60` | int ≥ `retry_min_s` | `RETRY_MAX_S` |
| `token_margin` | `1.25` | ≥ 1.0 | `TOKEN_MARGIN` |

- `thresholds` becomes **optional**: the "≥1 threshold" requirement is
  **removed** (zero-config is valid; an empty/absent tiered policy just means
  the tiered layer always allows).
- Removed: `auto_mode`, `auto_target_usage_pct`, `auto_token_margin`,
  `auto_base_retry_s`, `auto_max_retry_s`, `load_config(auto=...)`, the
  "ignore thresholds in auto mode" logic.

### Worked example

capacity = 100 000, target 85%, margin 1.25, retry 5/60.

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

**Tiered coexistence ("small prompts only").** thresholds
`[{50, 4096, 15}, {80, 1024, 30}]`, usage 55% ⇒ tiered governs on the 50-tier
(ctx ≤ 4096, retry 15); autoconfig anchor
`floor(100000 × (0.85 − 0.55)) = 30 000`. Small prompt (ctx 1 000 → effective
1 250): both allow → **forward**. Large prompt (ctx 5 000 → effective 6 250):
tiered rejects (5000 > 4096), autoconfig allows (6250 ≤ 30 000) → **429,
rejector = tiered, Retry-After 15**. Both reject: usage 90% ⇒ tiered 80-tier
retries 30; autoconfig anchor `floor(100000 × max(0.85 − 0.90, 0)) = 0` →
every request rejects at 60 ⇒ combined **429, Retry-After 60, rejector =
autoconfig** (60 > 30).

---

## 3. Files to Change/Create

The branch base is `main` (`ddb166a`, post-observability), which has **none** of
v1's capacity infrastructure (no `fetch_metrics`/`MetricsSample`/
`CapacityCache`/`decision_auto`/`auto_mode`/CLI). v2 therefore **builds** all of
that from scratch on top of `main`; the observability surface
(`GateStats`, `GET /metrics`, `extract_request_model`, per-model
`gate_kv_cache_usage_pct`) is already present and is only extended.

| Path | Action | Description |
|---|---|---|
| `src/gate/config.py` | modify | Add the 4 knobs (`target_kv_cache_pct` 85.0, `retry_min_s` 5, `retry_max_s` 60, `token_margin` 1.25) with defaults + validation + env overrides (§2 table); make `thresholds` **optional** (drop the ≥1 rule). Nothing to remove (no `auto_mode`/`auto_*`/`load_config(auto=)` in this base). |
| `src/gate/metrics.py` | modify | Add the capacity infra (absent in this base): `KV_CACHE_CAPACITY_METRIC`, `parse_kv_cache_capacity`, `MetricsSample`, `fetch_metrics` (one GET → usage + by-model + capacity; supersedes the separate `fetch_usage`/`fetch_usage_by_model`), `CapacityCache`, `CapacityUnavailableError`; plus the new `KvRemaining` (reanchor/subtract/value; single-loop contract; subtraction unclamped). |
| `src/gate/router.py` | modify | Add `decision_auto(remaining_tokens, ctx_tokens, policy)`, `AutoPolicy(token_margin, retry_min_s, retry_max_s)`, and the pure `anchor_remaining_tokens(capacity, target_frac, usage_frac) -> int`, `scaled_retry_after(effective, remaining, min_s, max_s) -> int`, `combine_decisions(tier, auto) -> Decision`; add the auto reason codes (`auto_unanchored`, `auto_within_headroom`, `auto_exceeds_headroom`). `decision()` untouched. |
| `src/gate/poller.py` | modify | Switch to `fetch_metrics` (single fetch); add `counter: KvRemaining`, `target_frac: float`, `capacity_cache` params; re-anchor the counter on observed usage+capacity; raise `CapacityUnavailableError` **unconditionally** on an observed body missing capacity; update both caches. |
| `src/gate/app.py` | modify | Add `CapacityCache` + `KvRemaining`; staleness gate bypasses autoconfig; `combine_decisions`; `counter.subtract(...)` on every forward (pre-proxy); combined 429 (max retry, rejector-aware message); `/healthz` adds `kv_cache_capacity_tokens` + `kv_cache_remaining_tokens`; `/metrics` sets the new gauge; add the fatal `os._exit(1)` callback. |
| `src/gate/stats.py` | modify | Add the `gate_kv_cache_remaining_tokens` gauge + `set_remaining(int \| None)` (None ⇒ NaN). |
| `src/gate/main.py` | modify | Add startup log of the autoconfig knobs + tier count. `load_config()` unchanged (no `auto` param in this base); no CLI to remove. |
| `tests/test_config.py` | modify | Add zero-config / no-thresholds-validity tests + the 4 knobs (defaults/env/validation/NaN guards); thresholds parsed when present. |
| `tests/test_metrics.py` | modify | Add capacity parsing / `fetch_metrics` / `CapacityCache` / `CapacityUnavailableError` tests (mirroring the existing usage/by-model style); add `KvRemaining` tests. |
| `tests/test_router.py` | modify | Add `decision_auto` / `scaled_retry_after` / `anchor_remaining_tokens` / `combine_decisions` tables; tiered `decision` tests unchanged. |
| `tests/test_poller.py` | modify | Add capacity / fatal / re-anchor / counter tests; keep the existing by-model tests. |
| `tests/test_api.py` | modify | Add counter-decrement / both-layers-reject / zero-config / `/healthz` / `/metrics` / fatal-callback tests; keep the existing tiered + observability tests. |
| `tests/test_stats.py` | modify | Add `set_remaining` (value, negative, None → NaN). |
| `tests/test_main.py` | modify | Update for the startup log lines. |
| `README.md` | modify | Add an "Autoconfig (always on)" section: counter model, two-layer AND + max-timeout rejector, knob table, zero-config quickstart (plain `docker run vllm-gate`, no flags), fail-closed note, worked example, `/healthz` body, 429 max-timeout rule, new metric row. |
| `config.example.yaml` | modify | Document `thresholds` as **optional**; add the 4 knobs. |
| `AGENTS.md` | modify | Add project-map lines (`KvRemaining`, `decision_auto`, `combine_decisions`); add the autoconfig gotchas (unconditional fail-closed; counter semantics — reanchor/subtract/staleness-override; AND + max-timeout rejector; zero-config). |
| `docker-compose.example.yaml` | modify | No `--auto` variant (autoconfig is always on); note zero-config. |

No new dependencies (`prometheus_client` already present).

---

## 4. Testing Plan

- **Pure unit (deterministic):** `decision_auto` table — remaining None /
  ctx None / allow-at-boundary (`effective == remaining`) / reject-one-over /
  margin 1.0 vs 1.25. `scaled_retry_after` — `remaining ≤ 0` → max (incl.
  negative), ratio 0 ⇒ min floor, ratio 1 ⇒ 2×min, cap at max, monotonic in
  deficit, integer result. `anchor_remaining_tokens` — usage ≥ target → 0,
  usage clamped at 0 and 1, floor of fractional product. `combine_decisions`
  — all four combinations, max-timeout, tie → tiered, `active_tier`
  propagation. `KvRemaining` — reanchor/reset, subtract, no-op-on-None,
  negative drift, no timestamp semantics.
- **Config:** zero config → valid defaults; no thresholds (file or env) →
  valid; each knob via YAML + env + default; malformed/NaN/Inf →
  `ConfigError` with clear messages; `retry_max_s < retry_min_s` → error;
  thresholds present → parsed and validated exactly as today.
- **Poller:** scripted `fetch_metrics` samples — (a) observed 200 without
  capacity → raises `CapacityUnavailableError` (no `require_capacity` needed);
  (b) `httpx.HTTPError` → no raise, caches and counter untouched; (c)
  observed with both → counter reanchored to `anchor_remaining_tokens(...)`
  exactly; (d) observed with usage but no capacity → fatal (per (a));
  (e) observed with capacity but no usage → no reanchor, no raise.
- **End-to-end ASGI:** fake vLLM `/metrics` carrying both gauges; drive the
  gate: forward decrements the counter (assert the second request's 429 vs
  the first's 200 with the same body); stale usage (monkeypatch `is_stale`
  or freeze the clock) → 200 fail-open even when the counter is
  over-committed; both layers reject → `Retry-After == max(...)` and the
  message names the higher-timeout rejector; the §2 "small prompts only"
  scenario; a zero-config app (no thresholds anywhere) serves both endpoints;
  `/healthz` carries `kv_cache_capacity_tokens` +
  `kv_cache_remaining_tokens` and **no** `mode`; `/metrics` renders
  `gate_kv_cache_remaining_tokens` reflecting the decrements.
- **Stats:** `set_remaining(123)` / `set_remaining(-5)` /
  `set_remaining(None)` (NaN) in the exposition.
- **Fatal-exit path:** `add_done_callback` handler unit test (monkeypatch
  `os._exit`; called with 1 on task exception; not called on clean
  shutdown).
- **Gates:** full `bash scripts/local-ci.sh` green (format, lint, mypy,
  pytest with 80% coverage floor — new code fully covered).

---

## 5. Documentation Updates

- **README.md**: replace the v1 "Auto mode" section with
  **"Autoconfig (always on)"**: the counter model (reanchor on poll, subtract
  on forward, staleness override), the two-layer AND + max-timeout rejector
  rule with the "small prompts only" example, the knob table (4 keys + env),
  the worked example from §2, the fail-closed caveat (live vLLM without
  `vllm:kv_cache_size_tokens` exits 1; unreachable ⇒ fail open),
  zero-config quickstart (drop the `--auto` example — plain
  `docker run ... vllm-gate`), `/healthz` sample body, 429 contract update,
  new metric row in the observability reference, v2 note (max-across-series
  for the capacity gauge).
- **config.example.yaml**: header note that `thresholds` is **optional**
  (autoconfig runs regardless); replace the `auto_*` block with
  `target_kv_cache_pct` / `retry_min_s` / `retry_max_s` / `token_margin` and
  comments.
- **AGENTS.md**: project-map lines (`KvRemaining`, reworked `decision_auto`,
  `combine_decisions`, no `auto_mode`/`--auto`); rewrite the autoconfig
  gotchas — (a) fail-closed is now **unconditional** on an observed body
  missing the capacity gauge (still fail-open while unreachable/stale);
  (b) counter semantics — reanchor resets, subtract only on forwards,
  unclamped (may go negative), staleness overrides the counter; (c) AND
  combination + max-timeout rejector (tie → tiered); (d) zero-config
  operation; (e) fraction-space vs percentage-space still applies (the single
  autoconfig conversion is `target_kv_cache_pct / 100.0` at poller start).
- **docker-compose.example.yaml**: remove the `--auto` comment variant.

---

## 6. Dev-Ops / Project Structure Updates

- **Dependencies:** none. (argparse disappears with the CLI.)
- **Dockerfile / entrypoint:** no change — `ENTRYPOINT
  ["python","-m","gate.main"]` runs with no args now (simpler than v1).
- **CI/workflows:** no new jobs; the existing 8-job pipeline and
  `scripts/local-ci.sh` cover everything.
- **Structure:** pure-core/thin-edge preserved — the anchor/retry/combine
  math is pure in `router.py`; `KvRemaining` is stateful but
  dependency-free (like `MetricsCache`); no new modules.
- **Ops semantics:** the only process-killing failure remains deterministic
  (live vLLM, no capacity gauge) and is logged fatally; orchestrators see
  exit 1 + log line.

---

## 7. Verification Checklist (for the implementer)

- [ ] Zero config (no file, no env) → valid; gate runs; only VLLM host/port
      required.
- [ ] No CLI at all: `python -m gate.main --anything` starts normally and
      ignores argv (no parser). Legacy deploys passing `--auto` keep working
      as a behavior-identical no-op (autoconfig is always on) — smooth
      upgrade. (If we instead want unknown flags to be a hard error, this is
      a one-argparse-call change — open question, see §2 main.)
- [ ] One `/metrics` GET per poll interval (unchanged single-fetch).
- [ ] Anchor formula: `floor(capacity × max(target_frac − usage, 0))` with
      usage clamped to [0, 1]; usage ≥ target → anchor 0.
- [ ] Reanchor **resets** the counter on every good poll (not additive).
- [ ] Subtraction only on **forwarded** requests, charging
      `ceil(ctx × margin)` (0 when ctx unknown); rejected requests charge
      nothing; subtraction before the proxy await; unclamped (may go
      negative).
- [ ] Stale/never-fetched usage → both layers fail open **regardless of the
      counter**; reason `metrics_unavailable`.
- [ ] Inclusive boundary: `effective == remaining` → allow.
- [ ] `Retry-After` integer in `[retry_min_s, retry_max_s]`;
      `remaining ≤ 0` → exactly `retry_max_s`; no `ZeroDivisionError`.
- [ ] Both layers reject → `Retry-After == max(tier, auto)`; message names
      the higher-timeout rejector (tie → tiered); exactly-one-rejects → that
      layer's retry/reason.
- [ ] Poller raises `CapacityUnavailableError` only on an observed (200) body
      without capacity; transport errors never raise; caches **and counter**
      untouched on error.
- [ ] Process exits 1 (logged) when the poller task dies from that error;
      clean shutdown does not trigger the exit callback.
- [ ] `target_kv_cache_pct / 100.0` is the only percentage→fraction step in
      the autoconfig path; tiered keeps `frac * 100.0` (gotcha #2 intact).
- [ ] `/healthz`: no `mode`; has `kv_cache_capacity_tokens` +
      `kv_cache_remaining_tokens`.
- [ ] `/metrics` includes `gate_kv_cache_remaining_tokens` (NaN before first
      anchor); recording never alters the decision.
- [ ] Tiered behavior byte-identical when thresholds are configured:
      existing `test_router.py` tiered assertions and tiered-only `test_api`
      scenarios pass unmodified.
- [ ] All 4 knobs: YAML + env + validation + defaults, documented with the
      same values as code.
- [ ] `bash scripts/local-ci.sh` fully green, coverage ≥ 80%.

---

## 8. Implementation Order & State of the Branch

**Branch state (2026-09-15):** `feat/auto-config` was **reset to `main`
(`ddb166a`)** and re-committed with only the v2 plan doc (option (c) from the
original decision list — the 8 v1 commits were discarded; they remain
recoverable via `git reflog` / the pre-reset ref `59973ce`). v2 is therefore
implemented **fresh on top of `main`**, not reworked in place.

```
Wave 1 (parallel — no cross-imports of new symbols):
  S1 config.py    (4 knobs, optional thresholds)
  S2 metrics.py   (capacity infra from scratch + KvRemaining)
  S2c router.py   (decision_auto, anchor/retry/combine pure helpers)
  S4b stats.py    (gate_kv_cache_remaining_tokens gauge)
Wave 2:
  S3 poller.py    (fetch_metrics, unconditional fatal, reanchor)   ← needs S2, S2c
Wave 3:
  S4 app.py       (staleness gate, AND, counter, healthz, fatal)   ← needs S1, S2, S2c, S3, S4b
Wave 4:
  S5 main.py      (startup logs)                                    ← needs S1, S4
Wave 5:
  S6 docs         (README, config.example, AGENTS.md, compose)      ← needs S5
```

- **Wave 1 is the max-parallelism point** (4 workstreams, each scoped to its
  own file + tests; the four leaf modules don't cross-import each other's new
  symbols). S2 is the largest workstream (capacity infra is built from
  scratch, not kept from v1).
- **Segments S1–S6 are the commit boundaries** (one commit per segment; tests
  are written with each segment — no separate test segment). Intermediate
  commits may not be independently full-suite-green (e.g. S2 adds
  `fetch_metrics` while `poller.py` still calls `fetch_usage_by_model`); the
  **final** state must be fully green.
- Before merge: full local CI, distinct-model review per the AGENTS.md gate,
  then `git merge --no-ff` into `main`.

---

## 9. Relationship to observability (merged in `main`)

The observability feature is **in `main`** (branch base `ddb166a`):
`GateStats` with the per-app registry, `GET /metrics`, `extract_request_model`,
and the per-model `gate_kv_cache_usage_pct` gauge are already wired into
`app.py`. v2's touches to the observability surface:

- **Fetch-layer unification (S2).** `main` currently fetches via two separate
  functions (`fetch_usage`, `fetch_usage_by_model` — two GETs per poll in the
  poller's design). v2 replaces them with a single `fetch_metrics() ->
  MetricsSample` (usage + by-model + capacity from **one** GET). This is the
  "A2 alignment" the observability plan anticipated; the by-model parse/cache
  behavior is preserved, only the fetch edge is unified.
- **New gauge (S4b).** `gate_kv_cache_remaining_tokens` (the gate's own live
  estimate) alongside the real `gate_kv_cache_usage_pct`.

`docs/plans/observability.md` is otherwise unaffected.
