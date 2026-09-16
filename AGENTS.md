# AGENTS.md — project context for fresh agent sessions

A small, single-purpose **reverse-proxy "gate"** container that sits in front of a
vLLM OpenAI-compatible server and protects its KV cache from over-subscription.
It speaks the OpenAI API, and per incoming request decides: **forward** (the KV
cache has room for this prompt's estimated size) or **reject with HTTP 429 +
`Retry-After`** (not enough room — retry in N seconds). It learns the cache's
headroom by polling vLLM's Prometheus `/metrics` endpoint and reading
`vllm:kv_cache_usage_perc`. Designed for safe, configuration-first, fail-open
operation: a monitoring outage never blocks inference traffic, and the gate is a
transparent pass-through for the endpoints it proxies. The full design lives in
the conversation/plan; the operator-facing reference is `README.md` (links at the
bottom). Read this file to know what to grep first and which invariants are load-bearing.

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
distinct-model review and attribution gates are concrete applications of this
section.)

## Project map

Single Python package (`gate`) under a src-layout; the deployable artifact is a
Docker image that runs `python -m gate.main`. The codebase follows a
**pure-core / thin-edges** structure: the decision logic and token estimation are
pure, dependency-free functions (trivially unit-testable); the I/O lives in thin
edge modules (HTTP proxy, metrics poller, config loader) that only the
FastAPI app wires together.

- **`src/gate/config.py`** — `Threshold` and `GateConfig` dataclasses;
  `load_config(path)` reads a **YAML** file (if any) and applies env-var
  overrides, then validates strictly. `kv_pct` is a **percentage (0–100)**;
  the vLLM metric is a **fraction (0–1)**. The `THRESHOLDS_JSON` env override
  is still parsed as JSON (a JSON list), even though the file is YAML.
  `thresholds` is **optional** (zero-config is valid; an empty/absent tiered
  policy just means the tiered layer always allows). Also carries the four
  autoconfig knobs with defaults + env overrides: `target_kv_cache_pct`
  (`85.0`, `TARGET_KV_CACHE_PCT`), `retry_min_s` (`5`, `RETRY_MIN_S`),
  `retry_max_s` (`60`, `RETRY_MAX_S`), `token_margin` (`1.25`, `TOKEN_MARGIN`).
- **`src/gate/tokens.py`** — `estimate_context_tokens(body, cfg) -> int |
  None`. Pure. Estimates prompt tokens as `ceil(prompt_chars / chars_per_token)`
  plus `max_tokens` headroom (default 256); returns `None` on an
  unparseable body (caller fails open). Chat vs completions prompt-text
  extraction. Intentionally a heuristic (no tokenizer dependency). Also
  `extract_request_model(body) -> str` (pure, never raises): the request
  body's `model` string, or `"unknown"` — the stats label and future
  routing key, never used in the decision.
- **`src/gate/metrics.py`** — `parse_kv_cache_usage(text) -> float | None`
  (max across all `vllm:kv_cache_usage_perc` series),
  `parse_kv_cache_usage_by_model(text) -> dict[str, float] | None` (per-model
  fractions keyed by `model_name`, `"default"` when unlabeled),
  `parse_kv_cache_capacity(text) -> int | None` (max positive
  `vllm:kv_cache_size_tokens` gauge sample, falling back to the
  `kv_cache_size_tokens` label on `vllm:cache_config_info` — the form current
  vLLM emits), and `fetch_metrics(client, url) ->
  MetricsSample` (a single GET that parses usage + by-model + capacity from one
  body). `MetricsCache` (last value + `fetched_at` + `age()`; `by_model()`
  stores the per-model breakdown while `_value` stays the MAX of it),
  `CapacityCache` (last good capacity token count; never stale, only unknown),
  `CapacityUnavailableError` (raised by the poller on an observed body missing
  a usable KV-cache capacity — the `vllm:kv_cache_size_tokens` gauge or the
  `kv_cache_size_tokens` label on `vllm:cache_config_info`), and `KvRemaining` — the in-memory estimated-remaining-KV
  counter (`reanchor(tokens)` resets, `subtract(tokens)` is unclamped and a
  no-op when never anchored, `value()` returns `int | None`).
- **`src/gate/poller.py`** — `run_poller(client, cache, url, interval_s, *,
  counter, target_frac, capacity_cache)`: async background task that makes ONE
  `GET` per tick via `fetch_metrics` and, on an observed (HTTP 200) body,
  updates the usage + per-model caches, **re-anchors the `KvRemaining` counter**
  to `anchor_remaining_tokens(capacity, target_frac, usage_frac)`, and updates
  the `CapacityCache`. **Fails open** on any transport error / non-200 (keeps
  ALL state — caches and counter — no crash). **Fails closed (unconditional):**
  an observed body missing a usable KV-cache capacity (the
  `vllm:kv_cache_size_tokens` gauge or the `kv_cache_size_tokens` label on
  `vllm:cache_config_info`) raises `CapacityUnavailableError` (the app's fatal
  callback exits the process).
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
- **`src/gate/proxy.py`** — `proxy_request(httpx, request) -> Response`.
  Transparent forward of method/path/query/headers/body; streams SSE verbatim;
  strips hop-by-hop headers; propagates upstream status (does not mask 5xx).
- **`src/gate/stats.py`** — `GateStats`: the stats edge. Owns a **per-app
  `prometheus_client.CollectorRegistry`** (never the global default), the
  `gate_*` metric objects, and `render()` (Prometheus text exposition for
  `GET /metrics`). In-memory only; resets on restart by design.
  `record_forwarded/rejected`, `set_kv_usage` (fraction→percent, removes stale
  model series), `set_freshness`, and `set_remaining(int | None)` (the
  `gate_kv_cache_remaining_tokens` gauge — the gate's own live estimate of
  remaining KV tokens; `None` renders `NaN`).
- **`src/gate/app.py`** — Builds the FastAPI app: routes
  (`POST /v1/chat/completions`, `POST /v1/completions`, `GET /healthz`,
  `GET /metrics`), the 429 builder, and wiring of poller + cache + proxy +
  stats (each decision is recorded as a pure side effect; `/metrics` renders
  the stats and always returns 200). Owns the `KvRemaining` counter and
  `CapacityCache`; applies the **staleness gate** (stale/never-fetched feed →
  both layers fail open regardless of the counter), AND-combines the tiered and
  autoconfig decisions via `combine_decisions`, charges the counter
  (`counter.subtract(ceil(ctx × margin) or 0)`) on every forward **before** the
  proxy await, builds the rejector-aware 429 (combined `Retry-After`, tiered vs
  autoconfig message text), and exits 1 (`os._exit(1)`) via the poller's fatal
  done-callback. `GET /healthz` adds `kv_cache_capacity_tokens` +
  `kv_cache_remaining_tokens`.
- **`src/gate/main.py`** — `main()` entrypoint: load config, log the autoconfig
  knobs (`target %s%% KV cache, token margin %s, retry %d-%ds`) and the tiered
  tier count (`N threshold tier(s)` or `none — autoconfig only`), build the app
  (the poller is started by the app lifespan, not here), and run uvicorn. There
  is **no CLI** — argv is ignored (autoconfig is always on).
- **`tests/`** — `test_tokens.py`, `test_router.py`, `test_metrics.py`,
  `test_stats.py`, `test_api.py` (ASGI end-to-end with a fake vLLM),
  `test_poller.py`.
- **`Dockerfile`**, **`docker-compose.example.yaml`**, **`config.example.yaml`**,
  **`Makefile`**, **`README.md`** — packaging, operator reference, and docs.

**Proxied endpoints (v1):** `POST /v1/chat/completions` and
`POST /v1/completions` (both consume KV cache). `GET /healthz` for liveness,
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

The `tests/test_api.py` suite drives the gate end-to-end against a fake vLLM
(`httpx.ASGITransport` + a mock upstream transport), so the allow / 429 /
fail-open / streaming paths are all exercised without a real model.

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
   gates (CI checks, per-commit distinct-model review, attribution block)
   apply to each commit individually at merge time, not to the feature as a
   whole.
5. **Keep the review pipeline honest.** Commits may be made on the branch
   before they are reviewed, but the review pipeline must not lag: review
   findings are addressed inline via history rewriting *within the unmerged
   branch only* (see rule 10), and no branch is merged to `main` or pushed
   as a PR until **every** commit on it has its individual review and proper
   attribution (see Git-management, steps 2–3 and 7).
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
   reviews may be batched over any number of the branch's commits (e.g.
   review commits 2–5 together) rather than one commit at a time, and
   feedback is addressed inline via history rewriting — amends and
   rebases — **WITHIN THE UNMERGED BRANCH ONLY**. Never rewrite history on
   `main` or on any branch already merged; those commits are final. Batching
   reviews does not relax the per-commit gate (Git-management steps 2–3 and
   7): each commit still needs its own review and attribution before merge
   or PR creation, but that requirement may be satisfied by a batched review
   that covers it.

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
2. **Require a code review by a distinct AI model as a mandatory gate before
   merging to `main` or pushing a PR.** Before any branch is merged to
   `main`, or pushed as a PR, obtain a review of the complete diff from a
   model other than the one that authored the change — per commit for
   single-commit branches, or a batched review covering multiple commits for
   multi-commit branches (see Dev-style rule 10: a pre-merge branch is a PR
   stack, and reviews may be batched over any number of the branch's commits,
   with feedback addressed inline via history rewriting within the unmerged
   branch only). Each commit **MUST** have its own review and proper
   attribution before merge or PR creation (step 7). Do **not** merge to
   `main` or push a PR until the required reviews have been produced.

   **Reviewer preference: both `qwen` and `gemma`.** Prefer obtaining
   reviews from **both** the `qwen` and `gemma` families; however, only one
   is **necessary** when the other is being flaky — a review from the
   available family satisfies the distinct-model requirement.

   The reviewer must be a genuinely different model — routing the review to a
   subagent category that defaults to the committer's own model is **not** a
   valid review. The reviewing model's name must appear in the review output
   (for example, "Review performed by `deepseek-v4-*`" or another
   non-committer provider model) so the separation is verifiable; record that
   name with the review findings.

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
3. **Present the review results to the user and ask how to proceed.** Surface
   the distinct reviewer's findings and recommendation to the user in exactly
   this shape: a `VERDICT: APPROVE | REQUEST CHANGES | REJECT` line and a
   `REVIEWER MODEL:` line naming the reviewing model, followed by a bulleted
   findings list in which every bullet is prefixed with its severity
   (`[blocker]`, `[major]`, `[minor]`, or `[nit]`); then ask the user how to
   proceed. Options include merging or pushing as-is, revising per the
   review, or abandoning. Do not treat a clean or critical review as an
   automatic decision — the user's final call is definitive and cannot be
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
   Reviewed by: [review-model] ([review agent]) — VERDICT
   ```

   One blank line separates the block from the message detail above (the
   explanatory body when one exists, otherwise the subject line) and from any
   footer below. `[model]` / `[review-model]` are human-readable display names,
   e.g. `GLM-5.3 Flash` / `Beast Qwen3.8 27B`; the parenthesized agent is the
   tool that produced the change or ran the review (e.g. `CodeBuff`, `OpenCode`,
   `FreeBuff`) and is included on both lines whenever the agent is known.
   `VERDICT` is the actual verdict surfaced in step 3 from the step-2 review
   (`APPROVE`, `REQUEST CHANGES`, or `REJECT`) — the final verdict when the
   change went through multiple review rounds.

   The `Reviewed by:` line is an evidence-bearing assertion, not a formality:
   it must name the step-2 reviewer of **this commit's diff**, obtained
   **before** the commit is merged to `main` or pushed as a PR, and the review
   model must differ from the authoring model. A commit created before its
   review is legitimate (see step 2 and Dev-style rules 5 and 10), but the
   `Reviewed by:` line may only ever describe a review that has actually
   happened: never include it on a change that has not been reviewed, never
   reuse another diff's review, and never fill it in prospectively — if the
   review has not happened yet, the merge or PR does not happen yet (step 2),
   and the line is added or corrected via history rewriting within the
   unmerged branch (Dev-style rule 10). Formatting-only or trivially
   mechanical changes are not exempt. This is an
   honesty-based gate: the message alone does not let a reader mechanically
    verify the claim, so hardening beyond wording (e.g. a pre-commit hook
    validating a review artifact) remains an option if violations recur.

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

5. **v1 takes the MAX across all `vllm:kv_cache_usage_perc` series.** vLLM may
   emit one series per `model_name`; v1 is single-instance and takes the max
   (conservative). v2 will key by `model_name`. Do not take the first or the
   mean. (`metrics.py`)

6. **The gate is a transparent proxy for the two generation endpoints only.**
   `/v1/chat/completions` and `/v1/completions` are forwarded verbatim,
   including `stream: true` SSE — do not buffer the response body in a way that
   breaks streaming, and do not mask upstream status codes (a vLLM 5xx must
   surface as a 5xx, not a 200 or a 429). Everything else except `/healthz` is
   404. (`proxy.py`, `app.py`)

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
   `model` is the request body's `model` field (the future routing key) while
   `model_name` is vLLM's served-model label — different strings, different
   label names, different metrics. And recording is a **pure side effect**:
   `record_*` must never change the decision, the 429, or the proxy path, and
   `/metrics` always returns 200 (never 429/404). (`stats.py`, `app.py`)

10. **Unconditional fail-closed on a missing KV-cache capacity — but fail-open
    while unreachable/stale.** The poller raises `CapacityUnavailableError`
    (and the app exits 1) **only** when an *observed* (HTTP 200) `/metrics` body
    lacks a usable KV-cache capacity (the `vllm:kv_cache_size_tokens` gauge or
    the `kv_cache_size_tokens` label on `vllm:cache_config_info`) — a live vLLM
    that cannot anchor the counter. A merely unreachable, erroring, or stale feed is **never
    fatal**: it keeps all state and the app fails open via staleness (gotcha #1
    outranks the counter — a metrics outage never blocks traffic). Do not
    "helpfully" make the unreachable path fatal, and do not gate the fatal path
    behind a flag: autoconfig is always on. (`poller.py`, `app.py`,
    `metrics.py`)

11. **Remaining-KV counter semantics.** `KvRemaining` is **re-anchored (reset,
    not additive)** on every good poll to
    `anchor_remaining_tokens(capacity, target_frac, usage_frac)`; it is
    **subtracted only on forwarded requests**, charging `ceil(ctx ×
    token_margin)` (`0` when `ctx` is unknown); subtraction is **unclamped** (the
    value may go negative — an over-committed signal); and **rejected requests
    charge nothing**. Subtraction happens on every forward even while the feed
    is stale (bookkeeping continues), but the *decision* fails open during
    staleness, so the counter cannot reject then. (`metrics.py`, `app.py`,
    `router.py`)

12. **Two admission layers are AND-combined; the rejector is the
    max-timeout one.** A request forwards only if the tiered layer **and** the
    autoconfig layer both allow (no tiers ⇒ tiered always allows). When both
    reject, `Retry-After = max(tier timeout, autoconfig retry)` and the
    **reported rejector** is the layer with the higher timeout (**tie →
    tiered**); the 429 message names the reported rejector. Do not OR the
    layers or pick the lower timeout. (`router.py`, `app.py`)

13. **Zero-config operation.** `thresholds` is **optional** — the gate runs
    with only the vLLM host/port (no config file, no flags, **no CLI**; argv is
    ignored, so a legacy `--auto` is a harmless no-op). The four autoconfig
    knobs (`target_kv_cache_pct` 85.0, `retry_min_s` 5, `retry_max_s` 60,
    `token_margin` 1.25) all have sane defaults. Do not reintroduce a
    "≥1 threshold" requirement or a mode/enablement switch. (`config.py`,
    `main.py`)

14. **The single percentage→fraction conversion is `target_kv_cache_pct /
    100.0` at poller start.** The autoconfig path (`anchor_remaining_tokens`,
    `decision_auto`, `scaled_retry_after`) works in **fraction (0–1) and token
    space** and never sees a percentage. Gotcha #2 still applies to the *tiered*
    path (`usage_frac * 100.0 >= kv_pct`); do not conflate the two conversions.
    (`app.py`, `router.py`)

## Authoritative docs (read on demand)

- `README.md` — quickstart, config reference, decision logic, 429 contract,
  observability (`/metrics` reference, PromQL, scrape config), operational
  notes, v2 roadmap.
- `docs/plans/observability.md` — the observability/stats design plan (metric
  table, two-model-namespaces rationale, median-as-histogram semantics).
- `docs/plans/autoconfig.md` — the autoconfig (always-on KV admission) design
  plan: the two AND-combined admission layers, the remaining-KV counter, the
  staleness gate, retry scaling, the combination rule, and the worked example.
- `config.example.yaml` — the reference configuration with the `thresholds`
  entry contract.
- `AGENTS.md` (this file) — architecture map, test commands, git workflow, and
  the load-bearing invariants above.
- `.github/workflows/tests.yml` / `release.yml` — the CI and release
  pipelines; `scripts/local-ci.sh` is their local equivalent (see "CI /
  release / local parity").
