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
accurate position must be on the record first. (The Git-management step 2/7
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
  `load_config(path)` with env-var overrides; strict startup validation.
  `kv_pct` is a **percentage (0–100)**; the vLLM metric is a **fraction (0–1)**.
- **`src/gate/tokens.py`** — `estimate_context_tokens(body, cfg) -> int`.
  Pure. Estimates prompt tokens as `ceil(prompt_chars / chars_per_token)` plus
  `max_tokens` headroom (default 256). Chat vs completions prompt-text
  extraction. Intentionally a heuristic (no tokenizer dependency).
- **`src/gate/metrics.py`** — `parse_kv_cache_usage(text) -> float | None`
  (regex over `vllm:kv_cache_usage_perc`, **max across all series**), and
  `MetricsCache` (last value + `fetched_at` + `age()`).
- **`src/gate/poller.py`** — `run_poller(...)`: async background task that fetches
  `http://VLLM_HOST:VLLM_PORT/metrics` every `metrics_poll_interval_s`, updates the
  cache, and **fails open** on error (keeps last value; no crash).
- **`src/gate/router.py`** — `decision(usage_pct, ctx_tokens, thresholds) ->
  Decision`. Pure. The heart of the gate: highest-tier-only selection +
  inclusive `ctx_tokens <= max_context` comparison.
- **`src/gate/proxy.py`** — `proxy_request(httpx, request) -> Response`.
  Transparent forward of method/path/query/headers/body; streams SSE verbatim;
  strips hop-by-hop headers; propagates upstream status (does not mask 5xx).
- **`src/gate/app.py`** — Builds the FastAPI app: routes
  (`POST /v1/chat/completions`, `POST /v1/completions`, `GET /healthz`), the 429
  builder, and wiring of poller + cache + proxy.
- **`src/gate/main.py`** — `main()` entrypoint: load config, start the poller as a
  background task, run uvicorn; graceful shutdown.
- **`tests/`** — `test_tokens.py`, `test_router.py`, `test_metrics.py`,
  `test_api.py` (ASGI end-to-end with a fake vLLM), `test_poller.py`.
- **`Dockerfile`**, **`docker-compose.example.yaml`**, **`config.example.json`**,
  **`Makefile`**, **`README.md`** — packaging, operator reference, and docs.

**Proxied endpoints (v1):** `POST /v1/chat/completions` and
`POST /v1/completions` (both consume KV cache). `GET /healthz` for liveness.
Everything else → 404. No auth, no TLS termination (v1).

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
(source-tracking, `ruff format --check`, `ruff check`, `mypy`, pytest with the
80% coverage gate, container build), with pass/fail tracking instead of
aborting on the first failure. A green local run should mean a green CI run;
the container job is skipped (with a warning) when docker is absent.

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
   gates (CI checks, distinct-model review, attribution block) apply to each
   commit, not to the feature as a whole.
5. **Commit after review, then continue.** A change that is ready for
   review/commit must not sit unreviewed while new work accumulates on the
   same branch on top of it: run the review, commit, and only then start the
   next segment of that branch.
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

## Git management

Before committing any complete change:

1. Run the full CI checks locally with the canonical local entry point —
   `bash scripts/local-ci.sh` (or `make ci`). That script is the exact local
   equivalent of `.github/workflows/tests.yml` and runs, in order: source
   tracking (`scripts/check-ignored-sources.sh`), `ruff format --check`,
   `ruff check .`, `mypy src/`, `pytest --cov=gate --cov-fail-under=80`, and
   `docker build` (skipped with a warning if docker is absent). Fix every
   issue reported by these checks before committing, then rerun
   `scripts/local-ci.sh` until it passes.
2. **Require a code review by a distinct AI model as a mandatory gate before
   committing.** After the CI checks pass and all changes are staged, obtain a
   review of the complete diff from a model other than the one that authored the
   change. Do **not** commit until this review has been produced. The reviewer
   must be a genuinely different model — routing the review to a subagent
   category that defaults to the committer's own model is **not** a valid
   review. The reviewing model's name must appear in the review output (for
   example, "Review performed by `deepseek-v4-*`" or another non-committer
   provider model) so the separation is verifiable; record that name with the
   review findings.

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
   proceed. Options include committing as-is, revising per the review, or
   abandoning. Do not treat a clean or critical review as an automatic decision
   — the user's final call is definitive and cannot be overridden.
4. Update all relevant documentation to reflect the new reality, including
   `AGENTS.md`, `README.md`, `config.example.json`, and any other checked-in
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

   The `Reviewed by:` line is an evidence-bearing assertion, not a formality: it
   must name the step-2 reviewer of **this diff**, obtained **before** the
   commit was created, and the review model must differ from the authoring
   model. Never include the line on a change that has not been reviewed, never
   reuse another diff's review, and never fill it in prospectively — if the
   review has not happened yet, the commit does not happen yet (step 2).
   Formatting-only or trivially mechanical changes are not exempt. This is an
   honesty-based gate: the message alone does not let a reader mechanically
    verify the claim, so hardening beyond wording (e.g. a pre-commit hook
    validating a review artifact) remains an option if violations recur.

## CI / release / local parity

- **CI** (`.github/workflows/tests.yml`): runs on push to `main`, PRs, and
  manual dispatch, as six jobs — `source-tracking` (fails if any `*.py` under
  `src/` is git-ignored), `format` (`ruff format --check src/ tests/`), `lint`
  (`ruff check .`), `typecheck` (`mypy src/`), `test` (pytest with the 80%
  coverage gate on a Python 3.11/3.12/3.13 matrix, coverage XML uploaded as an
  artifact), and `container` (docker build, no push). The `container` job is
  gated by `needs:` on all the others, so a code failure blocks the image
  build.
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
- **Pre-commit hooks** (`.pre-commit-config.yaml`): dev-time only — **not** a
  CI gate; CI is authoritative. ruff + ruff-format (pinned to the project's
  ruff version) plus hygiene hooks (check-yaml, end-of-file-fixer,
  trailing-whitespace, check-added-large-files, check-merge-conflict).
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
`uvicorn[standard]`, `httpx`, `prometheus_client` for its text parser). Do not
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

## Authoritative docs (read on demand)

- `README.md` — quickstart, config reference, decision logic, 429 contract,
  operational notes, v2 roadmap.
- `config.example.json` — the reference configuration with the `thresholds`
  entry contract.
- `AGENTS.md` (this file) — architecture map, test commands, git workflow, and
  the load-bearing invariants above.
- `.github/workflows/tests.yml` / `release.yml` — the CI and release
  pipelines; `scripts/local-ci.sh` is their local equivalent (see "CI /
  release / local parity").
