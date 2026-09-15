#!/usr/bin/env bash
# Local parity for .github/workflows/tests.yml: runs the same checks, in the
# same order, without aborting on the first failure. A green run here should
# mean a green CI run (the container job is skipped when docker is absent).
set -euo pipefail

cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ ! -x .venv/bin/python ]; then
  printf '%sERROR%s No .venv found — run `make venv && make install` first.\n' "$RED" "$NC" >&2
  exit 1
fi

PASS=0
FAILED=0

pass() {
  printf '%sPASS%s %s\n' "$GREEN" "$NC" "$1"
  PASS=$((PASS + 1))
}

fail() {
  printf '%sFAIL%s %s\n' "$RED" "$NC" "$1"
  FAILED=$((FAILED + 1))
}

warn() {
  printf '%sWARN%s %s\n' "$YELLOW" "$NC" "$1"
}

# CI uses Python 3.12 for the non-matrix jobs.
PY_VERSION="$(.venv/bin/python -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')"
if [ "$PY_VERSION" != "3.12" ]; then
  warn "Active Python is $PY_VERSION; CI non-matrix jobs use 3.12."
fi

# 1. Source tracking
if bash scripts/check-ignored-sources.sh; then
  pass "Source tracking"
else
  fail "Source tracking"
fi

# 2. Formatting
if .venv/bin/ruff format --check src/ tests/; then
  pass "Formatting"
else
  fail "Formatting"
fi

# 3. Linting
if .venv/bin/ruff check .; then
  pass "Linting"
else
  fail "Linting"
fi

# 4. Typecheck
if .venv/bin/mypy src/; then
  pass "Typecheck"
else
  fail "Typecheck"
fi

# 5. Unit tests (coverage gate)
if out=$(.venv/bin/pytest --cov=gate --cov-fail-under=80 -q 2>&1); then
  cov="$(printf '%s\n' "$out" | grep -E 'TOTAL' | awk '{print $NF}' || true)"
  if [ -n "$cov" ]; then
    pass "Unit tests (coverage gate) — coverage ${cov}"
  else
    pass "Unit tests (coverage gate)"
  fi
else
  printf '%s\n' "$out"
  fail "Unit tests (coverage gate)"
fi

# 6. Container build (skip, not fail, when docker is unavailable)
if command -v docker >/dev/null 2>&1; then
  if docker build -t vllm-gate:ci .; then
    pass "Container build"
  else
    fail "Container build"
  fi
else
  warn "docker not on PATH — skipping Container build (code-gate parity only)."
fi

echo
printf 'Local CI: %d passed, %d failed.\n' "$PASS" "$FAILED"
[ "$FAILED" = "0" ]
