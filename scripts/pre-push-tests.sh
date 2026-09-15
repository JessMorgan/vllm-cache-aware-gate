#!/usr/bin/env bash
# Run the full test suite against the committed state, temporarily stashing
# any unstaged or untracked changes. This lets you push one logical commit
# while unrelated work-in-progress remains in the working tree.
set -euo pipefail

cd "$(dirname "$0")/.."

STASH_CREATED=0

cleanup() {
    if [ "$STASH_CREATED" -eq 1 ]; then
        git stash pop --quiet || true
    fi
}
trap cleanup EXIT

if [ ! -x .venv/bin/python ]; then
    printf 'ERROR: No .venv found — run `make venv && make install` first\n' >&2
    exit 1
fi

# Only stash if the working tree differs from HEAD or there are untracked
# files. A repo with no commits yet has no HEAD, so skip stashing entirely
# and just run the tests against the working tree.
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    if ! git diff --quiet HEAD || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        git stash push --include-untracked --message "pre-push auto-stash" --quiet
        STASH_CREATED=1
    fi
fi

.venv/bin/pytest --cov=gate --cov-fail-under=80 -q
