#!/usr/bin/env bash
# Fails if any Python source file under src/ is git-ignored. Only src/ is
# checked: it is the shipped package; tests/ is dev-only and intentionally
# not guarded here.
set -euo pipefail
status=0
while IFS= read -r -d '' file; do
  if git check-ignore -q -- "$file"; then
    printf 'error: source file is ignored by Git: %s\n' "$file" >&2
    status=1
  fi
done < <(find src -type f -name '*.py' -print0 2>/dev/null)
exit "$status"
