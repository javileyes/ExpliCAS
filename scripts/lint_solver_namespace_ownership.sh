#!/usr/bin/env bash
# Lint: cas_solver namespace ownership
#
# Enforce that external consumer crates use explicit public namespaces from
# `cas_solver` instead of importing from the crate root.
#
# Allowed public namespaces:
# - cas_solver::api::*
# - cas_solver::runtime::*
# - cas_solver::wire::*
# - cas_solver::session_api::*
# - cas_solver::command_api::*
# - cas_solver::math::*
# - cas_solver::strategies::*

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== cas_solver Namespace Ownership ==="

TARGETS=(
  "$ROOT/crates/cas_cli"
  "$ROOT/crates/cas_session"
  "$ROOT/crates/cas_didactic"
  "$ROOT/crates/cas_android_ffi"
)

ALLOWED='cas_solver::(api|runtime|wire|session_api|command_api|math|strategies)(::|\b)'
ERRORS=0

while IFS= read -r hit; do
    file="${hit%%:*}"
    line="${hit#*:}"

    if [[ "$line" =~ $ALLOWED ]]; then
        continue
    fi

    echo "✘ ERROR: non-owned cas_solver root usage: $hit"
    ERRORS=$((ERRORS + 1))
done < <(
    rg -n --glob '*.{rs,kt}' 'cas_solver::[A-Za-z_][A-Za-z0-9_]*' "${TARGETS[@]}" || true
)

while IFS= read -r hit; do
    echo "✘ ERROR: root brace import from cas_solver: $hit"
    ERRORS=$((ERRORS + 1))
done < <(
    rg -n --glob '*.rs' 'use cas_solver::\{' "${TARGETS[@]}" || true
)

if [ "$ERRORS" -gt 0 ]; then
    echo
    echo "FAILED: $ERRORS ownership violation(s) found"
    echo "Use explicit namespaces from cas_solver instead of the crate root."
    exit 1
fi

echo
echo "✔ cas_solver namespace ownership passed"
