#!/usr/bin/env bash
# Lint: Substitute JSON Canonical Enforcement
#
# This script ensures that CLI and FFI always use the canonical
# cas_session substitute JSON bridge and don't hand-serialize JSON.
#
# POLICY: Single source of truth for substitute JSON API.
# SEE: docs/SUBSTITUTE_POLICY.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Substitute JSON Canonical Enforcement ==="

ERRORS=0

# 1) CLI must route JSON substitute through canonical cas_session bridge
#    Accepted paths:
#    - direct call from CLI
#    - CLI -> cas_session helper -> evaluate_substitute_json_canonical
if grep -rq "substitute_str_to_json" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI uses substitute_str_to_json (direct)"
elif grep -rq "evaluate_substitute_subcommand" "$ROOT/crates/cas_cli/src" \
    && grep -rq "pub fn evaluate_substitute_json_canonical" "$ROOT/crates/cas_session/src/json_bridge.rs"; then
    echo "✔ CLI routes substitute JSON via cas_session canonical helper"
else
    echo "✘ ERROR: CLI must call canonical substitute JSON path (direct or via cas_session)"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must call canonical substitute path
#    Accepted paths:
#    - direct call to substitute_str_to_json
#    - call to cas_session canonical wrapper
if grep -rq "substitute_str_to_json" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI uses substitute_str_to_json (direct)"
elif grep -rq "evaluate_substitute_json_canonical" "$ROOT/crates/cas_android_ffi/src" \
    && grep -rq "pub fn evaluate_substitute_json_canonical" "$ROOT/crates/cas_session/src/json_bridge.rs"; then
    echo "✔ FFI routes substitute JSON via cas_session canonical helper"
else
    echo "✘ ERROR: FFI must call canonical substitute JSON path"
    ERRORS=$((ERRORS + 1))
fi

# 3) No manual JSON serialization for substitute in CLI
# (Look for "substitute" context with serde_json::json! patterns that aren't just opts building)
if grep -n 'serde_json::json!' "$ROOT/crates/cas_cli/src/main.rs" | grep -v 'mode\|steps\|pretty' | grep -i 'substitute' >/dev/null 2>&1; then
    echo "✘ ERROR: CLI substitute should not hand-serialize JSON response"
    ERRORS=$((ERRORS + 1))
else
    echo "✔ CLI substitute doesn't hand-serialize response"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "FAILED: $ERRORS error(s) found"
    echo "Substitute JSON must use canonical substitute bridge path"
    exit 1
fi

echo ""
echo "✔ substitute JSON canonical enforcement passed"
