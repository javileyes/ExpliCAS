#!/usr/bin/env bash
# Lint: Substitute Wire Entrypoint Enforcement
#
# This script ensures that CLI and FFI always use the
# cas_session substitute wire entrypoint and don't hand-serialize JSON.
#
# POLICY: Single source of truth for substitute wire API.
# SEE: docs/SUBSTITUTE_POLICY.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Substitute Wire Entrypoint Enforcement ==="

ERRORS=0

# 1) CLI must route wire substitute through cas_session entrypoint
#    Accepted paths:
#    - direct call from CLI
#    - CLI -> cas_session helper -> evaluate_substitute_wire
if grep -rq "substitute_str_to_wire" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI uses substitute_str_to_wire (direct)"
elif grep -rq "evaluate_substitute_subcommand" "$ROOT/crates/cas_cli/src" \
    && grep -rq "evaluate_substitute_wire" "$ROOT/crates/cas_session/src"; then
    echo "✔ CLI routes substitute wire via cas_session entrypoint"
else
    echo "✘ ERROR: CLI must call substitute wire path (direct or via cas_session)"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must call substitute wire path
#    Accepted paths:
#    - direct call to substitute_str_to_wire
#    - call to cas_session wire wrapper
if grep -rq "substitute_str_to_wire" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI uses substitute_str_to_wire (direct)"
elif grep -rq "evaluate_substitute_wire" "$ROOT/crates/cas_android_ffi/src" \
    && grep -rq "evaluate_substitute_wire" "$ROOT/crates/cas_session/src"; then
    echo "✔ FFI routes substitute wire via cas_session entrypoint"
else
    echo "✘ ERROR: FFI must call substitute wire path"
    ERRORS=$((ERRORS + 1))
fi

# 3) No manual wire serialization for substitute in CLI
# (Look for "substitute" context with serde_json::json! patterns that aren't just opts building)
if grep -n 'serde_json::json!' "$ROOT/crates/cas_cli/src/main.rs" | grep -v 'mode\|steps\|pretty' | grep -i 'substitute' >/dev/null 2>&1; then
    echo "✘ ERROR: CLI substitute should not hand-serialize wire response"
    ERRORS=$((ERRORS + 1))
else
    echo "✔ CLI substitute doesn't hand-serialize response"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "FAILED: $ERRORS error(s) found"
    echo "Substitute wire must use the expected entrypoint path"
    exit 1
fi

echo ""
echo "✔ substitute wire entrypoint enforcement passed"
