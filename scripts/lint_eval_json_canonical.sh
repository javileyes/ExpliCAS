#!/usr/bin/env bash
# Lint: Eval JSON Canonical Enforcement
#
# This script ensures that CLI and FFI route eval JSON through session-layer
# canonical entry points and do not call solver JSON helpers directly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Eval JSON Canonical Enforcement ==="

ERRORS=0

# 1) CLI must route eval-json through cas_session command/bridge APIs.
if grep -rq "evaluate_eval_json_command_pretty_with_session" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI routes eval-json via cas_session command API"
elif grep -rq "evaluate_eval_json_canonical" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI routes eval-json via cas_session canonical bridge"
else
    echo "✘ ERROR: CLI must route eval-json through cas_session APIs"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must route through cas_session canonical eval bridge.
if grep -rq "evaluate_eval_json_canonical" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI routes eval-json via cas_session canonical bridge"
else
    echo "✘ ERROR: FFI must call evaluate_eval_json_canonical"
    ERRORS=$((ERRORS + 1))
fi

# 3) Frontends must not call solver-level JSON helper directly.
if grep -rq "eval_str_to_json" "$ROOT/crates/cas_cli/src" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✘ ERROR: Frontends must not call cas_solver::eval_str_to_json directly"
    ERRORS=$((ERRORS + 1))
else
    echo "✔ No direct cas_solver::eval_str_to_json calls in frontends"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "FAILED: $ERRORS error(s) found"
    echo "Eval JSON must use canonical cas_session entry points"
    exit 1
fi

echo ""
echo "✔ eval-json canonical enforcement passed"
