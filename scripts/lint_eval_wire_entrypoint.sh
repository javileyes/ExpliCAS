#!/usr/bin/env bash
# Lint: Eval Wire Entrypoint Enforcement
#
# This script ensures that CLI and FFI route eval wire through session-layer
# entry points and do not call solver wire helpers directly.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Eval Wire Entrypoint Enforcement ==="

ERRORS=0

# 1) CLI must route eval wire through cas_session command/bridge APIs.
if grep -rq "evaluate_eval_command_pretty_with_session" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI routes eval wire via cas_session command API"
elif grep -rq "evaluate_eval_wire" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI routes eval wire via cas_session entrypoint"
else
    echo "✘ ERROR: CLI must route eval wire through cas_session APIs"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must route through cas_session eval entrypoint.
if grep -rq "evaluate_eval_wire" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI routes eval wire via cas_session entrypoint"
else
    echo "✘ ERROR: FFI must call evaluate_eval_wire"
    ERRORS=$((ERRORS + 1))
fi

# 3) Frontends must not call solver-level JSON helper directly.
if grep -rq "eval_str_to_wire" "$ROOT/crates/cas_cli/src" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✘ ERROR: Frontends must not call cas_solver::eval_str_to_wire directly"
    ERRORS=$((ERRORS + 1))
else
    echo "✔ No direct cas_solver::eval_str_to_wire calls in frontends"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "FAILED: $ERRORS error(s) found"
    echo "Eval wire must use cas_session entry points"
    exit 1
fi

echo ""
echo "✔ eval wire entrypoint enforcement passed"
