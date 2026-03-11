#!/usr/bin/env bash
# Lint: Eval Wire Entrypoint Enforcement
#
# This script ensures that eval wire uses the intended ownership boundary:
# - CLI routes through the session command API
# - FFI calls the direct stateless `cas_solver::wire` entrypoint

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

# 2) FFI must call the direct stateless solver wire entrypoint.
if grep -rq "cas_solver::wire::eval_str_to_wire\\|use cas_solver::wire::eval_str_to_wire" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI routes eval wire via cas_solver::wire"
else
    echo "✘ ERROR: FFI must call cas_solver::wire::eval_str_to_wire"
    ERRORS=$((ERRORS + 1))
fi

# 3) CLI must not bypass cas_session and call solver wire directly.
if grep -rq "eval_str_to_wire" "$ROOT/crates/cas_cli/src"; then
    echo "✘ ERROR: CLI must not call cas_solver::wire::eval_str_to_wire directly"
    ERRORS=$((ERRORS + 1))
else
    echo "✔ CLI does not call cas_solver::wire::eval_str_to_wire directly"
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "FAILED: $ERRORS error(s) found"
    echo "Eval wire must use the expected ownership path"
    exit 1
fi

echo ""
echo "✔ eval wire entrypoint enforcement passed"
