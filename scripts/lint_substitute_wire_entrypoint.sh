#!/usr/bin/env bash
# Lint: Substitute Wire Entrypoint Enforcement
#
# This script ensures that substitute wire uses the intended ownership boundary:
# - CLI routes through `cas_solver::command_api::substitute`
# - FFI calls the direct stateless `cas_solver::wire` entrypoint
#
# POLICY: Single source of truth for substitute wire API.
# SEE: docs/SUBSTITUTE_POLICY.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Substitute Wire Entrypoint Enforcement ==="

ERRORS=0

# 1) CLI must route substitute through command_api.
if grep -rq "cas_solver::command_api::substitute::evaluate_substitute_subcommand\\|evaluate_substitute_subcommand" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI routes substitute via cas_solver::command_api::substitute"
else
    echo "✘ ERROR: CLI must call cas_solver::command_api::substitute"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must call the direct wire path.
if grep -rq "substitute_str_to_wire" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI uses substitute_str_to_wire (direct)"
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
