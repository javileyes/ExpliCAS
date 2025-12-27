#!/usr/bin/env bash
# Lint: Substitute JSON Canonical Enforcement
#
# This script ensures that CLI and FFI always use the canonical
# substitute_str_to_json entry point and don't hand-serialize JSON.
#
# POLICY: Single source of truth for substitute JSON API.
# SEE: docs/SUBSTITUTE_POLICY.md

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Substitute JSON Canonical Enforcement ==="

ERRORS=0

# 1) CLI must call substitute_str_to_json for JSON output
if grep -rq "substitute_str_to_json" "$ROOT/crates/cas_cli/src"; then
    echo "✔ CLI uses substitute_str_to_json"
else
    echo "✘ ERROR: CLI must call cas_engine::substitute_str_to_json for JSON output"
    ERRORS=$((ERRORS + 1))
fi

# 2) FFI must call substitute_str_to_json
if grep -rq "substitute_str_to_json" "$ROOT/crates/cas_android_ffi/src"; then
    echo "✔ FFI uses substitute_str_to_json"
else
    echo "✘ ERROR: FFI must call cas_engine::substitute_str_to_json"
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
    echo "Substitute JSON must use cas_engine::substitute_str_to_json"
    exit 1
fi

echo ""
echo "✔ substitute JSON canonical enforcement passed"
