#!/bin/bash
# Lint: Ensure no duplicate strip_hold implementations outside canonical module
# 
# This script prevents reintroduction of __hold handling bugs by detecting:
# 1. New strip_hold implementations (should use cas_ast::hold)
# 2. __hold appearing in JSON output (should be stripped at boundaries)

set -e

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "==> Checking for duplicate strip_hold implementations..."

# Find files with strip_hold functions that don't use the canonical implementation
# We check if the file contains both "fn strip" AND does NOT reference cas_ast::hold
VIOLATIONS=""
for file in $(grep -rl "fn strip.*hold" "$ROOT_DIR/crates" --include="*.rs" | grep -v "cas_ast/src/hold.rs"); do
    # Check if the file uses the canonical implementation
    if ! grep -q "cas_ast::hold::" "$file"; then
        VIOLATIONS="$VIOLATIONS\n$file"
    fi
done

if [ -n "$VIOLATIONS" ]; then
    echo "ERROR: Found strip_hold implementations not using canonical module:"
    echo -e "$VIOLATIONS"
    echo ""
    echo "Fix: Use cas_ast::hold::unwrap_hold or cas_ast::hold::strip_all_holds"
    exit 1
fi

echo "✔ No duplicate strip_hold implementations found"

echo "==> Checking for __hold in JSON test snapshots..."

# Check that no JSON output contains __hold
HOLD_IN_JSON=$(grep -rn "__hold" "$ROOT_DIR/crates" \
    --include="*.json" \
    || true)

if [ -n "$HOLD_IN_JSON" ]; then
    echo "WARNING: Found __hold in JSON files (may be test fixtures):"
    echo "$HOLD_IN_JSON"
fi

echo "✔ Lint passed"
