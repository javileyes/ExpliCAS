#!/usr/bin/env bash
set -euo pipefail

# scripts/lint_no_commented_asserts.sh
# 
# CI check: fail if there are commented-out assertions in test files.
# This prevents "weakened" tests where assertions are temporarily disabled.
#
# If you need to disable a test, use #[ignore = "reason"] instead.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Checking for commented assertions in test files..."

# Search for commented assert!/assert_eq!/assert_ne! in test directories
VIOLATIONS=$(grep -rn '^\s*//\s*assert' \
    "$ROOT_DIR/crates/cas_engine/tests" \
    "$ROOT_DIR/crates/cas_cli/tests" \
    2>/dev/null || true)

if [[ -n "$VIOLATIONS" ]]; then
    echo ""
    echo "ERROR: Commented assertions detected!"
    echo ""
    echo "The following lines contain commented-out assertions:"
    echo "$VIOLATIONS"
    echo ""
    echo "This is not allowed because it weakens test coverage silently."
    echo ""
    echo "Solutions:"
    echo "  1. Uncomment and fix the assertion"
    echo "  2. Use #[ignore = \"reason\"] to mark the whole test as pending"
    echo "  3. Remove the test if no longer needed"
    echo ""
    exit 1
fi

echo "  ✓ No commented assertions found"
