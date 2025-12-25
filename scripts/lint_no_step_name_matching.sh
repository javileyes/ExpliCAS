#!/bin/bash
# Lint: Ensure step.rs doesn't use string matching for importance classification
# This prevents regressions to heuristic-based importance inference

set -e

STEP_RS="crates/cas_engine/src/step.rs"

echo "==> Checking $STEP_RS for forbidden patterns..."

# Check for string matching patterns that infer importance from rule names
PATTERNS=(
    'contains("Canonicalize")'
    'contains("Factor")'
    'contains("Expand")'
    'contains("Evaluate")'
    'starts_with("Canonicalize")'
    'starts_with("Identity")'
    'TEMPORARY'
    '== "Canonicalize Even Power Base"'
)

FOUND=0
for pattern in "${PATTERNS[@]}"; do
    if grep -q "$pattern" "$STEP_RS" 2>/dev/null; then
        echo "✗ Found forbidden pattern: $pattern"
        grep -n "$pattern" "$STEP_RS"
        FOUND=1
    fi
done

if [ $FOUND -eq 1 ]; then
    echo ""
    echo "ERROR: step.rs contains string matching for importance."
    echo "       Importance should be declared in the Rule, not inferred from names."
    echo "       See POLICY.md 'Step Visibility Contract' for details."
    exit 1
fi

echo "✔ step.rs uses declarative importance (no forbidden patterns found)"
exit 0
