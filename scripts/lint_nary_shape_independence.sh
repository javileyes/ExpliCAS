#!/usr/bin/env bash
# lint_nary_shape_independence.sh
# 
# CI lint to detect binary Expr::Add/Mul pattern matching in n-ary rule modules.
# Use AddView/MulView from cas_engine::nary for shape-independent rules.
#
# This script checks only n-ary SUM-FOCUSED files (where AddView should be used).
# Files that do structural pattern matching on Mul for other purposes (e.g., 
# pattern recognition for Dirichlet kernel) are excluded.
#
# Usage: ./scripts/lint_nary_shape_independence.sh
# Exit 0 = pass, Exit 1 = binary pattern found in n-ary rules

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Checking for binary Expr::Add/Mul pattern matching in n-ary rule modules..."

# Patterns that suggest binary shape-dependent Add/Mul usage
# Only check Expr::Add (since Expr::Mul is common for structural patterns)
ADD_PATTERN='Expr::Add\s*\([a-z_]+\s*,\s*[a-z_]+\)'

# Files where n-ary sum patterns should be used
# These modules have been migrated to use AddView
NARY_SUM_FILES=(
    "crates/cas_engine/src/telescoping.rs"
    # polynomial.rs has legitimate binary Add/Sub for distribution logic
    # Add more as they get migrated
)

VIOLATIONS=""

for file in "${NARY_SUM_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    # Find Expr::Add matches in production code
    MATCHES=$(grep -n -E "$ADD_PATTERN" "$file" 2>/dev/null || echo "")
    
    if [[ -z "$MATCHES" ]]; then
        continue
    fi
    
    while IFS= read -r match_line; do
        [[ -z "$match_line" ]] && continue
        
        LINE_CONTENT=$(echo "$match_line" | cut -d: -f2-)
        
        # Skip comment lines
        if echo "$LINE_CONTENT" | grep -qE "^\s*//"; then
            continue
        fi
        
        # Skip test code (after #[cfg(test)] or mod tests)
        LINE_NUM=$(echo "$match_line" | cut -d: -f1)
        TEST_START=$(grep -n -E "(#\[cfg\(test\)\]|mod tests)" "$file" 2>/dev/null | head -1 | cut -d: -f1 || echo "999999")
        if [[ -n "$LINE_NUM" && -n "$TEST_START" && "$LINE_NUM" =~ ^[0-9]+$ && "$TEST_START" =~ ^[0-9]+$ ]]; then
            if (( LINE_NUM >= TEST_START )); then
                continue
            fi
        fi
        
        VIOLATIONS="${VIOLATIONS}${file}:${match_line}\n"
    done <<< "$MATCHES"
done

VIOLATIONS=$(echo -e "$VIOLATIONS" | grep -v "^$" || true)

if [[ -n "$VIOLATIONS" ]]; then
    echo -e "${RED}❌ Binary Expr::Add pattern found in n-ary sum modules:${NC}"
    echo ""
    echo -e "$VIOLATIONS"
    echo ""
    echo -e "${YELLOW}Use AddView from crate::nary for shape-independent sum handling.${NC}"
    echo -e "${YELLOW}Binary Expr::Mul patterns are allowed for structural matching.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ N-ary shape independence: no binary Expr::Add in n-ary sum modules.${NC}"
exit 0
