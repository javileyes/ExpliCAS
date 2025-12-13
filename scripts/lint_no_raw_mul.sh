#!/usr/bin/env bash
# lint_no_raw_mul.sh
# 
# CI lint to prevent reintroduction of ctx.add(Expr::Mul...) in production code.
# Use mul2_raw, mul2_simpl, or builders instead.
#
# Usage: ./scripts/lint_no_raw_mul.sh
# Exit 0 = pass, Exit 1 = forbidden pattern found

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Checking for forbidden ctx.add(Expr::Mul...) in cas_engine production code..."

# Find matches, excluding test files, add_raw calls, and helper definitions
MATCHES=$(grep -rn "ctx\.add(Expr::Mul" crates/cas_engine/src/ 2>/dev/null \
    | grep -v "_test\.rs:" \
    | grep -v "_tests\.rs:" \
    | grep -v "ctx\.add_raw" \
    | grep -v "fn mul2_raw" \
    || true)

PRODUCTION_MATCHES=""

while IFS= read -r match_line; do
    [[ -z "$match_line" ]] && continue
    
    FILE=$(echo "$match_line" | cut -d: -f1)
    MATCH_LINE_NUM=$(echo "$match_line" | cut -d: -f2)
    
    # Find test section start
    TEST_START=$(grep -n "mod tests" "$FILE" 2>/dev/null | head -1 | cut -d: -f1 || echo "")
    CFG_START=$(grep -n "#\[cfg(test)\]" "$FILE" 2>/dev/null | head -1 | cut -d: -f1 || echo "")
    
    # Default to very large number if not found
    [[ -z "$TEST_START" ]] && TEST_START=999999
    [[ -z "$CFG_START" ]] && CFG_START=999999
    
    # Use minimum
    if (( CFG_START < TEST_START )); then
        TEST_BOUNDARY=$CFG_START
    else
        TEST_BOUNDARY=$TEST_START
    fi
    
    # Skip if in test section (using arithmetic comparison)
    if (( MATCH_LINE_NUM >= TEST_BOUNDARY )); then
        continue
    fi
    
    # Skip comment lines
    LINE_CONTENT=$(echo "$match_line" | cut -d: -f3-)
    if echo "$LINE_CONTENT" | grep -qE "^\s*//"; then
        continue
    fi
    
    PRODUCTION_MATCHES="${PRODUCTION_MATCHES}${match_line}\n"
done <<< "$MATCHES"

PRODUCTION_MATCHES=$(echo -e "$PRODUCTION_MATCHES" | grep -v "^$" || true)

if [[ -n "$PRODUCTION_MATCHES" ]]; then
    echo -e "${RED}❌ FORBIDDEN: ctx.add(Expr::Mul...) found in production code:${NC}"
    echo ""
    echo -e "$PRODUCTION_MATCHES"
    echo ""
    echo -e "${YELLOW}Use mul2_raw(ctx, a, b) or mul2_simpl(ctx, a, b) instead.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ No forbidden ctx.add(Expr::Mul...) in cas_engine production code.${NC}"
exit 0
