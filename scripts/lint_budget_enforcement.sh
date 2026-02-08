#!/usr/bin/env bash
# lint_budget_enforcement.sh - Ensure hotspot modules have budget instrumentation
#
# Phase 6 of the unified anti-explosion budget policy.
# Checks that critical modules contain budget tracking code.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Hotspot modules that must have budget instrumentation
HOTSPOTS=(
    "crates/cas_engine/src/expand.rs"
    "crates/cas_engine/src/multinomial_expand.rs"
    "crates/cas_engine/src/engine/orchestration.rs"
    "crates/cas_engine/src/engine/transform/mod.rs"
    "crates/cas_engine/src/multipoly/arithmetic.rs"
    "crates/cas_engine/src/multipoly/gcd.rs"
    "crates/cas_engine/src/gcd_zippel_modp.rs"
)

# Budget-related patterns that indicate instrumentation
PATTERNS=(
    "PassStats"
    "_with_stats"
    "terms_materialized"
    "poly_ops"
    "budget.charge"
    "BudgetExceeded"
    "MultinomialExpandBudget"
    "PolyBudget"
    "GcdBudget"
    "ZippelBudget"
    "max_output_terms"
    "max_terms"
)

echo "==> Checking budget enforcement in hotspot modules..."

failed=0
for file in "${HOTSPOTS[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}✘ Missing hotspot: $file${NC}"
        failed=1
        continue
    fi
    
    found=0
    for pattern in "${PATTERNS[@]}"; do
        if grep -q "$pattern" "$file" 2>/dev/null; then
            found=1
            break
        fi
    done
    
    if [[ $found -eq 0 ]]; then
        echo -e "${RED}✘ No budget instrumentation in: $file${NC}"
        failed=1
    else
        echo -e "${GREEN}✔ $file${NC}"
    fi
done

if [[ $failed -eq 1 ]]; then
    echo -e "\n${RED}Budget enforcement lint failed!${NC}"
    echo "Hotspots must contain at least one of: ${PATTERNS[*]}"
    exit 1
fi

echo -e "\n${GREEN}✔ All hotspots have budget instrumentation${NC}"
