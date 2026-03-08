#!/usr/bin/env bash
# lint_no_solver_tls.sh
#
# CI lint: prevent new thread_local! declarations in solver modules.
#
# Allowlist (legacy, tracked for future migration):
#   - SOLVE_DEPTH         (recursion depth guard in solve runtime modules)
#
# Usage: ./scripts/lint_no_solver_tls.sh
# Exit 0 = pass, Exit 1 = new TLS found

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

SCAN_PATHS=(
    "crates/cas_engine/src/solve_core_runtime.rs"
    "crates/cas_engine/src/solve_isolation_runtime.rs"
    "crates/cas_engine/src/solve_runtime.rs"
    "crates/cas_engine/src/solve_runtime_adapters.rs"
)

# Allowlisted TLS variable names (currently empty; keep hook for future migrations)
ALLOWLIST=""

echo "==> Checking for new thread_local! declarations in solver..."

# Find all thread_local! blocks in solver, extract the variable names
VIOLATIONS=""

FILES_TO_SCAN=()
for path in "${SCAN_PATHS[@]}"; do
    if [[ -d "$path" ]]; then
        while IFS= read -r file; do
            FILES_TO_SCAN+=("$file")
        done < <(find "$path" -name '*.rs' -type f)
    elif [[ -f "$path" ]]; then
        FILES_TO_SCAN+=("$path")
    fi
done

# Search for thread_local! in all selected solver/runtime files
for file in "${FILES_TO_SCAN[@]}"; do
    # Get line numbers with thread_local!
    TL_LINES=$(grep -n 'thread_local!' "$file" 2>/dev/null || echo "")
    [[ -z "$TL_LINES" ]] && continue

    while IFS= read -r tl_line; do
        [[ -z "$tl_line" ]] && continue
        LINE_NUM=$(echo "$tl_line" | cut -d: -f1)

        # Check surrounding lines (up to 10 after) for the variable name
        BLOCK=$(sed -n "${LINE_NUM},$((LINE_NUM + 15))p" "$file")

        # Extract static variable names from the block
        VAR_NAMES=$(echo "$BLOCK" | grep -oE 'static\s+[A-Z_]+' | sed 's/static\s*//' || echo "")

        for var in $VAR_NAMES; do
            if echo "$var" | grep -qE "^($ALLOWLIST)$"; then
                continue  # Allowlisted
            fi
            VIOLATIONS="${VIOLATIONS}  ${file}:${LINE_NUM} → ${var}\n"
        done
    done <<< "$TL_LINES"
done

VIOLATIONS=$(echo -e "$VIOLATIONS" | grep -v "^$" || true)

if [[ -n "$VIOLATIONS" ]]; then
    echo -e "${RED}❌ New thread_local! declarations found in solver:${NC}"
    echo ""
    echo -e "$VIOLATIONS"
    echo ""
    echo "The solver is designed to be TLS-free (context via SolveCtx)."
    echo "Move state into SolveCtx or use an existing mechanism."
    echo ""
    if [[ -n "$ALLOWLIST" ]]; then
        echo "Allowlisted symbols: $ALLOWLIST"
    fi
    exit 1
fi

if [[ -n "$ALLOWLIST" ]]; then
    echo -e "${GREEN}✔${NC} No new thread_local! in solver/runtime scan set"
    echo "Allowlist entries: $ALLOWLIST"
else
    echo -e "${GREEN}✔${NC} No thread_local! declarations in solver/runtime scan set"
fi
exit 0
