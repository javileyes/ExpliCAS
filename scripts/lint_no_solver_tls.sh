#!/usr/bin/env bash
# lint_no_solver_tls.sh
#
# CI lint: prevent new thread_local! declarations in solver modules.
#
# Allowlist (legacy, tracked for future migration):
#   - SOLVE_DEPTH         (solve_core.rs via mod.rs)
#   - SOLVE_ASSUMPTIONS   (mod.rs)
#   - OUTPUT_SCOPES       (mod.rs)
#   - SOLVE_SEEN          (solve_core.rs)
#
# Usage: ./scripts/lint_no_solver_tls.sh
# Exit 0 = pass, Exit 1 = new TLS found

set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

SOLVER_DIR="crates/cas_engine/src/solver"

# Allowlisted TLS variable names (these already exist and are tracked)
ALLOWLIST="SOLVE_DEPTH|SOLVE_ASSUMPTIONS|OUTPUT_SCOPES|SOLVE_SEEN"

echo "==> Checking for new thread_local! declarations in solver..."

# Find all thread_local! blocks in solver, extract the variable names
VIOLATIONS=""

# Search recursively for thread_local! in all .rs files under solver/
while IFS= read -r file; do
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
done < <(find "$SOLVER_DIR" -name '*.rs' -type f)

VIOLATIONS=$(echo -e "$VIOLATIONS" | grep -v "^$" || true)

if [[ -n "$VIOLATIONS" ]]; then
    echo -e "${RED}❌ New thread_local! declarations found in solver:${NC}"
    echo ""
    echo -e "$VIOLATIONS"
    echo ""
    echo "The solver is designed to be TLS-free (context via SolveCtx)."
    echo "Move state into SolveCtx or use an existing mechanism."
    echo ""
    echo "Allowlisted (legacy): SOLVE_DEPTH, SOLVE_ASSUMPTIONS, OUTPUT_SCOPES, SOLVE_SEEN"
    exit 1
fi

echo -e "${GREEN}✔${NC} No new thread_local! in solver (4 legacy allowlisted)"
exit 0
