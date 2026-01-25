#!/usr/bin/env bash
# lint_string_compares_progress.sh — Prevent string comparisons for function names
#
# STRICT MODE (default): No function string comparisons allowed in cas_engine/cas_cli.
# This guardrail enforces is_builtin()/BuiltinFn pattern matching for function dispatch.
#
# Exception Policy:
#   - FUNCTION comparisons (Expr::Function) → BLOCKED (must use BuiltinFn)
#   - VARIABLE comparisons (Expr::Variable) → ALLOWED (intentional in solver, ~14 cases)
#     These compare against "x", "y", "var" for equation solving - not a migration target.
#
# Usage:
#   ./scripts/lint_string_compares_progress.sh           # strict mode (default in CI)
#   LINT_STRING_STRICT=0 ./scripts/lint_string_compares_progress.sh  # report only
#
# Environment variables:
#   LINT_STRING_STRICT=1  - Enable strict mode (default: 1)
#   STRING_COMPARE_LIMIT  - Maximum allowed comparisons (default: 0)

set -uo pipefail
# Note: NOT using set -e because grep returns 1 when no matches, which is expected

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="$ROOT_DIR/crates/cas_engine/src"
AST_DIR="$ROOT_DIR/crates/cas_ast/src"

# Colors
RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
BLU=$'\033[34m'
CYN=$'\033[36m'
RST=$'\033[0m'

# Config - strict mode ON by default (no function string comparisons allowed in production)
STRICT="${LINT_STRING_STRICT:-1}"
LIMIT="${STRING_COMPARE_LIMIT:-0}"

echo "${BLU}==>${RST} String comparisons progress (FunctionKind migration)"
echo ""

# Count sym_name comparisons in cas_engine
# We separate FUNCTION comparisons (the goal of this guardrail) from VARIABLE comparisons
SYM_NAME_ALL_LINES=$(grep -rn 'sym_name([^)]*) == "' "$ENGINE_DIR" 2>/dev/null || true)

# Filter: Variable comparisons (Expr::Variable pattern)
SYM_NAME_VAR_LINES=$(echo "$SYM_NAME_ALL_LINES" | grep -E 'Expr::Variable|Variable\(' || true)
SYM_NAME_VAR_COUNT=0
if [[ -n "$SYM_NAME_VAR_LINES" ]]; then
    SYM_NAME_VAR_COUNT=$(echo "$SYM_NAME_VAR_LINES" | wc -l | tr -d ' ')
fi

# Filter: Function comparisons (everything else - this is what we want to migrate)
SYM_NAME_FN_LINES=$(echo "$SYM_NAME_ALL_LINES" | grep -vE 'Expr::Variable|Variable\(' || true)
SYM_NAME_FN_COUNT=0
if [[ -n "$SYM_NAME_FN_LINES" ]]; then
    SYM_NAME_FN_COUNT=$(echo "$SYM_NAME_FN_LINES" | wc -l | tr -d ' ')
fi

SYM_NAME_COUNT=$SYM_NAME_FN_COUNT  # Use function count as the main metric

# Get per-file breakdown (FUNCTION comparisons only)
echo "${CYN}📊 Per-file breakdown (function sym_name(...) == \"...\"):${RST}"
echo ""

if [[ -n "$SYM_NAME_FN_LINES" ]]; then
    echo "$SYM_NAME_FN_LINES" | \
        sed 's|'"$ENGINE_DIR"'/||' | \
        cut -d: -f1 | \
        sort | uniq -c | sort -rn | head -15 | \
        while read -r count file; do
            printf "  %3d  %s\n" "$count" "$file"
        done
else
    echo "  (none found - function migration complete! 🎉)"
fi

echo ""

# Show variable comparisons separately (informational only)
echo "${YLW}📊 Variable comparisons (not targeted by this guardrail):${RST}"
echo "  Variable sym_name checks: ${BLU}$SYM_NAME_VAR_COUNT${RST} (these use Expr::Variable, not Function)"
echo ""

# NOTE: String comparisons like `name == "sin"` (where name is already a &str)
# are NOT what this guardrail targets. Those are pre-lookup patterns that could
# eventually migrate to BuiltinFn too, but are lower priority than sym_name() calls.

# Summary
echo "${CYN}📈 Summary:${RST}"
echo ""
echo "  ${YLW}sym_name(...) == \"...\"${RST} in cas_engine: ${BLU}$SYM_NAME_COUNT${RST}"
echo ""

# Check is_call_named usage (the O(1) alternative)
IS_CALL_NAMED_LINES=$(grep -rn 'is_call_named' "$ENGINE_DIR" 2>/dev/null | grep -v "fn is_call_named" || true)
IS_CALL_NAMED_COUNT=0
if [[ -n "$IS_CALL_NAMED_LINES" ]]; then
    IS_CALL_NAMED_COUNT=$(echo "$IS_CALL_NAMED_LINES" | wc -l | tr -d ' ')
fi
echo "  ${GRN}is_call_named() usages${RST} (O(1) alternative): ${BLU}$IS_CALL_NAMED_COUNT${RST}"
echo ""

# Check if FunctionKind/BuiltinFn exist
FUNC_KIND_FOUND=$(grep -rn 'FunctionKind\|BuiltinFn' "$AST_DIR" 2>/dev/null || true)
if [[ -n "$FUNC_KIND_FOUND" ]]; then
    echo "  ${GRN}✔${RST} FunctionKind/BuiltinFn: ${GRN}defined${RST}"
else
    echo "  ${YLW}○${RST} FunctionKind/BuiltinFn: ${YLW}not yet defined${RST}"
fi

echo ""

# Strict mode check
if [[ "$STRICT" -eq 1 ]]; then
    if [[ "$SYM_NAME_COUNT" -gt "$LIMIT" ]]; then
        echo "${RED}✘${RST} String comparisons ($SYM_NAME_COUNT) exceed limit ($LIMIT)"
        echo "  Set STRING_COMPARE_LIMIT=$SYM_NAME_COUNT to update the baseline."
        exit 1
    else
        echo "${GRN}✔${RST} String comparisons ($SYM_NAME_COUNT) within limit ($LIMIT)"
    fi
else
    echo "${GRN}✔${RST} string-compares guardrail: OK (report only mode)"
    echo "  Tip: Set LINT_STRING_STRICT=1 STRING_COMPARE_LIMIT=N to enforce a limit."
fi
