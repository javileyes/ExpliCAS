#!/usr/bin/env bash
# lint_string_compares_progress.sh — Track progress of FunctionKind migration
#
# This guardrail counts the remaining string comparisons for function names
# in cas_engine. The goal is to replace `sym_name(...) == "..."` with
# FunctionKind/BuiltinFn pattern matching (Phase 2 of interning).
#
# Usage:
#   ./scripts/lint_string_compares_progress.sh           # report only (CI-safe)
#   LINT_STRING_STRICT=1 ./scripts/lint_string_compares_progress.sh  # fail if over limit
#
# Environment variables:
#   LINT_STRING_STRICT=1  - Enable strict mode (fail if count exceeds STRING_COMPARE_LIMIT)
#   STRING_COMPARE_LIMIT  - Maximum allowed string comparisons (default: 999 = always pass)

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

# Config
STRICT="${LINT_STRING_STRICT:-0}"
LIMIT="${STRING_COMPARE_LIMIT:-999}"

echo "${BLU}==>${RST} String comparisons progress (FunctionKind migration)"
echo ""

# Count sym_name comparisons in cas_engine
SYM_NAME_LINES=$(grep -rn 'sym_name([^)]*) == "' "$ENGINE_DIR" 2>/dev/null || true)
SYM_NAME_COUNT=0
if [[ -n "$SYM_NAME_LINES" ]]; then
    SYM_NAME_COUNT=$(echo "$SYM_NAME_LINES" | wc -l | tr -d ' ')
fi

# Get per-file breakdown
echo "${CYN}📊 Per-file breakdown (sym_name(...) == \"...\"):${RST}"
echo ""

if [[ -n "$SYM_NAME_LINES" ]]; then
    echo "$SYM_NAME_LINES" | \
        sed 's|'"$ENGINE_DIR"'/||' | \
        cut -d: -f1 | \
        sort | uniq -c | sort -rn | head -15 | \
        while read -r count file; do
            printf "  %3d  %s\n" "$count" "$file"
        done
else
    echo "  (none found - migration complete! 🎉)"
fi

echo ""

# Count specific builtin names
echo "${CYN}📊 Top builtin names still compared as strings:${RST}"
echo ""

for fn in sqrt sin cos tan ln log exp abs asin acos atan; do
    fn_count=$(grep -rn "== \"$fn\"" "$ENGINE_DIR" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$fn_count" -gt 0 ]]; then
        printf "  %3d  %s\n" "$fn_count" "$fn"
    fi
done

echo ""

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
