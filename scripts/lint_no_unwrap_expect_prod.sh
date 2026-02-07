#!/usr/bin/env bash
set -euo pipefail

# Budget-based unwrap()/expect() guardrail for production code.
#
# Counts .unwrap() and .expect() in each production crate (excluding tests)
# and fails if any crate exceeds its budget. As developers fix unwrap calls,
# they lower the budget numbers below to ratchet down the count.
#
# Scope: all production crates (cas_engine, cas_cli, cas_ast, cas_android_ffi)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Budgets (current ceiling — lower these as you fix unwraps) ───
# Format: CRATE_DIR MAX_UNWRAP MAX_EXPECT
BUDGETS=(
    "crates/cas_engine/src    303  36"
    "crates/cas_ast/src        15  19"
    "crates/cas_cli/src        18   0"
    "crates/cas_android_ffi/src 0   1"
)

count_pattern() {
    local dir="$1" rg_pattern="$2" grep_pattern="$3"
    if [[ ! -d "$ROOT_DIR/$dir" ]]; then
        echo "0"
        return
    fi
    # Use rg if available, otherwise grep -F (fixed-string)
    if command -v rg >/dev/null 2>&1; then
        rg -c "$rg_pattern" "$ROOT_DIR/$dir" \
            --glob '*.rs' \
            --glob '!**/tests/**' \
            --glob '!**/test/**' \
            --glob '!**/*test*.rs' \
            --glob '!**/*tests*.rs' \
            2>/dev/null | awk -F: '{s+=$2} END {print s+0}'
    else
        find "$ROOT_DIR/$dir" -name '*.rs' \
            ! -path '*/tests/*' \
            ! -path '*/test/*' \
            ! -name '*test*.rs' \
            ! -name '*tests*.rs' \
            -exec grep -Fc "$grep_pattern" {} \; 2>/dev/null \
            | awk '{s+=$1} END {print s+0}'
    fi
}

echo "══════════════════════════════════════════════════════"
echo "  unwrap()/expect() budget lint"
echo "══════════════════════════════════════════════════════"
printf "  %-30s %8s %8s %8s %8s\n" "Crate" "unwrap" "budget" "expect" "budget"
echo "  ──────────────────────────────────────────────────"

failed=0

for entry in "${BUDGETS[@]}"; do
    read -r dir max_unwrap max_expect <<< "$entry"
    crate_name=$(echo "$dir" | sed 's|crates/||;s|/src||')

    actual_unwrap=$(count_pattern "$dir" '\.unwrap\(\)' '.unwrap()')
    actual_expect=$(count_pattern "$dir" '\.expect\(' '.expect(')

    # Status indicators
    u_status="✔"
    e_status="✔"
    if (( actual_unwrap > max_unwrap )); then
        u_status="✘"
        failed=1
    fi
    if (( actual_expect > max_expect )); then
        e_status="✘"
        failed=1
    fi

    printf "  %-30s %3s %4d/%-4d %3s %4d/%-4d\n" \
        "$crate_name" "$u_status" "$actual_unwrap" "$max_unwrap" \
        "$e_status" "$actual_expect" "$max_expect"
done

echo "  ──────────────────────────────────────────────────"

if [[ $failed -eq 1 ]]; then
    echo ""
    echo "✘ Budget exceeded! Lower the count or raise the budget in this script."
    exit 1
fi

echo ""
echo "✔ unwrap/expect budget: all crates within limits"
