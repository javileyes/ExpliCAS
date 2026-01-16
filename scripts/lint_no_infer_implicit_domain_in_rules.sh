#!/usr/bin/env bash
# lint_no_infer_implicit_domain_in_rules.sh
#
# PURPOSE: Prevent rules from calling infer_implicit_domain() directly.
# Rules should use parent_ctx.implicit_domain() (cached) instead.
#
# V2.15: This lint was added after discovering that direct calls from rules
# caused stack overflow (98+ nested calls) due to recomputation on each rule.
#
# ALLOWED:
# - Calls that are part of the "fallback pattern" (inside or_else/map after cache check)
# - Pattern: parent_ctx.implicit_domain().cloned().or_else(|| { ... infer_implicit_domain ... })

set -euo pipefail

echo "==> Checking for infer_implicit_domain calls in rules..."

# Find all occurrences
all_calls=$(grep -RIn 'infer_implicit_domain\s*(' crates/cas_engine/src/rules/ 2>/dev/null || true)

if [[ -z "${all_calls}" ]]; then
    echo "✔ No infer_implicit_domain calls in rules"
    exit 0
fi

# Check if each call is in the allowed "fallback" pattern
# The pattern we allow: call inside .or_else(|| { ... }) or .map(|root| { ... })
# We check by looking for "parent_ctx.root_expr().map" context
violations=""
while IFS= read -r line; do
    file=$(echo "$line" | cut -d: -f1)
    lineno=$(echo "$line" | cut -d: -f2)
    
    # Get a few lines of context before the call
    context=$(sed -n "$((lineno - 5)),$((lineno))p" "$file" 2>/dev/null || true)
    
    # If context contains the fallback pattern indicators, it's allowed
    if echo "$context" | grep -qE '\.or_else\s*\(\s*\|\s*\|' || \
       echo "$context" | grep -qE 'parent_ctx\.root_expr\(\)\.map'; then
        # Allowed: this is the fallback pattern
        continue
    fi
    
    violations="${violations}${line}\n"
done <<< "$all_calls"

if [[ -n "${violations}" ]]; then
    echo "❌ ERROR: infer_implicit_domain() called directly from rules."
    echo ""
    echo "Rules should use parent_ctx.implicit_domain() (cached)."
    echo "If cache is unavailable, use the fallback pattern:"
    echo "  parent_ctx.implicit_domain().cloned().or_else(|| {"
    echo "      parent_ctx.root_expr().map(|root| {"
    echo "          infer_implicit_domain(ctx, root, vd)"
    echo "      })"
    echo "  })"
    echo ""
    echo "Violations found:"
    echo -e "${violations}"
    exit 1
fi

echo "✔ All infer_implicit_domain calls in rules use the fallback pattern"
