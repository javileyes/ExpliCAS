#!/usr/bin/env bash
# Lint enforcement for const_fold module.
#
# This script ensures const_fold stays isolated from the simplification pipeline.
# It HARD FAILs if any prohibited patterns are found in actual code (not comments).
#
# ALLOWLIST (canonical helpers in const_fold):
# - is_constant_literal, is_imaginary_unit, is_neg_of_i
# - fold_neg, fold_sqrt, fold_mul_imaginary
# - fold_pow, literal_int, literal_rat, pow_rational (PR2.1)
#
# DENYLIST: const_fold must NOT call these functions
# - simplify_with_stats, Simplifier::new, apply_rules_loop → avoid simplify pipeline
# - rationalize(, expand_pow(, multinomial_expand( → avoid structural transforms
# - poly_add(, poly_mul(, poly_div(, gcd_poly(, gcd_zippel( → avoid poly ops
# - domain_assumption, prove_nonzero → no domain assumptions
# - DomainMode → const_fold depends only on ValueDomain/BranchPolicy

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/crates/cas_engine/src/const_fold"

# Check directory exists
if [ ! -d "$DIR" ]; then
    echo "ERROR: const_fold directory not found: $DIR"
    exit 1
fi

echo "==> Checking const_fold denylist..."

# Denylist patterns with function call syntax (excludes comment mentions)
deny_patterns=(
    "simplify_with_stats("
    "apply_rules_loop("
    "Simplifier::new("
    "rationalize("
    "expand_pow("
    "multinomial_expand("
    "poly_add("
    "poly_mul("
    "poly_div("
    "gcd_poly("
    "gcd_zippel("
    "div_exact("
    "domain_assumption("
    "prove_nonzero("
    "DomainMode"
)

for pat in "${deny_patterns[@]}"; do
    if grep -rn "$pat" "$DIR" 2>/dev/null; then
        echo ""
        echo "ERROR: const_fold must not call '$pat'"
        echo "       This violates the isolation contract."
        exit 1
    fi
done

echo "==> Checking style rules..."

# Style: no local is_zero/is_one definitions (use canonical helpers)
if grep -rn "fn\s*is_zero\s*(" "$DIR" 2>/dev/null; then
    echo ""
    echo "ERROR: const_fold should use canonical predicates, not local is_zero"
    exit 1
fi

if grep -rn "fn\s*is_one\s*(" "$DIR" 2>/dev/null; then
    echo ""
    echo "ERROR: const_fold should use canonical predicates, not local is_one"
    exit 1
fi

echo ""
echo "✔ const_fold enforcement passed"
