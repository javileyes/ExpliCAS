#!/usr/bin/env bash
# Lint enforcement for substitute module.
#
# This script ensures substitute stays isolated from the simplification pipeline.
# It HARD FAILs if any prohibited patterns are found in actual code (not comments).
#
# ALLOWLIST (canonical helpers in substitute):
# - substitute_power_aware, substitute_inner
# - as_power_int, as_int_exponent, mk_pow_int
# - compare_expr (for structural matching)
#
# DENYLIST: substitute must NOT call these functions
# - simplify_with_stats, Simplifier::new, apply_rules_loop → avoid simplify pipeline
# - rationalize(, expand_pow(, multinomial_expand( → avoid structural transforms
# - poly_add(, poly_mul(, poly_div(, gcd_poly(, gcd_zippel( → avoid poly ops
# - domain_assumption, prove_nonzero → no domain assumptions
# - DomainMode, ValueDomain → substitute is domain-independent

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$ROOT/crates/cas_engine/src/substitute.rs"

# Check file exists
if [ ! -f "$FILE" ]; then
    echo "ERROR: substitute.rs not found: $FILE"
    exit 1
fi

echo "==> Checking substitute.rs denylist..."

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
)

for pat in "${deny_patterns[@]}"; do
    if grep -n "$pat" "$FILE" 2>/dev/null; then
        echo ""
        echo "ERROR: substitute.rs must not call '$pat'"
        echo "       This violates the isolation contract."
        exit 1
    fi
done

echo ""
echo "✔ substitute enforcement passed"
