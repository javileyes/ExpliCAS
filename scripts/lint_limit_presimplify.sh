#!/usr/bin/env bash
# Lint: presimplify_safe must remain a minimal, auditable allowlist-only pipeline
#
# This lint enforces the Safe Feature Layering Pattern for limits/presimplify.rs:
# - No access to rationalization, expansion, or polynomial modules
# - No calls to the general simplifier engine
# - No domain assumptions
#
# POLICY: presimplify_safe may ONLY use:
# - helpers::{is_zero, is_one} (canonical predicates)
# - Direct structural transforms (Add/Sub/Mul/Neg/Div/Pow)
# - No external simplification or rule application

set -euo pipefail

FILE="crates/cas_engine/src/limits/presimplify.rs"

if [[ ! -f "$FILE" ]]; then
  echo "⚠️  presimplify.rs not found (skipping lint)"
  exit 0
fi

echo "===> Checking presimplify_safe isolation..."

ERRORS=0

# Denylist: any reference to these patterns is a violation
DENY_PATTERNS=(
  "rationalize"
  "expand_with_stats"
  "expand::"
  "multinomial"
  "gcd_"
  "poly_"
  "simplify_with_stats"
  "apply_rules_loop"
  "domain_assumption"
  "Simplifier"
  "engine::simplify"
  "::simplify("
)

for p in "${DENY_PATTERNS[@]}"; do
  if rg -n "$p" "$FILE" >/dev/null 2>&1; then
    echo "❌ presimplify_safe must NOT reference: $p"
    rg -n "$p" "$FILE"
    ((ERRORS++)) || true
  fi
done

# Allowlist check: should use canonical helpers (not local definitions)
if rg -n "^fn is_zero\b" "$FILE" >/dev/null 2>&1; then
  echo "❌ presimplify_safe has local is_zero; use crate::helpers::is_zero"
  ((ERRORS++)) || true
fi

if rg -n "^fn is_one\b" "$FILE" >/dev/null 2>&1; then
  echo "❌ presimplify_safe has local is_one; use crate::helpers::is_one"
  ((ERRORS++)) || true
fi

if [[ $ERRORS -gt 0 ]]; then
  echo ""
  echo "✗ presimplify_safe lint FAILED ($ERRORS error(s))"
  echo "  Fix: presimplify_safe must be a minimal allowlist pipeline."
  echo "  See: docs/Safe_Feature_Layering_Pattern.md"
  exit 1
fi

echo "✔ presimplify_safe lint OK (isolated, allowlist-only)"
