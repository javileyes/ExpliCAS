#!/usr/bin/env bash
# Lint: presimplify_safe must remain a minimal, auditable allowlist-only pipeline
#
# This lint enforces the Safe Feature Layering Pattern for
# cas_math::limits_support::presimplify_safe_for_limit:
# - No access to rationalization, expansion, or polynomial modules
# - No calls to the general simplifier engine
# - No domain assumptions
#
# POLICY: presimplify_safe may ONLY use:
# - helpers::{is_zero, is_one} (canonical predicates)
# - Direct structural transforms (Add/Sub/Mul/Neg/Div/Pow)
# - No external simplification or rule application

set -euo pipefail

FILE="crates/cas_math/src/limits_support.rs"

if [[ ! -f "$FILE" ]]; then
  echo "❌ presimplify source not found: $FILE"
  exit 1
fi

echo "===> Checking presimplify_safe isolation..."

ERRORS=0
TMP_FILE="$(mktemp "${TMPDIR:-/tmp}/presimplify_safe_for_limit.XXXXXX.rs")"
trap 'rm -f "$TMP_FILE"' EXIT

if command -v rg >/dev/null 2>&1; then
  search_regex() { rg -n "$1" "$2"; }
  search_fixed() { rg -n -F "$1" "$2"; }
else
  search_regex() { grep -nE "$1" "$2"; }
  search_fixed() { grep -nF "$1" "$2"; }
fi

START_LINE="$(search_regex '^const PRESIMPLIFY_MAX_DEPTH:' "$FILE" | head -n1 | cut -d: -f1)"
END_LINE="$(search_regex '^#\[cfg\(test\)\]' "$FILE" | awk -F: -v start="$START_LINE" '$1 > start { print $1; exit }')"

if [[ -z "$START_LINE" || -z "$END_LINE" ]]; then
  echo "❌ could not isolate presimplify_safe_for_limit in $FILE"
  exit 1
fi

sed -n "${START_LINE},$((END_LINE - 1))p" "$FILE" > "$TMP_FILE"

if ! search_fixed "pub fn presimplify_safe_for_limit" "$TMP_FILE" >/dev/null 2>&1; then
  echo "❌ presimplify_safe_for_limit not found in isolated lint slice"
  exit 1
fi

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
  if search_fixed "$p" "$TMP_FILE" >/dev/null 2>&1; then
    echo "❌ presimplify_safe must NOT reference: $p"
    search_fixed "$p" "$TMP_FILE"
    ((ERRORS++)) || true
  fi
done

# Allowlist check: should use canonical helpers (not local definitions)
if search_regex "^fn is_zero([^[:alnum:]_]|$)" "$TMP_FILE" >/dev/null 2>&1; then
  echo "❌ presimplify_safe has local is_zero; use crate::helpers::is_zero"
  ((ERRORS++)) || true
fi

if search_regex "^fn is_one([^[:alnum:]_]|$)" "$TMP_FILE" >/dev/null 2>&1; then
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
