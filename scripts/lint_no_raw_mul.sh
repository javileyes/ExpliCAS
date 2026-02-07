#!/usr/bin/env bash
set -euo pipefail

# lint_no_raw_mul.sh
# 
# CI lint to prevent reintroduction of ctx.add(Expr::Mul...) in production code.
# Use mul2_raw, mul2_simpl, or builders instead.
#
# Usage: ./scripts/lint_no_raw_mul.sh
# Exit 0 = pass, Exit 1 = forbidden pattern found

echo "🔍 Checking for forbidden raw Mul construction: ctx.add(Expr::Mul...)"

# What we forbid (raw Mul construction). Keep it narrow to reduce false positives.
PATTERN='ctx\.add\(\s*Expr::Mul'

# Where to search
SEARCH_DIR="crates/cas_engine/src"

# Allowlist: files that are legitimate low-level builders/infrastructure
# These files need to construct Mul directly as part of their core function.
ALLOWLIST=(
  "crates/cas_engine/src/nary.rs"
  "crates/cas_engine/src/engine/mod.rs"
  "crates/cas_engine/src/build.rs"
  "crates/cas_engine/src/session.rs"
  "crates/cas_engine/src/env.rs"
  "crates/cas_engine/src/rationalize.rs"
  "crates/cas_engine/src/multinomial_expand.rs"
  "crates/cas_engine/src/rules/inverse_trig.rs"
  "crates/cas_engine/src/rules/trigonometry/identities.rs"
  "crates/cas_engine/src/rules/canonicalization.rs"
  "crates/cas_engine/src/rules/functions.rs"
  "crates/cas_engine/src/rules/complex.rs"
  "crates/cas_engine/src/rules/integration.rs"
  "crates/cas_engine/src/rules/algebra/gcd_modp.rs"
  "crates/cas_engine/src/rules/algebra/roots.rs"
  "crates/cas_engine/src/rules/algebra/fractions.rs"
)

# A per-line escape hatch (use sparingly)
ALLOW_COMMENT="lint:allow-raw-mul"

# Collect matches
MATCHES=$(grep -rn "$PATTERN" "$SEARCH_DIR" 2>/dev/null \
    | grep -v "_test\.rs:" \
    | grep -v "_tests\.rs:" \
    || true)

if [[ -z "$MATCHES" ]]; then
  echo "✅ No forbidden raw Mul constructions found."
  exit 0
fi

failed=0
allowed_count=0

while IFS= read -r match_line; do
  [[ -z "$match_line" ]] && continue
  
  file=$(echo "$match_line" | cut -d: -f1)
  lineno=$(echo "$match_line" | cut -d: -f2)

  # 1) Check file allowlist
  allowed=false
  for a in "${ALLOWLIST[@]}"; do
    if [[ "$file" == "$a" ]]; then
      allowed=true
      break
    fi
  done

  # 2) Check for inline comment marker
  if [[ "$allowed" == false ]]; then
    line_text="$(sed -n "${lineno}p" "$file" 2>/dev/null || true)"
    if [[ "$line_text" == *"$ALLOW_COMMENT"* ]]; then
      allowed=true
    fi
  fi

  # 3) Check if in test section
  if [[ "$allowed" == false ]]; then
    TEST_START=$(grep -n "mod tests" "$file" 2>/dev/null | head -1 | cut -d: -f1 || echo "")
    CFG_START=$(grep -n "#\[cfg(test)\]" "$file" 2>/dev/null | head -1 | cut -d: -f1 || echo "")
    [[ -z "$TEST_START" ]] && TEST_START=999999
    [[ -z "$CFG_START" ]] && CFG_START=999999
    if (( CFG_START < TEST_START )); then
      TEST_BOUNDARY=$CFG_START
    else
      TEST_BOUNDARY=$TEST_START
    fi
    if (( lineno >= TEST_BOUNDARY )); then
      allowed=true
    fi
  fi

  # 4) Skip comment lines
  if [[ "$allowed" == false ]]; then
    line_text="$(sed -n "${lineno}p" "$file" 2>/dev/null || true)"
    if echo "$line_text" | grep -qE "^\s*//"; then
      allowed=true
    fi
  fi

  if [[ "$allowed" == true ]]; then
    ((allowed_count++))
  else
    echo "❌ forbidden: $match_line"
    failed=1
  fi
done <<< "$MATCHES"

if [[ "$failed" -ne 0 ]]; then
  echo
  echo "This repo forbids constructing Mul via ctx.add(Expr::Mul...) in production code."
  echo "Use mul2_raw(ctx, a, b) or mul2_simpl(ctx, a, b) instead."
  echo "If a low-level builder must do this, add its file to ALLOWLIST or add // $ALLOW_COMMENT."
  exit 1
fi

echo "✅ Raw Mul lint passed ($allowed_count allowlisted occurrences)."
