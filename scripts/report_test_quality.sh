#!/usr/bin/env bash
# Test Quality Report Script
# Reports on test patterns without failing - useful for tracking technical debt
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_TEST_DIR="$ROOT/crates/cas_engine/tests"
CLI_TEST_DIR="$ROOT/crates/cas_cli/tests"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                     TEST QUALITY REPORT                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
echo "Root: $ROOT"
echo

# Count pattern using grep
count_pattern() {
  local pat="$1"
  local dirs="$ENGINE_TEST_DIR $CLI_TEST_DIR"
  grep -r "$pat" $dirs --include="*.rs" 2>/dev/null | wc -l | tr -d ' '
}

TOTAL_FILES=$(find "$ENGINE_TEST_DIR" "$CLI_TEST_DIR" -name "*.rs" 2>/dev/null | wc -l | tr -d ' ')
echo "📁 Files scanned: $TOTAL_FILES"
echo

echo "━━━ ✅ STRONG PATTERNS (good) ━━━"
printf "  test_utils imports:     %4d\n" "$(count_pattern 'use.*test_utils')"
printf "  assert_equiv_numeric_*: %4d\n" "$(count_pattern 'assert_equiv_numeric_')"
printf "  assert_simplifies_to*:  %4d\n" "$(count_pattern 'assert_simplifies_to')"
printf "  assert_eq! (exact):     %4d\n" "$(count_pattern 'assert_eq!')"
echo

echo "━━━ ⚠️  WEAK PATTERNS (candidates for migration) ━━━"
printf "  .contains() in asserts: %4d\n" "$(count_pattern '\.contains(')"
printf "  !is_empty() checks:     %4d\n" "$(count_pattern '!.*is_empty()')"
printf "  is_ok() checks:         %4d\n" "$(count_pattern 'is_ok()')"
echo

echo "━━━ 📋 IGNORED/DISABLED ━━━"
printf "  #[ignore] annotations:  %4d\n" "$(count_pattern '#\[ignore')"
printf "  // assert (commented):  %4d\n" "$(count_pattern '// *assert')"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Note: Weak patterns are not always wrong (e.g., API contract tests),"
echo "but they are good candidates for migration to test_utils helpers."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
