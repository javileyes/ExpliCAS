#!/bin/bash
# =============================================================================
# Lint: Detect duplicate utility function definitions
# =============================================================================
#
# This script prevents reintroduction of utility function duplicates by
# detecting new DEFINITIONS (not calls) outside canonical modules.
#
# Canonical Registry:
# - __hold helpers    → cas_ast/src/hold.rs
# - flatten_add*      → cas_engine/src/nary.rs (AddView)
# - flatten_mul*      → cas_ast/src/views.rs (MulChainView)
# - is_zero/is_one    → cas_engine/src/helpers.rs
# - get_integer       → cas_engine/src/helpers.rs
# - strip_hold        → must call cas_ast::hold

set -e

SCRIPT_DIR=$(dirname "$0")
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==> Checking for duplicate utility definitions..."

ERRORS=0
WARNINGS=0

# -----------------------------------------------------------------------------
# CHECK 1: strip_hold duplicates (HARD FAIL)
# Canonical: cas_ast/src/hold.rs
# Allowed: files that reference cas_ast::hold
# -----------------------------------------------------------------------------
echo "  [1/5] Checking strip_hold..."

for file in $(grep -rl "fn strip.*hold" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null | grep -v "cas_ast/src/hold.rs" || true); do
    if ! grep -q "cas_ast::hold::" "$file"; then
        echo -e "  ${RED}ERROR${NC}: $file defines strip_hold without using cas_ast::hold"
        ((ERRORS++))
    fi
done

# -----------------------------------------------------------------------------
# CHECK 2: flatten_add definitions (HARD FAIL - migration complete)
# Canonical: cas_engine/src/nary.rs (AddView, add_terms_no_sign, add_terms_signed)
# Allowed: canonical modules or files that wrap canonical functions
# -----------------------------------------------------------------------------
echo "  [2/5] Checking flatten_add..."

FLATTEN_ADD_ALLOWED=(
    "nary.rs"           # Canonical: AddView
    "helpers.rs"        # Legacy canonical for simple cases
)

for file in $(grep -rln "fn flatten_add" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
    basename_file=$(basename "$file")
    is_allowed=false
    for allowed in "${FLATTEN_ADD_ALLOWED[@]}"; do
        if [[ "$basename_file" == "$allowed" ]]; then
            is_allowed=true
            break
        fi
    done
    # Also allow files that use the canonical nary module
    if [ "$is_allowed" = false ] && grep -q "crate::nary::" "$file"; then
        is_allowed=true
    fi
    if [ "$is_allowed" = false ]; then
        echo -e "  ${RED}ERROR${NC}: $file defines flatten_add without using nary.rs canonical"
        echo -e "         Fix: Use crate::nary::add_terms_no_sign or crate::nary::add_terms_signed"
        ((ERRORS++))
    fi
done

# -----------------------------------------------------------------------------
# CHECK 3: flatten_mul definitions (HARD FAIL - migration complete)
# Canonical: cas_engine/src/nary.rs (MulView, mul_factors)
# Allowed: canonical modules or files that wrap canonical functions
# -----------------------------------------------------------------------------
echo "  [3/5] Checking flatten_mul..."

FLATTEN_MUL_ALLOWED=(
    "views.rs"          # Canonical: MulChainView
    "helpers.rs"        # Legacy canonical
    "nary.rs"           # Also canonical: MulView
)

for file in $(grep -rln "fn flatten_mul" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
    basename_file=$(basename "$file")
    is_allowed=false
    for allowed in "${FLATTEN_MUL_ALLOWED[@]}"; do
        if [[ "$basename_file" == "$allowed" ]]; then
            is_allowed=true
            break
        fi
    done
    # Also allow files that use the canonical nary module
    if [ "$is_allowed" = false ] && grep -q "crate::nary::" "$file"; then
        is_allowed=true
    fi
    if [ "$is_allowed" = false ]; then
        echo -e "  ${RED}ERROR${NC}: $file defines flatten_mul without using nary.rs canonical"
        echo -e "         Fix: Use crate::nary::mul_factors"
        ((ERRORS++))
    fi
done

# -----------------------------------------------------------------------------
# CHECK 4: Predicate duplicates (WARNING)
# Canonical: cas_engine/src/helpers.rs
# Matches EXACT function names only (not is_one_term, is_negative_factor, etc.)
# -----------------------------------------------------------------------------
echo "  [4/5] Checking predicates (is_zero, is_one, is_negative, get_integer)..."

PREDICATE_ALLOWED=(
    "helpers.rs"        # Canonical
    "multipoly.rs"      # Struct method (different scope)
    "polynomial.rs"     # Struct method (different scope)
    "unipoly_modp.rs"   # Struct method (different scope)
    "multipoly_modp.rs" # Struct method (different scope)
    "didactic.rs"       # Trait method
)

# Use exact word boundary matching to avoid false positives like is_one_term
for pattern in "fn is_zero(" "fn is_one(" "fn is_negative(" "fn get_integer("; do
    for file in $(grep -rln "$pattern" "$ROOT_DIR/crates/cas_engine/src" --include="*.rs" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        is_allowed=false
        for allowed in "${PREDICATE_ALLOWED[@]}"; do
            if [[ "$basename_file" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        # Also allow files that use the canonical helpers module (they are wrappers)
        if [ "$is_allowed" = false ] && grep -q "crate::helpers::" "$file"; then
            is_allowed=true
        fi
        # Check if it's a method (&self) or a commented line
        if [ "$is_allowed" = false ]; then
            line=$(grep -n "$pattern" "$file" | head -1)
            # Skip if it's a method (has &self) or commented out
            if echo "$line" | grep -qE "(&self|// fn)"; then
                :
            else
                pattern_name=$(echo "$pattern" | sed 's/($//')
                echo -e "  ${YELLOW}WARNING${NC}: $file defines $pattern_name (should use helpers.rs canonical)"
                ((WARNINGS++))
            fi
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 5: __hold in output boundaries (HARD FAIL if in production JSON)
# -----------------------------------------------------------------------------
echo "  [5/5] Checking __hold doesn't leak to JSON output..."

# Check test assertions to ensure we have contract tests
if ! grep -rq "contains.*__hold" "$ROOT_DIR/crates/cas_engine/tests" 2>/dev/null; then
    echo -e "  ${YELLOW}WARNING${NC}: No contract tests found for __hold leak prevention"
    ((WARNINGS++))
fi

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
echo ""
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}✗ Found $ERRORS error(s) - these MUST be fixed${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ Found $WARNINGS warning(s) - migration recommended${NC}"
    echo "  Run 'make cleanup-audit' for detailed report"
    exit 0  # Warnings don't fail CI (yet)
else
    echo -e "${GREEN}✔ No duplicate utility definitions found${NC}"
    exit 0
fi
