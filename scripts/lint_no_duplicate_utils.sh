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
echo "  [1/10] Checking strip_hold..."

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
echo "  [2/10] Checking flatten_add..."

FLATTEN_ADD_ALLOWED=(
    "nary.rs"           # Canonical: AddView
    "helpers.rs"        # Legacy canonical for simple cases
    "trig_roots_flatten.rs"  # Part of helpers.rs via include!()
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
echo "  [3/10] Checking flatten_mul..."

FLATTEN_MUL_ALLOWED=(
    "views.rs"          # Canonical: MulChainView
    "helpers.rs"        # Legacy canonical
    "nary.rs"           # Also canonical: MulView
    "trig_roots_flatten.rs"  # Part of helpers.rs via include!()
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
# CHECK 4: Predicate duplicates (HARD FAIL - migration complete)
# Canonical: cas_engine/src/helpers.rs
# Matches EXACT function names only (not is_one_term, is_negative_factor, etc.)
# Allowed: canonical module or files that wrap canonical functions
# -----------------------------------------------------------------------------
echo "  [4/10] Checking predicates (is_zero, is_one, is_negative, get_integer)..."

PREDICATE_ALLOWED=(
    "helpers.rs"        # Canonical
    "predicates.rs"     # Part of helpers.rs via include!()
    "solver_domain.rs"  # Part of helpers.rs via include!()
    "extraction.rs"     # Renamed from solver_domain.rs (Phase 3.1)
    "mod.rs"            # multipoly struct methods (different scope)
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
                echo -e "  ${RED}ERROR${NC}: $file defines $pattern_name without using helpers.rs canonical"
                echo -e "         Fix: Use crate::helpers::$pattern_name or wrap canonical"
                ((ERRORS++))
            fi
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 5: __hold in output boundaries (HARD FAIL if in production JSON)
# -----------------------------------------------------------------------------
echo "  [5/10] Checking __hold doesn't leak to JSON output..."

# Check test assertions to ensure we have contract tests
if ! grep -rq "contains.*__hold" "$ROOT_DIR/crates/cas_engine/tests" 2>/dev/null; then
    echo -e "  ${YELLOW}WARNING${NC}: No contract tests found for __hold leak prevention"
    ((WARNINGS++))
fi

# -----------------------------------------------------------------------------
# CHECK 6: Builder duplicates (HARD FAIL - migration complete)
# Canonical: MulBuilder (right-fold), Context::build_balanced_mul (balanced)
# -----------------------------------------------------------------------------
echo "  [6/10] Checking builders (build_mul_from_factors)..."

BUILDER_ALLOWED=(
    "views.rs"       # MulBuilder canonical
    "expression.rs"  # Context::build_balanced_mul canonical
    "nary.rs"        # Wrapper (allowed)
)

for pattern in "fn build_mul_from_factors" "fn build_balanced_mul"; do
    for file in $(grep -rln "$pattern" "$ROOT_DIR/crates/cas_engine/src" --include="*.rs" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        is_allowed=false
        for allowed in "${BUILDER_ALLOWED[@]}"; do
            if [[ "$basename_file" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        # Also allow files that use canonical builders (wrappers)
        if [ "$is_allowed" = false ] && grep -qE "(MulBuilder|crate::nary::build_balanced_mul|ctx\.build_balanced_mul)" "$file"; then
            is_allowed=true
        fi
        if [ "$is_allowed" = false ]; then
            echo -e "  ${RED}ERROR${NC}: $file defines $pattern without using canonical builder"
            echo -e "         Fix: Use MulBuilder or Context::build_balanced_mul"
            ((ERRORS++))
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 7: Traversal duplicates (HARD FAIL - migration complete)
# Canonical: cas_ast::traversal::{count_all_nodes, count_nodes_matching, count_nodes_and_max_depth}
# -----------------------------------------------------------------------------
echo "  [7/10] Checking traversal (count_nodes*)..."

TRAVERSAL_ALLOWED=(
    "traversal.rs"  # Canonical module
)

for pattern in "fn count_nodes\>" "fn count_nodes_matching" "fn count_nodes_and_depth" "fn count_nodes_and_max_depth"; do
    for file in $(grep -rln "$pattern" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        is_allowed=false
        for allowed in "${TRAVERSAL_ALLOWED[@]}"; do
            if [[ "$basename_file" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        # Allow files that use canonical traversal (wrappers)
        if [ "$is_allowed" = false ] && grep -qE "(cas_ast::traversal::|crate::traversal::)" "$file"; then
            is_allowed=true
        fi
        if [ "$is_allowed" = false ]; then
            echo -e "  ${RED}ERROR${NC}: $file defines $pattern without using canonical traversal"
            echo -e "         Fix: Use cas_ast::traversal::count_all_nodes or count_nodes_matching"
            ((ERRORS++))
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 8: Factor-list helpers (HARD FAIL - migration complete)
# Canonical: cas_math::fraction_factors::{merge_factor_multiset, find_factor_exp}
# (the former `factors_to_vec` / `find_factor_exp` that the four div_* cancel
# modules each defined privately, byte-identical — consolidated 2026-07-02).
# -----------------------------------------------------------------------------
echo "  [8/10] Checking factor-list helpers (factors_to_vec/find_factor_exp)..."

FACTOR_ALLOWED=(
    "fraction_factors.rs"  # Canonical module
)

for pattern in "fn factors_to_vec\>" "fn find_factor_exp\>" "fn merge_factor_multiset\>"; do
    for file in $(grep -rln "$pattern" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        is_allowed=false
        for allowed in "${FACTOR_ALLOWED[@]}"; do
            if [[ "$basename_file" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        # Allow files that use the canonical fraction_factors helpers (wrappers).
        if [ "$is_allowed" = false ] && grep -qE "crate::fraction_factors::" "$file"; then
            is_allowed=true
        fi
        if [ "$is_allowed" = false ]; then
            echo -e "  ${RED}ERROR${NC}: $file defines $pattern without using canonical fraction_factors"
            echo -e "         Fix: Use crate::fraction_factors::{merge_factor_multiset, find_factor_exp}"
            ((ERRORS++))
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 9: Surd-sign kernel (HARD FAIL - migration complete)
# Canonical: cas_math::root_forms::sign_of_linear_surd (the value-level
# sign(p + q*sqrt(n)) kernel of the surd-sign chokepoint). Was re-implemented
# byte-identically as sign_of_linear_surd (cas_solver_core::solution_set) and
# linear_surd_sign (cas_solver::solve_backend_local) — consolidated 2026-07-02.
# -----------------------------------------------------------------------------
echo "  [9/10] Checking surd/nth-root exact comparators (root_forms canonical)..."

SURD_SIGN_ALLOWED=(
    "root_forms.rs"  # Canonical module
)

# All the exact surd/nth-root VALUE comparators live in cas_math::root_forms
# (chokepoint A, consolidated 2026-07-02). linear_surd_sign is the solver alias.
for pattern in \
    "fn sign_of_linear_surd\>" \
    "fn linear_surd_sign\>" \
    "fn sign_of_sum_two_surds\>" \
    "fn cmp_rational_to_quadratic_surd\>" \
    "fn cmp_rational_to_nth_root\>" \
    "fn compare_positive_nth_roots\>"; do
    for file in $(grep -rln "$pattern" "$ROOT_DIR/crates" --include="*.rs" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        is_allowed=false
        for allowed in "${SURD_SIGN_ALLOWED[@]}"; do
            if [[ "$basename_file" == "$allowed" ]]; then
                is_allowed=true
                break
            fi
        done
        # Allow files that use the canonical root_forms comparators (wrappers/aliases).
        if [ "$is_allowed" = false ] && grep -qE "root_forms::(sign_of_linear_surd|sign_of_sum_two_surds|cmp_rational_to_quadratic_surd|cmp_rational_to_nth_root|compare_positive_nth_roots)" "$file"; then
            is_allowed=true
        fi
        if [ "$is_allowed" = false ]; then
            echo -e "  ${RED}ERROR${NC}: $file defines $pattern without using canonical root_forms"
            echo -e "         Fix: Use cas_math::root_forms::<comparator>"
            ((ERRORS++))
        fi
    done
done

# -----------------------------------------------------------------------------
# CHECK 10: Python smoke-harness helpers (HARD FAIL - migration complete)
# Canonical homes:
#   - ensure_release_cas_cli  -> scripts/cas_cli_release.py (mtime-aware rebuild;
#     a weaker exists-only copy in the residual probe validated STALE binaries)
#   - extract_warning_messages -> scripts/engine_command_matrix_observability.py
#     (a drifted copy dropped rule/assumption-shaped warnings) — consolidated
#     2026-07-02. The old drifted name extract_warnings must not reappear.
# -----------------------------------------------------------------------------
echo "  [10/10] Checking Python smoke-harness helpers (single canonical home)..."

# pattern|canonical-basename ("" => no re-definition allowed anywhere)
for spec in \
    "def ensure_release_cas_cli\>|cas_cli_release.py" \
    "def extract_warning_messages\>|engine_command_matrix_observability.py" \
    "def extract_warnings\>|"; do
    pattern="${spec%%|*}"
    canonical="${spec##*|}"
    for file in $(grep -rln "$pattern" "$ROOT_DIR/scripts" --include="*.py" 2>/dev/null || true); do
        basename_file=$(basename "$file")
        if [[ -n "$canonical" && "$basename_file" == "$canonical" ]]; then
            continue
        fi
        echo -e "  ${RED}ERROR${NC}: $file defines '$pattern'"
        if [[ -n "$canonical" ]]; then
            echo -e "         Fix: import it from $canonical"
        else
            echo -e "         Fix: use extract_warning_messages from engine_command_matrix_observability.py"
        fi
        ((ERRORS++))
    done
done

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
