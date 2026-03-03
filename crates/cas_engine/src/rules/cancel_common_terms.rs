//! Equation-level additive term cancellation utilities.
//!
//! Provides `cancel_common_additive_terms()`, which compares terms from
//! two expression trees (LHS and RHS of an equation) and cancels matching
//! pairs using structural comparison.
//!
//! # Why this is NOT a simplifier rule
//!
//! The simplifier's `CanonicalizeNegationRule` converts `Sub(a, b)` to
//! `Add(a, Neg(b))` very early in the pipeline. A rule targeting `Sub`
//! nodes would never fire because the Sub is already gone.
//!
//! More fundamentally, equation-level cancellation is a **relational
//! operation** between two expression trees (LHS vs RHS of `lhs = rhs`).
//! A simplifier rule only sees a single expression node. These are
//! different abstractions and belong in different layers.
//!
//! # Preconditions
//!
//! For maximum effectiveness, both `lhs` and `rhs` should be in
//! canonical form before calling:
//! - Rational exponents: `Div(Number(p), Number(q))` → `Number(p/q)`
//!   (via CanonicalizeRationalDivRule)
//! - Flattened Add/Mul chains with stable ordering
//!   (via standard canonicalization rules)
//!
//! # Guarantees
//!
//! - **Strictly reductive**: total term count always decreases
//! - **Sound**: uses `compare_expr` for structural equality (no numeric
//!   approximation)
//! - **Deterministic**: order of cancellation is fixed (rhs terms scanned
//!   in order, matched against lhs terms in order)

use cas_ast::{Context, Expr, ExprId};
use cas_math::cancel_semantic_support::{
    try_cancel_additive_terms_semantic_with_state, SemanticCancelConfig,
};
use cas_math::cancel_support::{try_cancel_common_additive_terms_expr, CancelCommonAdditivePlan};

/// Result of equation-level additive term cancellation.
pub type CancelResult = CancelCommonAdditivePlan;

/// Cancel common additive terms between two expression trees.
///
/// Flattens both sides into multisets of `(term, is_positive)`, matches
/// pairs by structural equality (same term, same sign polarity), and
/// rebuilds both sides without the matched pairs.
///
/// Returns `None` if no terms could be cancelled.
///
/// # Example
/// ```text
/// lhs = Add(x², Add(x^(5/6), 1))    rhs = x^(5/6)
/// → lhs_terms = [(x², +), (x^(5/6), +), (1, +)]
/// → rhs_terms = [(x^(5/6), +)]
/// → cancel x^(5/6) pair
/// → new_lhs = Add(x², 1), new_rhs = 0, cancelled = 1
/// ```
pub fn cancel_common_additive_terms(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    try_cancel_common_additive_terms_expr(ctx, lhs, rhs)
}

/// Semantic fallback for equation-level term cancellation.
///
/// **2-phase candidate+proof pattern for soundness:**
///
/// 1. *Candidate generation* (Generic domain): normalize both sides
///    (split fractions, distribute scalar multiplication, split log
///    products/sqrt), then simplify each term with Generic domain to
///    expose structure (allows `x/x → 1`, etc.).
///
/// 2. *Pair proof* (Strict domain): for each candidate pair, verify
///    `simplify_strict(l - r) == 0` before cancelling. This ensures
///    the cancellation doesn't depend on unproven assumptions (e.g.,
///    `x ≠ 0` for `x/x → 1`). The Generic simplification only exposes
///    candidates; Strict validates them.
///
/// # Guards
/// - ≤ `MAX_TERMS` terms per side (avoids O(n²) explosion)
/// - ≤ `MAX_NODES` nodes per term (avoids expensive simplification)
pub fn cancel_additive_terms_semantic(
    simplifier: &mut crate::Simplifier,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<CancelResult> {
    use num_traits::Zero;

    const MAX_TERMS: usize = 12; // raised from 8: normalization may expand terms
    const MAX_NODES: usize = 200;

    // Simplify each term individually for canonical comparison.
    // Uses default domain (Generic) to allow x/x → 1 (essential for
    // Div(Mul(x, ln(..)), x) → ln(..) after fraction splitting).
    // Strict would block it — it requires proof of x ≠ 0.
    // Safe: this is purely for normalization, doesn't modify the solution set.
    // Phase 1: candidate generation — Generic allows x/x → 1, etc.
    let candidate_opts = crate::SimplifyOptions {
        collect_steps: false,
        ..Default::default()
    };
    // Phase 2: pair proof — Strict prevents domain expansion
    let strict_proof_opts = crate::SimplifyOptions {
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig::strict(),
            ..Default::default()
        },
        collect_steps: false,
        ..Default::default()
    };
    let result = try_cancel_additive_terms_semantic_with_state(
        simplifier,
        lhs,
        rhs,
        SemanticCancelConfig {
            max_terms: MAX_TERMS,
            max_nodes: MAX_NODES,
        },
        |state| &state.context,
        |state| &mut state.context,
        |state, term| state.simplify_with_stats(term, candidate_opts.clone()).0,
        crate::expand::expand,
        |state, lt, rt| {
            let diff = state.context.add(Expr::Sub(lt, rt));
            let (simplified_diff, _, _) =
                state.simplify_with_stats(diff, strict_proof_opts.clone());
            matches!(state.context.get(simplified_diff), Expr::Number(n) if n.is_zero())
        },
    )?;

    Some(CancelResult {
        new_lhs: result.new_lhs,
        new_rhs: result.new_rhs,
        cancelled_count: result.cancelled_count,
    })
}
