//! Helper functions for polynomial rules.
//!
//! Contains structural comparison helpers (`is_conjugate`, `poly_equal`),
//! additive-term flattening, and didactic focus selection utilities.

use cas_ast::{Context, ExprId};

// =============================================================================
// Conjugate / negation detection
// =============================================================================

pub(crate) fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    cas_math::expr_relations::is_conjugate_add_sub(ctx, a, b)
}

/// Unwrap __hold(X) to X, otherwise return the expression unchanged
/// Delegates to canonical implementation in cas_ast::hold
pub(super) fn unwrap_hold(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_hold(ctx, expr)
}

// =============================================================================
// Polynomial equality
// =============================================================================

/// Check if two expressions are polynomially equal (same after expansion)
pub(crate) fn poly_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    cas_math::expr_relations::poly_equal(ctx, a, b)
}

// =============================================================================
// Additive term flattening
// =============================================================================

/// Flatten an additive expression into a list of (term, is_negated) pairs
///
/// Uses canonical AddView from nary.rs for shape-independence and __hold transparency.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
pub(super) fn flatten_additive_terms(
    ctx: &Context,
    expr: ExprId,
    negated: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    use crate::nary::{add_terms_signed, Sign};

    // Use canonical AddView
    let signed_terms = add_terms_signed(ctx, expr);

    for (term, sign) in signed_terms {
        // XOR the incoming negation with the sign from AddView
        let is_negated = match sign {
            Sign::Pos => negated,
            Sign::Neg => !negated,
        };
        terms.push((term, is_negated));
    }
}

// =============================================================================
// Didactic focus helpers
// =============================================================================

/// Select focus for didactic display from a CollectResult
/// Shows ALL combined and cancelled groups together for complete picture
pub(super) fn select_best_focus(
    ctx: &mut Context,
    result: &crate::collect::CollectResult,
) -> (Option<ExprId>, Option<ExprId>, String) {
    // Collect all original terms and all result terms from all groups
    let mut all_before_terms: Vec<ExprId> = Vec::new();
    let mut all_after_terms: Vec<ExprId> = Vec::new();
    let mut has_cancellation = false;
    let mut has_combination = false;

    // Add cancelled groups (result is 0, but we skip adding 0 since it doesn't change sum)
    for cancelled in &result.cancelled {
        all_before_terms.extend(&cancelled.original_terms);
        has_cancellation = true;
        // Don't add 0 to after terms - it's implicit
    }

    // Add combined groups
    for combined in &result.combined {
        all_before_terms.extend(&combined.original_terms);
        all_after_terms.push(combined.combined_term);
        has_combination = true;
    }

    // If we have no groups, fallback
    if all_before_terms.is_empty() {
        return (None, None, "Combine like terms".to_string());
    }

    // Build the before expression from all original terms
    let focus_before = cas_math::expr_terms::build_sum(ctx, &all_before_terms);

    // Build the after expression
    let focus_after = if all_after_terms.is_empty() {
        // Only cancellations, result is 0
        ctx.num(0)
    } else {
        cas_math::expr_terms::build_sum(ctx, &all_after_terms)
    };

    // Choose appropriate description
    let description = if has_cancellation && has_combination {
        "Cancel and combine like terms".to_string()
    } else if has_cancellation {
        "Cancel opposite terms".to_string()
    } else {
        "Combine like terms".to_string()
    };

    (Some(focus_before), Some(focus_after), description)
}

/// Count the number of terms in a sum/difference expression
/// Returns the count of additive terms (flattening nested Add/Sub)
pub(super) fn count_additive_terms(ctx: &Context, expr: ExprId) -> usize {
    cas_math::expr_relations::count_additive_terms(ctx, expr)
}
