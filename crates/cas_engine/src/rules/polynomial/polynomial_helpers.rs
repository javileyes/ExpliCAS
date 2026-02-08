//! Helper functions for polynomial rules.
//!
//! Contains structural comparison helpers (`is_conjugate`, `is_negation`, `poly_equal`),
//! additive-term flattening, and didactic focus selection utilities.

use crate::ordering::compare_expr;
use crate::polynomial::Polynomial;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

// =============================================================================
// Conjugate / negation detection
// =============================================================================

pub(crate) fn is_conjugate(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Check for (A+B) and (A-B) or (A-B) and (A+B)
    let a_expr = ctx.get(a);
    let b_expr = ctx.get(b);

    match (a_expr, b_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // (A+B) vs (A-B)
            // Check if A=A and B=B
            // Or A=B and B=A (commutative)
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            // Direct match: A+B vs A-B
            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return true;
            }
            // Commutative A: B+A vs A-B (A matches A, B matches B)
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return true;
            }

            // What about -B+A? Canonicalization usually handles this to Sub(A,B) or Add(A, Neg(B)).
            // If we have Add(A, Neg(B)), it's not Sub.
            false
        }
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            // (A+B) vs (A+(-B)) or ((-B)+A)
            // Check if one term is negation of another
            let a1 = *a1;
            let a2 = *a2;
            let b1 = *b1;
            let b2 = *b2;

            // Case 1: b2 is neg(a2) -> (A+B)(A-B)
            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return true;
            }
            // Case 2: b1 is neg(a2) -> (A+B)(-B+A)
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return true;
            }
            // Case 3: b2 is neg(a1) -> (A+B)(B-A) -> No, that's -(A-B)(A+B)? No.
            // (A+B)(B-A) = B^2 - A^2. This IS a conjugate pair.
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return true;
            }
            // Case 4: b1 is neg(a1) -> (A+B)(-A+B)
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return true;
            }
            false
        }
        _ => false,
    }
}

pub(super) fn is_negation(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Check if b is Neg(a) or Mul(-1, a)
    if check_negation_structure(ctx, b, a) {
        return true;
    }
    // Check if a is Neg(b) or Mul(-1, b)
    if check_negation_structure(ctx, a, b) {
        return true;
    }
    false
}

fn check_negation_structure(ctx: &Context, potential_neg: ExprId, original: ExprId) -> bool {
    match ctx.get(potential_neg) {
        Expr::Neg(n) => compare_expr(ctx, original, *n) == Ordering::Equal,
        Expr::Mul(l, r) => {
            // Check for -1 * original
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == -BigRational::one() && compare_expr(ctx, *r, original) == Ordering::Equal {
                    return true;
                }
            }
            // Check for original * -1
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == -BigRational::one() && compare_expr(ctx, *l, original) == Ordering::Equal {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}

/// Unwrap __hold(X) to X, otherwise return the expression unchanged
/// Delegates to canonical implementation in cas_ast::hold
pub(super) fn unwrap_hold(ctx: &Context, expr: ExprId) -> ExprId {
    cas_ast::hold::unwrap_hold(ctx, expr)
}

// =============================================================================
// Term normalization
// =============================================================================

/// Normalize a term by extracting negation from leading coefficient
/// For example: (-15)*z with flag false → 15*z with flag true
/// Returns (normalized_expr, effective_negation_flag)
pub(super) fn normalize_term_sign(ctx: &Context, term: ExprId, neg: bool) -> (ExprId, bool) {
    // Check if it's a Mul with a negative number as first or second operand
    if let Expr::Mul(l, r) = ctx.get(term) {
        // Check left operand for negative number
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_negative() {
                // Flip the sign and negate the coefficient
                // We can't easily create a new expression here, so we'll compare differently
                return (term, !neg);
            }
        }
        // Check right operand for negative number
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_negative() {
                return (term, !neg);
            }
        }
    }

    // Check if it's a negative number itself
    if let Expr::Number(n) = ctx.get(term) {
        if n.is_negative() {
            return (term, !neg);
        }
    }

    (term, neg)
}

// =============================================================================
// Polynomial equality
// =============================================================================

/// Check if two expressions are polynomially equal (same after expansion)
pub(crate) fn poly_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Identical IDs
    if a == b {
        return true;
    }

    // First try structural comparison
    if compare_expr(ctx, a, b) == Ordering::Equal {
        return true;
    }

    let expr_a = ctx.get(a);
    let expr_b = ctx.get(b);

    // Try deep comparison for Pow expressions
    if let (Expr::Pow(base_a, exp_a), Expr::Pow(base_b, exp_b)) = (expr_a, expr_b) {
        if poly_equal(ctx, *exp_a, *exp_b) {
            return poly_equal(ctx, *base_a, *base_b);
        }
    }

    // Try deep comparison for Mul expressions (commutative)
    if let (Expr::Mul(l_a, r_a), Expr::Mul(l_b, r_b)) = (expr_a, expr_b) {
        // Try both orderings
        if (poly_equal(ctx, *l_a, *l_b) && poly_equal(ctx, *r_a, *r_b))
            || (poly_equal(ctx, *l_a, *r_b) && poly_equal(ctx, *r_a, *l_b))
        {
            return true;
        }

        // Also check for opposite coefficients: 15*z vs -15*z
        // Check if one left operand is the negation of the other
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*l_a), ctx.get(*l_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *r_a, *r_b) {
                return true; // Same up to sign
            }
        }
        if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(*r_a), ctx.get(*r_b)) {
            if n_a == &-n_b.clone() && poly_equal(ctx, *l_a, *l_b) {
                return true; // Same up to sign
            }
        }
    }

    // Try deep comparison for Neg expressions
    if let (Expr::Neg(inner_a), Expr::Neg(inner_b)) = (expr_a, expr_b) {
        return poly_equal(ctx, *inner_a, *inner_b);
    }

    // For any additive expressions, flatten and compare term sets
    // This handles Add, Sub, and mixed cases
    let is_additive_a = matches!(expr_a, Expr::Add(_, _) | Expr::Sub(_, _));
    let is_additive_b = matches!(expr_b, Expr::Add(_, _) | Expr::Sub(_, _));

    if is_additive_a && is_additive_b {
        let mut terms_a: Vec<(ExprId, bool)> = Vec::new();
        let mut terms_b: Vec<(ExprId, bool)> = Vec::new();
        flatten_additive_terms(ctx, a, false, &mut terms_a);
        flatten_additive_terms(ctx, b, false, &mut terms_b);

        if terms_a.len() == terms_b.len() {
            let mut matched = vec![false; terms_b.len()];
            for (term_a, neg_a) in &terms_a {
                let mut found = false;

                // Normalize term_a: extract negation from leading coefficient if any
                let (norm_a, eff_neg_a) = normalize_term_sign(ctx, *term_a, *neg_a);

                for (j, (term_b, neg_b)) in terms_b.iter().enumerate() {
                    if matched[j] {
                        continue;
                    }

                    // Normalize term_b: extract negation from leading coefficient if any
                    let (norm_b, eff_neg_b) = normalize_term_sign(ctx, *term_b, *neg_b);

                    // Now compare with effective negation
                    if eff_neg_a != eff_neg_b {
                        continue;
                    }

                    // Use poly_equal recursively for term comparison
                    if poly_equal(ctx, norm_a, norm_b) {
                        matched[j] = true;
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            if matched.iter().all(|&m| m) {
                return true;
            }
        }
    }

    // Fallback: try polynomial comparison for univariate case
    let vars_a: Vec<_> = cas_ast::collect_variables(ctx, a).into_iter().collect();
    let vars_b: Vec<_> = cas_ast::collect_variables(ctx, b).into_iter().collect();

    // Only compare if same single variable
    if vars_a.len() == 1 && vars_b.len() == 1 && vars_a[0] == vars_b[0] {
        let var = &vars_a[0];
        if let (Ok(poly_a), Ok(poly_b)) = (
            Polynomial::from_expr(ctx, a, var),
            Polynomial::from_expr(ctx, b, var),
        ) {
            return poly_a == poly_b;
        }
    }

    false
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

/// Build an additive expression from a list of terms (for focus display)
pub(super) fn build_additive_expr(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in &terms[1..] {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

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
    let focus_before = build_additive_expr(ctx, &all_before_terms);

    // Build the after expression
    let focus_after = if all_after_terms.is_empty() {
        // Only cancellations, result is 0
        ctx.num(0)
    } else {
        build_additive_expr(ctx, &all_after_terms)
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
    match ctx.get(expr) {
        Expr::Add(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Sub(l, r) => count_additive_terms(ctx, *l) + count_additive_terms(ctx, *r),
        Expr::Neg(inner) => count_additive_terms(ctx, *inner),
        _ => 1, // A single term (Variable, Number, Mul, Pow, etc.)
    }
}
