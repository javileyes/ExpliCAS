//! Power products and sum-to-product quotient rules.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_rewrite::smart_mul;
use cas_math::trig_power_identity_support::{
    coeff_is_three, extract_sin2_cos2_product, extract_trig_pow4, extract_trig_pow6,
};
use cas_math::trig_sum_product_support::{
    args_match_as_multiset, extract_trig_two_term_diff, extract_trig_two_term_sum,
    normalize_for_even_fn, simplify_numeric_div,
};
use std::cmp::Ordering;

// =============================================================================
// HIDDEN CUBIC TRIG IDENTITY
// sin^6(x) + cos^6(x) + 3*sin^2(x)*cos^2(x) = (sin^2(x) + cos^2(x))^3
// =============================================================================

define_rule!(
    TrigHiddenCubicIdentityRule,
    "Hidden Cubic Trig Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Only match Add nodes (sums)
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None;
        }

        // Flatten the sum to get all terms
        let terms = crate::nary::add_leaves(ctx, expr);

        // We need exactly 3 terms for the pattern
        if terms.len() != 3 {
            return None;
        }

        // Try to find: sin^6(arg), cos^6(arg), coeff*sin^2(arg)*cos^2(arg)
        let mut sin6_arg: Option<ExprId> = None;
        let mut cos6_arg: Option<ExprId> = None;
        let mut sin2cos2_info: Option<(ExprId, ExprId)> = None; // (coeff, arg)
        let mut sin6_idx: Option<usize> = None;
        let mut cos6_idx: Option<usize> = None;
        let mut sin2cos2_idx: Option<usize> = None;

        for (i, &term) in terms.iter().enumerate() {
            // Try to match sin^6 or cos^6
            if let Some((arg, name)) = extract_trig_pow6(ctx, term) {
                match name {
                    "sin" if sin6_arg.is_none() => {
                        sin6_arg = Some(arg);
                        sin6_idx = Some(i);
                    }
                    "cos" if cos6_arg.is_none() => {
                        cos6_arg = Some(arg);
                        cos6_idx = Some(i);
                    }
                    _ => {} // Already matched or duplicate
                }
            }
        }

        // Find the sin^2*cos^2 term (the remaining one)
        for (i, &term) in terms.iter().enumerate() {
            if Some(i) == sin6_idx || Some(i) == cos6_idx {
                continue;
            }

            if let Some((coeff, arg)) = extract_sin2_cos2_product(ctx, term) {
                sin2cos2_info = Some((coeff, arg));
                sin2cos2_idx = Some(i);
                break;
            }
        }

        // Verify we found all three pieces
        let sin6_a = sin6_arg?;
        let cos6_a = cos6_arg?;
        let (coeff, sin2cos2_a) = sin2cos2_info?;

        // Ensure we used all three terms (no extras)
        if sin6_idx.is_none() || cos6_idx.is_none() || sin2cos2_idx.is_none() {
            return None;
        }

        // Verify all arguments are the same
        if crate::ordering::compare_expr(ctx, sin6_a, cos6_a) != Ordering::Equal {
            return None;
        }
        if crate::ordering::compare_expr(ctx, sin6_a, sin2cos2_a) != Ordering::Equal {
            return None;
        }

        // Verify coefficient is 3
        if !coeff_is_three(ctx, coeff) {
            return None;
        }

        // All conditions met! Rewrite to (sin^2(arg) + cos^2(arg))^3
        let arg = sin6_a;
        let two = ctx.num(2);
        let three = ctx.num(3);

        let sin_arg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
        let cos_arg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
        let sin2 = ctx.add(Expr::Pow(sin_arg, two));
        let two_again = ctx.num(2);
        let cos2 = ctx.add(Expr::Pow(cos_arg, two_again));
        let sum = ctx.add(Expr::Add(sin2, cos2));
        let result = ctx.add(Expr::Pow(sum, three));

        Some(
            Rewrite::new(result).desc("sin⁶(x) + cos⁶(x) + 3sin²(x)cos²(x) = (sin²(x) + cos²(x))³"),
        )
    }
);

// =============================================================================
// QUARTIC TRIG IDENTITY
// sin^4(x) + cos^4(x) = 1 − 2·sin²(x)·cos²(x)
//
// Derivation: (sin²+cos²)² = sin⁴ + 2sin²cos² + cos⁴ = 1
// Therefore:  sin⁴ + cos⁴ = 1 − 2sin²cos²
// =============================================================================

define_rule!(
    SinCosQuarticSumRule,
    "Quartic Pythagorean Identity",
    None,
    crate::phase::PhaseMask::CORE,
    |ctx, expr| {
        // Only match Add nodes (sums)
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None;
        }

        // Flatten the sum to get all terms
        let terms = crate::nary::add_leaves(ctx, expr);

        if terms.len() < 2 {
            return None;
        }

        // Find sin^4(arg) and cos^4(arg) with matching arguments
        let mut sin4_arg: Option<ExprId> = None;
        let mut cos4_arg: Option<ExprId> = None;
        let mut sin4_idx: Option<usize> = None;
        let mut cos4_idx: Option<usize> = None;

        for (i, &term) in terms.iter().enumerate() {
            if let Some((arg, name)) = extract_trig_pow4(ctx, term) {
                match name {
                    "sin" if sin4_arg.is_none() => {
                        sin4_arg = Some(arg);
                        sin4_idx = Some(i);
                    }
                    "cos" if cos4_arg.is_none() => {
                        cos4_arg = Some(arg);
                        cos4_idx = Some(i);
                    }
                    _ => {}
                }
            }
        }

        let sin4_a = sin4_arg?;
        let cos4_a = cos4_arg?;
        let si = sin4_idx?;
        let ci = cos4_idx?;

        // Verify arguments match
        if crate::ordering::compare_expr(ctx, sin4_a, cos4_a) != Ordering::Equal {
            return None;
        }

        // Build replacement: 1 − 2·sin²(a)·cos²(a)
        let arg = sin4_a;
        let one = ctx.num(1);
        let two = ctx.num(2);
        let two_exp = ctx.num(2);
        let two_exp2 = ctx.num(2);

        let sin_a = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
        let cos_a = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
        let sin2 = ctx.add(Expr::Pow(sin_a, two_exp));
        let cos2 = ctx.add(Expr::Pow(cos_a, two_exp2));
        let product = ctx.add(Expr::Mul(sin2, cos2));
        let two_product = ctx.add(Expr::Mul(two, product));
        let replacement = ctx.add(Expr::Sub(one, two_product));

        // If there are other terms beyond sin⁴ and cos⁴, preserve them
        let remaining: Vec<ExprId> = terms
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != si && *i != ci)
            .map(|(_, &t)| t)
            .collect();

        let result = if remaining.is_empty() {
            replacement
        } else {
            let mut acc = replacement;
            for &t in &remaining {
                acc = ctx.add(Expr::Add(acc, t));
            }
            acc
        };

        Some(Rewrite::new(result).desc("sin⁴(x) + cos⁴(x) = 1 − 2·sin²(x)·cos²(x)"))
    }
);

// =============================================================================
// SUM-TO-PRODUCT QUOTIENT RULE
// (sin(A)+sin(B))/(cos(A)+cos(B)) → sin((A+B)/2)/cos((A+B)/2)
// =============================================================================

/// Build half_diff = (A-B)/2 preserving the order of A and B.
/// Use this for sin((A-B)/2) where the sign matters.
/// Pre-simplifies the difference to produce cleaner output.
pub fn build_half_diff(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    let diff = ctx.add(Expr::Sub(a, b));
    // Pre-simplify the difference (e.g., 5x - 3x → 2x)
    let diff_simplified = crate::collect::collect(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    // Try to simplify the division (e.g., 2x/2 → x)
    simplify_numeric_div(ctx, result)
}

/// Build canonical half_diff = (A-B)/2 with consistent ordering.
/// Since cos((A-B)/2) == cos((B-A)/2), we use canonical order to ensure
/// numerator and denominator produce identical expressions for cancellation.
/// Use this for cos((A-B)/2) where the sign doesn't matter.
fn build_canonical_half_diff(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // Use canonical order: if A > B, swap to (B-A)/2
    // This ensures consistent expression for cos(half_diff) in num and den
    let (first, second) = if compare_expr(ctx, a, b) == Ordering::Greater {
        (b, a)
    } else {
        (a, b)
    };

    let diff = ctx.add(Expr::Sub(first, second));
    // Pre-simplify the difference (e.g., x - 3x → -2x)
    let diff_simplified = crate::collect::collect(ctx, diff);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(diff_simplified, two));
    // Try to simplify the division (e.g., -2x/2 → -x)
    simplify_numeric_div(ctx, result)
}

/// Build avg = (A+B)/2, pre-simplifying sum for cleaner output
/// This eliminates the need for a separate "Combine Like Terms" step
pub fn build_avg(ctx: &mut cas_ast::Context, a: ExprId, b: ExprId) -> ExprId {
    let sum = ctx.add(Expr::Add(a, b));
    // Pre-simplify the sum (e.g., x + 3x → 4x)
    let sum_simplified = crate::collect::collect(ctx, sum);
    let two = ctx.num(2);
    let result = ctx.add(Expr::Div(sum_simplified, two));
    // Try to simplify the division (e.g., 4x/2 → 2x)
    simplify_numeric_div(ctx, result)
}

// SinCosSumQuotientRule: Handles two patterns:
// 1. (sin(A)+sin(B))/(cos(A)+cos(B)) → tan((A+B)/2)  [uses sin sum identity]
// 2. (sin(A)-sin(B))/(cos(A)+cos(B)) → tan((A-B)/2)  [uses sin diff identity]
//
// Sum-to-product identities:
//   sin(A) + sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)
//   sin(A) - sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)
//   cos(A) + cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)
//
// For sum case: common factor is 2·cos((A-B)/2) → result is tan((A+B)/2)
// For diff case: common factor is 2·cos((A+B)/2) → result is tan((A-B)/2)
//
// This rule runs BEFORE TripleAngleRule to avoid polynomial explosion.
define_rule!(
    SinCosSumQuotientRule,
    "Sum-to-Product Quotient",
    |ctx, expr| {
        // Only match Div nodes
        let (num_id, den_id) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // Extract cos(C) + cos(D) from denominator (required for both cases)
        let (cos_c, cos_d) = extract_trig_two_term_sum(ctx, den_id, "cos")?;

        // Try both patterns for numerator
        enum NumeratorPattern {
            Sum { sin_a: ExprId, sin_b: ExprId },
            Diff { sin_a: ExprId, sin_b: ExprId },
        }

        let pattern = if let Some((sin_a, sin_b)) = extract_trig_two_term_sum(ctx, num_id, "sin") {
            // Pattern 1: sin(A) + sin(B)
            NumeratorPattern::Sum { sin_a, sin_b }
        } else if let Some((sin_a, sin_b)) = extract_trig_two_term_diff(ctx, num_id, "sin") {
            // Pattern 2: sin(A) - sin(B)
            NumeratorPattern::Diff { sin_a, sin_b }
        } else {
            return None;
        };

        // Extract the sin arguments
        let (sin_a, sin_b, is_diff) = match pattern {
            NumeratorPattern::Sum { sin_a, sin_b } => (sin_a, sin_b, false),
            NumeratorPattern::Diff { sin_a, sin_b } => (sin_a, sin_b, true),
        };

        // Verify {A,B} == {C,D} as multisets
        if !args_match_as_multiset(ctx, sin_a, sin_b, cos_c, cos_d) {
            return None;
        }

        // Build avg = (A+B)/2 (commutative, order doesn't matter)
        let avg = build_avg(ctx, sin_a, sin_b);

        // Normalize avg for even functions (cos)
        let avg_normalized = normalize_for_even_fn(ctx, avg);

        use crate::rule::ChainedRewrite;

        if is_diff {
            // DIFFERENCE CASE: sin(A) - sin(B) = 2·cos(avg)·sin(half_diff)
            // half_diff = (A-B)/2 - ORDER MATTERS for sin! Use build_half_diff.
            let half_diff = build_half_diff(ctx, sin_a, sin_b);
            // For cos(half_diff), we can normalize since cos is even
            let half_diff_for_cos = normalize_for_even_fn(ctx, half_diff);
            // DIFFERENCE CASE: sin(A) - sin(B) = 2·cos(avg)·sin(half_diff)
            // cos(A) + cos(B) = 2·cos(avg)·cos(half_diff)
            // Cancel 2·cos(avg) → sin(half_diff)/cos(half_diff) = tan(half_diff)
            // Build sin/cos quotient form for intermediate display

            let sin_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);
            let cos_half_diff = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_for_cos]);
            let quotient_result = ctx.add(Expr::Div(sin_half_diff, cos_half_diff));

            // Intermediate states
            let two = ctx.num(2);
            let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg_normalized]);
            let sin_half = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![half_diff]);

            // Intermediate numerator: 2·cos(avg)·sin(half_diff)
            let num_product = smart_mul(ctx, cos_avg, sin_half);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_2 = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg_normalized]);
            let cos_half = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_for_cos]);
            let den_product = smart_mul(ctx, cos_avg_2, cos_half);
            let intermediate_den = smart_mul(ctx, two_2, den_product);

            let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
            let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

            let rewrite = Rewrite::new(state_after_step1)
                .desc("sin(A)−sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)")
                .local(num_id, intermediate_num)
                .chain(
                    ChainedRewrite::new(state_after_step2)
                        .desc("cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
                        .local(den_id, intermediate_den),
                )
                .chain(
                    ChainedRewrite::new(quotient_result)
                        .desc("Cancel common factors 2 and cos(avg)")
                        .local(state_after_step2, quotient_result),
                );

            Some(rewrite)
        } else {
            // SUM CASE: sin(A) + sin(B) = 2·sin(avg)·cos(half_diff)
            // cos(A) + cos(B) = 2·cos(avg)·cos(half_diff)
            // Cancel 2·cos(half_diff) → sin(avg)/cos(avg) = tan(avg)
            // For sum case, we use canonical half_diff since only cos uses it (even function)
            let half_diff = build_canonical_half_diff(ctx, sin_a, sin_b);
            let half_diff_normalized = normalize_for_even_fn(ctx, half_diff);

            // Final result: sin(avg)/cos(avg)
            let sin_avg = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
            let cos_avg = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
            let final_result = ctx.add(Expr::Div(sin_avg, cos_avg));

            // Intermediate states
            let two = ctx.num(2);
            let cos_half_diff =
                ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);

            // Intermediate numerator: 2·sin(avg)·cos(half_diff)
            let sin_avg_for_num = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![avg]);
            let num_product = smart_mul(ctx, sin_avg_for_num, cos_half_diff);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_for_den = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![avg]);
            let cos_half_diff_2 =
                ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![half_diff_normalized]);
            let den_product = smart_mul(ctx, cos_avg_for_den, cos_half_diff_2);
            let intermediate_den = smart_mul(ctx, two_2, den_product);

            let state_after_step1 = ctx.add(Expr::Div(intermediate_num, den_id));
            let state_after_step2 = ctx.add(Expr::Div(intermediate_num, intermediate_den));

            let rewrite = Rewrite::new(state_after_step1)
                .desc("sin(A)+sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)")
                .local(num_id, intermediate_num)
                .chain(
                    ChainedRewrite::new(state_after_step2)
                        .desc("cos(A)+cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)")
                        .local(den_id, intermediate_den),
                )
                .chain(
                    ChainedRewrite::new(final_result)
                        .desc("Cancel common factors 2 and cos(half_diff)")
                        .local(state_after_step2, final_result),
                );

            Some(rewrite)
        }
    }
);
