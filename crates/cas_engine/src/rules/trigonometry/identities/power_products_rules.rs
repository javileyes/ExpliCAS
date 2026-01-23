//! Power products and sum-to-product quotient rules.

use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use std::cmp::Ordering;

// =============================================================================
// HIDDEN CUBIC TRIG IDENTITY
// sin^6(x) + cos^6(x) + 3*sin^2(x)*cos^2(x) = (sin^2(x) + cos^2(x))^3
// =============================================================================

/// Extract sin(arg)^6 or cos(arg)^6 from a term.
/// Returns Some((arg, "sin"|"cos")) if matched.
fn extract_trig_pow6(ctx: &cas_ast::Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        // Check exponent is 6
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n.numer() == 6.into() {
                // Check base is sin(arg) or cos(arg)
                if let Expr::Function(name, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        match name.as_str() {
                            "sin" => return Some((args[0], "sin")),
                            "cos" => return Some((args[0], "cos")),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract sin(arg)^2 or cos(arg)^2 from terms.
/// Returns Some((arg, "sin"|"cos")) if matched.
fn extract_trig_pow2(ctx: &cas_ast::Context, term: ExprId) -> Option<(ExprId, &'static str)> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        // Check exponent is 2
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n.numer() == 2.into() {
                // Check base is sin(arg) or cos(arg)
                if let Expr::Function(name, args) = ctx.get(*base) {
                    if args.len() == 1 {
                        match name.as_str() {
                            "sin" => return Some((args[0], "sin")),
                            "cos" => return Some((args[0], "cos")),
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract coeff * sin(arg)^2 * cos(arg)^2 from a term.
/// Returns Some((coeff_expr, arg)) where coeff_expr is the coefficient expression.
/// The caller should verify coeff_expr simplifies to 3.
fn extract_sin2_cos2_product(ctx: &mut cas_ast::Context, term: ExprId) -> Option<(ExprId, ExprId)> {
    // Flatten the multiplication
    let factors = crate::helpers::flatten_mul_chain(ctx, term);

    if factors.len() < 2 {
        return None;
    }

    let mut sin2_arg: Option<ExprId> = None;
    let mut cos2_arg: Option<ExprId> = None;
    let mut other_factors: Vec<ExprId> = Vec::new();

    for factor in &factors {
        if let Some((arg, name)) = extract_trig_pow2(ctx, *factor) {
            match name {
                "sin" if sin2_arg.is_none() => sin2_arg = Some(arg),
                "cos" if cos2_arg.is_none() => cos2_arg = Some(arg),
                _ => other_factors.push(*factor), // Duplicate or already matched
            }
        } else {
            other_factors.push(*factor);
        }
    }

    // Must have exactly one sin^2 and one cos^2 with same argument
    let sin_arg = sin2_arg?;
    let cos_arg = cos2_arg?;

    // Verify same argument
    if crate::ordering::compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    // Build the coefficient expression from remaining factors
    let coeff = if other_factors.is_empty() {
        ctx.num(1)
    } else if other_factors.len() == 1 {
        other_factors[0]
    } else {
        // Build product of remaining factors
        let mut result = other_factors[0];
        for &f in &other_factors[1..] {
            result = ctx.add(Expr::Mul(result, f));
        }
        result
    };

    Some((coeff, sin_arg))
}

/// Check if a coefficient expression equals 3.
/// Uses simplification: coeff - 3 == 0
fn coeff_is_three(ctx: &mut cas_ast::Context, coeff: ExprId) -> bool {
    // Fast path: direct number check
    if let Expr::Number(n) = ctx.get(coeff) {
        return n.is_integer() && *n.numer() == 3.into();
    }

    // Use as_rational_const for expressions like 6/2
    if let Some(val) = crate::helpers::as_rational_const(ctx, coeff) {
        return val == num_rational::BigRational::from_integer(3.into());
    }

    false
}

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
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

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

        let sin_arg = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
        let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
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
// SUM-TO-PRODUCT QUOTIENT RULE
// (sin(A)+sin(B))/(cos(A)+cos(B)) → sin((A+B)/2)/cos((A+B)/2)
// =============================================================================

/// Extract the argument from a trig function: sin(arg) → Some(arg), else None
pub fn extract_trig_arg(ctx: &cas_ast::Context, id: ExprId, fn_name: &str) -> Option<ExprId> {
    if let Expr::Function(name, args) = ctx.get(id) {
        if name == fn_name && args.len() == 1 {
            return Some(args[0]);
        }
    }
    None
}

/// Extract two trig function args from a 2-term sum: sin(A) + sin(B) → Some((A, B))
/// Uses flatten_add for robustness against nested Add structures
fn extract_trig_two_term_sum(
    ctx: &cas_ast::Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    use crate::helpers::flatten_add;

    let mut terms = Vec::new();
    flatten_add(ctx, expr, &mut terms);

    // Must have exactly 2 terms (both same trig function)
    if terms.len() != 2 {
        return None;
    }

    // Check both are the target function (sin or cos)
    let arg1 = extract_trig_arg(ctx, terms[0], fn_name)?;
    let arg2 = extract_trig_arg(ctx, terms[1], fn_name)?;

    Some((arg1, arg2))
}

/// Extract two trig function args from a 2-term difference: sin(A) - sin(B) → Some((A, B))
/// Handles both Sub(sin(A), sin(B)) and Add(sin(A), Neg(sin(B)))
fn extract_trig_two_term_diff(
    ctx: &cas_ast::Context,
    expr: ExprId,
    fn_name: &str,
) -> Option<(ExprId, ExprId)> {
    // Pattern 1: Sub(sin(A), sin(B))
    if let Expr::Sub(l, r) = ctx.get(expr) {
        let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
        let arg2 = extract_trig_arg(ctx, *r, fn_name)?;
        return Some((arg1, arg2));
    }

    // Pattern 2: Add(sin(A), Neg(sin(B))) - normalized form
    if let Expr::Add(l, r) = ctx.get(expr) {
        // Check if one is Neg(sin(B))
        if let Expr::Neg(inner) = ctx.get(*r) {
            let arg1 = extract_trig_arg(ctx, *l, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            return Some((arg1, arg2));
        }
        if let Expr::Neg(inner) = ctx.get(*l) {
            let arg1 = extract_trig_arg(ctx, *r, fn_name)?;
            let arg2 = extract_trig_arg(ctx, *inner, fn_name)?;
            // Note: sin(A) - sin(B) vs -sin(B) + sin(A) = sin(A) - sin(B)
            return Some((arg1, arg2));
        }
    }

    None
}

/// Check if two pairs of args match as multisets: {A,B} == {C,D}
fn args_match_as_multiset(
    ctx: &cas_ast::Context,
    a1: ExprId,
    a2: ExprId,
    b1: ExprId,
    b2: ExprId,
) -> bool {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // Direct match: (A,B) == (C,D)
    let direct = compare_expr(ctx, a1, b1) == Ordering::Equal
        && compare_expr(ctx, a2, b2) == Ordering::Equal;

    // Crossed match: (A,B) == (D,C)
    let crossed = compare_expr(ctx, a1, b2) == Ordering::Equal
        && compare_expr(ctx, a2, b1) == Ordering::Equal;

    direct || crossed
}

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

/// Normalize an expression for even functions like cos.
/// For even functions: f(-x) = f(x), so we can strip the negation.
/// Returns the unwrapped inner expression if input is Neg(inner), else returns input.
pub fn normalize_for_even_fn(ctx: &cas_ast::Context, expr: ExprId) -> ExprId {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    // If expr is Neg(inner), return inner
    if let Expr::Neg(inner) = ctx.get(expr) {
        return *inner;
    }
    // Also handle Mul(-1, x) or Mul(x, -1)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let minus_one = BigRational::from_integer(BigInt::from(-1));
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &minus_one {
                return *r;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &minus_one {
                return *l;
            }
        }
    }
    expr
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

/// Try to simplify a division when numerator has a coefficient divisible by denominator
/// e.g., 4x/2 → 2x, -2x/2 → -x
fn simplify_numeric_div(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    use crate::helpers::as_i64;

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        // Check if denominator is a small integer
        if let Some(den_val) = as_i64(ctx, den) {
            if den_val == 0 {
                return expr; // Avoid division by zero
            }

            // Check if numerator is a product k*x where k is divisible by den
            if let Expr::Mul(l, r) = ctx.get(num).clone() {
                if let Some(coeff) = as_i64(ctx, l) {
                    if coeff % den_val == 0 {
                        let new_coeff = coeff / den_val;
                        if new_coeff == 1 {
                            return r; // 2x/2 → x
                        } else if new_coeff == -1 {
                            return ctx.add(Expr::Neg(r)); // -2x/2 → -x
                        } else {
                            let new_coeff_expr = ctx.num(new_coeff);
                            return ctx.add(Expr::Mul(new_coeff_expr, r)); // 4x/2 → 2x
                        }
                    }
                }
                if let Some(coeff) = as_i64(ctx, r) {
                    if coeff % den_val == 0 {
                        let new_coeff = coeff / den_val;
                        if new_coeff == 1 {
                            return l; // x*2/2 → x
                        } else if new_coeff == -1 {
                            return ctx.add(Expr::Neg(l)); // x*(-2)/2 → -x
                        } else {
                            let new_coeff_expr = ctx.num(new_coeff);
                            return ctx.add(Expr::Mul(l, new_coeff_expr)); // x*4/2 → x*2
                        }
                    }
                }
            }

            // Check if numerator is a plain number divisible by den
            if let Some(num_val) = as_i64(ctx, num) {
                if num_val % den_val == 0 {
                    return ctx.num(num_val / den_val); // 4/2 → 2
                }
            }
        }
    }
    expr
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
        let Expr::Div(num_id, den_id) = ctx.get(expr).clone() else {
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

            let sin_half_diff = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));
            let cos_half_diff = ctx.add(Expr::Function("cos".to_string(), vec![half_diff_for_cos]));
            let quotient_result = ctx.add(Expr::Div(sin_half_diff, cos_half_diff));

            // Intermediate states
            let two = ctx.num(2);
            let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg_normalized]));
            let sin_half = ctx.add(Expr::Function("sin".to_string(), vec![half_diff]));

            // Intermediate numerator: 2·cos(avg)·sin(half_diff)
            let num_product = smart_mul(ctx, cos_avg, sin_half);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_2 = ctx.add(Expr::Function("cos".to_string(), vec![avg_normalized]));
            let cos_half = ctx.add(Expr::Function("cos".to_string(), vec![half_diff_for_cos]));
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
            let sin_avg = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
            let cos_avg = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
            let final_result = ctx.add(Expr::Div(sin_avg, cos_avg));

            // Intermediate states
            let two = ctx.num(2);
            let cos_half_diff = ctx.add(Expr::Function(
                "cos".to_string(),
                vec![half_diff_normalized],
            ));

            // Intermediate numerator: 2·sin(avg)·cos(half_diff)
            let sin_avg_for_num = ctx.add(Expr::Function("sin".to_string(), vec![avg]));
            let num_product = smart_mul(ctx, sin_avg_for_num, cos_half_diff);
            let intermediate_num = smart_mul(ctx, two, num_product);

            // Intermediate denominator: 2·cos(avg)·cos(half_diff)
            let two_2 = ctx.num(2);
            let cos_avg_for_den = ctx.add(Expr::Function("cos".to_string(), vec![avg]));
            let cos_half_diff_2 = ctx.add(Expr::Function(
                "cos".to_string(),
                vec![half_diff_normalized],
            ));
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
