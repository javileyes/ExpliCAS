//! Secondary Pythagorean-adjacent rules: reciprocal conversions and even-power differences.
//!
//! Extracted from `pythagorean.rs` to reduce file size.
//! - SecToRecipCosRule, CscToRecipSinRule, CotToCosSinRule
//! - TrigEvenPowerDifferenceRule + extract_trig_fourth_power

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;

// =============================================================================
// SecToRecipCosRule: sec(x) → 1/cos(x) (canonical expansion)
// =============================================================================
// This ensures sec unifies with 1/cos forms from tan²+1 simplification.

define_rule!(
    SecToRecipCosRule,
    "Secant to Reciprocal Cosine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let builtin = ctx.builtin_of(*fn_id);
            if matches!(builtin, Some(cas_ast::BuiltinFn::Sec)) && args.len() == 1 {
                let arg = args[0];
                let cos_func = ctx.call("cos", vec![arg]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Div(one, cos_func));
                return Some(Rewrite::new(result).desc("sec(x) = 1/cos(x)"));
            }
        }
        None
    }
);

// =============================================================================
// CscToRecipSinRule: csc(x) → 1/sin(x) (canonical expansion)
// =============================================================================

define_rule!(
    CscToRecipSinRule,
    "Cosecant to Reciprocal Sine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let builtin = ctx.builtin_of(*fn_id);
            if matches!(builtin, Some(cas_ast::BuiltinFn::Csc)) && args.len() == 1 {
                let arg = args[0];
                let sin_func = ctx.call("sin", vec![arg]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Div(one, sin_func));
                return Some(Rewrite::new(result).desc("csc(x) = 1/sin(x)"));
            }
        }
        None
    }
);

// =============================================================================
// CotToCosSinRule: cot(x) → cos(x)/sin(x) (canonical expansion)
// =============================================================================
// This ensures cot unifies with cos/sin forms for csc²-cot²=1 simplification.

define_rule!(
    CotToCosSinRule,
    "Cotangent to Cosine over Sine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let builtin = ctx.builtin_of(*fn_id);
            if matches!(builtin, Some(cas_ast::BuiltinFn::Cot)) && args.len() == 1 {
                let arg = args[0];
                let cos_func = ctx.call("cos", vec![arg]);
                let sin_func = ctx.call("sin", vec![arg]);
                let result = ctx.add(Expr::Div(cos_func, sin_func));
                return Some(Rewrite::new(result).desc("cot(x) = cos(x)/sin(x)"));
            }
        }
        None
    }
);

// ============================================================================
// TrigEvenPowerDifferenceRule
// ============================================================================
// Detects pairs like k*sin⁴(u) + (-k)*cos⁴(u) in Add expressions and reduces
// them using the factorization: sin⁴ - cos⁴ = (sin² - cos²)(sin² + cos²) = sin² - cos²
//
// This is a degree-reducing rule (4 → 2) so it cannot loop.

define_rule!(
    TrigEvenPowerDifferenceRule,
    "Trig Fourth Power Difference",
    |ctx, expr| {
        // Collect all additive terms
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // For each term, try to extract (coefficient, trig_func, argument, power)
        // We're looking for sin⁴(u) or cos⁴(u) with coefficients
        let mut sin4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new(); // (coef, arg, term_index)
        let mut cos4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((coef, func_name, arg)) = extract_trig_fourth_power(ctx, term) {
                if func_name == "sin" {
                    sin4_terms.push((coef, arg, i));
                } else if func_name == "cos" {
                    cos4_terms.push((coef, arg, i));
                }
            }
        }

        // Find matching pairs: same argument, opposite coefficients
        for (sin_coef, sin_arg, sin_idx) in sin4_terms.iter() {
            for (cos_coef, cos_arg, cos_idx) in cos4_terms.iter() {
                // Same argument?
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Opposite coefficients? sin_coef = -cos_coef
                let sum = sin_coef.clone() + cos_coef.clone();
                if !sum.is_zero() {
                    continue;
                }

                // Found a match! k*sin⁴(u) + (-k)*cos⁴(u) → k*(sin²(u) - cos²(u))
                // Build replacement: coef * (sin²(arg) - cos²(arg))
                let arg = *sin_arg;
                let coef = sin_coef.clone();

                let sin_func = ctx.call("sin", vec![arg]);
                let cos_func = ctx.call("cos", vec![arg]);
                let two = ctx.num(2);
                let sin_sq = ctx.add(Expr::Pow(sin_func, two));
                let cos_sq = ctx.add(Expr::Pow(cos_func, two));
                let diff = ctx.add(Expr::Sub(sin_sq, cos_sq));

                // Apply coefficient
                let replacement = if coef == num_rational::BigRational::from_integer(1.into()) {
                    diff
                } else if coef == num_rational::BigRational::from_integer((-1).into()) {
                    ctx.add(Expr::Neg(diff))
                } else {
                    let coef_expr = ctx.add(Expr::Number(coef));
                    ctx.add(Expr::Mul(coef_expr, diff))
                };

                // Build new expression: remove the two matched terms, add replacement
                let mut new_terms: Vec<cas_ast::ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != *sin_idx && j != *cos_idx {
                        new_terms.push(t);
                    }
                }
                new_terms.push(replacement);

                // Build result as sum
                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("sin⁴(x) - cos⁴(x) = sin²(x) - cos²(x)"));
            }
        }

        None
    }
);

/// Extract (coefficient, trig_function_name, argument) from a term like k * sin(u)^4
/// Returns None if the term doesn't match this pattern.
fn extract_trig_fourth_power(
    ctx: &cas_ast::Context,
    term: cas_ast::ExprId,
) -> Option<(num_rational::BigRational, String, cas_ast::ExprId)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut coef = BigRational::one();
    let mut working = term;

    // Handle Neg wrapper
    if let Expr::Neg(inner) = ctx.get(term) {
        coef = -coef;
        working = *inner;
    }

    // Flatten multiplication
    let mut factors = Vec::new();
    let mut stack = vec![working];
    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            _ => factors.push(curr),
        }
    }

    // Look for trig⁴(arg) and extract numeric coefficients
    let mut trig_func_name = None;
    let mut trig_arg = None;
    let mut trig_idx = None;

    for (i, &f) in factors.iter().enumerate() {
        // Check for Number (accumulate into coefficient)
        if let Expr::Number(n) = ctx.get(f) {
            coef *= n.clone();
            continue;
        }

        // Check for Pow(trig_func, 4)
        if let Expr::Pow(base, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == BigRational::from_integer(4.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if matches!(
                            builtin,
                            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                        ) && args.len() == 1
                        {
                            trig_func_name = Some(builtin.unwrap().name().to_string());
                            trig_arg = Some(args[0]);
                            trig_idx = Some(i);
                        }
                    }
                }
            }
        }
    }

    // Verify we found exactly one trig⁴ and possibly some numeric factors
    let func_name = trig_func_name?;
    let arg = trig_arg?;
    let idx = trig_idx?;

    // Verify remaining factors are all numbers (already accumulated into coef)
    for (i, &f) in factors.iter().enumerate() {
        if i == idx {
            continue;
        }
        if !matches!(ctx.get(f), Expr::Number(_)) {
            // Non-numeric factor that isn't the trig⁴, pattern doesn't match cleanly
            return None;
        }
    }

    Some((coef, func_name, arg))
}

use num_traits::Zero;
