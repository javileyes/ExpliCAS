//! # Pythagorean Identity Simplification
//!
//! This module provides the `TrigPythagoreanSimplifyRule` which applies the
//! Pythagorean identity to simplify expressions of the form:
//!
//! - `k - k*sin²(x) → k*cos²(x)`
//! - `k - k*cos²(x) → k*sin²(x)`
//!
//! This rule was extracted from `CancelCommonFactorsRule` for better
//! step-by-step transparency and pedagogical clarity.

use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;

define_rule!(
    TrigPythagoreanSimplifyRule,
    "Pythagorean Factor Form",
    |ctx, expr| {
        // Only apply to Add expressions (Sub is represented as Add with Neg)

        // We need exactly 2 terms: constant and trig term
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);
        if terms.len() != 2 {
            return None;
        }

        let t1 = terms[0];
        let t2 = terms[1];

        // Try both orderings: (c_term, trig_term) or (trig_term, c_term)
        if let Some((result, desc)) = check_pythagorean_pattern(ctx, t1, t2) {
            return Some(Rewrite {
                new_expr: result,
                description: desc,
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            required_conditions: vec![],
            });
        }
        if let Some((result, desc)) = check_pythagorean_pattern(ctx, t2, t1) {
            return Some(Rewrite {
                new_expr: result,
                description: desc,
                before_local: None,
                after_local: None,
                assumption_events: Default::default(),
            required_conditions: vec![],
            });
        }

        None
    }
);

/// Check if (c_term, trig_term) matches the pattern k - k*trig²(x) = k*other²(x)
/// where trig is sin or cos, and other is the complementary function.
fn check_pythagorean_pattern(
    ctx: &mut Context,
    c_term: ExprId,
    t_term: ExprId,
) -> Option<(ExprId, String)> {
    // Parse t_term for k * trig^2 (possibly negated)
    let (base_term, is_neg) = if let Expr::Neg(inner) = ctx.get(t_term) {
        (*inner, true)
    } else {
        (t_term, false)
    };

    // Flatten multiplication factors
    let mut factors = Vec::new();
    let mut stack = vec![base_term];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(curr) {
            stack.push(*r);
            stack.push(*l);
        } else {
            factors.push(curr);
        }
    }

    // Find trig²(x) factor
    let mut trig_idx = None;
    let mut func_name = String::new();
    let mut arg = None;

    for (i, &f) in factors.iter().enumerate() {
        if let Expr::Pow(b, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Function(name, args) = ctx.get(*b) {
                        if (name == "sin" || name == "cos") && args.len() == 1 {
                            trig_idx = Some(i);
                            func_name = name.clone();
                            arg = Some(args[0]);
                            break;
                        }
                    }
                }
            }
        }
    }

    let idx = trig_idx?;
    let arg = arg?;

    // Collect coefficient factors (excluding trig²)
    let mut coeff_factors = Vec::new();
    if is_neg {
        coeff_factors.push(ctx.num(-1));
    }
    for (i, &f) in factors.iter().enumerate() {
        if i != idx {
            coeff_factors.push(f);
        }
    }

    // Build coefficient
    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut c = coeff_factors[0];
        for &f in coeff_factors.iter().skip(1) {
            c = smart_mul(ctx, c, f);
        }
        c
    };

    // The pattern is: c_term + (-coeff * trig²) where c_term == -(-coeff) = coeff
    // So we need c_term == -coeff (negative of the coefficient)
    let neg_coeff = if let Expr::Number(n) = ctx.get(coeff) {
        ctx.add(Expr::Number(-n.clone()))
    } else if let Expr::Neg(inner) = ctx.get(coeff) {
        *inner
    } else {
        ctx.add(Expr::Neg(coeff))
    };

    if crate::ordering::compare_expr(ctx, c_term, neg_coeff) == std::cmp::Ordering::Equal {
        // Pattern matches! Apply identity: k - k*sin² = k*cos², k - k*cos² = k*sin²
        let other_name = if func_name == "sin" { "cos" } else { "sin" };
        let func = ctx.add(Expr::Function(other_name.to_string(), vec![arg]));
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(func, two));
        let result = smart_mul(ctx, c_term, pow);

        let desc = format!("1 - {}²(x) = {}²(x)", func_name, other_name);

        return Some((result, desc));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::SimpleRule;

    #[test]
    fn test_one_minus_sin_squared() {
        let mut ctx = Context::new();
        // 1 - sin²(x) should become cos²(x)
        let x = ctx.var("x");
        let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x]));
        let two = ctx.num(2);
        let sin_sq = ctx.add(Expr::Pow(sin_x, two));
        let neg_sin_sq = ctx.add(Expr::Neg(sin_sq));
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(one, neg_sin_sq)); // 1 + (-sin²(x))

        let rule = TrigPythagoreanSimplifyRule;
        let result = rule.apply_simple(&mut ctx, expr);

        assert!(result.is_some(), "Rule should apply to 1 - sin²(x)");
        let rewrite = result.unwrap();

        // Result should be cos²(x)
        if let Expr::Mul(_coeff, pow) = ctx.get(rewrite.new_expr) {
            if let Expr::Pow(base, _) = ctx.get(*pow) {
                if let Expr::Function(name, _) = ctx.get(*base) {
                    assert_eq!(name, "cos");
                }
            }
        }
    }
}

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
        let mut sin4_terms: Vec<(num_rational::BigRational, ExprId, usize)> = Vec::new(); // (coef, arg, term_index)
        let mut cos4_terms: Vec<(num_rational::BigRational, ExprId, usize)> = Vec::new();

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

                let sin_func = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                let cos_func = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
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
                let mut new_terms: Vec<ExprId> = Vec::new();
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

                return Some(Rewrite {
                    new_expr: result,
                    description: "sin⁴(x) - cos⁴(x) = sin²(x) - cos²(x)".to_string(),
                    before_local: None,
                    after_local: None,
                    assumption_events: Default::default(),
            required_conditions: vec![],
                });
            }
        }

        None
    }
);

/// Extract (coefficient, trig_function_name, argument) from a term like k * sin(u)^4
/// Returns None if the term doesn't match this pattern.
fn extract_trig_fourth_power(
    ctx: &Context,
    term: ExprId,
) -> Option<(num_rational::BigRational, String, ExprId)> {
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
                    if let Expr::Function(name, args) = ctx.get(*base) {
                        if (name == "sin" || name == "cos") && args.len() == 1 {
                            trig_func_name = Some(name.clone());
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
