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
                after_local: None, domain_assumption: None,
            });
        }
        if let Some((result, desc)) = check_pythagorean_pattern(ctx, t2, t1) {
            return Some(Rewrite {
                new_expr: result,
                description: desc,
                before_local: None,
                after_local: None, domain_assumption: None,
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
