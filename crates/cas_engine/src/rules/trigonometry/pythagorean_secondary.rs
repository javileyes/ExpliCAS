//! Secondary Pythagorean-adjacent rules: reciprocal conversions and even-power differences.
//!
//! Extracted from `pythagorean.rs` to reduce file size.
//! - SecToRecipCosRule, CscToRecipSinRule, CotToCosSinRule
//! - TrigEvenPowerDifferenceRule (with extraction delegated to cas_math)

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::trig_power_identity_support::extract_coeff_trig_pow4;

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
                let cos_func = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
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
                let sin_func = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
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
                let cos_func = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
                let sin_func = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
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
        let terms = crate::nary::add_leaves(ctx, expr);

        if terms.len() < 2 {
            return None;
        }

        // For each term, try to extract (coefficient, trig_func, argument, power)
        // We're looking for sin⁴(u) or cos⁴(u) with coefficients
        let mut sin4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new(); // (coef, arg, term_index)
        let mut cos4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((coef, func_name, arg)) = extract_coeff_trig_pow4(ctx, term) {
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

                let sin_func = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
                let cos_func = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
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

// ============================================================================
// TrigEvenPowerSumRule
// ============================================================================
// Detects pairs like k*sin⁴(u) + k*cos⁴(u) in Add expressions and reduces
// them using: sin⁴ + cos⁴ = (sin² + cos²)² - 2·sin²·cos² = 1 - 2·sin²·cos²
//
// So: k·sin⁴(u) + k·cos⁴(u) → k·(1 - 2·sin²(u)·cos²(u))
//
// This is a degree-reducing rule (4 → 2) so it cannot loop.

define_rule!(
    TrigEvenPowerSumRule,
    "Trig Fourth Power Sum",
    |ctx, expr| {
        // Collect all additive terms
        let terms = crate::nary::add_leaves(ctx, expr);

        if terms.len() < 2 {
            return None;
        }

        let mut sin4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new();
        let mut cos4_terms: Vec<(num_rational::BigRational, cas_ast::ExprId, usize)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((coef, func_name, arg)) = extract_coeff_trig_pow4(ctx, term) {
                if func_name == "sin" {
                    sin4_terms.push((coef, arg, i));
                } else if func_name == "cos" {
                    cos4_terms.push((coef, arg, i));
                }
            }
        }

        // Find matching pairs: same argument, SAME coefficient (sum, not difference)
        for (sin_coef, sin_arg, sin_idx) in sin4_terms.iter() {
            for (cos_coef, cos_arg, cos_idx) in cos4_terms.iter() {
                if sin_idx == cos_idx {
                    continue;
                }
                // Same argument?
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Same coefficients? sin_coef == cos_coef
                if sin_coef != cos_coef {
                    continue;
                }

                // Found: k·sin⁴(u) + k·cos⁴(u) → k·(1 - 2·sin²(u)·cos²(u))
                let arg = *sin_arg;
                let coef = sin_coef.clone();

                let sin_func = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
                let cos_func = ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![arg]);
                let two = ctx.num(2);
                let sin_sq = ctx.add(Expr::Pow(sin_func, two));
                let two2 = ctx.num(2);
                let cos_sq = ctx.add(Expr::Pow(cos_func, two2));
                let sin2_cos2 = ctx.add(Expr::Mul(sin_sq, cos_sq));
                let two3 = ctx.num(2);
                let two_sin2_cos2 = ctx.add(Expr::Mul(two3, sin2_cos2));
                let one = ctx.num(1);
                let body = ctx.add(Expr::Sub(one, two_sin2_cos2));

                // Apply coefficient
                let replacement = if coef == num_rational::BigRational::from_integer(1.into()) {
                    body
                } else {
                    let coef_expr = ctx.add(Expr::Number(coef));
                    ctx.add(Expr::Mul(coef_expr, body))
                };

                // Build new expression: remove the two matched terms, add replacement
                let mut new_terms: Vec<cas_ast::ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != *sin_idx && j != *cos_idx {
                        new_terms.push(t);
                    }
                }
                new_terms.push(replacement);

                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(
                    Rewrite::new(result).desc("k·sin⁴(x) + k·cos⁴(x) = k·(1 - 2·sin²(x)·cos²(x))"),
                );
            }
        }

        None
    }
);

use num_traits::Zero;
