//! Angle expansion and product-to-sum identities.
//!
//! This module contains rules for:
//! - Product-to-sum: 2·sin(a)·cos(b) → sin(a+b) + sin(a-b)
//! - Trig phase shifts: sin(x + π/2) → cos(x)

use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};

// Import from parent (identities.rs) - function defined in half_angle_phase.rs include
use super::extract_phase_shift;

// =============================================================================
// PRODUCT-TO-SUM IDENTITIES
// =============================================================================
// 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
// 2*cos(a)*sin(b) → sin(a+b) - sin(a-b)
// 2*cos(a)*cos(b) → cos(a+b) + cos(a-b)
// 2*sin(a)*sin(b) → cos(a-b) - cos(a+b)

define_rule!(ProductToSumRule, "Product to Sum", |ctx, expr| {
    // Look for patterns like: 2 * sin(a) * cos(b)
    // or: sin(a) * cos(b) * 2
    let expr_data = ctx.get(expr).clone();

    if let Expr::Mul(_, _) = expr_data {
        let mut factors = Vec::new();
        crate::helpers::flatten_mul(ctx, expr, &mut factors);

        // Find the coefficient 2 and two trig functions
        let mut has_two = false;
        let mut two_idx = None;
        let mut trig_funcs: Vec<(usize, String, ExprId)> = Vec::new();

        for (i, &factor) in factors.iter().enumerate() {
            match ctx.get(factor) {
                Expr::Number(n) => {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        has_two = true;
                        two_idx = Some(i);
                    }
                }
                Expr::Function(name, args) => {
                    if args.len() == 1 && (name == "sin" || name == "cos") {
                        trig_funcs.push((i, name.clone(), args[0]));
                    }
                }
                _ => {}
            }
        }

        // Need exactly: coefficient 2 and exactly 2 trig functions
        if has_two && trig_funcs.len() == 2 {
            let (idx1, name1, arg1) = &trig_funcs[0];
            let (idx2, name2, arg2) = &trig_funcs[1];

            // Build remaining factors (everything except 2 and the two trigs)
            let mut remaining: Vec<ExprId> = Vec::new();
            for (i, &factor) in factors.iter().enumerate() {
                if Some(i) != two_idx && i != *idx1 && i != *idx2 {
                    remaining.push(factor);
                }
            }

            // Determine which identity to apply
            let (new_expr, description) = match (name1.as_str(), name2.as_str()) {
                ("sin", "cos") => {
                    // 2*sin(a)*cos(b) → sin(a+b) + sin(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let sin_sum = ctx.add(Expr::Function("sin".to_string(), vec![sum_arg]));
                    let sin_diff = ctx.add(Expr::Function("sin".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Add(sin_sum, sin_diff));
                    (result, "2·sin(a)·cos(b) → sin(a+b) + sin(a-b)")
                }
                ("cos", "sin") => {
                    // 2*cos(a)*sin(b) → sin(a+b) - sin(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let sin_sum = ctx.add(Expr::Function("sin".to_string(), vec![sum_arg]));
                    let sin_diff = ctx.add(Expr::Function("sin".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Sub(sin_sum, sin_diff));
                    (result, "2·cos(a)·sin(b) → sin(a+b) - sin(a-b)")
                }
                ("cos", "cos") => {
                    // 2*cos(a)*cos(b) → cos(a+b) + cos(a-b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let cos_sum = ctx.add(Expr::Function("cos".to_string(), vec![sum_arg]));
                    let cos_diff = ctx.add(Expr::Function("cos".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Add(cos_sum, cos_diff));
                    (result, "2·cos(a)·cos(b) → cos(a+b) + cos(a-b)")
                }
                ("sin", "sin") => {
                    // 2*sin(a)*sin(b) → cos(a-b) - cos(a+b)
                    let sum_arg = ctx.add(Expr::Add(*arg1, *arg2));
                    let diff_arg = ctx.add(Expr::Sub(*arg1, *arg2));
                    let cos_sum = ctx.add(Expr::Function("cos".to_string(), vec![sum_arg]));
                    let cos_diff = ctx.add(Expr::Function("cos".to_string(), vec![diff_arg]));
                    let result = ctx.add(Expr::Sub(cos_diff, cos_sum));
                    (result, "2·sin(a)·sin(b) → cos(a-b) - cos(a+b)")
                }
                _ => return None,
            };

            // If there are remaining factors, multiply them back
            let final_expr = if remaining.is_empty() {
                new_expr
            } else {
                let mut result = new_expr;
                for factor in remaining {
                    result = smart_mul(ctx, result, factor);
                }
                result
            };

            return Some(Rewrite::new(final_expr).desc(description));
        }
    }
    None
});
// ============================================================================
// Trig Phase Shift Rule
// ============================================================================
// sin(x + π/2) → cos(x)
// sin(x - π/2) → -cos(x)
// sin(x + π) → -sin(x)
// cos(x + π/2) → -sin(x)
// cos(x - π/2) → sin(x)
// cos(x + π) → -cos(x)
//
// Also handles canonical form: sin((2*x + π)/2) where arg = (2*x + π)/2

define_rule!(TrigPhaseShiftRule, "Trig Phase Shift", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();

    if let Expr::Function(name, args) = expr_data {
        if args.len() != 1 {
            return None;
        }

        let is_sin = name == "sin";
        let is_cos = name == "cos";
        if !is_sin && !is_cos {
            return None;
        }

        let arg = args[0];

        // Try to extract (base_term, pi_multiple) where arg = base_term + pi_multiple * π/2
        let (base_term, pi_multiple) = extract_phase_shift(ctx, arg)?;

        if pi_multiple == 0 {
            return None;
        }

        // Normalize k to 0..3 range (mod 4)
        let k = ((pi_multiple % 4) + 4) % 4;

        // Apply phase shift
        // sin(x + k*π/2): k=0→sin(x), k=1→cos(x), k=2→-sin(x), k=3→-cos(x)
        // cos(x + k*π/2): k=0→cos(x), k=1→-sin(x), k=2→-cos(x), k=3→sin(x)
        let (new_func, negate) = if is_sin {
            match k {
                0 => ("sin", false),
                1 => ("cos", false),
                2 => ("sin", true),
                3 => ("cos", true),
                _ => return None,
            }
        } else {
            match k {
                0 => ("cos", false),
                1 => ("sin", true),
                2 => ("cos", true),
                3 => ("sin", false),
                _ => return None,
            }
        };

        let new_trig = ctx.add(Expr::Function(new_func.to_string(), vec![base_term]));
        let new_expr = if negate {
            ctx.add(Expr::Neg(new_trig))
        } else {
            new_trig
        };

        let shift_desc = match pi_multiple {
            1 => "π/2",
            -1 => "-π/2",
            2 => "π",
            -2 => "-π",
            3 => "3π/2",
            -3 => "-3π/2",
            _ => "kπ/2",
        };

        return Some(
            Rewrite::new(new_expr).desc(format!("{}(x + {}) phase shift", name, shift_desc)),
        );
    }

    None
});
