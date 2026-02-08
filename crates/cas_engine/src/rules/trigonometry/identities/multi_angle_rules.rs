//! Multi-angle expansion rules (triple, quintuple, recursive) and trig-quotient helpers.
//!
//! Extracted from `expansion_rules.rs` to keep module size manageable.

use crate::define_rule;
use crate::helpers::{as_mul, as_pow, extract_triple_angle_arg};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{BuiltinFn, Expr, ExprId};
use num_traits::{One, Zero};

// Import helpers from sibling modules (via re-exports in parent)
use super::extract_trig_arg;

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(
    TripleAngleRule,
    "Triple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern (defer to SinCosSumQuotientRule)
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 3*x or x*3
                if let Some(inner_var) = extract_triple_angle_arg(ctx, args[0]) {
                    match ctx.builtin_of(*fn_id) {
                        Some(BuiltinFn::Sin) => {
                            // sin(3x) → 3sin(x) - 4sin³(x)
                            let three = ctx.num(3);
                            let four = ctx.num(4);
                            let exp_three = ctx.num(3); // Separate for Pow exponent
                            let sin_x = ctx.call("sin", vec![inner_var]);

                            // 3*sin(x)
                            let term1 = smart_mul(ctx, three, sin_x);

                            // sin³(x) = sin(x)^3
                            let sin_cubed = ctx.add(Expr::Pow(sin_x, exp_three));
                            // 4*sin³(x)
                            let term2 = smart_mul(ctx, four, sin_cubed);

                            // 3sin(x) - 4sin³(x)
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr).desc("sin(3x) → 3sin(x) - 4sin³(x)"),
                            );
                        }
                        Some(BuiltinFn::Cos) => {
                            // cos(3x) → 4cos³(x) - 3cos(x)
                            let three = ctx.num(3);
                            let four = ctx.num(4);
                            let exp_three = ctx.num(3); // Separate for Pow exponent
                            let cos_x = ctx.call("cos", vec![inner_var]);

                            // cos³(x) = cos(x)^3
                            let cos_cubed = ctx.add(Expr::Pow(cos_x, exp_three));
                            // 4*cos³(x)
                            let term1 = smart_mul(ctx, four, cos_cubed);

                            // 3*cos(x)
                            let term2 = smart_mul(ctx, three, cos_x);

                            // 4cos³(x) - 3cos(x)
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(
                                Rewrite::new(new_expr).desc("cos(3x) → 4cos³(x) - 3cos(x)"),
                            );
                        }
                        Some(BuiltinFn::Tan) => {
                            // tan(3x) → (3tan(x) - tan³(x)) / (1 - 3tan²(x))
                            let one = ctx.num(1);
                            let three = ctx.num(3);
                            let exp_two = ctx.num(2);
                            let exp_three = ctx.num(3);
                            let tan_x = ctx.call("tan", vec![inner_var]);

                            // Numerator: 3tan(x) - tan³(x)
                            let three_tan = smart_mul(ctx, three, tan_x);
                            let tan_cubed = ctx.add(Expr::Pow(tan_x, exp_three));
                            let numer = ctx.add(Expr::Sub(three_tan, tan_cubed));

                            // Denominator: 1 - 3tan²(x)
                            let tan_squared = ctx.add(Expr::Pow(tan_x, exp_two));
                            let three_tan_squared = smart_mul(ctx, three, tan_squared);
                            let denom = ctx.add(Expr::Sub(one, three_tan_squared));

                            let new_expr = ctx.add(Expr::Div(numer, denom));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("tan(3x) → (3tan(x) - tan³(x))/(1 - 3tan²(x))"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

// Quintuple Angle Rule: sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
// This is a direct expansion to avoid recursive explosion via double/triple angle.
define_rule!(
    QuintupleAngleRule,
    "Quintuple Angle Identity",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
        }

        // GUARD 2: Skip if inside sum-quotient pattern
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 5*x or x*5
                if let Some(inner_var) = crate::helpers::extract_quintuple_angle_arg(ctx, args[0]) {
                    match ctx.builtin_of(*fn_id) {
                        Some(BuiltinFn::Sin) => {
                            // sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let sin_x = ctx.call("sin", vec![inner_var]);

                            // 16sin⁵(x)
                            let sin_5 = ctx.add(Expr::Pow(sin_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, sin_5);

                            // 20sin³(x)
                            let sin_3 = ctx.add(Expr::Pow(sin_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, sin_3);

                            // 5sin(x)
                            let term3 = smart_mul(ctx, five, sin_x);

                            // 16sin⁵(x) - 20sin³(x) + 5sin(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)"),
                            );
                        }
                        Some(BuiltinFn::Cos) => {
                            // cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let five = ctx.num(5);
                            let sixteen = ctx.num(16);
                            let twenty = ctx.num(20);
                            let exp_three = ctx.num(3);
                            let exp_five = ctx.num(5);
                            let cos_x = ctx.call("cos", vec![inner_var]);

                            // 16cos⁵(x)
                            let cos_5 = ctx.add(Expr::Pow(cos_x, exp_five));
                            let term1 = smart_mul(ctx, sixteen, cos_5);

                            // 20cos³(x)
                            let cos_3 = ctx.add(Expr::Pow(cos_x, exp_three));
                            let term2 = smart_mul(ctx, twenty, cos_3);

                            // 5cos(x)
                            let term3 = smart_mul(ctx, five, cos_x);

                            // 16cos⁵(x) - 20cos³(x) + 5cos(x)
                            let sub1 = ctx.add(Expr::Sub(term1, term2));
                            let new_expr = ctx.add(Expr::Add(sub1, term3));
                            return Some(
                                Rewrite::new(new_expr)
                                    .desc("cos(5x) → 16cos⁵(x) - 20cos³(x) + 5cos(x)"),
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

/// Check if a trig function is inside a potential sum-quotient pattern
/// (sin(A)±sin(B)) / (cos(A)±cos(B))
/// Returns true if expansion should be deferred to SinCosSumQuotientRule
fn is_inside_trig_quotient_pattern(
    ctx: &cas_ast::Context,
    _expr: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    // Check if any ancestor is a Div with the sum-quotient pattern
    parent_ctx.has_ancestor_matching(ctx, |c, id| {
        if let Expr::Div(num, den) = c.get(id) {
            // Check if numerator is Add or Sub of sin functions
            let num_is_sin_sum_or_diff = is_binary_trig_op(c, *num, "sin");
            // Check if denominator is Add of cos functions
            let den_is_cos_sum = is_trig_sum(c, *den, "cos");
            num_is_sin_sum_or_diff && den_is_cos_sum
        } else {
            false
        }
    })
}

/// Check if expr is Add(trig(A), trig(B)) or Sub(trig(A), trig(B)) or Add(trig(A), Neg(trig(B)))
fn is_binary_trig_op(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Check for Add(sin(A), sin(B))
            if extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
            {
                return true;
            }
            // Check for Add(sin(A), Neg(sin(B)))
            if let Expr::Neg(inner) = ctx.get(*r) {
                if extract_trig_arg(ctx, *l, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                if extract_trig_arg(ctx, *r, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            false
        }
        Expr::Sub(l, r) => {
            extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
        }
        _ => false,
    }
}

/// Check if expr is Add(trig(A), trig(B))
fn is_trig_sum(ctx: &cas_ast::Context, expr: ExprId, fn_name: &str) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        return extract_trig_arg(ctx, *l, fn_name).is_some()
            && extract_trig_arg(ctx, *r, fn_name).is_some();
    }
    false
}

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr, parent_ctx| {
        // GUARD 1: Skip if this trig function is marked for protection
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_sum_quotient_protected(expr) {
                return None;
            }
            // GUARD 1b: Skip if sin(4x) identity pattern detected
            // This prevents sin(4*t) from expanding before Sin4xIdentityZeroRule can fire
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }

        // GUARD 2: Skip if we're inside a potential sum-quotient pattern
        // This heuristic checks: if the trig function is inside a Div, and both
        // numerator and denominator are Add/Sub of trig functions, defer to
        // SinCosSumQuotientRule instead of expanding.
        if is_inside_trig_quotient_pattern(ctx, expr, parent_ctx) {
            return None;
        }

        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        {
            let builtin = ctx.builtin_of(fn_id);
            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if args.len() == 1 && (is_sin || is_cos) {
                // Check for n * x where n is integer > 2
                let inner = args[0];

                let (n_val, x_val) = if let Some((l, r)) = as_mul(ctx, inner) {
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_integer() {
                            (n.to_integer(), r)
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_integer() {
                            (n.to_integer(), l)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };

                if n_val > num_bigint::BigInt::from(2) && n_val <= num_bigint::BigInt::from(6) {
                    // GUARD: Only expand sin(n*x) for small n (3-6).
                    // For n > 6, the expansion grows exponentially without benefit.
                    // This prevents catastrophic expansion like sin(671*x) → 670 recursive steps.

                    // Rewrite sin(nx) -> sin((n-1)x + x)

                    let n_minus_1 = n_val.clone() - 1;
                    let n_minus_1_expr = ctx.add(Expr::Number(
                        num_rational::BigRational::from_integer(n_minus_1),
                    ));
                    let term_nm1 = smart_mul(ctx, n_minus_1_expr, x_val);

                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)

                    let sin_nm1 = ctx.call("sin", vec![term_nm1]);
                    let cos_nm1 = ctx.call("cos", vec![term_nm1]);
                    let sin_x = ctx.call("sin", vec![x_val]);
                    let cos_x = ctx.call("cos", vec![x_val]);

                    if is_sin {
                        let t1 = smart_mul(ctx, sin_nm1, cos_x);
                        let t2 = smart_mul(ctx, cos_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("sin({}x) expansion", n_val)),
                        );
                    } else {
                        // cos
                        let t1 = smart_mul(ctx, cos_nm1, cos_x);
                        let t2 = smart_mul(ctx, sin_nm1, sin_x);
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(
                            Rewrite::new(new_expr).desc(format!("cos({}x) expansion", n_val)),
                        );
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        // cos^n(x) -> (1 - sin^2(x))^(n/2) for even n
        if let Some((base, exp)) = as_pow(ctx, expr) {
            let n_opt = if let Expr::Number(n) = ctx.get(exp) {
                Some(n.clone())
            } else {
                None
            };

            if let Some(n) = n_opt {
                if n.is_integer()
                    && n.to_integer() % 2 == 0.into()
                    && n > num_rational::BigRational::zero()
                {
                    // Limit power to avoid explosion? Let's say <= 4 for now.
                    if n <= num_rational::BigRational::from_integer(4.into()) {
                        if let Expr::Function(fn_id, args) = ctx.get(base) {
                            if matches!(ctx.builtin_of(*fn_id), Some(BuiltinFn::Cos)) && args.len() == 1 {
                                let arg = args[0];
                                // (1 - sin^2(x))^(n/2)
                                let one = ctx.num(1);
                                let sin_x = ctx.call("sin", vec![arg]);
                                let two = ctx.num(2);
                                let sin_sq = ctx.add(Expr::Pow(sin_x, two));
                                let base_term = ctx.add(Expr::Sub(one, sin_sq));

                                let half_n = n / num_rational::BigRational::from_integer(2.into());

                                if half_n.is_one() {
                                    return Some(Rewrite::new(base_term).desc("cos^2(x) -> 1 - sin^2(x)"));
                                } else {
                                    let half_n_expr = ctx.add(Expr::Number(half_n));
                                    let new_expr = ctx.add(Expr::Pow(base_term, half_n_expr));
                                    return Some(Rewrite::new(new_expr).desc("cos^2k(x) -> (1 - sin^2(x))^k"));
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
);
