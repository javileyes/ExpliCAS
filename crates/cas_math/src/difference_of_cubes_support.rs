//! Support for detecting and planning cube-root difference cancellations.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelCubeRootDifferencePlan {
    pub numerator: ExprId,
    pub denominator: ExprId,
    pub factored_numerator: ExprId,
    pub intermediate: ExprId,
    pub final_factor: ExprId,
    pub cube_value: BigRational,
    pub b_value: BigRational,
}

/// Plan the factor-and-cancel rewrite:
/// `(x - b^3) / (x^(2/3) + b*x^(1/3) + b^2)` -> `x^(1/3) - b`.
pub fn try_plan_cancel_cube_root_difference_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<CancelCubeRootDifferencePlan> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(n, d) => (*n, *d),
        _ => return None,
    };

    let (base_x, cube_value) = match_x_minus_const(ctx, numerator)?;
    let b_value = perfect_cube_root(&cube_value)?;
    let b_squared = &b_value * &b_value;

    let denominator_terms = collect_add_terms(ctx, denominator);
    if denominator_terms.len() != 3 {
        return None;
    }

    let mut found_x_two_thirds = false;
    let mut found_cbrt_term = false;
    let mut found_const = false;

    for term in &denominator_terms {
        if let Some(base) = extract_cbrt_squared_base(ctx, *term) {
            if compare_expr(ctx, base, base_x) == Ordering::Equal {
                found_x_two_thirds = true;
                continue;
            }
        }

        if let Some(base) = extract_cbrt_base(ctx, *term) {
            if compare_expr(ctx, base, base_x) == Ordering::Equal && b_value == BigRational::one() {
                found_cbrt_term = true;
                continue;
            }
        }

        if let Expr::Mul(l, r) = ctx.get(*term) {
            if let Some(coeff) = get_rational(ctx, *l) {
                if let Some(base) = extract_cbrt_base(ctx, *r) {
                    if compare_expr(ctx, base, base_x) == Ordering::Equal && coeff == b_value {
                        found_cbrt_term = true;
                        continue;
                    }
                }
            }
            if let Some(coeff) = get_rational(ctx, *r) {
                if let Some(base) = extract_cbrt_base(ctx, *l) {
                    if compare_expr(ctx, base, base_x) == Ordering::Equal && coeff == b_value {
                        found_cbrt_term = true;
                        continue;
                    }
                }
            }
        }

        if let Some(value) = get_rational(ctx, *term) {
            if value == b_squared {
                found_const = true;
                continue;
            }
        }
    }

    if !found_x_two_thirds || !found_cbrt_term || !found_const {
        return None;
    }

    let one_third = ctx.add(Expr::Number(BigRational::new(
        BigInt::from(1),
        BigInt::from(3),
    )));
    let cbrt_x = ctx.add(Expr::Pow(base_x, one_third));
    let b_expr = ctx.add(Expr::Number(b_value.clone()));
    let neg_b = ctx.add(Expr::Neg(b_expr));
    let final_factor = ctx.add(Expr::Add(cbrt_x, neg_b));

    let factored_numerator = ctx.add(Expr::Mul(final_factor, denominator));
    let intermediate = ctx.add(Expr::Div(factored_numerator, denominator));

    Some(CancelCubeRootDifferencePlan {
        numerator,
        denominator,
        factored_numerator,
        intermediate,
        final_factor,
        cube_value,
        b_value,
    })
}

fn is_one_third(ctx: &Context, exp: ExprId) -> bool {
    match ctx.get(exp) {
        Expr::Number(n) => n.numer() == &BigInt::from(1) && n.denom() == &BigInt::from(3),
        Expr::Div(num, den) => {
            let Expr::Number(num_n) = ctx.get(*num) else {
                return false;
            };
            let Expr::Number(den_n) = ctx.get(*den) else {
                return false;
            };
            num_n == &BigRational::from_integer(1.into())
                && den_n == &BigRational::from_integer(3.into())
        }
        _ => false,
    }
}

fn is_two_thirds(ctx: &Context, exp: ExprId) -> bool {
    match ctx.get(exp) {
        Expr::Number(n) => n.numer() == &BigInt::from(2) && n.denom() == &BigInt::from(3),
        Expr::Div(num, den) => {
            let Expr::Number(num_n) = ctx.get(*num) else {
                return false;
            };
            let Expr::Number(den_n) = ctx.get(*den) else {
                return false;
            };
            num_n == &BigRational::from_integer(2.into())
                && den_n == &BigRational::from_integer(3.into())
        }
        _ => false,
    }
}

fn extract_cbrt_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if is_one_third(ctx, *exp) {
            return Some(*base);
        }
    }
    None
}

fn extract_cbrt_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if is_two_thirds(ctx, *exp) {
            return Some(*base);
        }
    }
    None
}

fn get_rational(ctx: &Context, id: ExprId) -> Option<BigRational> {
    if let Expr::Number(n) = ctx.get(id) {
        Some(n.clone())
    } else {
        None
    }
}

fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.get(expr) {
        Expr::Add(a, b) => {
            let mut left = collect_add_terms(ctx, *a);
            left.extend(collect_add_terms(ctx, *b));
            left
        }
        _ => vec![expr],
    }
}

fn match_x_minus_const(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    if let Expr::Sub(left, right) = ctx.get(expr) {
        if let Expr::Variable(_) = ctx.get(*left) {
            if let Expr::Number(n) = ctx.get(*right) {
                return Some((*left, n.clone()));
            }
        }
    }

    let terms = collect_add_terms(ctx, expr);
    if terms.len() != 2 {
        return None;
    }

    let mut variable_term: Option<ExprId> = None;
    let mut neg_const: Option<BigRational> = None;

    for term in terms {
        match ctx.get(term) {
            Expr::Variable(_) => variable_term = Some(term),
            Expr::Neg(inner) => {
                if let Expr::Number(n) = ctx.get(*inner) {
                    neg_const = Some(n.clone());
                }
            }
            Expr::Number(n) if n.is_negative() => neg_const = Some(-n.clone()),
            _ => {}
        }
    }

    Some((variable_term?, neg_const?))
}

fn perfect_cube_root(value: &BigRational) -> Option<BigRational> {
    if !value.is_integer() {
        return None;
    }

    let integer_value = value.to_integer();
    for b in 1..=100i32 {
        let cube = BigInt::from(b).pow(3);
        if cube == integer_value {
            return Some(BigRational::from_integer(BigInt::from(b)));
        }
        if cube == -integer_value.clone() {
            return Some(BigRational::from_integer(BigInt::from(-b)));
        }
    }
    None
}

/// Exact rational cube root: `Some(∛r)` only when the numerator and denominator are BOTH perfect
/// cubes, carrying the real sign (`∛(-8) = -2`, the real odd root). `None` otherwise, so `∛2`,
/// `∛9`, `∛16` (a non-perfect cube) stay symbolic. EXACT — verified by re-cubing the candidate
/// roots, never an `f64` estimate.
pub(crate) fn rational_cbrt(r: &BigRational) -> Option<BigRational> {
    if r.is_zero() {
        return Some(BigRational::from_integer(0.into()));
    }
    let negative = r.is_negative();
    let numer = r.numer().abs();
    let denom = r.denom().clone(); // a normalized BigRational has a positive denominator
    let numer_root = numer.cbrt();
    if &numer_root * &numer_root * &numer_root != numer {
        return None;
    }
    let denom_root = denom.cbrt();
    if &denom_root * &denom_root * &denom_root != denom {
        return None;
    }
    let root = BigRational::new(numer_root, denom_root);
    Some(if negative { -root } else { root })
}

/// Fold `cbrt(n)` to its exact rational value when `n` is a perfect-cube rational
/// (`cbrt(8) → 2`, `cbrt(-8) → -2`, `cbrt(8/27) → 2/3`, `cbrt(1) → 1`, `cbrt(0) → 0`); a
/// non-cube argument (`cbrt(2)`, `cbrt(16)`) is left symbolic. Mirrors the perfect-square
/// folding of `sqrt`, closing the gap where `root(8,3) = 2` evaluated but `cbrt(8)` did not.
pub fn try_rewrite_cbrt_perfect_cube_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let arg = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Cbrt) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };
    let value = crate::numeric_eval::as_rational_const(ctx, arg)?;
    let root = rational_cbrt(&value)?;
    Some(ctx.add(Expr::Number(root)))
}

#[cfg(test)]
mod tests {
    use super::{rational_cbrt, try_plan_cancel_cube_root_difference_expr};
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;
    use num_rational::BigRational;

    #[test]
    fn rational_cbrt_folds_perfect_cubes_and_keeps_non_cubes_symbolic() {
        let r = |n: i64, d: i64| BigRational::new(n.into(), d.into());
        // Perfect cubes (incl. negative and rational) resolve exactly.
        assert_eq!(rational_cbrt(&r(8, 1)), Some(r(2, 1)));
        assert_eq!(rational_cbrt(&r(27, 1)), Some(r(3, 1)));
        assert_eq!(rational_cbrt(&r(-8, 1)), Some(r(-2, 1)));
        assert_eq!(rational_cbrt(&r(64, 1)), Some(r(4, 1)));
        assert_eq!(rational_cbrt(&r(1000, 1)), Some(r(10, 1)));
        assert_eq!(rational_cbrt(&r(8, 27)), Some(r(2, 3)));
        assert_eq!(rational_cbrt(&r(-27, 8)), Some(r(-3, 2)));
        assert_eq!(rational_cbrt(&r(0, 1)), Some(r(0, 1)));
        assert_eq!(rational_cbrt(&r(1, 1)), Some(r(1, 1)));
        // Non-cubes (incl. a partial cube `16 = 2³·2`) stay symbolic.
        assert_eq!(rational_cbrt(&r(2, 1)), None);
        assert_eq!(rational_cbrt(&r(9, 1)), None);
        assert_eq!(rational_cbrt(&r(16, 1)), None);
        assert_eq!(rational_cbrt(&r(8, 9)), None); // numerator cube, denominator not
    }

    #[test]
    fn plan_cancel_cube_root_difference_matches_basic_shape() {
        let mut ctx = Context::new();
        let expr = parse("(x - 27) / (x^(2/3) + 3*x^(1/3) + 9)", &mut ctx).expect("parse");
        let plan =
            try_plan_cancel_cube_root_difference_expr(&mut ctx, expr).expect("expected plan");

        let final_text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: plan.final_factor
            }
        );
        assert!(final_text.contains("x"));
        assert!(final_text.contains("1/3"));
        assert_eq!(
            plan.b_value,
            num_rational::BigRational::from_integer(3.into())
        );
        assert_eq!(
            plan.cube_value,
            num_rational::BigRational::from_integer(27.into())
        );
    }

    #[test]
    fn plan_cancel_cube_root_difference_matches_unit_coefficient_variant() {
        let mut ctx = Context::new();
        let expr = parse("(x - 1) / (x^(2/3) + x^(1/3) + 1)", &mut ctx).expect("parse");
        let plan =
            try_plan_cancel_cube_root_difference_expr(&mut ctx, expr).expect("expected plan");

        let final_text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: plan.final_factor
            }
        );
        assert!(final_text.contains("x"));
        assert!(final_text.contains("1/3"));
        assert_eq!(
            plan.b_value,
            num_rational::BigRational::from_integer(1.into())
        );
    }

    #[test]
    fn plan_cancel_cube_root_difference_rejects_non_matching_denominator() {
        let mut ctx = Context::new();
        let expr = parse("(x - 27) / (x^(2/3) + 9)", &mut ctx).expect("parse");
        assert!(try_plan_cancel_cube_root_difference_expr(&mut ctx, expr).is_none());
    }
}
