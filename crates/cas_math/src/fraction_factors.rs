//! Structural factor helpers for fraction/rationalization workflows.
//!
//! These utilities operate on AST structure only (no polynomial expansion).

use cas_ast::{Context, Expr, ExprId};

/// Collect multiplicative factors with integer exponents from an expression.
///
/// Rules:
/// - `Mul(...)` is flattened
/// - `Pow(base, k)` with integer `k` becomes `(base, k)`
/// - top-level `Neg(x)` is unwrapped for intersection purposes
/// - everything else becomes `(expr, 1)`
pub fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    let actual_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = integer_exponent(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => factors.push((expr, mult)),
    }
}

fn integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Build a product from factors with integer exponents.
///
/// Negative exponents are ignored (the caller typically manages denominator
/// factors separately).
pub fn build_mul_from_factors_int_pow(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    use cas_ast::views::MulBuilder;

    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
    }
    builder.build(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn collect_strips_top_level_neg() {
        let mut ctx = Context::new();
        let expr = parse("-(x*y^2)", &mut ctx).expect("parse");
        let factors = collect_mul_factors_int_pow(&ctx, expr);

        assert_eq!(factors.len(), 2);
        let mut exponents: Vec<i64> = factors.iter().map(|(_, e)| *e).collect();
        exponents.sort_unstable();
        assert_eq!(exponents, vec![1, 2]);
    }

    #[test]
    fn collect_recognizes_negative_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse("x^(-3)", &mut ctx).expect("parse");
        let factors = collect_mul_factors_int_pow(&ctx, expr);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, -3);
    }

    #[test]
    fn build_product_from_factors() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");
        let factors = vec![(x, 2), (y, 1), (x, -1)];

        let built = build_mul_from_factors_int_pow(&mut ctx, &factors);
        let expected = parse("x^2*y", &mut ctx).expect("parse expected");
        assert!(poly_eq(&ctx, built, expected));
    }
}
