//! Builders for addition of fraction pairs.

use crate::build::mul2_raw;
use crate::expr_predicates::{is_minus_one_expr, is_one_expr};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct AddFractionBuildResult {
    pub numerator: ExprId,
    pub denominator: ExprId,
}

/// Build `(n1/d1) + (n2/d2)` as a single fraction.
///
/// When `related_denominators` is true, uses `(n1 + n2) / common_den`.
/// Otherwise uses `(n1*d2 + n2*d1) / (d1*d2)` with `1*x` and `-1*x` shortcuts.
pub fn build_add_fraction_rewrite(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
    common_den: ExprId,
    related_denominators: bool,
) -> AddFractionBuildResult {
    if related_denominators {
        return AddFractionBuildResult {
            numerator: ctx.add(Expr::Add(n1, n2)),
            denominator: common_den,
        };
    }

    let ad = if is_one_expr(ctx, n1) {
        d2
    } else if is_minus_one_expr(ctx, n1) {
        ctx.add(Expr::Neg(d2))
    } else {
        mul2_raw(ctx, n1, d2)
    };
    let bc = if is_one_expr(ctx, n2) {
        d1
    } else if is_minus_one_expr(ctx, n2) {
        ctx.add(Expr::Neg(d1))
    } else {
        mul2_raw(ctx, n2, d1)
    };

    AddFractionBuildResult {
        numerator: ctx.add(Expr::Add(ad, bc)),
        denominator: mul2_raw(ctx, d1, d2),
    }
}

#[cfg(test)]
mod tests {
    use super::build_add_fraction_rewrite;
    use crate::poly_compare::poly_eq;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn builds_same_denominator_addition() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("b", &mut ctx).expect("parse");
        let d = parse("x+1", &mut ctx).expect("parse");
        let built = build_add_fraction_rewrite(&mut ctx, n1, n2, d, d, d, true);
        let expected_num = parse("a+b", &mut ctx).expect("parse");
        assert!(poly_eq(&ctx, built.numerator, expected_num));
        assert_eq!(compare_expr(&ctx, built.denominator, d), Ordering::Equal);
    }

    #[test]
    fn builds_cross_product_addition() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse");
        let n2 = parse("c", &mut ctx).expect("parse");
        let d1 = parse("b", &mut ctx).expect("parse");
        let d2 = parse("d", &mut ctx).expect("parse");
        let built = build_add_fraction_rewrite(&mut ctx, n1, n2, d1, d2, d1, false);
        let expected_num = parse("a*d+c*b", &mut ctx).expect("parse");
        let expected_den = parse("b*d", &mut ctx).expect("parse");
        assert!(poly_eq(&ctx, built.numerator, expected_num));
        assert!(poly_eq(&ctx, built.denominator, expected_den));
    }
}
