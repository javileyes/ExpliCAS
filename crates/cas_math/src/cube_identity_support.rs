//! Support utilities for sum/difference-of-cubes product identities.

use crate::expr_nary::{AddView, Sign};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeIdentityMatch {
    pub base: ExprId,
    pub constant_cubed: BigRational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SumDiffCubesContractionRewrite {
    pub rewritten: ExprId,
}

/// Return true when `a*b` matches one of:
/// - `(X + c) * (X^2 - c*X + c^2)`
/// - `(X - c) * (X^2 + c*X + c^2)`
pub fn is_cube_identity_product(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    try_extract_cube_identity(ctx, a, b).is_some()
}

/// Extract cube-identity components from a binomial/trinomial product.
///
/// The factor order is commutative (`a*b` or `b*a`).
pub fn try_extract_cube_identity(ctx: &Context, a: ExprId, b: ExprId) -> Option<CubeIdentityMatch> {
    try_extract_cube_identity_ordered(ctx, a, b)
        .or_else(|| try_extract_cube_identity_ordered(ctx, b, a))
}

/// Rewrite a cube-identity product into compact sum/difference-of-cubes form.
pub fn try_rewrite_sum_diff_cubes_product_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SumDiffCubesContractionRewrite> {
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(l, r) => (*l, *r),
        _ => return None,
    };
    let matched = try_extract_cube_identity(ctx, left, right)?;

    let three = ctx.add(Expr::Number(BigRational::from_integer(3.into())));
    let x_cubed = ctx.add(Expr::Pow(matched.base, three));
    let c_cubed_node = ctx.add(Expr::Number(matched.constant_cubed));
    let rewritten = ctx.add(Expr::Add(x_cubed, c_cubed_node));

    Some(SumDiffCubesContractionRewrite { rewritten })
}

fn try_extract_cube_identity_ordered(
    ctx: &Context,
    binomial: ExprId,
    trinomial: ExprId,
) -> Option<CubeIdentityMatch> {
    let (base, c_value, is_sum) = extract_binomial_components(ctx, binomial)?;
    if !matches_cube_identity_trinomial(ctx, trinomial, base, &c_value, is_sum) {
        return None;
    }

    let constant_cubed = &c_value * &c_value * &c_value;
    Some(CubeIdentityMatch {
        base,
        constant_cubed,
    })
}

fn extract_binomial_components(
    ctx: &Context,
    binomial: ExprId,
) -> Option<(ExprId, BigRational, bool)> {
    match ctx.get(binomial) {
        Expr::Add(l, r) => {
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), true))
            } else if let Expr::Number(n) = ctx.get(*l) {
                Some((*r, n.clone(), true))
            } else {
                None
            }
        }
        Expr::Sub(l, r) => {
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), false))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_square_of_base(ctx: &Context, term: ExprId, base: ExprId) -> bool {
    let Expr::Pow(pow_base, exp) = ctx.get(term) else {
        return false;
    };
    if compare_expr(ctx, *pow_base, base) != Ordering::Equal {
        return false;
    }
    matches!(
        ctx.get(*exp),
        Expr::Number(n) if *n == BigRational::from_integer(2.into())
    )
}

fn matches_middle_term(
    ctx: &Context,
    term: ExprId,
    is_negative: bool,
    base: ExprId,
    c_value: &BigRational,
    is_sum: bool,
) -> bool {
    let c_is_negative = c_value.is_negative();
    let expected_negative = is_sum ^ c_is_negative;
    if is_negative != expected_negative {
        return false;
    }

    let c_abs = c_value.abs();
    if c_abs.is_one() && compare_expr(ctx, term, base) == Ordering::Equal {
        return true;
    }

    let Expr::Mul(l, r) = ctx.get(term) else {
        return false;
    };

    if let Expr::Number(n) = ctx.get(*l) {
        if *n == c_abs && compare_expr(ctx, *r, base) == Ordering::Equal {
            return true;
        }
    }
    if let Expr::Number(n) = ctx.get(*r) {
        if *n == c_abs && compare_expr(ctx, *l, base) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn matches_cube_identity_trinomial(
    ctx: &Context,
    trinomial: ExprId,
    base: ExprId,
    c_value: &BigRational,
    is_sum: bool,
) -> bool {
    let terms = AddView::from_expr(ctx, trinomial).terms;
    if terms.len() != 3 {
        return false;
    }

    let c_squared = c_value * c_value;
    let mut found_base_square = false;
    let mut found_middle = false;
    let mut found_constant_square = false;

    for (term, sign) in terms {
        let is_negative = sign == Sign::Neg;

        if !found_base_square && !is_negative && is_square_of_base(ctx, term, base) {
            found_base_square = true;
            continue;
        }

        if !found_middle && matches_middle_term(ctx, term, is_negative, base, c_value, is_sum) {
            found_middle = true;
            continue;
        }

        if !found_constant_square && !is_negative {
            if let Expr::Number(n) = ctx.get(term) {
                if *n == c_squared {
                    found_constant_square = true;
                    continue;
                }
            }
        }
    }

    found_base_square && found_middle && found_constant_square
}

#[cfg(test)]
mod tests {
    use super::{
        is_cube_identity_product, try_extract_cube_identity,
        try_rewrite_sum_diff_cubes_product_expr,
    };
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn extracts_sum_cube_identity() {
        let mut ctx = Context::new();
        let lhs = parse("x + 1", &mut ctx).expect("parse");
        let rhs = parse("x^2 - x + 1", &mut ctx).expect("parse");
        let matched = try_extract_cube_identity(&ctx, lhs, rhs).expect("match");

        assert_eq!(
            matched.constant_cubed,
            num_rational::BigRational::from_integer(1.into())
        );
    }

    #[test]
    fn extracts_difference_cube_identity() {
        let mut ctx = Context::new();
        let lhs = parse("x - 2", &mut ctx).expect("parse");
        let rhs = parse("x^2 + 2*x + 4", &mut ctx).expect("parse");
        let matched = try_extract_cube_identity(&ctx, lhs, rhs).expect("match");

        assert_eq!(
            matched.constant_cubed,
            num_rational::BigRational::from_integer(8.into())
        );
    }

    #[test]
    fn rejects_non_identity_product() {
        let mut ctx = Context::new();
        let lhs = parse("x + 1", &mut ctx).expect("parse");
        let rhs = parse("x^2 + x + 1", &mut ctx).expect("parse");
        assert!(!is_cube_identity_product(&ctx, lhs, rhs));
    }

    #[test]
    fn rewrites_sum_cube_identity_product() {
        let mut ctx = Context::new();
        let expr = parse("(x + 2) * (x^2 - 2*x + 4)", &mut ctx).expect("parse");
        let expected = parse("x^3 + 8", &mut ctx).expect("parse");
        let rewrite = try_rewrite_sum_diff_cubes_product_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }
}
