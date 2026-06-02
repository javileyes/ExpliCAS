use cas_ast::{Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn positive_shift_denominator_scale(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly.coeffs.first()?.clone();
        let slope = poly.coeffs.get(1)?;
        if !slope.is_positive() {
            return None;
        }
        let candidate_offset = constant / slope.clone();
        if !candidate_offset.is_positive() {
            return None;
        }
        scale *= slope.clone();
        if offset.replace(candidate_offset).is_some() {
            return None;
        }
    }
    Some((scale, offset?))
}

pub(super) fn x_plus_one_linear_scale_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let offset = poly.coeffs.first()?;
    let slope = poly.coeffs.get(1)?;
    (offset == slope).then_some(offset.clone())
}

pub(super) fn is_calculus_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

pub(super) fn is_x_plus_one_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok();
    poly.is_some_and(|poly| {
        poly.degree() == 1
            && poly
                .coeffs
                .first()
                .is_some_and(|offset| offset == &BigRational::one())
            && poly
                .coeffs
                .get(1)
                .is_some_and(|slope| slope == &BigRational::one())
    })
}
