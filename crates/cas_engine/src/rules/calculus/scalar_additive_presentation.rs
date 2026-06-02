use cas_ast::{Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::scalar_presentation::rational_const_for_calculus_presentation;

pub(super) fn add_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    add_rational_for_calculus_presentation(ctx, expr, BigRational::one())
}

pub(super) fn subtract_from_one_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let raw = ctx.add(Expr::Sub(one, expr));
    bounded_polynomial_additive_presentation(ctx, raw)
}

pub(super) fn subtract_from_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Sub(constant, expr));
    bounded_polynomial_additive_presentation(ctx, raw)
}

pub(super) fn subtract_expr_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let raw = ctx.add(Expr::Sub(left, right));
    bounded_polynomial_additive_presentation(ctx, raw)
}

pub(super) fn add_rational_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    if value.is_zero() {
        return expr;
    }

    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Add(expr, constant));
    bounded_polynomial_additive_presentation(ctx, raw)
}

pub(super) fn add_rational_combining_additive_constant_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return add_rational_for_calculus_presentation(ctx, expr, value);
    }

    let mut constant = value;
    let mut saw_constant = false;
    let mut rebuilt_terms = Vec::new();
    for (term, sign) in terms {
        if let Some(term_value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            saw_constant = true;
            if sign == cas_math::expr_nary::Sign::Neg {
                constant -= term_value;
            } else {
                constant += term_value;
            }
            continue;
        }

        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_terms.push(ctx.add(Expr::Neg(term)));
        } else {
            rebuilt_terms.push(term);
        }
    }

    if !saw_constant {
        return add_rational_for_calculus_presentation(ctx, expr, constant);
    }
    if !constant.is_zero() {
        rebuilt_terms.push(rational_const_for_calculus_presentation(ctx, constant));
    }

    match rebuilt_terms.len() {
        0 => ctx.num(0),
        1 => rebuilt_terms[0],
        _ => cas_math::expr_nary::build_balanced_add(ctx, &rebuilt_terms),
    }
}

fn bounded_polynomial_additive_presentation(ctx: &mut Context, raw: ExprId) -> ExprId {
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}
