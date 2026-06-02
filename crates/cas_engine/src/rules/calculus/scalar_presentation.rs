use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) use super::scalar_additive_presentation::{
    add_one_for_calculus_presentation,
    add_rational_combining_additive_constant_for_calculus_presentation,
    add_rational_for_calculus_presentation, subtract_expr_for_calculus_presentation,
    subtract_from_one_for_calculus_presentation, subtract_from_rational_for_calculus_presentation,
};
pub(super) use super::scalar_positive_quotient_presentation::positive_constant_over_inverse_sqrt_arg_for_calculus_presentation;

pub(super) use super::scalar_numeric_fold_presentation::{
    fold_numeric_mul_constants_for_hold, fold_numeric_mul_constants_for_hold_additive_terms,
};
pub(super) use super::scalar_rational_sqrt_presentation::{
    exact_positive_rational_sqrt_for_calculus_presentation,
    positive_rational_sqrt_denominator_factor_for_calculus_presentation,
    reciprocal_integer_radicand_content_for_calculus_presentation,
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    split_square_factor_positive_rational_for_calculus_presentation,
    sqrt_positive_rational_expr_for_calculus_presentation,
};
pub(super) use super::scalar_scale_presentation::{
    nonzero_rational_parts, rational_scaled_single_factor,
    rational_scaled_single_factor_allow_unit,
    scale_compact_fraction_numerator_by_rational_for_calculus_presentation,
    scale_fraction_for_calculus_presentation, signed_numerator_for_calculus_presentation,
    signed_rational_const_for_calculus_presentation, split_numeric_scale_single_core,
    split_outer_numeric_mul_for_calculus_presentation,
};

pub(super) fn rational_const_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if value == BigRational::one() {
        ctx.num(1)
    } else {
        ctx.add(Expr::Number(value))
    }
}

pub(super) fn scale_expr_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff.is_one() {
        return expr;
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if value == BigRational::one() {
            return coeff;
        }
        if let Some(coeff_value) = cas_ast::views::as_rational_const(ctx, coeff, 8) {
            return rational_const_for_calculus_presentation(ctx, coeff_value * value);
        }
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[coeff, expr])
}

pub(super) fn negate_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        Expr::Div(numerator, denominator) => {
            let numerator = match ctx.get(numerator).clone() {
                Expr::Neg(inner) => inner,
                _ => ctx.add(Expr::Neg(numerator)),
            };
            ctx.add(Expr::Div(numerator, denominator))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        fold_numeric_mul_constants_for_hold, fold_numeric_mul_constants_for_hold_additive_terms,
    };

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_collapses_rational_noise() {
        let mut ctx = Context::new();
        let expr = parse("(atanh(x^2/sqrt(3)) * 1/2 * 2)/sqrt(3)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_absorbs_outer_scale_into_quotient() {
        let mut ctx = Context::new();
        let expr = parse("2 * ((atanh(x^2/sqrt(3))/2)/sqrt(3))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "atanh(x^2 / sqrt(3)) / sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_extracts_scaled_sqrt_square_factor() {
        let mut ctx = Context::new();
        let expr = parse("25*sqrt(12/25)", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "10 * sqrt(3)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_keeps_fractional_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1/(3*(x^2+x-1))", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, expr);

        assert_eq!(rendered(&ctx, folded), "-1 / (3 * (x^2 + x - 1))");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_additive_terms_recurses_into_terms() {
        let mut ctx = Context::new();
        let expr = parse("1/2*ln(abs(x+1)) + 1/2*(x^2/2) - 1/2*x", &mut ctx).unwrap();
        let folded = fold_numeric_mul_constants_for_hold_additive_terms(&mut ctx, expr);

        assert_eq!(
            rendered(&ctx, folded),
            "1/2 * ln(|x + 1|) + 1/4 * x^2 - 1/2 * x"
        );
    }
}
