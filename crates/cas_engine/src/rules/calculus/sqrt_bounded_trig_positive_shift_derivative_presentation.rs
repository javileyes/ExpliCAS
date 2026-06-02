use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::differentiation::differentiate;
use super::presentation_compaction::{
    bounded_sin_cos_shift_margin_for_calculus_presentation,
    compact_double_angle_sine_products_for_calculus_presentation,
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
    distribute_half_over_additive_numerator_for_calculus_presentation,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, split_numeric_scale_single_core,
};

pub(super) fn sqrt_bounded_trig_positive_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, radicand)?;
    let presentation_radicand =
        compact_double_angle_sine_products_for_calculus_presentation(ctx, radicand)
            .filter(|candidate| {
                bounded_sin_cos_shift_margin_for_calculus_presentation(ctx, *candidate).is_some()
            })
            .unwrap_or(radicand);

    let derivative = differentiate(ctx, presentation_radicand, var_name)?;
    let derivative = compact_small_power_exponents_for_calculus_presentation(ctx, derivative);
    let derivative = compact_numeric_mul_factors_for_calculus_presentation(ctx, derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
    let distributed_numerator = if coefficient == BigRational::new(1.into(), 2.into()) {
        distribute_half_over_additive_numerator_for_calculus_presentation(ctx, derivative_core)
    } else {
        None
    };
    let (numerator, denominator_coeff) = if let Some(numerator) = distributed_numerator {
        (numerator, BigRational::one())
    } else {
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        (
            scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core),
            denominator_coeff,
        )
    };
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![presentation_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_bounded_trig_positive_shift_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn bounded_trig_positive_shift_sqrt_derivative_presentation_accepts_multi_function_sum() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(cos(x)+2*sin(x)*cos(x)+4)", &mut ctx).unwrap();
        let compact = sqrt_bounded_trig_positive_shift_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("positive shifted bounded trig root should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(cos(2 * x) - 1/2 * sin(x)) / sqrt(sin(2 * x) + cos(x) + 4)"
        );
    }
}
