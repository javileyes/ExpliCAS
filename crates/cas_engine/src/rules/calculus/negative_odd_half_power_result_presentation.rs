use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation, strictly_positive_quadratic_on_reals,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

pub(super) fn compact_negative_three_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if matches_supported_negative_odd_half_power(&exponent) {
            if !quadratic_for_fractional_power_calculus_presentation(
                ctx,
                base,
                var_name,
                allow_conditional_positive_quadratic,
            ) {
                return None;
            }
            let one = ctx.num(1);
            let denominator =
                negative_odd_half_power_denominator_for_presentation(ctx, base, -exponent)?;
            return Some(ctx.add(Expr::Div(one, denominator)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            ctx,
            inner,
            var_name,
            allow_conditional_positive_quadratic,
        )?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let numerator_value = cas_ast::views::as_rational_const(ctx, num, 8)?;
        let mut denominator_scale = BigRational::one();
        let mut base = None;
        for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
            if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                denominator_scale *= value;
                continue;
            }
            let Expr::Pow(pow_base, exp) = ctx.get(factor).clone() else {
                return None;
            };
            let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
            if !matches_supported_positive_odd_half_power(&exponent) || base.is_some() {
                return None;
            }
            base = Some((pow_base, exponent));
        }
        if denominator_scale.is_zero() {
            return None;
        }
        let (base, denominator_power) = base?;
        if !quadratic_for_fractional_power_calculus_presentation(
            ctx,
            base,
            var_name,
            allow_conditional_positive_quadratic,
        ) {
            return None;
        }
        let coefficient = numerator_value / denominator_scale;
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let one = ctx.num(1);
        let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
        let mut denominator_parts = Vec::new();
        if !denominator_coeff.is_one() {
            denominator_parts.push(rational_const_for_calculus_presentation(
                ctx,
                denominator_coeff,
            ));
        }
        denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
            ctx,
            base,
            denominator_power,
        )?);
        let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };
    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut coefficient = BigRational::one();
    let mut base = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Pow(pow_base, exp) => {
                let exponent = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
                if !matches_supported_negative_odd_half_power(&exponent) || base.is_some() {
                    return None;
                }
                base = Some((*pow_base, -exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, denominator_power) = base?;
    if !quadratic_for_fractional_power_calculus_presentation(
        ctx,
        base,
        var_name,
        allow_conditional_positive_quadratic,
    ) {
        return None;
    }
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let one = ctx.num(1);
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, one);
    let mut denominator_parts = Vec::new();
    if !denominator_coeff.is_one() {
        denominator_parts.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_parts.push(negative_odd_half_power_denominator_for_presentation(
        ctx,
        base,
        denominator_power,
    )?);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_parts);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn matches_supported_negative_odd_half_power(exponent: &BigRational) -> bool {
    exponent.is_negative() && odd_half_denominator_base_power(&(-exponent.clone())).is_some()
}

fn matches_supported_positive_odd_half_power(exponent: &BigRational) -> bool {
    odd_half_denominator_base_power(exponent).is_some()
}

fn quadratic_for_fractional_power_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    allow_conditional_positive_quadratic: bool,
) -> bool {
    if strictly_positive_quadratic_for_calculus_presentation(ctx, expr, var_name) {
        return true;
    }
    if !allow_conditional_positive_quadratic {
        return false;
    }
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    if poly.degree() != 2 || poly.coeffs.len() < 3 {
        return false;
    }
    let leading = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    leading.is_positive()
}

fn negative_odd_half_power_denominator_for_presentation(
    ctx: &mut Context,
    base: ExprId,
    denominator_power: BigRational,
) -> Option<ExprId> {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    let base_power = odd_half_denominator_base_power(&denominator_power)?;
    let base_factor = if base_power == 1 {
        base
    } else {
        let exponent = ctx.num(base_power);
        ctx.add(Expr::Pow(base, exponent))
    };
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[base_factor, sqrt],
    ))
}

fn odd_half_denominator_base_power(denominator_power: &BigRational) -> Option<i64> {
    if denominator_power.denom() != &BigInt::from(2) {
        return None;
    }
    let numerator = denominator_power.numer();
    if !numerator.is_positive() || !numerator.is_odd() {
        return None;
    }
    let numerator = numerator.to_i64()?;
    if numerator < 3 {
        return None;
    }
    Some((numerator - 1) / 2)
}

fn strictly_positive_quadratic_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let Some(poly) = polynomial_radicand_for_calculus_presentation(ctx, expr, var_name) else {
        return false;
    };
    strictly_positive_quadratic_on_reals(&poly)
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::scalar_presentation::fold_numeric_mul_constants_for_hold;
    use super::compact_negative_three_half_power_result_for_integration_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2+x+1)^(3/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }

    #[test]
    fn compact_negative_five_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(5*(x^2+x+1)^(5/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (5 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn compact_negative_seven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(7*(x^2+x+1)^(7/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (7 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^3)"
        );
    }

    #[test]
    fn compact_negative_nine_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(9*(x^2+x+1)^(9/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (9 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^4)"
        );
    }

    #[test]
    fn compact_negative_eleven_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("-2/(11*(x^2+x+1)^(11/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (11 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^5)"
        );
    }

    #[test]
    fn compact_negative_thirteen_half_power_result_for_integration_presentation_uses_sqrt_product()
    {
        let mut ctx = Context::new();
        let expr = parse("-2/(13*(x^2+x+1)^(13/2))", &mut ctx).unwrap();
        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", false,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (13 * sqrt(x^2 + x + 1) * (x^2 + x + 1)^6)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_requires_conditional_domain_signal() {
        let mut ctx = Context::new();
        let expr = parse("-2/(3*(x^2-1)^(3/2))", &mut ctx).unwrap();

        assert!(
            compact_negative_three_half_power_result_for_integration_presentation(
                &mut ctx, expr, "x", false,
            )
            .is_none()
        );

        let compact = compact_negative_three_half_power_result_for_integration_presentation(
            &mut ctx, expr, "x", true,
        )
        .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-2 / (3 * sqrt(x^2 - 1) * (x^2 - 1))"
        );
    }
}
