use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};

pub(super) fn polynomial_times_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target) else {
        return None;
    };

    let mut polynomial_factors = Vec::new();
    let mut radicand = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if let Some(factor_radicand) = extract_square_root_base(ctx, factor) {
            if radicand.replace(factor_radicand).is_some() {
                return None;
            }
        } else {
            polynomial_factors.push(factor);
        }
    }

    let radicand = radicand?;
    if polynomial_factors.is_empty() {
        return None;
    }

    let polynomial_expr = if polynomial_factors.len() == 1 {
        polynomial_factors[0]
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &polynomial_factors)
    };
    let multiplier_poly =
        polynomial_radicand_for_calculus_presentation(ctx, polynomial_expr, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let multiplier_derivative = multiplier_poly.derivative();
    let radicand_derivative = radicand_poly.derivative();

    let two_poly = Polynomial::new(
        vec![BigRational::from_integer(2.into())],
        var_name.to_string(),
    );
    let numerator_poly = multiplier_derivative
        .mul(&radicand_poly)
        .mul(&two_poly)
        .add(&multiplier_poly.mul(&radicand_derivative));
    if numerator_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let raw_numerator = numerator_poly.to_expr(ctx);
    let (numerator_core, numerator_content) =
        split_polynomial_content_for_calculus_presentation(ctx, raw_numerator);
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_content * BigRational::new(1.into(), 2.into())))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_radicand])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::polynomial_times_sqrt_polynomial_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn polynomial_times_sqrt_polynomial_presentation_handles_single_multiplier() {
        let mut ctx = Context::new();
        let expr = parse("x*sqrt(x)", &mut ctx).unwrap();
        let compact = polynomial_times_sqrt_polynomial_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("polynomial times sqrt polynomial should be recognized"));

        assert_eq!(rendered(&ctx, compact), "3 * x / (2 * sqrt(x))");
    }

    #[test]
    fn polynomial_times_sqrt_polynomial_presentation_handles_product_multiplier() {
        let mut ctx = Context::new();
        let expr = parse("(x+1)*(x+2)*sqrt(x+3)", &mut ctx).unwrap();
        let compact = polynomial_times_sqrt_polynomial_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("polynomial times sqrt polynomial should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "(5 * x^2 + 21 * x + 20) / (2 * sqrt(x + 3))"
        );
    }
}
