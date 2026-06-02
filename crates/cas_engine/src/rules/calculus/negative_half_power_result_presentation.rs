use super::presentation_utils::{
    is_calculus_presentation_one, negative_half_power_base_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) fn compact_negative_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new((-1).into(), 2.into()) {
            let one = ctx.num(1);
            let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
            return Some(ctx.add(Expr::Div(one, sqrt)));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_result_for_integration_presentation(ctx, inner)?;
        if let Expr::Div(num, den) = ctx.get(compact).clone() {
            let numerator = ctx.add(Expr::Neg(num));
            return Some(ctx.add(Expr::Div(numerator, den)));
        }
        return Some(ctx.add(Expr::Neg(compact)));
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
                if exponent != BigRational::new((-1).into(), 2.into()) || base.is_some() {
                    return None;
                }
                base = Some(*pow_base);
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let base = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let numerator = ctx.add(Expr::Number(coefficient));
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    Some(ctx.add(Expr::Div(numerator, sqrt)))
}

pub(super) fn compact_negative_half_power_product_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = compact_negative_half_power_product_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_fraction_product_factors_for_calculus_presentation(
        ctx,
        expr,
        &mut numerator_factors,
        &mut denominator_factors,
    );

    let mut base = None;
    let mut retained_numerator = Vec::new();
    for factor in numerator_factors {
        if base.is_none() {
            if let Some(pow_base) = negative_half_power_base_for_calculus_presentation(ctx, factor)
            {
                base = Some(pow_base);
                continue;
            }
        }
        retained_numerator.push(factor);
    }

    let base = base?;
    denominator_factors.push(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));

    let one = ctx.num(1);
    let numerator = if retained_numerator.is_empty() {
        one
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &retained_numerator)
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

fn collect_fraction_product_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                left,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                right,
                numerator_factors,
                denominator_factors,
            );
        }
        Expr::Div(num, den) => {
            collect_fraction_product_factors_for_calculus_presentation(
                ctx,
                num,
                numerator_factors,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                den,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            numerator_factors,
            denominator_factors,
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => numerator_factors.push(expr),
        _ => {}
    }
}

fn collect_fraction_denominator_factors_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    denominator_factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                left,
                denominator_factors,
            );
            collect_fraction_denominator_factors_for_calculus_presentation(
                ctx,
                right,
                denominator_factors,
            );
        }
        _ if split_rational_factor_for_calculus_presentation(
            ctx,
            expr,
            denominator_factors,
            &mut Vec::new(),
        ) => {}
        _ if !is_calculus_presentation_one(ctx, expr) => denominator_factors.push(expr),
        _ => {}
    }
}

fn split_rational_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) -> bool {
    let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) else {
        return false;
    };
    if value.is_zero() {
        numerator_factors.push(expr);
        return true;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    if !numerator.is_one() {
        numerator_factors.push(ctx.add(Expr::Number(numerator)));
    }
    if !denominator.is_one() {
        denominator_factors.push(ctx.add(Expr::Number(denominator)));
    }
    true
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::scalar_presentation::fold_numeric_mul_constants_for_hold;
    use super::{
        compact_negative_half_power_product_for_calculus_presentation,
        compact_negative_half_power_result_for_integration_presentation,
    };

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_negative_half_power_result_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("-2*(x^2+x+1)^(-1/2)", &mut ctx).unwrap();
        let compact =
            compact_negative_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "-2 / sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_negative_half_power_product_for_calculus_presentation_uses_sqrt_denominator() {
        let cases = [
            (
                "cos(x)/2*(sin(x)+1)^(1/2-1)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            (
                "((ln(x)+1)^(1/2-1)/2)*(1/x)",
                "1 / (2 * x * sqrt(ln(x) + 1))",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact =
                compact_negative_half_power_product_for_calculus_presentation(&mut ctx, expr)
                    .unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }
}
