use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn compact_positive_half_power_result_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        let exponent = cas_ast::views::as_rational_const(ctx, exp, 8)?;
        if exponent == BigRational::new(1.into(), 2.into()) {
            return Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![base]));
        }
        if exponent == BigRational::new(3.into(), 2.into()) {
            return Some(product_with_sqrt_for_positive_three_half_power(ctx, base));
        }
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        if expression_contains_positive_three_half_power(ctx, inner) {
            return None;
        }
        let compact = compact_positive_half_power_result_for_integration_presentation(ctx, inner)?;
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
                if !(exponent == BigRational::new(1.into(), 2.into())
                    || exponent == BigRational::new(3.into(), 2.into()))
                    || base.is_some()
                {
                    return None;
                }
                base = Some((*pow_base, exponent));
            }
            _ => {
                coefficient *= cas_ast::views::as_rational_const(ctx, factor, 8)?;
            }
        }
    }

    let (base, exponent) = base?;
    if coefficient.is_zero() {
        return Some(ctx.num(0));
    }
    let core = if exponent == BigRational::new(1.into(), 2.into()) {
        ctx.call_builtin(BuiltinFn::Sqrt, vec![base])
    } else if exponent == BigRational::new(3.into(), 2.into()) {
        let product = product_with_sqrt_for_positive_three_half_power(ctx, base);
        return Some(scale_three_half_power_product_for_presentation(
            ctx,
            coefficient,
            product,
        ));
    } else {
        return None;
    };
    Some(scale_expr_for_calculus_presentation(ctx, coefficient, core))
}

fn expression_contains_positive_three_half_power(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Pow(_, exp) => cas_ast::views::as_rational_const(ctx, *exp, 8)
            .is_some_and(|exponent| exponent == BigRational::new(3.into(), 2.into())),
        Expr::Mul(_, _) => cas_math::expr_nary::mul_leaves(ctx, expr)
            .iter()
            .any(|factor| expression_contains_positive_three_half_power(ctx, *factor)),
        _ => false,
    }
}

fn product_with_sqrt_for_positive_three_half_power(ctx: &mut Context, base: ExprId) -> ExprId {
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
    cas_math::expr_nary::build_balanced_mul(ctx, &[base, sqrt])
}

fn scale_three_half_power_product_for_presentation(
    ctx: &mut Context,
    coefficient: BigRational,
    product: ExprId,
) -> ExprId {
    if coefficient.is_negative() {
        let positive = scale_expr_for_calculus_presentation(ctx, -coefficient, product);
        return ctx.add(Expr::Neg(positive));
    }
    scale_expr_for_calculus_presentation(ctx, coefficient, product)
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::scalar_presentation::fold_numeric_mul_constants_for_hold;
    use super::compact_positive_half_power_result_for_integration_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_positive_half_power_result_for_integration_presentation_uses_sqrt() {
        let mut ctx = Context::new();
        let expr = parse("2*(x^2+x+1)^(1/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(rendered(&ctx, folded), "2 * sqrt(x^2 + x + 1)");
    }

    #[test]
    fn compact_positive_three_half_power_result_for_integration_presentation_uses_sqrt_product() {
        let mut ctx = Context::new();
        let expr = parse("2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1)"
        );
    }

    #[test]
    fn compact_negative_three_half_power_result_for_integration_presentation_keeps_outer_sign() {
        let mut ctx = Context::new();
        let expr = parse("-2/3*(x^2+x+1)^(3/2)", &mut ctx).unwrap();
        let compact =
            compact_positive_half_power_result_for_integration_presentation(&mut ctx, expr)
                .unwrap();
        let folded = fold_numeric_mul_constants_for_hold(&mut ctx, compact);

        assert_eq!(
            rendered(&ctx, folded),
            "-(2/3 * sqrt(x^2 + x + 1) * (x^2 + x + 1))"
        );
    }
}
