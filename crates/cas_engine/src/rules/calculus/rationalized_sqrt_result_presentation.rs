//! Post-calculus presentation for rationalized square-root denominator shapes.
//!
//! This module owns compact result rendering for expressions where a positive
//! rational square root was rationalized into the numerator, plus the narrow
//! `acosh` argument presentation that uses the same detector.

use super::presentation_utils::{
    negative_half_power_base_for_calculus_presentation, positive_rational_scale_between_exprs,
    unwrap_internal_hold_for_calculus,
};
use super::scalar_presentation::sqrt_positive_rational_expr_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };

    let mut numerator_factors = cas_math::expr_nary::mul_leaves(ctx, numerator);
    for idx in 0..numerator_factors.len() {
        let Some(base) = extract_square_root_base(ctx, numerator_factors[idx]) else {
            continue;
        };
        let Some(denominator_scale) = positive_rational_scale_between_exprs(ctx, denominator, base)
        else {
            continue;
        };

        numerator_factors.remove(idx);
        let compact_numerator = match numerator_factors.as_slice() {
            [] => ctx.num(1),
            [single] => *single,
            _ => cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors),
        };

        let sqrt_base = ctx.call_builtin(BuiltinFn::Sqrt, vec![base]);
        let mut denominator_factors = Vec::new();
        if !denominator_scale.is_one() {
            denominator_factors.push(ctx.add(Expr::Number(denominator_scale)));
        }
        denominator_factors.push(sqrt_base);
        let compact_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
        return Some(ctx.add(Expr::Div(compact_numerator, compact_denominator)));
    }

    None
}

pub(super) fn compact_acosh_surd_width_arg_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let unwrapped = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(unwrapped).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) || args.len() != 1 {
        return None;
    }

    let compact_arg =
        compact_rationalized_sqrt_denominator_arg_for_calculus_presentation(ctx, args[0])?;
    let compact = ctx.call_builtin(BuiltinFn::Acosh, vec![compact_arg]);
    if unwrapped == expr {
        Some(compact)
    } else {
        Some(cas_ast::hold::wrap_hold(ctx, compact))
    }
}

fn compact_rationalized_sqrt_denominator_arg_for_calculus_presentation(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(arg).clone() {
        let denominator_value = cas_ast::views::as_rational_const(ctx, den, 8)?;
        if !denominator_value.is_positive() {
            return None;
        }
        return compact_rationalized_sqrt_product_for_calculus_presentation(
            ctx,
            num,
            denominator_value,
        );
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    if let Some((scale_idx, denominator_value)) =
        factors.iter().enumerate().find_map(|(idx, factor)| {
            let base = negative_half_power_base_for_calculus_presentation(ctx, *factor)?;
            let value = cas_ast::views::as_rational_const(ctx, base, 8)?;
            value.is_positive().then_some((idx, value))
        })
    {
        let mut numerator_factors = factors;
        numerator_factors.remove(scale_idx);
        let numerator = match numerator_factors.as_slice() {
            [] => ctx.num(1),
            [single] => *single,
            _ => cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors),
        };
        let denominator =
            sqrt_positive_rational_expr_for_calculus_presentation(ctx, denominator_value);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
    let (scale_idx, denominator_value) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let value = cas_ast::views::as_rational_const(ctx, *factor, 8)?;
        if !value.is_positive() || value >= BigRational::one() {
            return None;
        }
        let denominator_value = BigRational::one() / value;
        Some((idx, denominator_value))
    })?;
    let mut numerator_factors = factors;
    numerator_factors.remove(scale_idx);
    let numerator = match numerator_factors.as_slice() {
        [] => ctx.num(1),
        [single] => *single,
        _ => cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors),
    };
    compact_rationalized_sqrt_product_for_calculus_presentation(ctx, numerator, denominator_value)
}

fn compact_rationalized_sqrt_product_for_calculus_presentation(
    ctx: &mut Context,
    num: ExprId,
    denominator_value: BigRational,
) -> Option<ExprId> {
    let mut factors = cas_math::expr_nary::mul_leaves(ctx, num);
    for idx in 0..factors.len() {
        let Some(sqrt_value) =
            sqrt_positive_rational_factor_value_for_calculus_presentation(ctx, factors[idx])
        else {
            continue;
        };
        if sqrt_value != denominator_value {
            continue;
        }

        factors.remove(idx);
        let numerator = match factors.as_slice() {
            [] => ctx.num(1),
            [single] => *single,
            _ => cas_math::expr_nary::build_balanced_mul(ctx, &factors),
        };
        let denominator =
            sqrt_positive_rational_expr_for_calculus_presentation(ctx, denominator_value);
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    None
}

fn sqrt_positive_rational_factor_value_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, expr)) {
        Expr::Function(fn_id, args)
            if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            let value = cas_ast::views::as_rational_const(ctx, args[0], 8)?;
            value.is_positive().then_some(value)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new(1.into(), 2.into())) =>
        {
            let value = cas_ast::views::as_rational_const(ctx, *base, 8)?;
            value.is_positive().then_some(value)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compact_acosh_surd_width_arg_for_integration_presentation,
        compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation,
    };
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_acosh_surd_width_arg_for_integration_presentation_uses_sqrt_denominator() {
        let mut ctx = Context::new();
        let expr = parse("acosh(sqrt(5)*(x^2+x)/5)", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, expr).unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized = parse("acosh(1/5*sqrt(5)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let normalized_power = parse("acosh(1/5*5^(1/2)*(x^2+x))", &mut ctx).unwrap();
        let compact =
            compact_acosh_surd_width_arg_for_integration_presentation(&mut ctx, normalized_power)
                .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");

        let negative_half_power = parse("acosh(5^(-1/2)*(x^2+x))", &mut ctx).unwrap();
        let compact = compact_acosh_surd_width_arg_for_integration_presentation(
            &mut ctx,
            negative_half_power,
        )
        .unwrap();

        assert_eq!(rendered(&ctx, compact), "acosh((x^2 + x) / sqrt(5))");
    }

    #[test]
    fn compact_rationalized_sqrt_denominator_quotient_uses_sqrt_denominator() {
        let cases = [
            (
                "cos(x)*sqrt(sin(x)+1)/(2*sin(x)+2)",
                "cos(x) / (2 * sqrt(sin(x) + 1))",
            ),
            ("sqrt(ln(x)+1)/(ln(x)+1)", "1 / sqrt(ln(x) + 1)"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = compact_rationalized_sqrt_denominator_quotient_for_calculus_presentation(
                &mut ctx, expr,
            )
            .unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }
}
