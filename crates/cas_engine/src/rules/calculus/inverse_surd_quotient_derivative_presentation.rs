use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;

use crate::symbolic_calculus_call_support::NamedVarCall;
use crate::Rewrite;

use super::arctan_surd_derivative_presentation::arctan_surd_quotient_compact_derivative;
use super::asinh_surd_derivative_presentation::asinh_surd_quotient_compact_derivative;
use super::atanh_surd_derivative_presentation::atanh_surd_quotient_compact_derivative;
use super::derivative_result_scaling_presentation::{
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation,
};
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::inverse_trig_derivative_presentation::bounded_inverse_trig_surd_quotient_compact_derivative;

pub(super) fn inverse_surd_quotient_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx, target, var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, target, var_name)
    {
        return Some(compact);
    }

    asinh_surd_quotient_compact_derivative(ctx, target, var_name)
}

pub(super) fn constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Div(inner, outer_den) = ctx.get(target).clone() else {
        return None;
    };
    if contains_named_var(ctx, outer_den, var_name) {
        return None;
    }

    let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
    let inner_derivative =
        bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
            .or_else(|| arctan_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
            .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))?;
    Some(divide_compact_derivative_by_constant_factor(
        ctx,
        inner_derivative,
        outer_den,
    ))
}

pub(super) fn constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let result = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
        ctx,
        target,
        &call.var_name,
    )?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        Vec::new(),
    ))
}

pub(super) fn reciprocal_constant_scaled_bounded_inverse_trig_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    for idx in 0..factors.len() {
        let inner = factors[idx];
        let mut constant_factors = factors.clone();
        constant_factors.remove(idx);
        let [constant_factor] = constant_factors.as_slice() else {
            continue;
        };
        let Some(outer_den) = reciprocal_constant_denominator_for_calculus_presentation(
            ctx,
            *constant_factor,
            var_name,
        ) else {
            continue;
        };
        let Some(inner_derivative) =
            bounded_inverse_trig_surd_quotient_compact_derivative(ctx, inner, var_name)
                .or_else(|| asinh_surd_quotient_compact_derivative(ctx, inner, var_name))
                .or_else(|| atanh_surd_quotient_compact_derivative(ctx, inner, var_name))
        else {
            continue;
        };
        return Some(divide_compact_derivative_by_constant_factor(
            ctx,
            inner_derivative,
            outer_den,
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_asinh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("asinh((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(-2 * x - 1) / (sqrt(5) * sqrt((1 - x - x^2)^2 + 5))"
        );
    }

    #[test]
    fn constant_divisor_compact_derivative_accepts_atanh_surd_quotient() {
        let mut ctx = Context::new();
        let expr = parse("atanh((x^2+x+1)/sqrt(7))/sqrt(7)", &mut ctx).unwrap();
        let derivative = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "(2 * x + 1) / (7 - (x^2 + x + 1)^2)"
        );
    }

    #[test]
    fn constant_divisor_rewrite_preserves_result_and_empty_conditions() {
        let mut ctx = Context::new();
        let target = parse("asinh((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let route_result = constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative(
            &mut ctx, target, "x",
        )
        .unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite =
            constant_divisor_bounded_inverse_trig_surd_quotient_compact_derivative_rewrite(
                &mut ctx, &call, target,
            )
            .unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            rendered(&ctx, route_result)
        );
        assert!(rewrite.required_conditions.is_empty());
    }
}
