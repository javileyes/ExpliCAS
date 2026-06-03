use super::derivative_result_scaling_presentation::{
    divide_compact_derivative_by_constant_factor,
    reciprocal_constant_denominator_for_calculus_presentation,
    remove_unit_mul_factors_for_calculus_presentation,
};
use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::domain_checks::inverse_reciprocal_trig_affine_abs_required_conditions;
use super::gap_presentation::{primitive_positive_gap, reciprocal_positive_rational};
use super::polynomial_support::nonzero_affine_variable_derivative;
use super::presentation_utils::squared_expr;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation,
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
};
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn inverse_reciprocal_trig_affine_abs_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsec | BuiltinFn::Asec) => BigRational::one(),
        Some(BuiltinFn::Arccsc | BuiltinFn::Acsc) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let arg = args[0];
    let derivative_scale = nonzero_affine_variable_derivative(ctx, arg, var_name)?;
    let numerator = rational_const_for_calculus_presentation(ctx, sign * derivative_scale);
    let arg_sq = squared_expr(ctx, arg);
    let one = ctx.num(1);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let (gap, gap_content) = primitive_positive_gap(ctx, raw_gap);
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        reciprocal_positive_rational(&gap_content),
        numerator,
    );
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[abs_arg, sqrt_gap]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    if let Expr::Div(inner, outer_den) = ctx.get(target).clone() {
        if contains_named_var(ctx, outer_den, var_name) {
            return None;
        }

        let inner = remove_unit_mul_factors_for_calculus_presentation(ctx, inner);
        let inner_derivative =
            inverse_reciprocal_trig_affine_abs_presentation(ctx, inner, var_name)?;
        let required_conditions =
            inverse_reciprocal_trig_affine_abs_required_conditions(ctx, inner, var_name);
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

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
            inverse_reciprocal_trig_affine_abs_presentation(ctx, inner, var_name)
        else {
            continue;
        };
        let required_conditions =
            inverse_reciprocal_trig_affine_abs_required_conditions(ctx, inner, var_name);
        let result = divide_compact_derivative_by_constant_factor(ctx, inner_derivative, outer_den);
        let result = cas_ast::hold::wrap_hold(ctx, result);
        return Some((result, required_conditions));
    }

    None
}

pub(super) fn constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let (result, required_conditions) =
        constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(
            ctx,
            target,
            &call.var_name,
        )?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        required_conditions,
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        constant_scaled_inverse_reciprocal_trig_affine_abs_presentation,
        constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite,
    };
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn constant_scaled_arcsec_affine_abs_route_preserves_domain_condition() {
        let mut ctx = Context::new();
        let target = parse("arcsec(2*x+1)/3", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_reciprocal_trig_affine_abs_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (3 * |2 * x + 1| * sqrt(x^2 + x))"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(required_conditions[0].display(&ctx), "x < -1 or x > 0");
    }

    #[test]
    fn constant_scaled_arcsec_affine_abs_rewrite_preserves_domain_condition() {
        let mut ctx = Context::new();
        let target = parse("arcsec(2*x+1)/3", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite =
            constant_scaled_inverse_reciprocal_trig_affine_abs_rewrite(&mut ctx, &call, target)
                .unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "1 / (3 * |2 * x + 1| * sqrt(x^2 + x))"
        );
        assert_eq!(rewrite.required_conditions.len(), 1);
        assert_eq!(
            rewrite.required_conditions[0].display(&ctx),
            "x < -1 or x > 0"
        );
    }
}
