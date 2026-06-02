use super::derivative_result_scaling_presentation::scale_compact_derivative_by_rational;
use super::gap_presentation::{
    primitive_positive_gap, reciprocal_positive_rational, squared_expr_for_compact_gap_presentation,
};
use super::polynomial_support::{
    expanded_polynomial_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, exact_positive_rational_sqrt_for_calculus_presentation,
    rational_scaled_single_factor, scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::atanh_arg_over_sqrt_parts;
use crate::ImplicitCondition;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Signed};

pub(super) fn acosh_polynomial_over_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    if num_poly.degree() > 2 {
        return None;
    }

    let derivative_poly = num_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);

    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let raw_gap = ctx.add(Expr::Sub(num_square, radicand));
    let expanded_gap = expanded_polynomial_expr_for_calculus_presentation(ctx, raw_gap, 4);
    let (primitive_gap, gap_content) = primitive_positive_gap(ctx, expanded_gap);
    let (gap, numerator) = if gap_content.is_one()
        || exact_positive_rational_sqrt_for_calculus_presentation(&gap_content).is_some()
    {
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, derivative_content, derivative_core);
        (raw_gap, numerator)
    } else {
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, derivative_content, derivative_core);
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_positive_rational(&gap_content),
            numerator,
        );
        (primitive_gap, numerator)
    };
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let result = ctx.add(Expr::Div(numerator, sqrt_gap));

    let branch_gap = if let Some(root) =
        exact_positive_rational_sqrt_for_calculus_presentation(&radicand_value)
    {
        add_rational_for_calculus_presentation(ctx, num, -root)
    } else {
        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        ctx.add(Expr::Sub(num, sqrt_radicand))
    };

    Some((
        result,
        vec![
            ImplicitCondition::Positive(gap),
            ImplicitCondition::Positive(branch_gap),
        ],
    ))
}

pub(super) fn constant_scaled_acosh_polynomial_over_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        acosh_polynomial_over_sqrt_derivative_presentation(ctx, inner, var_name)?;
    let result = scale_compact_derivative_by_rational(ctx, derivative, scale);

    Some((result, required_conditions))
}
