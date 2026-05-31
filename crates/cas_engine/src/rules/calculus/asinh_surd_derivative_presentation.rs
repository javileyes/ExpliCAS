use super::differentiation::differentiate;
use super::gap_presentation::{
    primitive_positive_gap, reciprocal_positive_rational, squared_expr_for_compact_gap_presentation,
};
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    exact_positive_rational_sqrt_for_calculus_presentation, fold_numeric_mul_constants_for_hold,
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::atanh_arg_over_sqrt_parts;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn asinh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Asinh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
        .or_else(|| differentiate(ctx, num, var_name))?;
    let (d_num_core, d_num_content) =
        split_polynomial_content_for_calculus_presentation(ctx, d_num);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, num);
    let raw_gap = ctx.add(Expr::Add(radicand, num_square));
    let (positive_gap, gap_content) = if radicand_value.is_integer() {
        (raw_gap, BigRational::one())
    } else {
        primitive_positive_gap(ctx, raw_gap)
    };
    let numerator_scale = if gap_content.is_one() {
        d_num_content
    } else if let Some(sqrt_content) =
        exact_positive_rational_sqrt_for_calculus_presentation(&gap_content)
    {
        d_num_content / sqrt_content
    } else {
        let reciprocal_content = reciprocal_positive_rational(&gap_content);
        let numerator = signed_numerator_for_calculus_presentation(ctx, d_num_content, d_num_core);
        let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
            ctx,
            reciprocal_content,
            numerator,
        );
        let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![positive_gap]);
        let compact = ctx.add(Expr::Div(numerator, denominator));
        let compact = fold_numeric_mul_constants_for_hold(ctx, compact);
        return Some(compact);
    };
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_scale, d_num_core);
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![positive_gap]);
    let compact = ctx.add(Expr::Div(numerator, denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}
