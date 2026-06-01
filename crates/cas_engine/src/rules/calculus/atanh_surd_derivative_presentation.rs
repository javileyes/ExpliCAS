use super::differentiation::differentiate;
use super::domain_checks::{
    atanh_open_interval_condition, atanh_self_normalized_surd_quotient_positive_gap,
};
use super::gap_presentation::{
    compact_squared_affine_gap_for_calculus_presentation, primitive_positive_gap,
};
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::{
    arctan_self_normalized_surd_quotient_parts, atanh_arg_over_sqrt_parts,
};
use super::surd_quotient_presentation::compact_surd_quotient_polynomial_presentation_parts;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn atanh_self_normalized_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
        return None;
    }

    let (num, radicand) = arctan_self_normalized_surd_quotient_parts(ctx, args[0])?;
    atanh_self_normalized_surd_quotient_positive_gap(ctx, args[0], var_name)?;

    let (d_num, _) = compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let compact = ctx.add(Expr::Div(d_num, sqrt_radicand));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn atanh_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };

    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) || args.len() != 1 {
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
    let raw_denominator = atanh_open_interval_condition(ctx, args[0]);
    let (compact_denominator, denominator_content) = if radicand_value.is_integer() {
        (raw_denominator, BigRational::one())
    } else {
        primitive_positive_gap(ctx, raw_denominator)
    };
    let compact_denominator =
        compact_squared_affine_gap_for_calculus_presentation(ctx, compact_denominator, var_name);
    let d_num = signed_numerator_for_calculus_presentation(
        ctx,
        d_num_content / denominator_content,
        d_num_core,
    );
    let compact_sqrt_radicand = ctx.add(Expr::Number(radicand_value));
    let compact_sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![compact_sqrt_radicand]);
    let compact_numerator = ctx.add(Expr::Mul(compact_sqrt_radicand, d_num));
    let compact = ctx.add(Expr::Div(compact_numerator, compact_denominator));
    let compact = fold_numeric_mul_constants_for_hold(ctx, compact);

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::atanh_self_normalized_surd_quotient_compact_derivative;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn atanh_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("atanh(((2*x+1)^2+3)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            atanh_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / sqrt((2 * x + 1)^2 + 3)");
    }
}
