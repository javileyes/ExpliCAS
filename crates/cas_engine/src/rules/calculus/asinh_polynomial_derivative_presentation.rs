use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    rational_polynomial_content_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn asinh_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Asinh) {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    let arg_content = rational_polynomial_content_for_calculus_presentation(&arg_poly);
    if arg_content.is_zero() {
        return None;
    }
    let primitive_arg_poly = if arg_content.is_one() {
        arg_poly
    } else {
        arg_poly.div_scalar(&arg_content)
    };
    let derivative_poly = primitive_arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let content_num = BigRational::from_integer(arg_content.numer().clone());
    let numerator = scale_expr_for_calculus_presentation(
        ctx,
        derivative_content * content_num,
        derivative_core,
    );

    let one = ctx.num(1);
    let square_arg_poly =
        if !arg_content.is_one() && primitive_arg_poly.leading_coeff().is_negative() {
            primitive_arg_poly.neg()
        } else {
            primitive_arg_poly.clone()
        };
    let primitive_arg = square_arg_poly.to_expr(ctx);
    let primitive_arg_sq = squared_expr_for_compact_gap_presentation(ctx, primitive_arg);
    let radicand = if arg_content.is_one() {
        ctx.add(Expr::Add(primitive_arg_sq, one))
    } else {
        let content_num_sq =
            BigRational::from_integer(arg_content.numer().clone() * arg_content.numer().clone());
        let content_den_sq =
            BigRational::from_integer(arg_content.denom().clone() * arg_content.denom().clone());
        let scaled_arg_sq =
            scale_expr_for_calculus_presentation(ctx, content_num_sq, primitive_arg_sq);
        let den_sq = rational_const_for_calculus_presentation(ctx, content_den_sq);
        ctx.add(Expr::Add(scaled_arg_sq, den_sq))
    };
    let denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
