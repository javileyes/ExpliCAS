use super::polynomial_support::{
    polynomial_is_strictly_positive_everywhere, polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn acosh_strictly_positive_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Acosh) {
        return None;
    }

    let arg = args[0];
    let arg_poly = polynomial_radicand_for_calculus_presentation(ctx, arg, var_name)?;
    if arg_poly.degree() != 2 {
        return None;
    }

    let one_poly = Polynomial::one(arg_poly.var.clone());
    let lower_poly = arg_poly.sub(&one_poly);
    if !polynomial_is_strictly_positive_everywhere(&lower_poly) {
        return None;
    }

    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some((ctx.num(0), Vec::new()));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&derivative_content)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let lower_branch = lower_poly.to_expr(ctx);
    let upper_branch = arg_poly.add(&one_poly).to_expr(ctx);
    let sqrt_lower = ctx.call_builtin(BuiltinFn::Sqrt, vec![lower_branch]);
    let sqrt_upper = ctx.call_builtin(BuiltinFn::Sqrt, vec![upper_branch]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_lower, sqrt_upper]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));
    Some((cas_ast::hold::wrap_hold(ctx, compact), Vec::new()))
}
