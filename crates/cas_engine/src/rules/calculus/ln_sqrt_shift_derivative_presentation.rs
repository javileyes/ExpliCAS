//! Derivative presentation for `ln(sqrt(radicand) + shift)`.
//!
//! This module owns the polynomial shifted log-root route used by
//! post-calculus differentiation presentation.

use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};
use super::shifted_sqrt_args::supported_sqrt_shift_constant_parts;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn ln_sqrt_shift_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let shifted_sqrt = add_rational_for_calculus_presentation(ctx, sqrt_radicand, shift);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, shifted_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sqrt_shift_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sqrt_shift_derivative_keeps_compact_denominator() {
        let mut ctx = Context::new();
        let target = parse("ln(sqrt(x)+1)", &mut ctx).unwrap();
        let derivative = ln_sqrt_shift_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (2 * sqrt(x) * (sqrt(x) + 1))"
        );

        let target = parse("ln(sqrt(2*x)+1)", &mut ctx).unwrap();
        let derivative = ln_sqrt_shift_derivative_presentation(&mut ctx, target, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "1 / (sqrt(2 * x) * (sqrt(2 * x) + 1))"
        );
    }
}
