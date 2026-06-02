use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

use super::polynomial_support::split_polynomial_content_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_rational_const_for_calculus_presentation,
};

pub(super) fn sqrt_reciprocal_trig_function_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = extract_square_root_base(ctx, target)?;
    let Expr::Function(fn_id, args) = ctx.get(radicand).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (derivative_fn, sign) = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sec) => (BuiltinFn::Tan, BigRational::one()),
        Some(BuiltinFn::Csc) => (BuiltinFn::Cot, -BigRational::one()),
        _ => return None,
    };

    let arg_poly = Polynomial::from_expr(ctx, args[0], var_name).ok()?;
    let derivative_poly = arg_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (mut derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let mut coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    if let Some(core_value) = signed_rational_const_for_calculus_presentation(ctx, derivative_core)
    {
        coefficient *= core_value;
        derivative_core = ctx.num(1);
    }
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let trig_factor = ctx.call_builtin(derivative_fn, vec![args[0]]);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let mut numerator_factors = Vec::new();
    if !derivative_core_is_one {
        numerator_factors.push(derivative_core);
    }
    numerator_factors.push(trig_factor);
    numerator_factors.push(sqrt_radicand);
    let numerator_core = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    if denominator_coeff == BigRational::one() {
        return Some(numerator);
    }

    let denominator = rational_const_for_calculus_presentation(ctx, denominator_coeff);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::sqrt_reciprocal_trig_function_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn sqrt_reciprocal_trig_derivative_presentation_handles_sec_radicand() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(sec(2*x+1))", &mut ctx).unwrap();
        let compact = sqrt_reciprocal_trig_function_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("sqrt(sec(u)) derivative should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "tan(2 * x + 1) * sqrt(sec(2 * x + 1))"
        );
    }

    #[test]
    fn sqrt_reciprocal_trig_derivative_presentation_handles_csc_radicand() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(csc(2*x+1))", &mut ctx).unwrap();
        let compact = sqrt_reciprocal_trig_function_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| panic!("sqrt(csc(u)) derivative should be recognized"));

        assert_eq!(
            rendered(&ctx, compact),
            "-cot(2 * x + 1) * sqrt(csc(2 * x + 1))"
        );
    }
}
