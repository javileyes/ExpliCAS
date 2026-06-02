//! Derivative presentation for `ln(c + sqrt(f(x)))` with non-polynomial radicands.

use super::differentiation::differentiate;
use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
    split_numeric_scale_single_core,
};
use super::shifted_sqrt_args::supported_sqrt_shift_constant_parts;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

pub(super) fn ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (sqrt_arg, shift) = supported_sqrt_shift_constant_parts(ctx, args[0])?;
    if !shift.is_positive() {
        return None;
    }
    let radicand = extract_square_root_base(ctx, sqrt_arg)?;
    if polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some() {
        return None;
    }

    let derivative = differentiate(ctx, radicand, var_name)?;
    let derivative =
        remove_unit_log_e_factor_for_calculus_presentation(ctx, derivative).unwrap_or(derivative);
    if cas_ast::views::as_rational_const(ctx, derivative, 8).is_some_and(|value| value.is_zero()) {
        return Some((ctx.num(0), Vec::new()));
    }
    let (derivative_scale, derivative_core) = split_numeric_scale_single_core(ctx, derivative)
        .unwrap_or((BigRational::one(), derivative));
    let coefficient = derivative_scale * BigRational::new(1.into(), 2.into());
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
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        vec![crate::ImplicitCondition::Positive(radicand)],
    ))
}

fn remove_unit_log_e_factor_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if is_ln_e_for_calculus_presentation(ctx, expr) {
        return Some(ctx.num(1));
    }
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let compact = remove_unit_log_e_factor_for_calculus_presentation(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(compact)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let retained = factors
        .into_iter()
        .filter(|factor| !is_ln_e_for_calculus_presentation(ctx, *factor))
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Some(ctx.num(1));
    }
    if retained.len() == cas_math::expr_nary::mul_leaves(ctx, expr).len() {
        return None;
    }
    Some(cas_math::expr_nary::build_balanced_mul(ctx, &retained))
}

fn is_ln_e_for_calculus_presentation(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Ln) => {
            matches!(ctx.get(args[0]), Expr::Constant(Constant::E))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::ln_sqrt_positive_shift_nonpolynomial_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sqrt_positive_shift_nonpolynomial_diff_uses_direct_denominator() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(sin(x)+2))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "cos(x) / (2 * sqrt(sin(x) + 2) * (sqrt(sin(x) + 2) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "sin(x) + 2 > 0");
    }

    #[test]
    fn ln_sqrt_positive_shift_exp_diff_does_not_reintroduce_ln_e() {
        let mut ctx = Context::new();
        let target = parse("ln(1+sqrt(exp(x)+1))", &mut ctx).unwrap();
        let (derivative, conditions) =
            ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "e^x / (2 * sqrt(e^x + 1) * (sqrt(e^x + 1) + 1))"
        );
        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].display(&ctx), "e^x + 1 > 0");
    }
}
