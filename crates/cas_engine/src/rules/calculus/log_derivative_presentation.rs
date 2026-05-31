use super::differentiation::differentiate;
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::variable_named;
use super::scalar_presentation::{
    fold_numeric_mul_constants_for_hold, scale_expr_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use num_traits::{One, Zero};

pub(super) fn variable_base_constant_argument_log_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Log) || args.len() != 2 {
        return None;
    }

    let base = args[0];
    let arg = args[1];
    if !contains_named_var(ctx, base, var_name) || contains_named_var(ctx, arg, var_name) {
        return None;
    }
    if variable_named(ctx, base, var_name) {
        return None;
    }

    let d_base = polynomial_derivative_expr_for_calculus_presentation(ctx, base, var_name)
        .or_else(|| differentiate(ctx, base, var_name))?;
    if cas_ast::views::as_rational_const(ctx, d_base, 8).is_some_and(|value| value.is_zero()) {
        return None;
    }

    let ln_arg = ctx.call_builtin(BuiltinFn::Ln, vec![arg]);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let two = ctx.num(2);
    let ln_base_sq = ctx.add(Expr::Pow(ln_base, two));
    let (d_base_core, d_base_coeff) =
        split_polynomial_content_for_calculus_presentation(ctx, d_base);
    let numerator_core = if cas_ast::views::as_rational_const(ctx, d_base_core, 8)
        .is_some_and(|value| value.is_one())
    {
        ln_arg
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[ln_arg, d_base_core])
    };
    let numerator = if d_base_coeff.is_one() {
        ctx.add(Expr::Neg(numerator_core))
    } else {
        scale_expr_for_calculus_presentation(ctx, -d_base_coeff, numerator_core)
    };
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[base, ln_base_sq]);

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn ln_power_derivative_numeric_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Pow(_, exp) = ctx.get(args[0]) else {
        return None;
    };
    let exp_value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    if !exp_value.is_integer() || exp_value.is_zero() {
        return None;
    }

    let compact = fold_numeric_mul_constants_for_hold(ctx, result);
    (compact != result).then_some(compact)
}
