use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::Zero;

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn one_expr(ctx: &mut Context) -> ExprId {
    ctx.num(1)
}

fn expr_is_one(ctx: &mut Context, expr: ExprId) -> bool {
    let one = one_expr(ctx);
    expr_eq(ctx, expr, one)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1).then_some(args[0])
}

fn diff_call_with_optional_divisor(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr).is_some() {
        let one = one_expr(ctx);
        return Some((expr, one));
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;
    crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, num)?;
    Some((num, den))
}

fn is_constant_scaled_hyperbolic_reciprocal_target(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(expr) {
        return is_constant_scaled_hyperbolic_reciprocal_target(ctx, *inner);
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 4).is_none() {
        return false;
    }

    let mut matched_hyperbolic = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if unary_builtin_arg(ctx, factor, BuiltinFn::Sinh).is_some()
            || unary_builtin_arg(ctx, factor, BuiltinFn::Cosh).is_some()
        {
            if matched_hyperbolic {
                return false;
            }
            matched_hyperbolic = true;
            continue;
        }

        let Some(factor_scale) = cas_ast::views::as_rational_const(ctx, factor, 4) else {
            return false;
        };
        if factor_scale.is_zero() {
            return false;
        }
    }

    matched_hyperbolic
}

fn scaled_expected_matches(
    ctx: &mut Context,
    scale: ExprId,
    builtin: BuiltinFn,
    hyperbolic_arg: ExprId,
    right: ExprId,
) -> bool {
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
    match builtin {
        BuiltinFn::Sinh if expr_is_one(ctx, scale) => {
            let one = one_expr(ctx);
            let reciprocal_tanh = ctx.add(Expr::Div(one, tanh));
            if expr_eq(ctx, reciprocal_tanh, right) {
                return true;
            }

            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let coth_quotient = ctx.add(Expr::Div(cosh, sinh));
            expr_eq(ctx, coth_quotient, right)
        }
        BuiltinFn::Sinh => {
            let expected = ctx.add(Expr::Div(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        BuiltinFn::Cosh if expr_is_one(ctx, scale) => {
            if expr_eq(ctx, tanh, right) {
                return true;
            }

            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let tanh_quotient = ctx.add(Expr::Div(sinh, cosh));
            expr_eq(ctx, tanh_quotient, right)
        }
        BuiltinFn::Cosh => {
            let expected = ctx.add(Expr::Mul(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        _ => false,
    }
}

fn log_abs_hyperbolic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let ln_arg = unary_builtin_arg(ctx, call.target, BuiltinFn::Ln)?;
    let abs_arg = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs)?;

    let (builtin, hyperbolic_arg) =
        if let Some(arg) = unary_builtin_arg(ctx, abs_arg, BuiltinFn::Sinh) {
            (BuiltinFn::Sinh, arg)
        } else {
            (
                BuiltinFn::Cosh,
                unary_builtin_arg(ctx, abs_arg, BuiltinFn::Cosh)?,
            )
        };

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        hyperbolic_arg,
        &call.var_name,
    )?;
    let scale = if expr_is_one(ctx, divisor) {
        derivative
    } else if expr_eq(ctx, derivative, divisor) {
        one_expr(ctx)
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(scaled_expected_matches(
        ctx,
        scale,
        builtin,
        hyperbolic_arg,
        right,
    ))
}

pub(crate) fn try_diff_log_abs_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    log_abs_hyperbolic_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

fn constant_scaled_hyperbolic_reciprocal_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !is_constant_scaled_hyperbolic_reciprocal_target(ctx, call.target) {
        return None;
    }

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        call.target,
        &call.var_name,
    )?;
    let expected = if expr_is_one(ctx, divisor) {
        derivative
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(expr_eq(ctx, expected, right))
}

pub(crate) fn try_diff_hyperbolic_reciprocal_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    constant_scaled_hyperbolic_reciprocal_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

pub(crate) fn try_diff_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    try_diff_log_abs_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_hyperbolic_reciprocal_residual_zero_preorder(ctx, left, right))
}

pub(crate) fn try_diff_hyperbolic_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_hyperbolic_residual_zero_preorder(ctx, right, left))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_hyperbolic_residual_root_zero(&mut ctx, expr).map(|result| render(&ctx, result))
    }

    fn simplify_text(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::new();
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _steps) = simplifier.simplify(expr);
        render(&simplifier.context, result)
    }

    #[test]
    fn diff_log_abs_hyperbolic_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(ln(abs(sinh(2*x+1))), x)/2 - 1/tanh(2*x+1)",
            "diff(ln(abs(sinh(2*x+1))), x)/2 - cosh(2*x+1)/sinh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - tanh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - sinh(2*x+1)/cosh(2*x+1)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_hyperbolic_reciprocal_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(-1/(2*cosh(2*x+1)), x) - sinh(2*x+1)/cosh(2*x+1)^2",
            "diff(-1/(2*sinh(2*x+1)), x) - cosh(2*x+1)/sinh(2*x+1)^2",
            "sinh(2*x+1)/cosh(2*x+1)^2 - diff(-1/(2*cosh(2*x+1)), x)",
            "cosh(2*x+1)/sinh(2*x+1)^2 - diff(-1/(2*sinh(2*x+1)), x)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }
}
