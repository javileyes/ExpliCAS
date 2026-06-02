use super::shifted_sqrt_argument_presentation::compact_shifted_sqrt_argument_for_integration_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};

pub(super) fn compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(ctx, num, var_name);
            let den = compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, den, var_name)
                .unwrap_or_else(|| {
                    compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                        ctx, den, var_name,
                    )
                });
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        _ => compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name)
            .unwrap_or(expr),
    }
}

pub(super) fn has_compactable_sqrt_hyperbolic_reciprocal_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_sqrt_hyperbolic_call_for_integration_presentation(ctx, expr, var_name).is_some() {
        return true;
    }
    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, left, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, num, var_name)
                || has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_sqrt_hyperbolic_reciprocal_result(ctx, inner, var_name),
        _ => false,
    }
}

fn compact_sqrt_hyperbolic_call_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh | BuiltinFn::Tanh) || args.len() != 1 {
        return None;
    }

    let compact_arg =
        compact_shifted_sqrt_argument_for_integration_presentation(ctx, args[0], var_name)?;
    Some(ctx.call_builtin(builtin, vec![compact_arg]))
}

pub(super) fn compact_positive_cosh_log_abs_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(compact) = compact_ln_abs_cosh_sqrt(ctx, expr, var_name) {
        return compact;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, left, var_name);
            let right =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, right, var_name);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(num, den) => {
            let num =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, num, var_name);
            let den =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, den, var_name);
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner =
                compact_positive_cosh_log_abs_for_integration_presentation(ctx, inner, var_name);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn compact_ln_abs_cosh_sqrt(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let (hyperbolic_expr, wrapped_in_abs) = match ctx.get(args[0]).clone() {
        Expr::Function(abs_fn, abs_args)
            if ctx.builtin_of(abs_fn) == Some(BuiltinFn::Abs) && abs_args.len() == 1 =>
        {
            (abs_args[0], true)
        }
        Expr::Function(hyperbolic_fn, hyperbolic_args)
            if matches!(
                ctx.builtin_of(hyperbolic_fn),
                Some(BuiltinFn::Cosh | BuiltinFn::Sinh)
            ) && hyperbolic_args.len() == 1 =>
        {
            (args[0], false)
        }
        _ => return None,
    };
    let Expr::Function(hyperbolic_fn, hyperbolic_args) = ctx.get(hyperbolic_expr).clone() else {
        return None;
    };
    let hyperbolic_builtin = ctx.builtin_of(hyperbolic_fn)?;
    if !matches!(hyperbolic_builtin, BuiltinFn::Cosh | BuiltinFn::Sinh)
        || hyperbolic_args.len() != 1
    {
        return None;
    }

    let compact_arg = compact_shifted_sqrt_argument_for_integration_presentation(
        ctx,
        hyperbolic_args[0],
        var_name,
    )?;
    let hyperbolic_expr = ctx.call_builtin(hyperbolic_builtin, vec![compact_arg]);
    let log_arg = if wrapped_in_abs && hyperbolic_builtin != BuiltinFn::Cosh {
        ctx.call_builtin(BuiltinFn::Abs, vec![hyperbolic_expr])
    } else {
        hyperbolic_expr
    };
    Some(ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]))
}

pub(super) fn has_compactable_ln_abs_cosh_sqrt(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if compact_ln_abs_cosh_sqrt(ctx, expr, var_name).is_some() {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, left, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, right, var_name)
        }
        Expr::Div(num, den) => {
            has_compactable_ln_abs_cosh_sqrt(ctx, num, var_name)
                || has_compactable_ln_abs_cosh_sqrt(ctx, den, var_name)
        }
        Expr::Neg(inner) => has_compactable_ln_abs_cosh_sqrt(ctx, inner, var_name),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compact_positive_cosh_log_abs_for_integration_presentation,
        compact_sqrt_hyperbolic_reciprocal_for_integration_presentation,
    };
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_positive_cosh_log_presentation_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse("ln(cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_positive_cosh_log_abs_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "ln(cosh(sqrt(3 * x + 1)))");
    }

    #[test]
    fn compact_positive_cosh_log_presentation_preserves_sinh_abs_shift() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(sinh(x^(1/2)-b)))", &mut ctx).unwrap();
        let compact =
            compact_positive_cosh_log_abs_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "ln(|sinh(sqrt(x) - b)|)");
    }

    #[test]
    fn compact_sqrt_hyperbolic_reciprocal_preserves_shifted_sqrt_display() {
        let mut ctx = Context::new();
        let expr = parse("-k/cosh(x^(1/2)-b)", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "-k / cosh(sqrt(x) - b)");
    }

    #[test]
    fn fold_numeric_mul_constants_for_hold_cancels_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("-1*3/(3*cosh((3*x+1)^(1/2)))", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");
        let folded = super::super::scalar_presentation::fold_numeric_mul_constants_for_hold(
            &mut ctx, compact,
        );

        assert_eq!(rendered(&ctx, folded), "-1 / cosh(sqrt(3 * x + 1))");
    }

    #[test]
    fn compact_sqrt_hyperbolic_reciprocal_preserves_negative_shift_orientation() {
        let mut ctx = Context::new();
        let expr = parse("-k/sinh(b-x^(1/2))", &mut ctx).unwrap();
        let compact =
            compact_sqrt_hyperbolic_reciprocal_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(rendered(&ctx, compact), "-k / sinh(b - sqrt(x))");
    }
}
