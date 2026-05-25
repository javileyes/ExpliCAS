use crate::runtime::Step;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

pub(super) struct RenderedStepWireExprs {
    pub(super) before: String,
    pub(super) after: String,
}

pub(super) fn render_step_wire_exprs(context: &Context, step: &Step) -> RenderedStepWireExprs {
    let mut temp_ctx = context.clone();
    let snapshots =
        crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step);
    let compact_before = cleanup_step_prefers_compact_nested_reciprocal_display(
        &temp_ctx,
        step,
        snapshots.global_before_expr,
    );

    RenderedStepWireExprs {
        before: if residual_limit_step_prefers_direct_display(step) {
            render_human_expr_without_display_normalization(
                context,
                step.global_before.unwrap_or(step.before),
            )
        } else if symbolic_integration_step_prefers_reciprocal_sqrt_before_display(
            &temp_ctx,
            step,
            snapshots.global_before_expr,
        ) {
            render_human_expr_with_reciprocal_sqrt_display_cleanup(
                &temp_ctx,
                snapshots.global_before_expr,
            )
        } else if compact_before {
            render_human_expr_with_symbolic_diff_display_cleanup(
                &temp_ctx,
                snapshots.global_before_expr,
            )
        } else {
            render_human_expr(&temp_ctx, snapshots.global_before_expr)
        },
        after: if residual_limit_step_prefers_direct_display(step) {
            render_human_expr_without_display_normalization(
                context,
                step.global_after.unwrap_or(step.after),
            )
        } else if symbolic_differentiation_step_prefers_compact_after_display(
            &temp_ctx,
            step,
            snapshots.global_after_expr,
        ) || cleanup_step_prefers_compact_nested_reciprocal_display(
            &temp_ctx,
            step,
            snapshots.global_after_expr,
        ) {
            render_human_expr_with_symbolic_diff_display_cleanup(
                &temp_ctx,
                snapshots.global_after_expr,
            )
        } else if repeated_by_parts_step_prefers_direct_after_display(context, step) {
            render_human_expr_without_display_normalization(
                context,
                step.after_local().unwrap_or(step.after),
            )
        } else if polynomial_exp_by_parts_step_prefers_latex_after_display(context, step) {
            render_polynomial_exp_by_parts_after(context, step.after_local().unwrap_or(step.after))
        } else {
            render_human_expr(&temp_ctx, snapshots.global_after_expr)
        },
    }
}

fn residual_limit_step_prefers_direct_display(step: &Step) -> bool {
    step.rule_name == "Conservar límite residual"
}

fn symbolic_integration_step_prefers_reciprocal_sqrt_before_display(
    context: &Context,
    step: &Step,
    expr: ExprId,
) -> bool {
    step.rule_name == "Symbolic Integration" && contains_negative_half_power(context, expr)
}

fn symbolic_differentiation_step_prefers_compact_after_display(
    context: &Context,
    step: &Step,
    expr: ExprId,
) -> bool {
    step.rule_name == "Symbolic Differentiation"
        && (contains_ln_e_call(context, expr)
            || contains_positive_integer_power_exponent_arithmetic(context, expr)
            || contains_nested_reciprocal_division(context, expr))
}

pub(super) fn cleanup_step_prefers_compact_nested_reciprocal_display(
    context: &Context,
    step: &Step,
    expr: ExprId,
) -> bool {
    matches!(
        step.rule_name.as_str(),
        "Cancel Exact Additive Pairs"
            | "Cancel Reciprocal Exponents"
            | "Cancel opposite terms"
            | "Root Power Cancel"
            | "Square of Square Root"
    ) && contains_nested_reciprocal_division(context, expr)
}

fn repeated_by_parts_step_prefers_direct_after_display(context: &Context, step: &Step) -> bool {
    if step.rule_name != "Symbolic Integration" {
        return false;
    }

    let before = step.before_local().unwrap_or(step.before);
    let Expr::Function(fn_id, args) = context.get(before) else {
        return false;
    };
    if context.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    let Expr::Variable(var_sym) = context.get(args[1]) else {
        return false;
    };
    let var_name = context.sym_name(*var_sym);
    let mut scratch = context.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
        &mut scratch,
        args[0],
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
        &mut scratch,
        args[0],
        var_name,
    )
}

fn polynomial_exp_by_parts_step_prefers_latex_after_display(
    context: &Context,
    step: &Step,
) -> bool {
    if step.rule_name != "Symbolic Integration" {
        return false;
    }

    let before = step.before_local().unwrap_or(step.before);
    let Expr::Function(fn_id, args) = context.get(before) else {
        return false;
    };
    if context.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return false;
    }
    let Expr::Variable(var_sym) = context.get(args[1]) else {
        return false;
    };
    let var_name = context.sym_name(*var_sym);
    let mut scratch = context.clone();
    cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_exp_linear_target(
        &mut scratch,
        args[0],
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_exp_linear_target(
        &mut scratch,
        args[0],
        var_name,
    )
}

pub(crate) fn render_human_expr(context: &Context, expr: ExprId) -> String {
    let mut temp_ctx = context.clone();
    let normalized =
        cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut temp_ctx, expr);
    let latex = cas_formatter::LaTeXExpr {
        context: &temp_ctx,
        id: normalized,
    }
    .to_latex();
    let human = cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(&latex));
    if human.trim().is_empty() {
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &temp_ctx,
                id: normalized
            }
        ))
    } else {
        human
    }
}

fn render_human_expr_with_symbolic_diff_display_cleanup(context: &Context, expr: ExprId) -> String {
    let mut temp_ctx = context.clone();
    let collapsed = cleanup_symbolic_diff_after_for_display(&mut temp_ctx, expr);
    render_human_expr(&temp_ctx, collapsed)
}

fn render_human_expr_with_reciprocal_sqrt_display_cleanup(
    context: &Context,
    expr: ExprId,
) -> String {
    let mut temp_ctx = context.clone();
    let collapsed = collapse_negative_half_powers_for_display(&mut temp_ctx, expr);
    render_human_expr(&temp_ctx, collapsed)
}

pub(super) fn cleanup_symbolic_diff_after_for_display(
    context: &mut Context,
    expr: ExprId,
) -> ExprId {
    let collapsed = collapse_ln_e_for_display(context, expr);
    let collapsed = collapse_positive_integer_power_exponents_for_display(context, collapsed);
    collapse_nested_reciprocal_division_for_display(context, collapsed)
}

pub(super) fn collapse_ln_e_for_display(context: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr).clone() {
        Expr::Function(fn_id, args)
            if context.is_builtin(fn_id, BuiltinFn::Ln)
                && args.len() == 1
                && is_e_constant(context, args[0]) =>
        {
            context.num(1)
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| collapse_ln_e_for_display(context, arg))
                .collect();
            context.add(Expr::Function(fn_id, args))
        }
        Expr::Add(lhs, rhs) => {
            let lhs = collapse_ln_e_for_display(context, lhs);
            let rhs = collapse_ln_e_for_display(context, rhs);
            context.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = collapse_ln_e_for_display(context, lhs);
            let rhs = collapse_ln_e_for_display(context, rhs);
            context.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = collapse_ln_e_for_display(context, lhs);
            let rhs = collapse_ln_e_for_display(context, rhs);
            context.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = collapse_ln_e_for_display(context, lhs);
            let rhs = collapse_ln_e_for_display(context, rhs);
            context.add(Expr::Div(lhs, rhs))
        }
        Expr::Pow(lhs, rhs) => {
            let lhs = collapse_ln_e_for_display(context, lhs);
            let rhs = collapse_ln_e_for_display(context, rhs);
            context.add(Expr::Pow(lhs, rhs))
        }
        Expr::Neg(inner) => {
            let inner = collapse_ln_e_for_display(context, inner);
            context.add(Expr::Neg(inner))
        }
        Expr::Hold(inner) => {
            let inner = collapse_ln_e_for_display(context, inner);
            context.add(Expr::Hold(inner))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| collapse_ln_e_for_display(context, item))
                .collect();
            context.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

pub(super) fn contains_ln_e_call(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Function(fn_id, args) => {
            (context.is_builtin(*fn_id, BuiltinFn::Ln)
                && args.len() == 1
                && is_e_constant(context, args[0]))
                || args
                    .iter()
                    .copied()
                    .any(|arg| contains_ln_e_call(context, arg))
        }
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            contains_ln_e_call(context, *lhs) || contains_ln_e_call(context, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_ln_e_call(context, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|item| contains_ln_e_call(context, item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn collapse_positive_integer_power_exponents_for_display(
    context: &mut Context,
    expr: ExprId,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr).clone() {
        Expr::Pow(base, exp) => {
            let base = collapse_positive_integer_power_exponents_for_display(context, base);
            let exp = collapse_positive_integer_power_exponents_for_display(context, exp);
            if let Some(value) = positive_integer_display_constant(context, exp) {
                if value.is_one() {
                    base
                } else {
                    let exp = context.add(Expr::Number(value));
                    context.add(Expr::Pow(base, exp))
                }
            } else {
                context.add(Expr::Pow(base, exp))
            }
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| collapse_positive_integer_power_exponents_for_display(context, arg))
                .collect();
            context.add(Expr::Function(fn_id, args))
        }
        Expr::Add(lhs, rhs) => {
            let lhs = collapse_positive_integer_power_exponents_for_display(context, lhs);
            let rhs = collapse_positive_integer_power_exponents_for_display(context, rhs);
            context.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = collapse_positive_integer_power_exponents_for_display(context, lhs);
            let rhs = collapse_positive_integer_power_exponents_for_display(context, rhs);
            context.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = collapse_positive_integer_power_exponents_for_display(context, lhs);
            let rhs = collapse_positive_integer_power_exponents_for_display(context, rhs);
            context.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = collapse_positive_integer_power_exponents_for_display(context, lhs);
            let rhs = collapse_positive_integer_power_exponents_for_display(context, rhs);
            context.add(Expr::Div(lhs, rhs))
        }
        Expr::Neg(inner) => {
            let inner = collapse_positive_integer_power_exponents_for_display(context, inner);
            context.add(Expr::Neg(inner))
        }
        Expr::Hold(inner) => {
            let inner = collapse_positive_integer_power_exponents_for_display(context, inner);
            context.add(Expr::Hold(inner))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| collapse_positive_integer_power_exponents_for_display(context, item))
                .collect();
            context.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

pub(super) fn contains_positive_integer_power_exponent_arithmetic(
    context: &Context,
    expr: ExprId,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Pow(_, exp) if is_positive_integer_exponent_arithmetic(context, *exp) => true,
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| contains_positive_integer_power_exponent_arithmetic(context, arg)),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            contains_positive_integer_power_exponent_arithmetic(context, *lhs)
                || contains_positive_integer_power_exponent_arithmetic(context, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_positive_integer_power_exponent_arithmetic(context, *inner)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|item| contains_positive_integer_power_exponent_arithmetic(context, item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

pub(super) fn contains_negative_half_power(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Pow(_, exp) if is_negative_half_display_constant(context, *exp) => true,
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| contains_negative_half_power(context, arg)),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            contains_negative_half_power(context, *lhs)
                || contains_negative_half_power(context, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_negative_half_power(context, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|item| contains_negative_half_power(context, item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

pub(super) fn collapse_negative_half_powers_for_display(
    context: &mut Context,
    expr: ExprId,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr).clone() {
        Expr::Pow(base, exp) => {
            let base = collapse_negative_half_powers_for_display(context, base);
            let exp = collapse_negative_half_powers_for_display(context, exp);
            if is_negative_half_display_constant(context, exp) {
                let one = context.num(1);
                let sqrt = context.call_builtin(BuiltinFn::Sqrt, vec![base]);
                context.add(Expr::Div(one, sqrt))
            } else {
                context.add(Expr::Pow(base, exp))
            }
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| collapse_negative_half_powers_for_display(context, arg))
                .collect();
            context.add(Expr::Function(fn_id, args))
        }
        Expr::Add(lhs, rhs) => {
            let lhs = collapse_negative_half_powers_for_display(context, lhs);
            let rhs = collapse_negative_half_powers_for_display(context, rhs);
            context.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = collapse_negative_half_powers_for_display(context, lhs);
            let rhs = collapse_negative_half_powers_for_display(context, rhs);
            context.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = collapse_negative_half_powers_for_display(context, lhs);
            let rhs = collapse_negative_half_powers_for_display(context, rhs);
            context.add(Expr::Mul(lhs, rhs))
        }
        Expr::Div(lhs, rhs) => {
            let lhs = collapse_negative_half_powers_for_display(context, lhs);
            let rhs = collapse_negative_half_powers_for_display(context, rhs);
            context.add(Expr::Div(lhs, rhs))
        }
        Expr::Neg(inner) => {
            let inner = collapse_negative_half_powers_for_display(context, inner);
            context.add(Expr::Neg(inner))
        }
        Expr::Hold(inner) => {
            let inner = collapse_negative_half_powers_for_display(context, inner);
            context.add(Expr::Hold(inner))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| collapse_negative_half_powers_for_display(context, item))
                .collect();
            context.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

pub(super) fn contains_nested_reciprocal_division(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Div(num, den)
            if signed_reciprocal_division(context, *num).is_some()
                && !is_one_expr(context, *den) =>
        {
            true
        }
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| contains_nested_reciprocal_division(context, arg)),
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            contains_nested_reciprocal_division(context, *lhs)
                || contains_nested_reciprocal_division(context, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_nested_reciprocal_division(context, *inner)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|item| contains_nested_reciprocal_division(context, item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn collapse_nested_reciprocal_division_for_display(context: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr).clone() {
        Expr::Div(num, den) => {
            let num = collapse_nested_reciprocal_division_for_display(context, num);
            let den = collapse_nested_reciprocal_division_for_display(context, den);
            if let Some((negative, inner_den)) = signed_reciprocal_division(context, num) {
                if !is_one_expr(context, den) {
                    let one = context.num(1);
                    let inner_num = if negative {
                        context.add(Expr::Neg(one))
                    } else {
                        one
                    };
                    let denominator = context.add(Expr::Mul(inner_den, den));
                    return context.add(Expr::Div(inner_num, denominator));
                }
            }
            context.add(Expr::Div(num, den))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| collapse_nested_reciprocal_division_for_display(context, arg))
                .collect();
            context.add(Expr::Function(fn_id, args))
        }
        Expr::Add(lhs, rhs) => {
            let lhs = collapse_nested_reciprocal_division_for_display(context, lhs);
            let rhs = collapse_nested_reciprocal_division_for_display(context, rhs);
            context.add(Expr::Add(lhs, rhs))
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = collapse_nested_reciprocal_division_for_display(context, lhs);
            let rhs = collapse_nested_reciprocal_division_for_display(context, rhs);
            context.add(Expr::Sub(lhs, rhs))
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = collapse_nested_reciprocal_division_for_display(context, lhs);
            let rhs = collapse_nested_reciprocal_division_for_display(context, rhs);
            context.add(Expr::Mul(lhs, rhs))
        }
        Expr::Pow(lhs, rhs) => {
            let lhs = collapse_nested_reciprocal_division_for_display(context, lhs);
            let rhs = collapse_nested_reciprocal_division_for_display(context, rhs);
            context.add(Expr::Pow(lhs, rhs))
        }
        Expr::Neg(inner) => {
            let inner = collapse_nested_reciprocal_division_for_display(context, inner);
            context.add(Expr::Neg(inner))
        }
        Expr::Hold(inner) => {
            let inner = collapse_nested_reciprocal_division_for_display(context, inner);
            context.add(Expr::Hold(inner))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| collapse_nested_reciprocal_division_for_display(context, item))
                .collect();
            context.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

fn signed_reciprocal_division(context: &Context, expr: ExprId) -> Option<(bool, ExprId)> {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Div(num, den) if is_signed_one_expr(context, *num) && !is_one_expr(context, *den) => {
            Some((is_negative_one_expr(context, *num), *den))
        }
        Expr::Neg(inner) => {
            let (negative, den) = signed_reciprocal_division(context, *inner)?;
            Some((!negative, den))
        }
        _ => None,
    }
}

fn is_signed_one_expr(context: &Context, expr: ExprId) -> bool {
    is_one_expr(context, expr) || is_negative_one_expr(context, expr)
}

fn is_one_expr(context: &Context, expr: ExprId) -> bool {
    matches!(
        context.get(cas_ast::hold::unwrap_internal_hold(context, expr)),
        Expr::Number(value) if value.is_one()
    )
}

fn is_negative_one_expr(context: &Context, expr: ExprId) -> bool {
    match context.get(cas_ast::hold::unwrap_internal_hold(context, expr)) {
        Expr::Number(value) => value == &-BigRational::one(),
        Expr::Neg(inner) => is_one_expr(context, *inner),
        _ => false,
    }
}

fn is_positive_integer_exponent_arithmetic(context: &Context, expr: ExprId) -> bool {
    !matches!(
        context.get(cas_ast::hold::unwrap_internal_hold(context, expr)),
        Expr::Number(_)
    ) && positive_integer_display_constant(context, expr).is_some()
}

fn is_negative_half_display_constant(context: &Context, expr: ExprId) -> bool {
    rational_display_constant(context, expr)
        .as_ref()
        .is_some_and(|value| value == &BigRational::new((-1).into(), 2.into()))
}

fn positive_integer_display_constant(context: &Context, expr: ExprId) -> Option<BigRational> {
    let value = rational_display_constant(context, expr)?;
    if value.denom().is_one() && value > BigRational::zero() {
        Some(value)
    } else {
        None
    }
}

fn rational_display_constant(context: &Context, expr: ExprId) -> Option<BigRational> {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Neg(inner) => Some(-rational_display_constant(context, *inner)?),
        Expr::Add(lhs, rhs) => Some(
            rational_display_constant(context, *lhs)? + rational_display_constant(context, *rhs)?,
        ),
        Expr::Sub(lhs, rhs) => Some(
            rational_display_constant(context, *lhs)? - rational_display_constant(context, *rhs)?,
        ),
        Expr::Div(num, den) => {
            let denominator = rational_display_constant(context, *den)?;
            if denominator.is_zero() {
                None
            } else {
                Some(rational_display_constant(context, *num)? / denominator)
            }
        }
        _ => None,
    }
}

fn is_e_constant(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    matches!(context.get(expr), Expr::Constant(Constant::E))
}

fn render_human_expr_without_display_normalization(context: &Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr { context, id: expr }
    ))
}

fn render_polynomial_exp_by_parts_after(context: &Context, expr: ExprId) -> String {
    let mut temp_ctx = context.clone();
    let presented = present_polynomial_exp_by_parts_product(&mut temp_ctx, expr);
    let human = if contains_exp_function_call(&temp_ctx, presented) {
        render_latex_human_expr_without_display_normalization(&temp_ctx, presented)
    } else {
        render_human_expr_without_display_normalization(&temp_ctx, presented)
    };
    move_leading_exp_power_after_parenthesized_factor(&human).unwrap_or(human)
}

fn present_polynomial_exp_by_parts_product(context: &mut Context, expr: ExprId) -> ExprId {
    let denominator_lifted = lift_numeric_denominator_to_front(context, expr);
    move_exp_factor_after_single_companion(context, denominator_lifted)
}

fn lift_numeric_denominator_to_front(context: &mut Context, expr: ExprId) -> ExprId {
    let Expr::Div(num, den) = context.get(expr).clone() else {
        return expr;
    };
    if cas_ast::views::as_rational_const(context, den, 4).is_none() {
        return expr;
    }

    let one = context.num(1);
    let reciprocal = context.add(Expr::Div(one, den));
    cas_math::expr_nary::build_balanced_mul(context, &[reciprocal, num])
}

fn move_exp_factor_after_single_companion(context: &mut Context, expr: ExprId) -> ExprId {
    let factors = cas_math::expr_nary::mul_leaves(context, expr);
    if factors.len() != 2 {
        return expr;
    }
    let [left, right] = [factors[0], factors[1]];
    if is_exp_call(context, left) && !is_exp_call(context, right) {
        return cas_math::expr_nary::build_balanced_mul(context, &[right, left]);
    }
    expr
}

fn is_exp_call(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Function(fn_id, _) => context.is_builtin(*fn_id, BuiltinFn::Exp),
        Expr::Pow(base, _) => matches!(context.get(*base), Expr::Constant(Constant::E)),
        _ => false,
    }
}

fn contains_exp_function_call(context: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(context, expr);
    match context.get(expr) {
        Expr::Function(fn_id, args) => {
            context.is_builtin(*fn_id, BuiltinFn::Exp)
                || args
                    .iter()
                    .copied()
                    .any(|arg| contains_exp_function_call(context, arg))
        }
        Expr::Add(lhs, rhs)
        | Expr::Sub(lhs, rhs)
        | Expr::Mul(lhs, rhs)
        | Expr::Div(lhs, rhs)
        | Expr::Pow(lhs, rhs) => {
            contains_exp_function_call(context, *lhs) || contains_exp_function_call(context, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_exp_function_call(context, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|item| contains_exp_function_call(context, item)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn render_latex_human_expr_without_display_normalization(
    context: &Context,
    expr: ExprId,
) -> String {
    let latex = cas_formatter::LaTeXExpr { context, id: expr }.to_latex();
    let human = cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(&latex));
    if human.trim().is_empty() {
        render_human_expr_without_display_normalization(context, expr)
    } else {
        human
    }
}

fn move_leading_exp_power_after_parenthesized_factor(human: &str) -> Option<String> {
    let prefix = "e^(";
    if !human.starts_with(prefix) {
        return None;
    }

    let mut depth = 0_i32;
    let mut exp_end = None;
    for (index, ch) in human.char_indices().skip(prefix.len() - 1) {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    exp_end = Some(index + ch.len_utf8());
                    break;
                }
            }
            _ => {}
        }
    }

    let exp_end = exp_end?;
    let exp_part = &human[..exp_end];
    let rest = human[exp_end..].strip_prefix(" · ")?;
    if !rest.starts_with('(') || !rest.ends_with(')') {
        return None;
    }

    Some(format!("{rest} · {exp_part}"))
}
