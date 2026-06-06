use cas_ast::{BuiltinFn, Context, Expr, ExprId};

pub(super) fn collect_reciprocal_trig_log_denominator_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Vec<ExprId> {
    let mut conditions = Vec::new();
    collect_from_expr(ctx, expr, &mut conditions);
    conditions
}

fn collect_from_expr(ctx: &mut Context, expr: ExprId, conditions: &mut Vec<ExprId>) {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
        {
            let log_arg = unwrap_abs_arg(ctx, args[0]);
            if let Some((required_builtin, arg)) =
                reciprocal_trig_log_denominator_condition_arg(ctx, log_arg)
            {
                conditions.push(ctx.call_builtin(required_builtin, vec![arg]));
            }
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_from_expr(ctx, arg, conditions);
            }
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_from_expr(ctx, left, conditions);
            collect_from_expr(ctx, right, conditions);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_from_expr(ctx, inner, conditions);
        }
        Expr::Matrix { data, .. } => {
            for entry in data {
                collect_from_expr(ctx, entry, conditions);
            }
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
    }
}

fn unwrap_abs_arg(ctx: &Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Abs) =>
        {
            args[0]
        }
        _ => expr,
    }
}

fn reciprocal_trig_log_denominator_condition_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    sec_tan_sum_arg(ctx, expr)
        .map(|arg| (BuiltinFn::Cos, arg))
        .or_else(|| csc_cot_difference_arg(ctx, expr).map(|arg| (BuiltinFn::Sin, arg)))
}

fn sec_tan_sum_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    unordered_same_arg_unary_pair(ctx, *left, BuiltinFn::Sec, *right, BuiltinFn::Tan)
}

fn csc_cot_difference_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, *right, BuiltinFn::Cot)
        }
        Expr::Add(left, right) => unordered_same_arg_unary_negated_pair(
            ctx,
            *left,
            BuiltinFn::Csc,
            *right,
            BuiltinFn::Cot,
        ),
        _ => None,
    }
}

fn unordered_same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    same_arg_unary_pair(ctx, left, left_builtin, right, right_builtin)
        .or_else(|| same_arg_unary_pair(ctx, right, left_builtin, left, right_builtin))
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = unary_builtin_arg(ctx, left, left_builtin)?;
    let right_arg = unary_builtin_arg(ctx, right, right_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, left_arg, right_arg).then_some(left_arg)
}

fn unordered_same_arg_unary_negated_pair(
    ctx: &Context,
    left: ExprId,
    positive_builtin: BuiltinFn,
    right: ExprId,
    negative_builtin: BuiltinFn,
) -> Option<ExprId> {
    same_arg_unary_negated_pair(ctx, left, positive_builtin, right, negative_builtin).or_else(
        || same_arg_unary_negated_pair(ctx, right, positive_builtin, left, negative_builtin),
    )
}

fn same_arg_unary_negated_pair(
    ctx: &Context,
    positive: ExprId,
    positive_builtin: BuiltinFn,
    negative: ExprId,
    negative_builtin: BuiltinFn,
) -> Option<ExprId> {
    let positive_arg = unary_builtin_arg(ctx, positive, positive_builtin)?;
    let negative_arg = negated_unary_builtin_arg(ctx, negative, negative_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, positive_arg, negative_arg).then_some(positive_arg)
}

fn negated_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    expected_builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Neg(inner) = ctx.get(expr) else {
        return None;
    };
    unary_builtin_arg(ctx, *inner, expected_builtin)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, expected_builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (args.len() == 1 && ctx.builtin_of(*fn_id) == Some(expected_builtin)).then_some(args[0])
}
