use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::{One, Signed, Zero};

pub(crate) fn public_required_condition_displays(
    ctx: &mut Context,
    cond: &crate::ImplicitCondition,
) -> Vec<String> {
    match cond {
        crate::ImplicitCondition::Positive(expr) => public_positive_condition_expr(ctx, *expr)
            .map(|expr| crate::ImplicitCondition::Positive(expr).display(ctx))
            .map(|display| vec![display])
            .unwrap_or_else(|| vec![cond.display(ctx)]),
        crate::ImplicitCondition::NonZero(expr) => public_nonzero_condition_exprs(ctx, *expr)
            .into_iter()
            .map(|expr| crate::ImplicitCondition::NonZero(expr).display(ctx))
            .collect(),
        _ => vec![cond.display(ctx)],
    }
}

pub(crate) fn public_condition_wires(
    ctx: &mut Context,
    cond: &crate::ImplicitCondition,
) -> Vec<(&'static str, ExprId)> {
    match cond {
        crate::ImplicitCondition::NonNegative(expr) => vec![("NonNegative", *expr)],
        crate::ImplicitCondition::LowerBound(expr, _) => vec![("LowerBound", *expr)],
        crate::ImplicitCondition::Positive(expr) => {
            vec![(
                "Positive",
                public_positive_condition_expr(ctx, *expr).unwrap_or(*expr),
            )]
        }
        crate::ImplicitCondition::NonZero(expr) => public_nonzero_condition_exprs(ctx, *expr)
            .into_iter()
            .map(|expr| ("NonZero", expr))
            .collect(),
        crate::ImplicitCondition::PrincipalBranch { arg, .. } => {
            vec![("PrincipalBranch", *arg)]
        }
    }
}

fn public_positive_condition_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (builtin, arg) = positive_trig_quotient_source_condition(ctx, expr)?;
    Some(source_builtin_expr(ctx, builtin, arg))
}

fn positive_trig_quotient_source_condition(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)) else {
        return None;
    };

    trig_quotient_arg(
        ctx,
        *numerator,
        *denominator,
        BuiltinFn::Sin,
        BuiltinFn::Cos,
    )
    .map(|arg| (BuiltinFn::Tan, arg))
    .or_else(|| {
        trig_quotient_arg(
            ctx,
            *numerator,
            *denominator,
            BuiltinFn::Cos,
            BuiltinFn::Sin,
        )
        .map(|arg| (BuiltinFn::Cot, arg))
    })
}

fn trig_quotient_arg(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let numerator_arg = unary_builtin_arg(ctx, numerator, numerator_builtin)?;
    let denominator_arg = unary_builtin_arg(ctx, denominator, denominator_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, numerator_arg, denominator_arg)
        .then_some(numerator_arg)
}

fn public_nonzero_condition_exprs(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    if let Some(exprs) = public_log_scaled_trig_shift_nonzero_exprs(ctx, expr) {
        return exprs;
    }

    if let Some(exprs) = public_linear_trig_combination_nonzero_exprs(ctx, expr) {
        return exprs;
    }

    if let Some(exprs) = public_shifted_trig_quotient_nonzero_exprs(ctx, expr) {
        return exprs;
    }

    let Some((kind, arg)) = reciprocal_trig_source_defined_condition(ctx, expr) else {
        return vec![expr];
    };

    match kind {
        ReciprocalTrigSourceCondition::Sec => vec![source_builtin_expr(ctx, BuiltinFn::Cos, arg)],
        ReciprocalTrigSourceCondition::Csc => vec![source_builtin_expr(ctx, BuiltinFn::Sin, arg)],
        ReciprocalTrigSourceCondition::Tan => vec![
            source_builtin_expr(ctx, BuiltinFn::Sin, arg),
            source_builtin_expr(ctx, BuiltinFn::Cos, arg),
        ],
        ReciprocalTrigSourceCondition::Cot => vec![
            source_builtin_expr(ctx, BuiltinFn::Cos, arg),
            source_builtin_expr(ctx, BuiltinFn::Sin, arg),
        ],
    }
}

fn public_log_scaled_trig_shift_nonzero_exprs(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ExprId>> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        Expr::Add(left, right) => {
            if let Some(positive_right) = negative_unit_factor_inner(ctx, *right) {
                (*left, positive_right)
            } else if let Some(positive_left) = negative_unit_factor_inner(ctx, *left) {
                (*right, positive_left)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let (offset, log_arg) = scaled_ln_arg(ctx, left)?;
    if numeric_expr_is_one(ctx, offset) {
        return None;
    }

    let (source_defined_builtin, source_builtin, arg, product_log_arg) =
        trig_quotient_log_product_divisor(ctx, right)?;
    if !cas_math::expr_domain::exprs_equivalent(ctx, log_arg, product_log_arg) {
        return None;
    }

    let source_defined = source_builtin_expr(ctx, source_defined_builtin, arg);
    let source = source_builtin_expr(ctx, source_builtin, arg);
    let shifted = ctx.add(Expr::Sub(offset, source));
    Some(vec![source_defined, shifted])
}

fn scaled_ln_arg(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Some(log_arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Ln) {
        let one = ctx.num(1);
        return Some((one, log_arg));
    }

    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(left, right) => (*left, *right),
        _ => return None,
    };
    scaled_ln_arg_from_factor(ctx, left, right)
        .or_else(|| scaled_ln_arg_from_factor(ctx, right, left))
}

fn scaled_ln_arg_from_factor(
    ctx: &Context,
    scale: ExprId,
    log_term: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Number(value) = ctx.get(cas_ast::hold::unwrap_hold(ctx, scale)) else {
        return None;
    };
    if value.is_zero() {
        return None;
    }
    let log_arg = unary_builtin_arg(ctx, log_term, BuiltinFn::Ln)?;
    Some((scale, log_arg))
}

fn trig_quotient_log_product_divisor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let (log_arg, source_builtin, arg) = trig_quotient_log_product(ctx, *numerator)?;
    let source_defined_builtin = match source_builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        _ => return None,
    };
    let denominator_arg = unary_builtin_arg(ctx, *denominator, source_defined_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, arg, denominator_arg).then_some((
        source_defined_builtin,
        source_builtin,
        arg,
        log_arg,
    ))
}

fn trig_quotient_log_product(ctx: &Context, expr: ExprId) -> Option<(ExprId, BuiltinFn, ExprId)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(left, right) => (*left, *right),
        _ => return None,
    };
    trig_quotient_log_product_from_factors(ctx, left, right)
        .or_else(|| trig_quotient_log_product_from_factors(ctx, right, left))
}

fn trig_quotient_log_product_from_factors(
    ctx: &Context,
    trig_term: ExprId,
    log_term: ExprId,
) -> Option<(ExprId, BuiltinFn, ExprId)> {
    let log_arg = unary_builtin_arg(ctx, log_term, BuiltinFn::Ln)?;
    let (source_builtin, source_arg) = positive_trig_quotient_source_condition(ctx, log_arg)?;
    let numerator_builtin = match source_builtin {
        BuiltinFn::Tan => BuiltinFn::Sin,
        BuiltinFn::Cot => BuiltinFn::Cos,
        _ => return None,
    };
    let trig_arg = unary_builtin_arg(ctx, trig_term, numerator_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, source_arg, trig_arg).then_some((
        log_arg,
        source_builtin,
        source_arg,
    ))
}

fn numeric_expr_is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)),
        Expr::Number(value) if value.is_one()
    )
}

fn negative_unit_factor_inner(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(left, right) if numeric_expr_is_negative_one(ctx, *left) => Some(*right),
        Expr::Mul(left, right) if numeric_expr_is_negative_one(ctx, *right) => Some(*left),
        _ => None,
    }
}

fn numeric_expr_is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)),
        Expr::Number(value) if value == &(-num_rational::BigRational::one())
    )
}

fn public_linear_trig_combination_nonzero_exprs(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ExprId>> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let (left, right, is_addition) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right, false),
        Expr::Add(left, right) => (*left, *right, true),
        _ => return None,
    };

    if is_addition {
        return linear_trig_additive_combination_nonzero_exprs(ctx, left, right)
            .or_else(|| linear_trig_additive_combination_nonzero_exprs(ctx, right, left));
    }

    linear_trig_combination_nonzero_exprs(
        ctx,
        left,
        right,
        ShiftedTrigQuotientOrientation::SourceMinusOffset,
    )
    .or_else(|| {
        linear_trig_combination_nonzero_exprs(
            ctx,
            right,
            left,
            ShiftedTrigQuotientOrientation::OffsetMinusSource,
        )
    })
}

fn linear_trig_additive_combination_nonzero_exprs(
    ctx: &mut Context,
    source_term: ExprId,
    negative_scaled_denominator_term: ExprId,
) -> Option<Vec<ExprId>> {
    linear_trig_negative_quotient_parts(
        ctx,
        source_term,
        negative_scaled_denominator_term,
        BuiltinFn::Sin,
        BuiltinFn::Cos,
        BuiltinFn::Tan,
    )
    .or_else(|| {
        linear_trig_negative_quotient_parts(
            ctx,
            source_term,
            negative_scaled_denominator_term,
            BuiltinFn::Cos,
            BuiltinFn::Sin,
            BuiltinFn::Cot,
        )
    })
    .map(|(source_defined_builtin, source_builtin, arg, offset)| {
        let source_defined = source_builtin_expr(ctx, source_defined_builtin, arg);
        let source = source_builtin_expr(ctx, source_builtin, arg);
        let shifted = ctx.add(Expr::Sub(source, offset));
        vec![source_defined, shifted]
    })
}

fn linear_trig_combination_nonzero_exprs(
    ctx: &mut Context,
    source_term: ExprId,
    scaled_denominator_term: ExprId,
    orientation: ShiftedTrigQuotientOrientation,
) -> Option<Vec<ExprId>> {
    linear_trig_quotient_parts(
        ctx,
        source_term,
        scaled_denominator_term,
        BuiltinFn::Sin,
        BuiltinFn::Cos,
        BuiltinFn::Tan,
    )
    .or_else(|| {
        linear_trig_quotient_parts(
            ctx,
            source_term,
            scaled_denominator_term,
            BuiltinFn::Cos,
            BuiltinFn::Sin,
            BuiltinFn::Cot,
        )
    })
    .map(|(source_defined_builtin, source_builtin, arg, offset)| {
        let source_defined = source_builtin_expr(ctx, source_defined_builtin, arg);
        let source = source_builtin_expr(ctx, source_builtin, arg);
        let shifted = match orientation {
            ShiftedTrigQuotientOrientation::SourceMinusOffset => ctx.add(Expr::Sub(source, offset)),
            ShiftedTrigQuotientOrientation::OffsetMinusSource => ctx.add(Expr::Sub(offset, source)),
        };
        vec![source_defined, shifted]
    })
}

fn linear_trig_quotient_parts(
    ctx: &mut Context,
    source_term: ExprId,
    scaled_denominator_term: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
    source_builtin: BuiltinFn,
) -> Option<(BuiltinFn, BuiltinFn, ExprId, ExprId)> {
    let source_arg = unary_builtin_arg(ctx, source_term, numerator_builtin)?;
    let (offset, denominator_arg) =
        scaled_numeric_builtin_arg(ctx, scaled_denominator_term, denominator_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, source_arg, denominator_arg).then_some((
        denominator_builtin,
        source_builtin,
        source_arg,
        offset,
    ))
}

fn linear_trig_negative_quotient_parts(
    ctx: &mut Context,
    source_term: ExprId,
    negative_scaled_denominator_term: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
    source_builtin: BuiltinFn,
) -> Option<(BuiltinFn, BuiltinFn, ExprId, ExprId)> {
    let source_arg = unary_builtin_arg(ctx, source_term, numerator_builtin)?;
    let (offset, denominator_arg) = negative_scaled_numeric_builtin_arg(
        ctx,
        negative_scaled_denominator_term,
        denominator_builtin,
    )?;
    cas_math::expr_domain::exprs_equivalent(ctx, source_arg, denominator_arg).then_some((
        denominator_builtin,
        source_builtin,
        source_arg,
        offset,
    ))
}

fn scaled_numeric_builtin_arg(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, builtin) {
        let one = ctx.num(1);
        return Some((one, arg));
    }

    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    scaled_numeric_builtin_arg_from_factor(ctx, *left, *right, builtin)
        .or_else(|| scaled_numeric_builtin_arg_from_factor(ctx, *right, *left, builtin))
}

fn negative_scaled_numeric_builtin_arg(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    if let Expr::Neg(inner) = ctx.get(expr) {
        return scaled_numeric_builtin_arg(ctx, *inner, builtin);
    }

    let (left, right) = match ctx.get(expr) {
        Expr::Mul(left, right) => (*left, *right),
        _ => return None,
    };
    negative_scaled_numeric_builtin_arg_from_factor(ctx, left, right, builtin)
        .or_else(|| negative_scaled_numeric_builtin_arg_from_factor(ctx, right, left, builtin))
}

fn scaled_numeric_builtin_arg_from_factor(
    ctx: &Context,
    scale: ExprId,
    trig_term: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let Expr::Number(value) = ctx.get(cas_ast::hold::unwrap_hold(ctx, scale)) else {
        return None;
    };
    if value.is_zero() {
        return None;
    }
    let arg = unary_builtin_arg(ctx, trig_term, builtin)?;
    Some((scale, arg))
}

fn negative_scaled_numeric_builtin_arg_from_factor(
    ctx: &mut Context,
    scale: ExprId,
    trig_term: ExprId,
    builtin: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let Expr::Number(value) = ctx.get(cas_ast::hold::unwrap_hold(ctx, scale)) else {
        return None;
    };
    if !value.is_negative() {
        return None;
    }
    let positive_scale = ctx.add(Expr::Number(-value.clone()));
    let arg = unary_builtin_arg(ctx, trig_term, builtin)?;
    Some((positive_scale, arg))
}

fn public_shifted_trig_quotient_nonzero_exprs(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<ExprId>> {
    let (quotient, offset, orientation) = shifted_trig_quotient_difference(ctx, expr)?;
    let (source_builtin, arg) = positive_trig_quotient_source_condition(ctx, quotient)?;
    let source_defined_builtin = match source_builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        _ => return None,
    };
    let source_defined = source_builtin_expr(ctx, source_defined_builtin, arg);
    let source = source_builtin_expr(ctx, source_builtin, arg);
    let shifted = match orientation {
        ShiftedTrigQuotientOrientation::SourceMinusOffset => ctx.add(Expr::Sub(source, offset)),
        ShiftedTrigQuotientOrientation::OffsetMinusSource => ctx.add(Expr::Sub(offset, source)),
    };
    Some(vec![source_defined, shifted])
}

#[derive(Clone, Copy)]
enum ShiftedTrigQuotientOrientation {
    SourceMinusOffset,
    OffsetMinusSource,
}

fn shifted_trig_quotient_difference(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ShiftedTrigQuotientOrientation)> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) if positive_trig_quotient_source_condition(ctx, *left).is_some() => {
            Some((
                *left,
                *right,
                ShiftedTrigQuotientOrientation::SourceMinusOffset,
            ))
        }
        Expr::Sub(left, right)
            if positive_trig_quotient_source_condition(ctx, *right).is_some() =>
        {
            Some((
                *right,
                *left,
                ShiftedTrigQuotientOrientation::OffsetMinusSource,
            ))
        }
        Expr::Add(left, right) => shifted_trig_quotient_additive_difference(ctx, *left, *right),
        _ => None,
    }
}

fn shifted_trig_quotient_additive_difference(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId, ShiftedTrigQuotientOrientation)> {
    if positive_trig_quotient_source_condition(ctx, left).is_some() {
        if let Expr::Neg(offset) = ctx.get(right) {
            return Some((
                left,
                *offset,
                ShiftedTrigQuotientOrientation::SourceMinusOffset,
            ));
        }
    }
    if positive_trig_quotient_source_condition(ctx, right).is_some() {
        if let Expr::Neg(offset) = ctx.get(left) {
            return Some((
                right,
                *offset,
                ShiftedTrigQuotientOrientation::SourceMinusOffset,
            ));
        }
    }
    if let Expr::Neg(quotient) = ctx.get(right) {
        if positive_trig_quotient_source_condition(ctx, *quotient).is_some() {
            return Some((
                *quotient,
                left,
                ShiftedTrigQuotientOrientation::OffsetMinusSource,
            ));
        }
    }
    if let Expr::Neg(quotient) = ctx.get(left) {
        if positive_trig_quotient_source_condition(ctx, *quotient).is_some() {
            return Some((
                *quotient,
                right,
                ShiftedTrigQuotientOrientation::OffsetMinusSource,
            ));
        }
    }
    None
}

#[derive(Clone, Copy)]
enum ReciprocalTrigSourceCondition {
    Tan,
    Cot,
    Sec,
    Csc,
}

fn reciprocal_trig_source_defined_condition(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ReciprocalTrigSourceCondition, ExprId)> {
    for (builtin, kind) in [
        (BuiltinFn::Tan, ReciprocalTrigSourceCondition::Tan),
        (BuiltinFn::Cot, ReciprocalTrigSourceCondition::Cot),
        (BuiltinFn::Sec, ReciprocalTrigSourceCondition::Sec),
        (BuiltinFn::Csc, ReciprocalTrigSourceCondition::Csc),
    ] {
        if let Some(arg) = unary_builtin_arg(ctx, expr, builtin) {
            return Some((kind, arg));
        }
    }
    None
}

fn source_builtin_expr(ctx: &mut Context, builtin: BuiltinFn, arg: ExprId) -> ExprId {
    ctx.add(Expr::Function(ctx.builtin_id(builtin), vec![arg]))
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}
