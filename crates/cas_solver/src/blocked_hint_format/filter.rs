use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_traits::Zero;
use std::collections::BTreeSet;

fn implicit_domain_condition_displays(ctx: &Context, expr: ExprId) -> BTreeSet<String> {
    crate::infer_implicit_domain(ctx, expr, crate::ValueDomain::RealOnly)
        .conditions()
        .iter()
        .map(|condition| condition.display(ctx))
        .collect()
}

fn explicit_condition_displays(
    ctx: &Context,
    conditions: &[crate::ImplicitCondition],
) -> BTreeSet<String> {
    conditions
        .iter()
        .map(|condition| condition.display(ctx))
        .collect()
}

fn reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)) {
        Expr::Div(numerator, denominator) if integer_literal_is(ctx, *numerator, 1) => {
            Some(*denominator)
        }
        Expr::Pow(base, exponent) if integer_literal_is(ctx, *exponent, -1) => Some(*base),
        _ => None,
    }
}

fn integer_literal_is(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)),
        Expr::Number(n) if n == &num_rational::BigRational::from_integer(value.into())
    )
}

fn reciprocal_defined_hint_is_covered_by_required_nonzero(
    ctx: &Context,
    conditions: &[crate::ImplicitCondition],
    hint: &crate::BlockedHint,
) -> bool {
    if hint.key.kind() != "defined" || !hint.suggestion.starts_with("cycle detected") {
        return false;
    }

    conditions.iter().any(|condition| {
        let crate::ImplicitCondition::NonZero(required) = condition else {
            return false;
        };
        reciprocal_denominator(ctx, hint.expr_id).is_some_and(|denominator| {
            cas_math::expr_domain::exprs_equivalent(ctx, *required, denominator)
        }) || reciprocal_trig_defined_hint_is_covered_by_required_nonzero(
            ctx,
            hint.expr_id,
            *required,
        )
    })
}

fn reciprocal_defined_hint_is_covered_by_result_domain(
    ctx: &Context,
    result: ExprId,
    hint: &crate::BlockedHint,
) -> bool {
    if hint.key.kind() != "defined" || !hint.suggestion.starts_with("cycle detected") {
        return false;
    }

    let domain = crate::infer_implicit_domain(ctx, result, crate::ValueDomain::RealOnly);
    domain.conditions().iter().any(|condition| {
        let crate::ImplicitCondition::NonZero(required) = condition else {
            return false;
        };
        reciprocal_denominator(ctx, hint.expr_id).is_some_and(|denominator| {
            cas_math::expr_domain::exprs_equivalent(ctx, *required, denominator)
        }) || reciprocal_trig_defined_hint_is_covered_by_required_nonzero(
            ctx,
            hint.expr_id,
            *required,
        )
    })
}

fn reciprocal_trig_defined_hint_is_covered_by_required_nonzero(
    ctx: &Context,
    hint_expr: ExprId,
    required_nonzero: ExprId,
) -> bool {
    unary_builtin_arg(ctx, hint_expr, BuiltinFn::Sec)
        .zip(unary_builtin_arg(ctx, required_nonzero, BuiltinFn::Cos))
        .or_else(|| {
            unary_builtin_arg(ctx, hint_expr, BuiltinFn::Csc).zip(unary_builtin_arg(
                ctx,
                required_nonzero,
                BuiltinFn::Sin,
            ))
        })
        .is_some_and(|(hint_arg, required_arg)| {
            cas_math::expr_domain::exprs_equivalent(ctx, hint_arg, required_arg)
        })
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(cas_ast::hold::unwrap_hold(ctx, expr)) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) => {
            Some(args[0])
        }
        _ => None,
    }
}

fn cycle_defined_hint_is_covered_by_result_domain(
    ctx: &Context,
    result: ExprId,
    required_conditions: &[crate::ImplicitCondition],
    hint: &crate::BlockedHint,
) -> bool {
    if hint.key.kind() != "defined" || !hint.suggestion.starts_with("cycle detected") {
        return false;
    }

    let hint_conditions = implicit_domain_condition_displays(ctx, hint.expr_id);
    if hint_conditions.is_empty() {
        return false;
    }

    let result_conditions = implicit_domain_condition_displays(ctx, result);
    let explicit_conditions = explicit_condition_displays(ctx, required_conditions);
    hint_conditions.iter().all(|condition| {
        result_conditions.contains(condition) || explicit_conditions.contains(condition)
    })
}

/// Filter blocked hints for eval display.
///
/// When the resolved result is `Undefined`, drops `defined` hints because
/// they are often cycle-artifacts and not actionable.
pub fn filter_blocked_hints_for_eval(
    ctx: &Context,
    resolved: ExprId,
    required_conditions: &[crate::ImplicitCondition],
    hints: &[crate::BlockedHint],
) -> Vec<crate::BlockedHint> {
    let result_is_undefined = matches!(
        ctx.get(resolved),
        cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
    );

    let result_is_zero = matches!(ctx.get(resolved), Expr::Number(value) if value.is_zero());

    hints
        .iter()
        .filter(|hint| {
            !(hint.key.kind() == "defined"
                && (result_is_undefined
                    || (result_is_zero && hint.suggestion.starts_with("cycle detected"))
                    || cycle_defined_hint_is_covered_by_result_domain(
                        ctx,
                        resolved,
                        required_conditions,
                        hint,
                    )
                    || reciprocal_defined_hint_is_covered_by_required_nonzero(
                        ctx,
                        required_conditions,
                        hint,
                    )
                    || reciprocal_defined_hint_is_covered_by_result_domain(ctx, resolved, hint)))
        })
        .cloned()
        .collect()
}
