use cas_api_models::{AssumptionDto, BlockedHintDto, RequiredConditionWire, WarningWire};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashSet;

use crate::eval_output_condition_filter::AssumedConditionFilter;

pub(crate) fn collect_output_warnings(
    domain_warnings: &[crate::DomainWarning],
    assumptions_used: &[AssumptionDto],
) -> Vec<WarningWire> {
    let assumed_display: HashSet<&str> = assumptions_used
        .iter()
        .map(|assumption| assumption.display.as_str())
        .collect();
    domain_warnings
        .iter()
        .filter(|w| !assumed_display.contains(w.message.as_str()))
        .map(|w| WarningWire {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_output_required_conditions(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<RequiredConditionWire> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    let normalized = normalize_and_dedupe_conditions(ctx, required_conditions);
    let visible_conditions = visible_required_conditions_after_public_suppression(
        ctx,
        &normalized,
        &assumed_filter,
        result_display,
    );

    visible_conditions
        .iter()
        .filter(|cond| !required_condition_wire_is_redundant(ctx, cond, &visible_conditions))
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| {
            let cond = *cond;
            let (kind, expr_id) = match cond {
                crate::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                crate::ImplicitCondition::LowerBound(e, _) => ("LowerBound", *e),
                crate::ImplicitCondition::Positive(e) => ("Positive", *e),
                crate::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = expr_display(ctx, expr_id);
            let expr_display =
                apply_input_inverse_trig_alias_preferences(&expr_str, raw_input, result_display);
            RequiredConditionWire {
                kind: kind.to_string(),
                expr_display,
                expr_canonical: expr_str,
            }
        })
        .collect()
}

fn required_condition_wire_is_redundant(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let crate::ImplicitCondition::NonZero(nonzero_expr) = cond else {
        return matches!(cond, crate::ImplicitCondition::Positive(positive_expr) if
            reciprocal_trig_log_positive_quotient_condition_is_redundant(
                ctx,
                *positive_expr,
                visible_conditions
            )
        );
    };

    if calculus_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    reciprocal_trig_log_argument_condition_is_redundant(ctx, *nonzero_expr, visible_conditions)
}

pub(crate) fn collect_output_required_display(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
    raw_input: &str,
    result_display: Option<&str>,
) -> Vec<String> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    let normalized = normalize_and_dedupe_conditions(ctx, required_conditions);
    let visible_conditions = visible_required_conditions_after_public_suppression(
        ctx,
        &normalized,
        &assumed_filter,
        result_display,
    );

    visible_conditions
        .iter()
        .filter(|cond| !required_condition_is_redundant(ctx, cond, &visible_conditions))
        .map(|cond| {
            apply_input_inverse_trig_alias_preferences(
                &cond.display(ctx),
                raw_input,
                result_display,
            )
        })
        .collect()
}

fn visible_required_conditions_after_public_suppression<'a>(
    ctx: &Context,
    normalized: &'a [crate::ImplicitCondition],
    assumed_filter: &AssumedConditionFilter,
    result_display: Option<&str>,
) -> Vec<&'a crate::ImplicitCondition> {
    normalized
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .filter(|cond| !should_suppress_public_required_condition(ctx, cond, result_display))
        .collect()
}

fn should_suppress_public_required_condition(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    result_display: Option<&str>,
) -> bool {
    let Some(display) = result_display.map(str::trim_start) else {
        return false;
    };
    if !(display.starts_with("limit(") || display == "undefined") {
        return false;
    }

    matches!(
        cond,
        crate::ImplicitCondition::NonZero(expr) if is_integer_literal(ctx, *expr, 0)
    )
}

fn required_condition_is_redundant(
    ctx: &Context,
    cond: &crate::ImplicitCondition,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let crate::ImplicitCondition::NonZero(nonzero_expr) = cond else {
        return matches!(cond, crate::ImplicitCondition::Positive(positive_expr) if
            reciprocal_trig_log_positive_quotient_condition_is_redundant(
                ctx,
                *positive_expr,
                visible_conditions
            )
        );
    };

    if unit_interval_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    if calculus_nonzero_condition_is_redundant(ctx, *nonzero_expr, visible_conditions) {
        return true;
    }

    reciprocal_trig_log_argument_condition_is_redundant(ctx, *nonzero_expr, visible_conditions)
}

#[derive(Clone, Copy)]
enum ReciprocalTrigLogPositiveKind {
    SecTan,
    CscCot,
}

fn reciprocal_trig_log_positive_quotient_condition_is_redundant(
    ctx: &Context,
    positive_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    if reciprocal_trig_log_positive_quotient_display_condition_is_redundant(
        ctx,
        positive_expr,
        visible_conditions,
    ) {
        return true;
    }

    let Some((kind, arg)) = reciprocal_trig_log_positive_quotient_arg(ctx, positive_expr) else {
        return false;
    };

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::Positive(candidate_expr) = candidate else {
            return false;
        };
        if *candidate_expr == positive_expr {
            return false;
        }

        let candidate_arg = match kind {
            ReciprocalTrigLogPositiveKind::SecTan => sec_tan_sum_arg(ctx, *candidate_expr),
            ReciprocalTrigLogPositiveKind::CscCot => csc_cot_difference_arg(ctx, *candidate_expr),
        };
        candidate_arg.is_some_and(|candidate_arg| {
            cas_math::expr_domain::exprs_equivalent(ctx, candidate_arg, arg)
        })
    })
}

fn reciprocal_trig_log_positive_quotient_display_condition_is_redundant(
    ctx: &Context,
    positive_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let positive_display = expr_display(ctx, positive_expr);
    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::Positive(candidate_expr) = candidate else {
            return false;
        };
        if *candidate_expr == positive_expr {
            return false;
        }

        if let Some(arg) = sec_tan_sum_arg(ctx, *candidate_expr) {
            let arg_display = expr_display(ctx, arg);
            return positive_display == format!("(sin({arg_display}) + 1) / cos({arg_display})")
                || positive_display == format!("(1 + sin({arg_display})) / cos({arg_display})");
        }

        if let Some(arg) = csc_cot_difference_arg(ctx, *candidate_expr) {
            let arg_display = expr_display(ctx, arg);
            return positive_display == format!("(1 - cos({arg_display})) / sin({arg_display})");
        }

        false
    })
}

fn reciprocal_trig_log_positive_quotient_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ReciprocalTrigLogPositiveKind, ExprId)> {
    sec_tan_compact_quotient_arg(ctx, expr)
        .map(|arg| (ReciprocalTrigLogPositiveKind::SecTan, arg))
        .or_else(|| {
            csc_cot_compact_quotient_arg(ctx, expr)
                .map(|arg| (ReciprocalTrigLogPositiveKind::CscCot, arg))
        })
}

fn sec_tan_compact_quotient_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let denominator_arg = unary_builtin_arg(ctx, *denominator, BuiltinFn::Cos)?;
    let numerator_arg = one_plus_unary_builtin_arg(ctx, *numerator, BuiltinFn::Sin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, numerator_arg)
        .then_some(denominator_arg)
}

fn csc_cot_compact_quotient_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let denominator_arg = unary_builtin_arg(ctx, *denominator, BuiltinFn::Sin)?;
    let numerator_arg = one_minus_unary_builtin_arg(ctx, *numerator, BuiltinFn::Cos)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, numerator_arg)
        .then_some(denominator_arg)
}

fn one_plus_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    if is_integer_literal(ctx, *left, 1) {
        unary_builtin_arg(ctx, *right, builtin)
    } else if is_integer_literal(ctx, *right, 1) {
        unary_builtin_arg(ctx, *left, builtin)
    } else {
        None
    }
}

fn one_minus_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => {
            unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_unary_builtin_arg(ctx, *left, builtin)
        }
        _ => None,
    }
}

fn unit_interval_nonzero_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonNegative(gap_expr) = candidate else {
            return false;
        };
        let Some(denominator) = exterior_unit_interval_denominator(ctx, *gap_expr) else {
            return false;
        };
        cas_math::expr_domain::exprs_equivalent(ctx, denominator, nonzero_expr)
    })
}

fn reciprocal_trig_log_argument_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    let Some((required_builtin, arg)) = reciprocal_trig_log_argument_denominator(ctx, nonzero_expr)
    else {
        return false;
    };

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonZero(candidate_expr) = candidate else {
            return false;
        };
        unary_builtin_arg(ctx, *candidate_expr, required_builtin).is_some_and(|candidate_arg| {
            cas_math::expr_domain::exprs_equivalent(ctx, candidate_arg, arg)
        })
    })
}

fn calculus_nonzero_condition_is_redundant(
    ctx: &Context,
    nonzero_expr: ExprId,
    visible_conditions: &[&crate::ImplicitCondition],
) -> bool {
    if !expr_contains_calculus_call(ctx, nonzero_expr) {
        return false;
    }

    visible_conditions.iter().any(|candidate| {
        let crate::ImplicitCondition::NonZero(candidate_expr) = candidate else {
            return false;
        };
        *candidate_expr != nonzero_expr
            && (cas_math::expr_domain::exprs_equivalent(ctx, *candidate_expr, nonzero_expr)
                || nonzero_condition_is_candidate_plus_antiderivative_residual(
                    ctx,
                    nonzero_expr,
                    *candidate_expr,
                ))
    })
}

#[derive(Clone, Copy)]
struct SignedTerm {
    expr: ExprId,
    positive: bool,
}

fn nonzero_condition_is_candidate_plus_antiderivative_residual(
    ctx: &Context,
    nonzero_expr: ExprId,
    candidate_expr: ExprId,
) -> bool {
    let mut terms = Vec::new();
    collect_signed_add_terms(ctx, nonzero_expr, true, &mut terms);

    let mut candidate_terms = Vec::new();
    collect_signed_add_terms(ctx, candidate_expr, true, &mut candidate_terms);

    for candidate_term in candidate_terms {
        let Some(index) = terms.iter().position(|term| {
            term.positive == candidate_term.positive
                && cas_math::expr_domain::exprs_equivalent(ctx, term.expr, candidate_term.expr)
        }) else {
            return false;
        };
        terms.remove(index);
    }

    signed_terms_are_antiderivative_residual(ctx, &terms)
}

fn collect_signed_add_terms(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<SignedTerm>,
) {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_add_terms(ctx, *left, positive, terms);
            collect_signed_add_terms(ctx, *right, positive, terms);
        }
        Expr::Sub(left, right) => {
            collect_signed_add_terms(ctx, *left, positive, terms);
            collect_signed_add_terms(ctx, *right, !positive, terms);
        }
        Expr::Neg(inner) => collect_signed_add_terms(ctx, *inner, !positive, terms),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => {
            collect_signed_add_terms(ctx, *right, !positive, terms)
        }
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => {
            collect_signed_add_terms(ctx, *left, !positive, terms)
        }
        _ => terms.push(SignedTerm { expr, positive }),
    }
}

fn signed_terms_are_antiderivative_residual(ctx: &Context, terms: &[SignedTerm]) -> bool {
    if terms.len() != 2 || terms[0].positive == terms[1].positive {
        return false;
    }

    diff_integrate_integrand(ctx, terms[0].expr).is_some_and(|integrand| {
        cas_math::expr_domain::exprs_equivalent(ctx, integrand, terms[1].expr)
    }) || diff_integrate_integrand(ctx, terms[1].expr).is_some_and(|integrand| {
        cas_math::expr_domain::exprs_equivalent(ctx, integrand, terms[0].expr)
    })
}

fn diff_integrate_integrand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Function(diff_fn, diff_args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*diff_fn) != "diff" || diff_args.len() != 2 {
        return None;
    }

    let Expr::Function(integrate_fn, integrate_args) =
        ctx.get(cas_ast::hold::unwrap_hold(ctx, diff_args[0]))
    else {
        return None;
    };
    if ctx.sym_name(*integrate_fn) != "integrate" || integrate_args.len() != 2 {
        return None;
    }

    cas_math::expr_domain::exprs_equivalent(ctx, diff_args[1], integrate_args[1])
        .then_some(integrate_args[0])
}

fn expr_contains_calculus_call(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Function(fn_id, args) => {
            matches!(ctx.sym_name(*fn_id), "diff" | "integrate" | "limit")
                || args
                    .iter()
                    .any(|arg| expr_contains_calculus_call(ctx, *arg))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expr_contains_calculus_call(ctx, *left) || expr_contains_calculus_call(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_calculus_call(ctx, *inner),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| expr_contains_calculus_call(ctx, *entry)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn reciprocal_trig_log_argument_denominator(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    sec_tan_sum_arg(ctx, expr)
        .or_else(|| sec_tan_quotient_sum_arg(ctx, expr))
        .map(|arg| (BuiltinFn::Cos, arg))
        .or_else(|| {
            csc_cot_difference_arg(ctx, expr)
                .or_else(|| csc_cot_quotient_difference_arg(ctx, expr))
                .map(|arg| (BuiltinFn::Sin, arg))
        })
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
        Expr::Add(left, right) => unary_builtin_arg(ctx, *left, BuiltinFn::Csc)
            .zip(negated_unary_builtin_arg(ctx, *right, BuiltinFn::Cot))
            .or_else(|| {
                unary_builtin_arg(ctx, *right, BuiltinFn::Csc).zip(negated_unary_builtin_arg(
                    ctx,
                    *left,
                    BuiltinFn::Cot,
                ))
            })
            .and_then(|(left_arg, right_arg)| {
                cas_math::expr_domain::exprs_equivalent(ctx, left_arg, right_arg)
                    .then_some(left_arg)
            }),
        _ => None,
    }
}

fn sec_tan_quotient_sum_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    reciprocal_plus_ratio_arg(ctx, *left, *right, BuiltinFn::Sin, BuiltinFn::Cos)
        .or_else(|| reciprocal_plus_ratio_arg(ctx, *right, *left, BuiltinFn::Sin, BuiltinFn::Cos))
}

fn csc_cot_quotient_difference_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            reciprocal_plus_ratio_arg(ctx, *left, *right, BuiltinFn::Cos, BuiltinFn::Sin)
        }
        Expr::Add(left, right) => reciprocal_plus_ratio_arg_with_negated_ratio(
            ctx,
            *left,
            *right,
            BuiltinFn::Cos,
            BuiltinFn::Sin,
        )
        .or_else(|| {
            reciprocal_plus_ratio_arg_with_negated_ratio(
                ctx,
                *right,
                *left,
                BuiltinFn::Cos,
                BuiltinFn::Sin,
            )
        }),
        _ => None,
    }
}

fn reciprocal_plus_ratio_arg(
    ctx: &Context,
    reciprocal: ExprId,
    ratio: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let denominator_arg = reciprocal_builtin_denominator_arg(ctx, reciprocal, denominator_builtin)?;
    let ratio_arg =
        ratio_builtin_denominator_arg(ctx, ratio, numerator_builtin, denominator_builtin)?;
    cas_math::expr_domain::exprs_equivalent(ctx, denominator_arg, ratio_arg)
        .then_some(denominator_arg)
}

fn reciprocal_plus_ratio_arg_with_negated_ratio(
    ctx: &Context,
    reciprocal: ExprId,
    ratio: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let ratio = match ctx.get(cas_ast::hold::unwrap_hold(ctx, ratio)) {
        Expr::Neg(inner) => *inner,
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => *right,
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => *left,
        _ => return None,
    };
    reciprocal_plus_ratio_arg(
        ctx,
        reciprocal,
        ratio,
        numerator_builtin,
        denominator_builtin,
    )
}

fn reciprocal_builtin_denominator_arg(
    ctx: &Context,
    expr: ExprId,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *numerator, 1) {
        unary_builtin_arg(ctx, *denominator, denominator_builtin)
    } else {
        None
    }
}

fn ratio_builtin_denominator_arg(
    ctx: &Context,
    expr: ExprId,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    same_arg_unary_pair(
        ctx,
        *numerator,
        numerator_builtin,
        *denominator,
        denominator_builtin,
    )
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

fn negated_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Neg(inner) => unary_builtin_arg(ctx, *inner, builtin),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => {
            unary_builtin_arg(ctx, *right, builtin)
        }
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => {
            unary_builtin_arg(ctx, *left, builtin)
        }
        _ => None,
    }
}

fn exterior_unit_interval_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let base = unit_interval_base(ctx, expr)?;
    match ctx.get(base) {
        Expr::Div(numerator, denominator) if is_integer_literal(ctx, *numerator, 1) => {
            Some(*denominator)
        }
        Expr::Pow(inner, exponent) if is_integer_literal(ctx, *exponent, -1) => Some(*inner),
        _ => None,
    }
}

fn unit_interval_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_integer_literal(ctx, *left, 1) => squared_base(ctx, *right),
        Expr::Add(left, right) if is_integer_literal(ctx, *left, 1) => {
            negated_squared_base(ctx, *right)
        }
        Expr::Add(left, right) if is_integer_literal(ctx, *right, 1) => {
            negated_squared_base(ctx, *left)
        }
        _ => None,
    }
}

fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if is_integer_literal(ctx, *exponent, 2) {
        Some(*base)
    } else {
        None
    }
}

fn negated_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => squared_base(ctx, *inner),
        Expr::Mul(left, right) if is_integer_literal(ctx, *left, -1) => squared_base(ctx, *right),
        Expr::Mul(left, right) if is_integer_literal(ctx, *right, -1) => squared_base(ctx, *left),
        _ => None,
    }
}

fn is_integer_literal(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n) if n == &num_rational::BigRational::from_integer(value.into())
    )
}

pub(crate) fn collect_output_blocked_hints(
    ctx: &mut Context,
    resolved: cas_ast::ExprId,
    required_conditions: &[crate::ImplicitCondition],
    blocked_hints: &[crate::BlockedHint],
) -> Vec<BlockedHintDto> {
    let normalized_required_conditions = normalize_and_dedupe_conditions(ctx, required_conditions);
    crate::filter_blocked_hints_for_eval(
        ctx,
        resolved,
        &normalized_required_conditions,
        blocked_hints,
    )
    .iter()
    .map(|hint| BlockedHintDto {
        rule: hint.rule.clone(),
        requires: vec![crate::format_blocked_hint_condition(ctx, hint)],
        tip: hint.suggestion.to_string(),
    })
    .collect()
}

pub(crate) fn collect_output_assumptions_used(steps: &[crate::Step]) -> Vec<AssumptionDto> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut assumptions = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let kind = event.key.kind().to_string();
            let rule = step.rule_name.to_string();
            let display = event.message.clone();
            let expr_canonical = event.expr_display.clone();
            if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
                continue;
            }

            assumptions.push(AssumptionDto {
                kind,
                display,
                expr_canonical,
                rule,
            });
        }
    }

    assumptions
}

fn expr_display(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr_id,
    }
    .to_string()
}

pub(crate) fn apply_input_inverse_trig_alias_preferences(
    display: &str,
    raw_input: &str,
    result_display: Option<&str>,
) -> String {
    let mut adjusted = display.to_string();
    let result_lookup = result_display.map(normalize_alias_lookup_text);
    for (short, long) in [
        ("asin", "arcsin"),
        ("acos", "arccos"),
        ("atan", "arctan"),
        ("asec", "arcsec"),
        ("acsc", "arccsc"),
        ("acot", "arccot"),
    ] {
        adjusted = apply_single_input_inverse_trig_alias_preference(
            &adjusted,
            raw_input,
            result_lookup.as_deref(),
            short,
            long,
        );
    }
    adjusted
}

fn apply_single_input_inverse_trig_alias_preference(
    display: &str,
    raw_input: &str,
    result_lookup: Option<&str>,
    short: &str,
    long: &str,
) -> String {
    let long_call_prefix = format!("{long}(");
    let raw_lookup = normalize_alias_lookup_text(raw_input);
    let mut out = String::with_capacity(display.len());
    let mut cursor = 0;

    while let Some(relative_start) = display[cursor..].find(&long_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(display, start + long.len()) else {
            break;
        };

        out.push_str(&display[cursor..start]);
        let long_call = &display[start..end];
        let short_call = format!("{short}{}", &long_call[long.len()..]);
        if result_lookup_contains_call(result_lookup, &short_call) {
            out.push_str(&short_call);
        } else if result_lookup_contains_call(result_lookup, long_call) {
            out.push_str(long_call);
        } else if raw_input_contains_short_alias_call(&raw_lookup, short, &short_call) {
            out.push_str(&short_call);
        } else {
            out.push_str(long_call);
        }
        cursor = end;
    }

    out.push_str(&display[cursor..]);
    out
}

fn result_lookup_contains_call(result_lookup: Option<&str>, call: &str) -> bool {
    result_lookup.is_some_and(|lookup| lookup.contains(&normalize_alias_lookup_text(call)))
}

fn raw_input_contains_short_alias_call(raw_lookup: &str, short: &str, short_call: &str) -> bool {
    let short_call_lookup = normalize_alias_lookup_text(short_call);
    if raw_lookup.contains(&short_call_lookup) {
        return true;
    }

    let Some(short_call_arg) = call_argument(&short_call_lookup, short.len()) else {
        return false;
    };
    let short_call_arg = strip_redundant_outer_parens(short_call_arg);
    let short_call_prefix = format!("{short}(");
    let mut cursor = 0;

    while let Some(relative_start) = raw_lookup[cursor..].find(&short_call_prefix) {
        let start = cursor + relative_start;
        let Some(end) = matching_call_end(raw_lookup, start + short.len()) else {
            break;
        };
        let raw_call = &raw_lookup[start..end];
        if call_argument(raw_call, short.len()).map(strip_redundant_outer_parens)
            == Some(short_call_arg)
        {
            return true;
        }
        cursor = end;
    }

    false
}

fn call_argument(call: &str, name_len: usize) -> Option<&str> {
    let open_paren = name_len;
    let end = matching_call_end(call, open_paren)?;
    if end != call.len() {
        return None;
    }
    Some(&call[open_paren + 1..end - 1])
}

fn strip_redundant_outer_parens(mut text: &str) -> &str {
    loop {
        if !text.starts_with('(') || !text.ends_with(')') {
            return text;
        }
        if matching_call_end(text, 0) != Some(text.len()) {
            return text;
        }
        text = &text[1..text.len() - 1];
    }
}

fn matching_call_end(text: &str, open_paren: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open_paren) != Some(&b'(') {
        return None;
    }
    let mut depth = 0usize;
    for (offset, byte) in bytes[open_paren..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open_paren + offset + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn normalize_alias_lookup_text(text: &str) -> String {
    text.chars()
        .filter_map(|ch| {
            if ch.is_whitespace() {
                None
            } else if ch == '·' {
                Some('*')
            } else {
                Some(ch)
            }
        })
        .collect()
}
