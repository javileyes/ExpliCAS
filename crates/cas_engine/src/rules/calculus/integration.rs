//! Symbolic integration adapter.
//!
//! Core integration logic lives in `cas_math::symbolic_integration_support`.

use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::{render_integrate_desc_with, NamedVarCall};
use cas_ast::{ConditionPredicate, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::general_integration_backend::{
    try_algorithmic_integration_backend, AlgorithmicIntegrationBackendBudget,
    AlgorithmicIntegrationBackendConfig, AlgorithmicIntegrationMethod, IntegrationConstantPolicy,
};
use num_rational::BigRational;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IntegrationTraceKind {
    EducationalRule,
    AlgorithmicBackendSummary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IntegrationOutcome {
    pub result: ExprId,
    pub trace_kind: IntegrationTraceKind,
    pub required_conditions: Vec<ConditionPredicate>,
}

pub(crate) fn integrate(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    integrate_with_trace(ctx, expr, var).map(|outcome| outcome.result)
}

pub(crate) fn integrate_with_trace(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<IntegrationOutcome> {
    integrate_with_trace_with_backend_config(
        ctx,
        expr,
        var,
        AlgorithmicIntegrationBackendConfig::residual_fallback(),
    )
}

pub(crate) fn integrate_with_trace_with_backend_config(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    backend_config: AlgorithmicIntegrationBackendConfig,
) -> Option<IntegrationOutcome> {
    if let Some(result) =
        cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, expr, var)
    {
        return Some(IntegrationOutcome {
            result,
            trace_kind: IntegrationTraceKind::EducationalRule,
            required_conditions: Vec::new(),
        });
    }

    public_algorithmic_backend_fallback(ctx, expr, var, backend_config).map(
        |(result, required_conditions)| IntegrationOutcome {
            result,
            trace_kind: IntegrationTraceKind::AlgorithmicBackendSummary,
            required_conditions,
        },
    )
}

fn public_algorithmic_backend_fallback(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    backend_config: AlgorithmicIntegrationBackendConfig,
) -> Option<(ExprId, Vec<ConditionPredicate>)> {
    if !backend_config.mode.attempts_backend() {
        return None;
    }
    if !is_public_algorithmic_backend_fallback_shape(ctx, expr, 0) {
        return None;
    }

    let allow_positive_quadratic_hermite =
        is_public_algorithmic_backend_symbolic_positive_quadratic_fallback_shape(ctx, expr, var, 0);
    let allow_indefinite_square_hermite =
        is_public_algorithmic_backend_symbolic_indefinite_square_fallback_shape(ctx, expr, var, 0);
    let budget = if allow_positive_quadratic_hermite || allow_indefinite_square_hermite {
        AlgorithmicIntegrationBackendBudget::two_probes()
    } else {
        AlgorithmicIntegrationBackendBudget::single_probe()
    };
    let config = backend_config.with_budget(AlgorithmicIntegrationBackendBudget::new(
        backend_config
            .budget
            .max_method_probes
            .min(budget.max_method_probes),
        backend_config
            .budget
            .max_verification_checks
            .min(budget.max_verification_checks),
    ));
    let candidate = try_algorithmic_integration_backend(ctx, expr, var, config);
    let result = candidate.fallback_antiderivative(config)?;

    match candidate.method {
        AlgorithmicIntegrationMethod::Rational => {
            if !public_rational_backend_conditions_allowed(&candidate.required_conditions) {
                return None;
            }
        }
        AlgorithmicIntegrationMethod::Hermite if allow_positive_quadratic_hermite => {
            if !public_positive_quadratic_hermite_conditions_allowed(&candidate.required_conditions)
            {
                return None;
            }
        }
        AlgorithmicIntegrationMethod::Hermite if allow_indefinite_square_hermite => {
            if candidate.constant_policy != IntegrationConstantPolicy::ComponentLocalConstant
                || !public_indefinite_square_hermite_conditions_allowed(
                    &candidate.required_conditions,
                )
            {
                return None;
            }
        }
        _ => return None,
    }
    Some((result, candidate.required_conditions))
}

fn public_rational_backend_conditions_allowed(conditions: &[ConditionPredicate]) -> bool {
    (1..=2).contains(&conditions.len())
        && conditions
            .iter()
            .all(|condition| matches!(condition, ConditionPredicate::NonZero(_)))
}

fn public_positive_quadratic_hermite_conditions_allowed(conditions: &[ConditionPredicate]) -> bool {
    (1..=2).contains(&conditions.len())
        && conditions
            .iter()
            .any(|condition| matches!(condition, ConditionPredicate::Positive(_)))
        && conditions.iter().all(|condition| {
            matches!(
                condition,
                ConditionPredicate::Positive(_) | ConditionPredicate::NonZero(_)
            )
        })
}

fn public_indefinite_square_hermite_conditions_allowed(conditions: &[ConditionPredicate]) -> bool {
    (3..=4).contains(&conditions.len())
        && conditions
            .iter()
            .all(|condition| matches!(condition, ConditionPredicate::NonZero(_)))
}

fn is_public_algorithmic_backend_fallback_shape(ctx: &Context, expr: ExprId, depth: usize) -> bool {
    if depth > 1 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Div(_, _) => true,
        Expr::Mul(left, right) => {
            matches!(ctx.get(*left), Expr::Div(_, _)) || matches!(ctx.get(*right), Expr::Div(_, _))
        }
        Expr::Neg(inner) => is_public_algorithmic_backend_fallback_shape(ctx, *inner, depth + 1),
        _ => false,
    }
}

pub(crate) fn is_public_algorithmic_backend_symbolic_indefinite_square_fallback_shape(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    depth: usize,
) -> bool {
    if depth > 1 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Div(_, denominator) => {
            is_public_symbolic_indefinite_square_denominator(ctx, *denominator, var)
        }
        Expr::Neg(inner) => {
            is_public_algorithmic_backend_symbolic_indefinite_square_fallback_shape(
                ctx,
                *inner,
                var,
                depth + 1,
            )
        }
        _ => false,
    }
}

pub(crate) fn is_public_algorithmic_backend_symbolic_positive_quadratic_fallback_shape(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    depth: usize,
) -> bool {
    if depth > 1 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Div(_, denominator) => {
            is_public_symbolic_positive_quadratic_denominator(ctx, *denominator, var)
        }
        Expr::Neg(inner) => {
            is_public_algorithmic_backend_symbolic_positive_quadratic_fallback_shape(
                ctx,
                *inner,
                var,
                depth + 1,
            )
        }
        _ => false,
    }
}

fn is_public_symbolic_indefinite_square_denominator(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            is_public_affine_variable_square(ctx, *left, var)
                && is_public_symbolic_square_radius_candidate(ctx, *right, var)
        }
        _ => false,
    }
}

fn is_public_symbolic_positive_quadratic_denominator(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            is_public_symbolic_positive_quadratic_denominator_pair(ctx, *left, *right, var)
                || is_public_symbolic_positive_quadratic_denominator_pair(ctx, *right, *left, var)
                || is_public_expanded_symbolic_positive_quadratic_denominator(ctx, expr, var)
        }
        _ => false,
    }
}

fn is_public_symbolic_square_radius_candidate(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            is_numeric_two(ctx, *exponent)
                && !contains_named_var(ctx, *base, var)
                && !matches!(ctx.get(*base), Expr::Number(_))
        }
        _ => false,
    }
}

fn is_public_symbolic_positive_quadratic_denominator_pair(
    ctx: &Context,
    square_candidate: ExprId,
    radius_candidate: ExprId,
    var: &str,
) -> bool {
    is_public_affine_variable_square(ctx, square_candidate, var)
        && is_symbolic_external_positive_radius_candidate(ctx, radius_candidate, var)
}

fn is_public_affine_variable_square(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            is_public_affine_variable_expr(ctx, *base, var, 0) && is_numeric_two(ctx, *exponent)
        }
        _ => false,
    }
}

fn is_public_affine_variable_expr(ctx: &Context, expr: ExprId, var: &str, depth: usize) -> bool {
    if depth > 3 {
        return false;
    }
    if is_named_variable(ctx, expr, var) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            (is_public_affine_variable_expr(ctx, *left, var, depth + 1)
                && is_public_external_affine_offset(ctx, *right, var, depth + 1))
                || (is_public_external_affine_offset(ctx, *left, var, depth + 1)
                    && is_public_affine_variable_expr(ctx, *right, var, depth + 1))
        }
        Expr::Mul(left, right) => {
            (is_public_external_affine_scale(ctx, *left, var, depth + 1)
                && is_public_affine_variable_expr(ctx, *right, var, depth + 1))
                || (is_public_affine_variable_expr(ctx, *left, var, depth + 1)
                    && is_public_external_affine_scale(ctx, *right, var, depth + 1))
        }
        Expr::Neg(inner) => is_public_affine_variable_expr(ctx, *inner, var, depth + 1),
        _ => false,
    }
}

fn is_public_external_affine_offset(ctx: &Context, expr: ExprId, var: &str, depth: usize) -> bool {
    if depth > 3 || contains_named_var(ctx, expr, var) {
        return false;
    }
    match ctx.get(expr) {
        Expr::Variable(_) | Expr::Number(_) => true,
        Expr::Neg(inner) => is_public_external_affine_offset(ctx, *inner, var, depth + 1),
        _ => false,
    }
}

fn is_public_external_affine_scale(ctx: &Context, expr: ExprId, var: &str, depth: usize) -> bool {
    if !is_public_external_affine_offset(ctx, expr, var, depth) {
        return false;
    }
    !matches!(ctx.get(expr), Expr::Number(value) if *value == BigRational::from_integer(0.into()))
}

fn is_public_expanded_symbolic_positive_quadratic_denominator(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    contains_named_variable_square_factor(ctx, expr, var, 0)
        && has_symbolic_external_additive_radius_term(ctx, expr, var, 0)
}

fn contains_named_variable_square_factor(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    depth: usize,
) -> bool {
    if depth > 5 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            is_named_variable(ctx, *base, var) && is_numeric_two(ctx, *exponent)
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            contains_named_variable_square_factor(ctx, *left, var, depth + 1)
                || contains_named_variable_square_factor(ctx, *right, var, depth + 1)
        }
        Expr::Neg(inner) => contains_named_variable_square_factor(ctx, *inner, var, depth + 1),
        _ => false,
    }
}

fn has_symbolic_external_additive_radius_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    depth: usize,
) -> bool {
    if depth > 5 {
        return false;
    }
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            has_symbolic_external_additive_radius_term(ctx, *left, var, depth + 1)
                || has_symbolic_external_additive_radius_term(ctx, *right, var, depth + 1)
        }
        Expr::Neg(inner) => has_symbolic_external_additive_radius_term(ctx, *inner, var, depth + 1),
        _ => is_symbolic_external_positive_radius_candidate(ctx, expr, var),
    }
}

fn is_named_variable(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
}

fn is_symbolic_external_positive_radius_candidate(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) != var)
        && !contains_named_var(ctx, expr, var)
}

fn is_numeric_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if *value == BigRational::from_integer(2.into()))
}

pub(super) fn integrate_rewrite_with_conditions<I>(
    ctx: &mut Context,
    call: &NamedVarCall,
    result: ExprId,
    required_conditions: I,
) -> Rewrite
where
    I: IntoIterator<Item = crate::ImplicitCondition>,
{
    let desc = render_integrate_desc_with(call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Rewrite::new(result)
        .desc(desc)
        .requires_all(required_conditions)
}

#[cfg(test)]
mod tests;
