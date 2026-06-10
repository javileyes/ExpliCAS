//! Method probes (rational, Hermite, heurisch) and the public backend entry point.

use super::verification_normalization::*;
use super::*;

use crate::expr_nary::{add_terms_signed, Sign};
use crate::expr_predicates::contains_named_var;
use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{BuiltinFn, ConditionPredicate, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn try_rational_reciprocal_affine_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let parts = match affine_denominator_linear_numerator_parts(ctx, integrand, variable) {
        Ok(parts) => parts,
        Err(reason) => return AlgorithmicIntegrationProbeResult::NoMatch(reason),
    };

    let variable_expr = ctx.var(variable);
    let quotient_antiderivative =
        build_backend_product(ctx, parts.quotient_coefficient, variable_expr);
    let log_antiderivative = build_affine_denominator_remainder_antiderivative(
        ctx,
        parts.remainder,
        parts.denominator,
        &parts.denominator_slope,
    );
    let antiderivative = build_backend_sum(ctx, quotient_antiderivative, log_antiderivative);
    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::Rational,
    );
    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(parts.denominator));
    if let Some(condition) = parts.denominator_slope.required_condition() {
        candidate.required_conditions.push(condition);
    }
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

pub(super) struct AffineDenominatorLinearNumeratorParts {
    pub(super) quotient_coefficient: ExprId,
    pub(super) remainder: ExprId,
    pub(super) denominator: ExprId,
    pub(super) denominator_slope: BackendAffineSlope,
}

fn try_hermite_positive_quadratic_log_derivative_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let mut pole_conditions = Vec::new();
    let mut constant_policy = IntegrationConstantPolicy::ArbitraryConstantOmitted;
    let (antiderivative, slope_condition, radius_condition) =
        if let Some((coefficient, variable_slope, denominator, required_condition)) =
            positive_quadratic_log_derivative_parts(ctx, integrand, variable)
        {
            (
                build_positive_quadratic_log_derivative_antiderivative(
                    ctx,
                    coefficient,
                    &variable_slope,
                    denominator,
                ),
                variable_slope.required_condition(),
                required_condition,
            )
        } else if let Some(parts) =
            positive_quadratic_linear_numerator_parts(ctx, integrand, variable)
        {
            (
                build_positive_quadratic_linear_numerator_antiderivative(
                    ctx,
                    parts.variable_coefficient,
                    parts.constant_term,
                    parts.variable_expr,
                    &parts.variable_slope,
                    parts.denominator,
                    parts.radius,
                ),
                parts.variable_slope.required_condition(),
                parts.required_condition,
            )
        } else if let Some(parts) =
            indefinite_square_denominator_reciprocal_parts(ctx, integrand, variable)
        {
            pole_conditions.push(ConditionPredicate::NonZero(parts.left_pole));
            pole_conditions.push(ConditionPredicate::NonZero(parts.right_pole));
            constant_policy = IntegrationConstantPolicy::ComponentLocalConstant;
            (
                build_indefinite_square_denominator_linear_numerator_antiderivative(
                    ctx,
                    parts.variable_coefficient,
                    parts.constant_term,
                    parts.variable_expr,
                    &parts.variable_slope,
                    parts.denominator,
                    parts.radius,
                ),
                parts.variable_slope.required_condition(),
                parts.radius_condition,
            )
        } else {
            return AlgorithmicIntegrationProbeResult::NoMatch(
                positive_quadratic_log_derivative_no_match_reason(ctx, integrand, variable),
            );
        };

    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::Hermite,
    );
    candidate.constant_policy = constant_policy;
    if let Some(condition) = radius_condition {
        candidate.required_conditions.push(condition);
    }
    candidate.required_conditions.extend(pole_conditions);
    if let Some(condition) = slope_condition {
        candidate.required_conditions.push(condition);
    }
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

fn try_heurisch_sine_log_derivative_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let Some(denominator) = sine_log_derivative_denominator(ctx, integrand, variable) else {
        return AlgorithmicIntegrationProbeResult::NoMatch(
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
        );
    };

    let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
    let antiderivative = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::HeurischProbe,
    );
    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(denominator));
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

pub(super) fn affine_denominator_linear_numerator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Result<AffineDenominatorLinearNumeratorParts, AlgorithmicIntegrationProbeNoMatchReason> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            affine_denominator_linear_numerator_div_parts(ctx, numerator, denominator, variable)
        }
        Expr::Mul(left, right) => {
            if let Some(parts) =
                scaled_affine_denominator_linear_numerator_parts(ctx, left, right, variable)
                    .or_else(|| {
                        scaled_affine_denominator_linear_numerator_parts(ctx, right, left, variable)
                    })
            {
                return Ok(parts);
            }
            if matches!(ctx.get(left), Expr::Div(_, _)) || matches!(ctx.get(right), Expr::Div(_, _))
            {
                Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch)
            } else {
                Err(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch)
            }
        }
        Expr::Neg(inner) => {
            let negative_one = ctx.num(-1);
            scaled_affine_denominator_linear_numerator_parts(ctx, negative_one, inner, variable)
                .ok_or(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch)
        }
        _ => Err(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch),
    }
}

fn affine_denominator_linear_numerator_div_parts(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
) -> Result<AffineDenominatorLinearNumeratorParts, AlgorithmicIntegrationProbeNoMatchReason> {
    if is_supported_scaled_affine_reciprocal_numerator(ctx, numerator, variable) {
        let Some(denominator_slope) = affine_denominator_slope(ctx, denominator, variable) else {
            return Err(AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch);
        };
        return Ok(AffineDenominatorLinearNumeratorParts {
            quotient_coefficient: ctx.num(0),
            remainder: numerator,
            denominator,
            denominator_slope,
        });
    }

    let Some(denominator_slope) = affine_denominator_slope(ctx, denominator, variable) else {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    };
    let Some((quotient_coefficient, remainder)) =
        linear_numerator_decomposition_terms(ctx, numerator, denominator, variable).or_else(|| {
            affine_quotient_remainder_from_linear_terms(
                ctx,
                numerator,
                denominator,
                &denominator_slope,
                variable,
            )
        })
    else {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    };
    if (is_zero(ctx, quotient_coefficient) && is_zero(ctx, remainder))
        || !is_supported_backend_linear_coefficient_for_affine_slope(
            ctx,
            quotient_coefficient,
            variable,
            &denominator_slope,
        )
        || !is_supported_backend_linear_coefficient_for_affine_slope(
            ctx,
            remainder,
            variable,
            &denominator_slope,
        )
    {
        return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
    }
    Ok(AffineDenominatorLinearNumeratorParts {
        quotient_coefficient,
        remainder,
        denominator,
        denominator_slope,
    })
}

fn scaled_affine_denominator_linear_numerator_parts(
    ctx: &mut Context,
    scale: ExprId,
    quotient: ExprId,
    variable: &str,
) -> Option<AffineDenominatorLinearNumeratorParts> {
    if contains_named_var(ctx, scale, variable)
        || !is_supported_backend_linear_coefficient(ctx, scale, variable)
    {
        return None;
    }
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };
    let scaled_numerator = build_backend_product(ctx, scale, numerator);
    affine_denominator_linear_numerator_div_parts(ctx, scaled_numerator, denominator, variable).ok()
}

fn affine_quotient_remainder_from_linear_terms(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    denominator_slope: &BackendAffineSlope,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    let (numerator_slope, numerator_intercept) =
        backend_affine_linear_terms(ctx, numerator, variable)?;
    let (_, denominator_intercept) = backend_affine_linear_terms(ctx, denominator, variable)?;
    if is_zero(ctx, numerator_slope) {
        return None;
    }

    let quotient_coefficient =
        divide_backend_coefficient_by_slope(ctx, numerator_slope, denominator_slope);
    let scaled_denominator_intercept =
        build_backend_product(ctx, quotient_coefficient, denominator_intercept);
    let remainder =
        build_backend_difference(ctx, numerator_intercept, scaled_denominator_intercept);
    Some((quotient_coefficient, remainder))
}

pub(super) fn backend_affine_linear_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    if is_variable(ctx, expr, variable) {
        let one = ctx.num(1);
        let zero = ctx.num(0);
        return Some((one, zero));
    }
    if is_supported_backend_linear_coefficient(ctx, expr, variable) {
        let zero = ctx.num(0);
        return Some((zero, expr));
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let (left_slope, left_intercept) = backend_affine_linear_terms(ctx, left, variable)?;
            let (right_slope, right_intercept) = backend_affine_linear_terms(ctx, right, variable)?;
            Some((
                build_backend_sum(ctx, left_slope, right_slope),
                build_backend_sum(ctx, left_intercept, right_intercept),
            ))
        }
        Expr::Sub(left, right) => {
            let (left_slope, left_intercept) = backend_affine_linear_terms(ctx, left, variable)?;
            let (right_slope, right_intercept) = backend_affine_linear_terms(ctx, right, variable)?;
            Some((
                build_backend_difference(ctx, left_slope, right_slope),
                build_backend_difference(ctx, left_intercept, right_intercept),
            ))
        }
        Expr::Neg(inner) => {
            let (slope, intercept) = backend_affine_linear_terms(ctx, inner, variable)?;
            Some((
                negate_backend_expr(ctx, slope),
                negate_backend_expr(ctx, intercept),
            ))
        }
        Expr::Mul(left, right) => {
            if is_supported_backend_linear_coefficient(ctx, left, variable)
                && !contains_named_var(ctx, left, variable)
            {
                let (slope, intercept) = backend_affine_linear_terms(ctx, right, variable)?;
                return Some((
                    build_backend_product(ctx, left, slope),
                    build_backend_product(ctx, left, intercept),
                ));
            }
            if is_supported_backend_linear_coefficient(ctx, right, variable)
                && !contains_named_var(ctx, right, variable)
            {
                let (slope, intercept) = backend_affine_linear_terms(ctx, left, variable)?;
                return Some((
                    build_backend_product(ctx, right, slope),
                    build_backend_product(ctx, right, intercept),
                ));
            }
            None
        }
        _ => None,
    }
}

fn build_affine_denominator_remainder_antiderivative(
    ctx: &mut Context,
    remainder: ExprId,
    denominator: ExprId,
    denominator_slope: &BackendAffineSlope,
) -> ExprId {
    if is_zero(ctx, remainder) {
        return ctx.num(0);
    }

    let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
    let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
    let antiderivative_scale =
        divide_backend_coefficient_by_slope(ctx, remainder, denominator_slope);
    if is_one(ctx, antiderivative_scale) {
        log_denominator
    } else {
        build_backend_product(ctx, antiderivative_scale, log_denominator)
    }
}

fn is_supported_scaled_affine_reciprocal_numerator(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
        || is_supported_external_coefficient(ctx, expr, variable)
}

fn is_supported_backend_linear_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    is_supported_backend_linear_coefficient_inner(ctx, expr, variable, 0, None)
}

pub(super) fn is_supported_backend_linear_coefficient_for_affine_slope(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    affine_slope: &BackendAffineSlope,
) -> bool {
    is_supported_backend_linear_coefficient_inner(ctx, expr, variable, 0, Some(affine_slope))
}

fn is_supported_backend_linear_coefficient_inner(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
    allowed_symbolic_divisor: Option<&BackendAffineSlope>,
) -> bool {
    if depth >= BACKEND_EXTERNAL_COEFFICIENT_DEPTH {
        return false;
    }
    if is_zero(ctx, expr) {
        return true;
    }
    if numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
    {
        return true;
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        return true;
    }
    if contains_named_var(ctx, expr, variable) {
        return false;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            is_supported_backend_linear_coefficient_inner(
                ctx,
                *left,
                variable,
                depth + 1,
                allowed_symbolic_divisor,
            ) && is_supported_backend_linear_coefficient_inner(
                ctx,
                *right,
                variable,
                depth + 1,
                allowed_symbolic_divisor,
            )
        }
        Expr::Div(numerator, denominator) => {
            if let Some(denominator_value) = numeric_value(ctx, *denominator) {
                return !denominator_value.is_zero()
                    && is_supported_backend_linear_coefficient_inner(
                        ctx,
                        *numerator,
                        variable,
                        depth + 1,
                        allowed_symbolic_divisor,
                    );
            }
            backend_affine_slope_allows_divisor(ctx, *denominator, allowed_symbolic_divisor)
                && is_supported_backend_linear_coefficient_inner(
                    ctx,
                    *numerator,
                    variable,
                    depth + 1,
                    allowed_symbolic_divisor,
                )
        }
        Expr::Neg(inner) => is_supported_backend_linear_coefficient_inner(
            ctx,
            *inner,
            variable,
            depth + 1,
            allowed_symbolic_divisor,
        ),
        _ => false,
    }
}

fn backend_affine_slope_allows_divisor(
    ctx: &Context,
    divisor: ExprId,
    allowed_symbolic_divisor: Option<&BackendAffineSlope>,
) -> bool {
    let Some(BackendAffineSlope::Symbolic(allowed_divisor)) = allowed_symbolic_divisor else {
        return false;
    };

    divisor == *allowed_divisor
        || SemanticEqualityChecker::new(ctx).are_equal(divisor, *allowed_divisor)
}

fn is_supported_nonzero_backend_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    numeric_value(ctx, expr)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
        || is_supported_external_coefficient(ctx, expr, variable)
}

fn positive_quadratic_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius_square, required_condition) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)?;
            let coefficient =
                affine_variable_coefficient_expr(ctx, numerator, variable_expr, variable)?;
            let denominator =
                build_positive_quadratic_denominator(ctx, variable_expr, radius_square);
            Some((coefficient, variable_slope, denominator, required_condition))
        }
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositiveQuadraticLinearNumeratorParts {
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
    required_condition: Option<ConditionPredicate>,
}

fn positive_quadratic_linear_numerator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<PositiveQuadraticLinearNumeratorParts> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius_square, required_condition) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)?;
            let (variable_coefficient, constant_term) =
                linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)?;
            if is_zero(ctx, constant_term) {
                return None;
            }
            if !is_supported_backend_linear_coefficient_for_affine_slope(
                ctx,
                variable_coefficient,
                variable,
                &variable_slope,
            ) || !is_supported_backend_linear_coefficient_for_affine_slope(
                ctx,
                constant_term,
                variable,
                &variable_slope,
            ) {
                return None;
            }
            let radius = positive_radius_expr(ctx, radius_square, &required_condition)?;
            let denominator =
                build_positive_quadratic_denominator(ctx, variable_expr, radius_square);
            Some(PositiveQuadraticLinearNumeratorParts {
                variable_coefficient,
                constant_term,
                variable_expr,
                variable_slope,
                denominator,
                radius,
                required_condition,
            })
        }
        _ => None,
    }
}

fn build_positive_quadratic_denominator(
    ctx: &mut Context,
    variable_expr: ExprId,
    radius_square: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let variable_square = ctx.add(Expr::Pow(variable_expr, two));
    build_backend_sum(ctx, variable_square, radius_square)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IndefiniteSquareDenominatorReciprocalParts {
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
    radius_condition: Option<ConditionPredicate>,
    left_pole: ExprId,
    right_pole: ExprId,
}

fn indefinite_square_denominator_reciprocal_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<IndefiniteSquareDenominatorReciprocalParts> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let (variable_expr, variable_slope, radius, radius_condition) =
                indefinite_square_denominator_parts(ctx, denominator, variable)?;
            let (variable_coefficient, constant_term) =
                linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)?;
            if is_zero(ctx, variable_coefficient) && is_zero(ctx, constant_term) {
                return None;
            }
            if !is_supported_backend_linear_coefficient(ctx, variable_coefficient, variable)
                || !is_supported_backend_linear_coefficient(ctx, constant_term, variable)
            {
                return None;
            }
            let left_pole = build_backend_difference(ctx, variable_expr, radius);
            let right_pole = build_backend_sum(ctx, variable_expr, radius);
            Some(IndefiniteSquareDenominatorReciprocalParts {
                variable_coefficient,
                constant_term,
                variable_expr,
                variable_slope,
                denominator,
                radius,
                radius_condition,
                left_pole,
                right_pole,
            })
        }
        _ => None,
    }
}

fn indefinite_square_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Sub(left, right) => {
            let (variable_expr, variable_slope) = affine_variable_from_square(ctx, left, variable)?;
            let radius_condition = positive_radius_square_required_condition(ctx, right, variable)?;
            let radius = positive_radius_expr(ctx, right, &radius_condition)?;
            Some((variable_expr, variable_slope, radius, radius_condition))
        }
        _ => None,
    }
}

fn positive_quadratic_log_derivative_no_match_reason(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationProbeNoMatchReason {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            let Some((variable_expr, _, radius_square, required_condition)) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)
            else {
                return positive_quadratic_denominator_no_match_reason(ctx, denominator, variable);
            };
            if affine_variable_coefficient_expr(ctx, numerator, variable_expr, variable).is_none() {
                if linear_numerator_decomposition_terms(ctx, numerator, variable_expr, variable)
                    .is_some()
                    && positive_radius_expr(ctx, radius_square, &required_condition).is_none()
                {
                    return AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch;
                }
                return AlgorithmicIntegrationProbeNoMatchReason::NumeratorDerivativeMismatch;
            }
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
        }
        _ => AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
    }
}

fn positive_quadratic_denominator_no_match_reason(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationProbeNoMatchReason {
    if positive_quadratic_radius_policy_mismatch(ctx, expr, variable) {
        AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch
    } else {
        AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch
    }
}

fn positive_quadratic_radius_policy_mismatch(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            positive_quadratic_radius_policy_mismatch_pair(ctx, left, right, variable)
                || positive_quadratic_radius_policy_mismatch_pair(ctx, right, left, variable)
        }
        Expr::Sub(left, right) => {
            affine_variable_from_square(ctx, left, variable).is_some()
                && backend_radius_policy_candidate(ctx, right, variable)
        }
        _ => false,
    }
}

fn positive_quadratic_radius_policy_mismatch_pair(
    ctx: &mut Context,
    square_candidate: ExprId,
    radius_candidate: ExprId,
    variable: &str,
) -> bool {
    affine_variable_from_square(ctx, square_candidate, variable).is_some()
        && backend_radius_policy_candidate(ctx, radius_candidate, variable)
        && positive_radius_square_required_condition(ctx, radius_candidate, variable).is_none()
}

fn backend_radius_policy_candidate(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    !contains_named_var(ctx, expr, variable)
}

fn build_positive_quadratic_log_derivative_antiderivative(
    ctx: &mut Context,
    numerator_coefficient: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
) -> ExprId {
    let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![denominator]);
    let halved_coefficient = halve_backend_coefficient(ctx, numerator_coefficient);
    let antiderivative_coefficient =
        divide_backend_coefficient_by_slope(ctx, halved_coefficient, variable_slope);
    if is_one(ctx, antiderivative_coefficient) {
        log_denominator
    } else {
        ctx.add(Expr::Mul(antiderivative_coefficient, log_denominator))
    }
}

fn build_positive_quadratic_linear_numerator_antiderivative(
    ctx: &mut Context,
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
) -> ExprId {
    let log_part = if is_zero(ctx, variable_coefficient) {
        ctx.num(0)
    } else {
        build_positive_quadratic_log_derivative_antiderivative(
            ctx,
            variable_coefficient,
            variable_slope,
            denominator,
        )
    };
    let arctan_part = if is_zero(ctx, constant_term) {
        ctx.num(0)
    } else {
        build_positive_quadratic_constant_numerator_antiderivative(
            ctx,
            constant_term,
            variable_expr,
            variable_slope,
            radius,
        )
    };
    build_backend_sum(ctx, log_part, arctan_part)
}

fn build_positive_quadratic_constant_numerator_antiderivative(
    ctx: &mut Context,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    radius: ExprId,
) -> ExprId {
    let slope_scaled_constant =
        divide_backend_coefficient_by_slope(ctx, constant_term, variable_slope);
    if is_one(ctx, radius) {
        let arctan_variable = ctx.call_builtin(BuiltinFn::Arctan, vec![variable_expr]);
        return build_backend_product(ctx, slope_scaled_constant, arctan_variable);
    }

    let scaled_variable = ctx.add(Expr::Div(variable_expr, radius));
    let arctan_scaled_variable = ctx.call_builtin(BuiltinFn::Arctan, vec![scaled_variable]);
    let scaled_constant =
        divide_backend_coefficient_by_symbolic(ctx, slope_scaled_constant, radius);
    build_backend_product(ctx, scaled_constant, arctan_scaled_variable)
}

fn build_indefinite_square_denominator_linear_numerator_antiderivative(
    ctx: &mut Context,
    variable_coefficient: ExprId,
    constant_term: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    denominator: ExprId,
    radius: ExprId,
) -> ExprId {
    let log_derivative_part = if is_zero(ctx, variable_coefficient) {
        ctx.num(0)
    } else {
        let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
        let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
        let halved_coefficient = halve_backend_coefficient(ctx, variable_coefficient);
        let antiderivative_scale =
            divide_backend_coefficient_by_slope(ctx, halved_coefficient, variable_slope);
        build_backend_product(ctx, antiderivative_scale, log_denominator)
    };

    let reciprocal_part = if is_zero(ctx, constant_term) {
        ctx.num(0)
    } else {
        build_indefinite_square_denominator_reciprocal_antiderivative(
            ctx,
            constant_term,
            variable_expr,
            variable_slope,
            radius,
        )
    };

    build_backend_sum(ctx, log_derivative_part, reciprocal_part)
}

fn build_indefinite_square_denominator_reciprocal_antiderivative(
    ctx: &mut Context,
    numerator: ExprId,
    variable_expr: ExprId,
    variable_slope: &BackendAffineSlope,
    radius: ExprId,
) -> ExprId {
    let left_pole = build_backend_difference(ctx, variable_expr, radius);
    let right_pole = build_backend_sum(ctx, variable_expr, radius);
    let abs_left = ctx.call_builtin(BuiltinFn::Abs, vec![left_pole]);
    let abs_right = ctx.call_builtin(BuiltinFn::Abs, vec![right_pole]);
    let log_left = ctx.call_builtin(BuiltinFn::Ln, vec![abs_left]);
    let log_right = ctx.call_builtin(BuiltinFn::Ln, vec![abs_right]);
    let log_difference = build_backend_difference(ctx, log_left, log_right);
    let slope_scaled_numerator =
        divide_backend_coefficient_by_slope(ctx, numerator, variable_slope);
    let radius_scaled_numerator =
        divide_backend_coefficient_by_symbolic(ctx, slope_scaled_numerator, radius);
    let antiderivative_scale = divide_backend_coefficient_by_numeric(
        ctx,
        radius_scaled_numerator,
        BigRational::from_integer(2.into()),
    );
    build_backend_product(ctx, antiderivative_scale, log_difference)
}

fn sine_log_derivative_denominator(ctx: &Context, expr: ExprId, variable: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator)
            if is_builtin_of_variable(ctx, *numerator, BuiltinFn::Cos, variable)
                && is_builtin_of_variable(ctx, *denominator, BuiltinFn::Sin, variable) =>
        {
            Some(*denominator)
        }
        _ => None,
    }
}

/// Public recognizer for denominators the Hermite positive-quadratic method
/// accepts: compact `(s*x + b)^2 + a` or expanded
/// `s^2*x^2 + 2*b*s*x + b^2 + a` with a variable-free radius. Returns the
/// radius expression so condition presentation can drop source-denominator
/// conditions that are already implied by the displayed `Positive(radius)`
/// backend condition, without duplicating center reconstruction outside the
/// backend.
pub fn backend_positive_quadratic_denominator_radius(
    ctx: &mut Context,
    denominator: ExprId,
    variable: &str,
) -> Option<ExprId> {
    positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)
        .map(|(_, _, radius, _)| radius)
}

pub(super) fn positive_shifted_quadratic_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if let Some(required_condition) =
                positive_radius_square_required_condition(ctx, right, variable)
            {
                let (variable_expr, variable_slope) =
                    affine_variable_from_square(ctx, left, variable)?;
                return Some((variable_expr, variable_slope, right, required_condition));
            }
            if let Some(required_condition) =
                positive_radius_square_required_condition(ctx, left, variable)
            {
                let (variable_expr, variable_slope) =
                    affine_variable_from_square(ctx, right, variable)?;
                return Some((variable_expr, variable_slope, left, required_condition));
            }
            expanded_positive_shifted_quadratic_denominator_parts(ctx, expr, variable)
        }
        _ => expanded_positive_shifted_quadratic_denominator_parts(ctx, expr, variable),
    }
}

fn expanded_positive_shifted_quadratic_denominator_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(
    ExprId,
    BackendAffineSlope,
    ExprId,
    Option<ConditionPredicate>,
)> {
    let terms = add_terms_signed(ctx, expr);
    if terms.len() != 4 {
        return None;
    }

    for (radius_index, (radius_candidate, radius_sign)) in terms.iter().copied().enumerate() {
        if radius_sign != Sign::Pos {
            continue;
        }
        let Some(required_condition) =
            positive_radius_square_required_condition(ctx, radius_candidate, variable)
        else {
            continue;
        };

        for (quadratic_index, (quadratic_candidate, quadratic_sign)) in
            terms.iter().copied().enumerate()
        {
            if quadratic_index == radius_index || quadratic_sign != Sign::Pos {
                continue;
            }
            let Some(variable_slope) =
                expanded_affine_square_quadratic_slope(ctx, quadratic_candidate, variable)
            else {
                continue;
            };

            for (intercept_index, (intercept_square_candidate, intercept_sign)) in
                terms.iter().copied().enumerate()
            {
                if intercept_index == radius_index
                    || intercept_index == quadratic_index
                    || intercept_sign != Sign::Pos
                {
                    continue;
                }
                let Some(intercept) =
                    squared_external_radius_base(ctx, intercept_square_candidate, variable)
                else {
                    continue;
                };

                let Some((cross_candidate, cross_sign)) = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(index, _)| {
                        *index != radius_index
                            && *index != quadratic_index
                            && *index != intercept_index
                    })
                    .map(|(_, term)| term)
                else {
                    continue;
                };
                if !expanded_affine_square_cross_term_matches(
                    ctx,
                    cross_candidate,
                    &variable_slope,
                    intercept,
                    variable,
                ) {
                    continue;
                }

                let variable_expr = build_expanded_affine_square_variable_expr(
                    ctx,
                    &variable_slope,
                    intercept,
                    cross_sign,
                    variable,
                );
                return Some((
                    variable_expr,
                    variable_slope,
                    radius_candidate,
                    required_condition,
                ));
            }
        }
    }

    None
}

fn expanded_affine_square_quadratic_slope(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if is_named_variable_square_factor(ctx, expr, variable) {
        return Some(BackendAffineSlope::Numeric(BigRational::one()));
    }

    let mut factors = backend_mul_factors(ctx, expr);
    let variable_square_index = factors
        .iter()
        .position(|factor| is_named_variable_square_factor(ctx, *factor, variable))?;
    factors.remove(variable_square_index);
    if factors.is_empty() {
        return Some(BackendAffineSlope::Numeric(BigRational::one()));
    }
    if factors.len() != 1 {
        return None;
    }
    let slope_square_candidate = factors[0];
    let slope = squared_external_radius_base(ctx, slope_square_candidate, variable)?;
    affine_slope_coefficient(ctx, slope, variable)
}

fn is_named_variable_square_factor(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => is_variable(ctx, *base, variable),
        _ => false,
    }
}

fn expanded_affine_square_cross_term_matches(
    ctx: &mut Context,
    expr: ExprId,
    variable_slope: &BackendAffineSlope,
    intercept: ExprId,
    variable: &str,
) -> bool {
    let two = ctx.num(2);
    let slope = backend_affine_slope_expr(ctx, variable_slope);
    let variable_expr = ctx.var(variable);
    let expected = build_backend_factor_product(ctx, vec![two, slope, intercept, variable_expr]);
    expr == expected || SemanticEqualityChecker::new(ctx).are_equal(expr, expected)
}

fn build_expanded_affine_square_variable_expr(
    ctx: &mut Context,
    variable_slope: &BackendAffineSlope,
    intercept: ExprId,
    cross_sign: Sign,
    variable: &str,
) -> ExprId {
    let slope = backend_affine_slope_expr(ctx, variable_slope);
    let variable_expr = ctx.var(variable);
    let variable_term = build_backend_product(ctx, slope, variable_expr);
    let signed_intercept = match cross_sign {
        Sign::Pos => intercept,
        Sign::Neg => negate_backend_expr(ctx, intercept),
    };
    build_backend_sum(ctx, variable_term, signed_intercept)
}

fn backend_affine_slope_expr(ctx: &mut Context, slope: &BackendAffineSlope) -> ExprId {
    match slope {
        BackendAffineSlope::Numeric(value) => ctx.add(Expr::Number(value.clone())),
        BackendAffineSlope::Symbolic(expr) => *expr,
    }
}

fn positive_radius_square_required_condition(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<Option<ConditionPredicate>> {
    if let Some(value) = numeric_value(ctx, expr) {
        return value.is_positive().then_some(None);
    }
    if let Some(required_condition) =
        squared_external_radius_required_condition(ctx, expr, variable)
    {
        return Some(required_condition);
    }
    is_supported_external_coefficient(ctx, expr, variable)
        .then_some(Some(ConditionPredicate::Positive(expr)))
}

fn squared_external_radius_required_condition(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<Option<ConditionPredicate>> {
    let radius = squared_external_radius_base(ctx, expr, variable)?;
    if let Some(value) = numeric_value(ctx, radius) {
        return (!value.is_zero()).then_some(None);
    }
    Some(Some(ConditionPredicate::NonZero(radius)))
}

fn squared_external_radius_base(ctx: &Context, expr: ExprId, variable: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent)
            if is_two(ctx, *exponent)
                && is_supported_external_coefficient(ctx, *base, variable) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn affine_variable_from_square(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, BackendAffineSlope)> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => {
            affine_variable_expr(ctx, *base, variable)
        }
        _ => None,
    }
}

pub(super) fn affine_variable_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<(ExprId, BackendAffineSlope)> {
    affine_denominator_slope(ctx, expr, variable).map(|slope| (expr, slope))
}

fn affine_variable_coefficient_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if expr == variable_expr || SemanticEqualityChecker::new(ctx).are_equal(expr, variable_expr) {
        return Some(ctx.num(1));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            if (right == variable_expr
                || SemanticEqualityChecker::new(ctx).are_equal(right, variable_expr))
                && is_supported_nonzero_backend_coefficient(ctx, left, variable)
            {
                return Some(left);
            }
            if (left == variable_expr
                || SemanticEqualityChecker::new(ctx).are_equal(left, variable_expr))
                && is_supported_nonzero_backend_coefficient(ctx, right, variable)
            {
                return Some(right);
            }
            None
        }
        _ => None,
    }
}

pub(super) fn linear_numerator_decomposition_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    if let Some(coefficient) = affine_variable_coefficient_expr(ctx, expr, variable_expr, variable)
    {
        let zero = ctx.num(0);
        return Some((coefficient, zero));
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        let zero = ctx.num(0);
        return Some((zero, expr));
    }
    if let Some(decomposition) =
        affine_linear_numerator_decomposition_terms(ctx, expr, variable_expr, variable)
    {
        return Some(decomposition);
    }

    let direct = match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let (left_coefficient, left_constant) =
                linear_numerator_decomposition_terms(ctx, left, variable_expr, variable)?;
            let (right_coefficient, right_constant) =
                linear_numerator_decomposition_terms(ctx, right, variable_expr, variable)?;
            let coefficient = build_backend_sum(ctx, left_coefficient, right_coefficient);
            let constant = build_backend_sum(ctx, left_constant, right_constant);
            Some((coefficient, constant))
        }
        Expr::Sub(left, right) => {
            let (left_coefficient, left_constant) =
                linear_numerator_decomposition_terms(ctx, left, variable_expr, variable)?;
            let (right_coefficient, right_constant) =
                linear_numerator_decomposition_terms(ctx, right, variable_expr, variable)?;
            let coefficient = build_backend_difference(ctx, left_coefficient, right_coefficient);
            let constant = build_backend_difference(ctx, left_constant, right_constant);
            Some((coefficient, constant))
        }
        Expr::Neg(inner) => {
            let (coefficient, constant) =
                linear_numerator_decomposition_terms(ctx, inner, variable_expr, variable)?;
            Some((
                negate_backend_expr(ctx, coefficient),
                negate_backend_expr(ctx, constant),
            ))
        }
        _ => None,
    };
    direct
}

fn affine_linear_numerator_decomposition_terms(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    let variable_slope = affine_denominator_slope(ctx, variable_expr, variable)?;
    let (_, variable_intercept) = backend_affine_linear_terms(ctx, variable_expr, variable)?;
    let (numerator_slope, numerator_intercept) = backend_affine_linear_terms(ctx, expr, variable)?;
    if is_zero(ctx, numerator_slope) {
        return None;
    }

    let coefficient = divide_backend_coefficient_by_slope(ctx, numerator_slope, &variable_slope);
    let scaled_variable_intercept = build_backend_product(ctx, coefficient, variable_intercept);
    let constant = build_backend_difference_canceling_sum_term(
        ctx,
        numerator_intercept,
        scaled_variable_intercept,
    );
    Some((coefficient, constant))
}

fn halve_backend_coefficient(ctx: &mut Context, coefficient: ExprId) -> ExprId {
    if let Some(value) = numeric_value(ctx, coefficient) {
        let half = value / BigRational::from_integer(2.into());
        return ctx.add(Expr::Number(half));
    }

    let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
    ctx.add(Expr::Div(coefficient, two))
}

fn divide_backend_coefficient_by_numeric(
    ctx: &mut Context,
    coefficient: ExprId,
    divisor: BigRational,
) -> ExprId {
    multiply_backend_numeric_coefficient(ctx, BigRational::one() / divisor, coefficient)
}

fn divide_backend_coefficient_by_slope(
    ctx: &mut Context,
    coefficient: ExprId,
    slope: &BackendAffineSlope,
) -> ExprId {
    match slope {
        BackendAffineSlope::Numeric(value) => {
            divide_backend_coefficient_by_numeric(ctx, coefficient, value.clone())
        }
        BackendAffineSlope::Symbolic(divisor) => {
            divide_backend_coefficient_by_symbolic(ctx, coefficient, *divisor)
        }
    }
}

fn divide_backend_coefficient_by_symbolic(
    ctx: &mut Context,
    coefficient: ExprId,
    divisor: ExprId,
) -> ExprId {
    if coefficient == divisor || SemanticEqualityChecker::new(ctx).are_equal(coefficient, divisor) {
        return ctx.num(1);
    }
    if let Some(stripped) = strip_backend_exact_factor(ctx, coefficient, divisor, "") {
        return stripped;
    }
    ctx.add(Expr::Div(coefficient, divisor))
}

fn is_symbolic_external_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) != variable)
        && !contains_named_var(ctx, expr, variable)
}

pub(super) fn is_supported_external_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> bool {
    !contains_named_var(ctx, expr, variable)
        && is_supported_external_coefficient_inner(ctx, expr, variable, 0)
}

fn is_supported_external_coefficient_inner(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
) -> bool {
    if depth >= BACKEND_EXTERNAL_COEFFICIENT_DEPTH {
        return false;
    }

    match ctx.get(expr) {
        Expr::Number(value) => !value.is_zero(),
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) != variable,
        Expr::Mul(left, right) => {
            is_supported_external_coefficient_inner(ctx, *left, variable, depth + 1)
                && is_supported_external_coefficient_inner(ctx, *right, variable, depth + 1)
        }
        Expr::Div(numerator, denominator) => {
            let Some(denominator_value) = numeric_value(ctx, *denominator) else {
                return false;
            };
            !denominator_value.is_zero()
                && is_supported_external_coefficient_inner(ctx, *numerator, variable, depth + 1)
        }
        Expr::Neg(inner) => {
            is_supported_external_coefficient_inner(ctx, *inner, variable, depth + 1)
        }
        _ => false,
    }
}

pub(super) fn affine_denominator_slope(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    match ctx.get(expr).clone() {
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == variable => {
            Some(BackendAffineSlope::Numeric(BigRational::one()))
        }
        Expr::Mul(left, right) => affine_linear_term_coefficient(ctx, left, right, variable),
        Expr::Neg(inner) => {
            let slope = affine_denominator_slope(ctx, inner, variable)?;
            negate_affine_slope(ctx, slope)
        }
        Expr::Add(left, right) => {
            if is_affine_intercept(ctx, right, variable) {
                return affine_denominator_slope(ctx, left, variable);
            }
            if is_affine_intercept(ctx, left, variable) {
                return affine_denominator_slope(ctx, right, variable);
            }
            None
        }
        Expr::Sub(left, right) => {
            if is_affine_intercept(ctx, right, variable) {
                return affine_denominator_slope(ctx, left, variable);
            }
            if is_affine_intercept(ctx, left, variable) {
                let slope = affine_denominator_slope(ctx, right, variable)?;
                return negate_affine_slope(ctx, slope);
            }
            None
        }
        _ => None,
    }
}

fn affine_linear_term_coefficient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if is_variable(ctx, right, variable) {
        return affine_slope_coefficient(ctx, left, variable);
    }
    if is_variable(ctx, left, variable) {
        return affine_slope_coefficient(ctx, right, variable);
    }
    if is_supported_external_coefficient(ctx, left, variable) {
        if let Some(slope) = affine_denominator_slope(ctx, right, variable) {
            return multiply_affine_slope(ctx, left, slope);
        }
    }
    if is_supported_external_coefficient(ctx, right, variable) {
        if let Some(slope) = affine_denominator_slope(ctx, left, variable) {
            return multiply_affine_slope(ctx, right, slope);
        }
    }
    None
}

fn affine_slope_coefficient(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
) -> Option<BackendAffineSlope> {
    if let Some(value) = numeric_value(ctx, expr) {
        return (!value.is_zero()).then_some(BackendAffineSlope::Numeric(value));
    }
    if is_supported_external_coefficient(ctx, expr, variable) {
        return Some(BackendAffineSlope::Symbolic(expr));
    }
    None
}

fn multiply_affine_slope(
    ctx: &mut Context,
    coefficient: ExprId,
    slope: BackendAffineSlope,
) -> Option<BackendAffineSlope> {
    match slope {
        BackendAffineSlope::Numeric(value) => {
            if let Some(coefficient_value) = numeric_value(ctx, coefficient) {
                let product = coefficient_value * value;
                return (!product.is_zero()).then_some(BackendAffineSlope::Numeric(product));
            }
            let value_expr = ctx.add(Expr::Number(value));
            Some(BackendAffineSlope::Symbolic(build_backend_product(
                ctx,
                coefficient,
                value_expr,
            )))
        }
        BackendAffineSlope::Symbolic(slope_expr) => Some(BackendAffineSlope::Symbolic(
            build_backend_product(ctx, coefficient, slope_expr),
        )),
    }
}

fn negate_affine_slope(ctx: &mut Context, slope: BackendAffineSlope) -> Option<BackendAffineSlope> {
    match slope {
        BackendAffineSlope::Numeric(value) => Some(BackendAffineSlope::Numeric(-value)),
        BackendAffineSlope::Symbolic(expr) => {
            Some(BackendAffineSlope::Symbolic(ctx.add(Expr::Neg(expr))))
        }
    }
}

fn is_affine_intercept(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    is_numeric_constant(ctx, expr)
        || (is_symbolic_external_coefficient(ctx, expr, variable)
            && !contains_named_var(ctx, expr, variable))
}

fn is_builtin_of_variable(ctx: &Context, expr: ExprId, builtin: BuiltinFn, variable: &str) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, builtin) && args.len() == 1 => {
            is_variable(ctx, args[0], variable)
        }
        _ => false,
    }
}

pub(super) fn is_variable(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == variable)
}

fn is_numeric_constant(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

pub(super) fn numeric_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        _ => None,
    }
}

pub(super) fn backend_numeric_constant_value(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> Option<BigRational> {
    if depth > 4 {
        return None;
    }
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Add(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                + backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Sub(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                - backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Mul(left, right) => Some(
            backend_numeric_constant_value(ctx, *left, depth + 1)?
                * backend_numeric_constant_value(ctx, *right, depth + 1)?,
        ),
        Expr::Div(left, right) => {
            let numerator = backend_numeric_constant_value(ctx, *left, depth + 1)?;
            let denominator = backend_numeric_constant_value(ctx, *right, depth + 1)?;
            (!denominator.is_zero()).then(|| numerator / denominator)
        }
        Expr::Neg(inner) => Some(-backend_numeric_constant_value(ctx, *inner, depth + 1)?),
        _ => None,
    }
}

fn positive_rational_radius_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let value = numeric_value(ctx, expr)?;
    if !value.is_positive() {
        return None;
    }
    if let Some(exact_radius) = exact_positive_rational_sqrt_expr(ctx, expr) {
        Some(exact_radius)
    } else {
        Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]))
    }
}

pub(super) fn positive_radius_expr(
    ctx: &mut Context,
    expr: ExprId,
    required_condition: &Option<ConditionPredicate>,
) -> Option<ExprId> {
    if let Some(radius) = positive_rational_radius_expr(ctx, expr) {
        return Some(radius);
    }
    if let Some(radius) = squared_radius_expr(ctx, expr, required_condition) {
        return Some(radius);
    }
    matches!(required_condition, Some(ConditionPredicate::Positive(condition_expr)) if *condition_expr == expr || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr))
        .then(|| ctx.call_builtin(BuiltinFn::Sqrt, vec![expr]))
}

fn squared_radius_expr(
    ctx: &Context,
    expr: ExprId,
    required_condition: &Option<ConditionPredicate>,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_two(ctx, *exponent) => {
            if numeric_value(ctx, *base)
                .map(|value| !value.is_zero())
                .unwrap_or(false)
            {
                return Some(*base);
            }
            matches!(required_condition, Some(ConditionPredicate::NonZero(condition_expr)) if *condition_expr == *base || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, *base))
                .then_some(*base)
        }
        _ => None,
    }
}

pub(super) fn positive_radius_square_value(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<BackendRadiusSquareValue> {
    if let Some(radius_value) = numeric_value(ctx, expr) {
        return radius_value
            .is_positive()
            .then_some(BackendRadiusSquareValue::Numeric(
                radius_value.clone() * radius_value,
            ));
    }

    if let Some(radicand) = positive_numeric_sqrt_radicand(ctx, expr) {
        return Some(BackendRadiusSquareValue::Numeric(radicand));
    }

    if let Some(radicand_expr) = crate::root_forms::extract_square_root_base(ctx, expr) {
        if required_positive_condition_matches(ctx, radicand_expr, required_conditions) {
            return Some(BackendRadiusSquareValue::ConditionalSymbolic(radicand_expr));
        }
    }

    if required_nonzero_condition_matches(ctx, expr, required_conditions)
        && !contains_named_var(ctx, expr, variable)
    {
        let two = ctx.add(Expr::Number(BigRational::from_integer(2.into())));
        return Some(BackendRadiusSquareValue::ConditionalSymbolic(
            ctx.add(Expr::Pow(expr, two)),
        ));
    }

    None
}

fn positive_numeric_sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let radicand_expr = crate::root_forms::extract_square_root_base(ctx, expr)?;
    let radicand = numeric_value(ctx, radicand_expr)?;
    radicand.is_positive().then_some(radicand)
}

fn required_positive_condition_matches(
    ctx: &Context,
    expr: ExprId,
    required_conditions: &[ConditionPredicate],
) -> bool {
    required_conditions.iter().any(|condition| {
        let ConditionPredicate::Positive(condition_expr) = condition else {
            return false;
        };
        *condition_expr == expr
            || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr)
    })
}

fn required_nonzero_condition_matches(
    ctx: &Context,
    expr: ExprId,
    required_conditions: &[ConditionPredicate],
) -> bool {
    required_conditions.iter().any(|condition| {
        let ConditionPredicate::NonZero(condition_expr) = condition else {
            return false;
        };
        *condition_expr == expr
            || SemanticEqualityChecker::new(ctx).are_equal(*condition_expr, expr)
    })
}

fn exact_positive_rational_sqrt_expr(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let value = numeric_value(ctx, expr)?;
    if !value.is_positive() {
        return None;
    }
    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(ctx.add(Expr::Number(BigRational::new(sqrt_num, sqrt_den))))
    } else {
        None
    }
}

pub(super) fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_one())
}

pub(super) fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_zero())
}

pub(super) fn is_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if *value == BigRational::from_integer(2.into())
    )
}

pub fn try_algorithmic_integration_backend(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    config: AlgorithmicIntegrationBackendConfig,
) -> AlgorithmicIntegrationCandidate {
    if !config.mode.attempts_backend() {
        return AlgorithmicIntegrationCandidate::disabled(integrand, variable);
    }

    let mut probe_runner = AlgorithmicIntegrationProbeRunner::new(config.budget);
    if let Some(candidate) = probe_runner
        .try_method_probe(AlgorithmicIntegrationMethod::Rational, |probe_runner| {
            try_rational_reciprocal_affine_probe(ctx, integrand, variable, probe_runner)
        })
    {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }
    if let Some(candidate) =
        probe_runner.try_method_probe(AlgorithmicIntegrationMethod::Hermite, |probe_runner| {
            try_hermite_positive_quadratic_log_derivative_probe(
                ctx,
                integrand,
                variable,
                probe_runner,
            )
        })
    {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }
    if let Some(candidate) = probe_runner.try_method_probe(
        AlgorithmicIntegrationMethod::HeurischProbe,
        |probe_runner| {
            try_heurisch_sine_log_derivative_probe(ctx, integrand, variable, probe_runner)
        },
    ) {
        let mut candidate = candidate;
        candidate.record_probe_usage(&probe_runner);
        return candidate;
    }

    let mut candidate = if probe_runner.method_budget_exhausted() {
        AlgorithmicIntegrationCandidate::budget_exceeded(integrand, variable)
    } else {
        AlgorithmicIntegrationCandidate::unsupported(integrand, variable)
    };
    candidate.record_probe_usage(&probe_runner);
    candidate
}
