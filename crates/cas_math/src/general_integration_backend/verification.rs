//! Antiderivative verification service: diff(candidate, x) ~ integrand with structured outcomes.

use super::methods::*;
use super::verification_algebraic::*;
use super::verification_normalization::*;
use super::*;

use crate::expr_domain::exprs_equivalent;
use crate::expr_predicates::contains_named_var;
use crate::semantic_equality::SemanticEqualityChecker;
use crate::symbolic_differentiation_support::differentiate_symbolic_expr;
use cas_ast::{ConditionPredicate, Context, Expr, ExprId};
use num_traits::One;

pub(crate) fn verify_antiderivative_by_differentiation(
    ctx: &mut Context,
    candidate: &mut AlgorithmicIntegrationCandidate,
) -> AlgorithmicIntegrationVerificationOutcome {
    let report = antiderivative_verification_report(ctx, candidate);
    report.apply_to_candidate(candidate);
    report.into_outcome()
}

pub(crate) fn antiderivative_verification_report(
    ctx: &mut Context,
    candidate: &AlgorithmicIntegrationCandidate,
) -> AlgorithmicIntegrationVerificationReport {
    let Some(antiderivative) = candidate.antiderivative else {
        return AlgorithmicIntegrationVerificationReport {
            status: AlgorithmicIntegrationVerificationStatus::Inconclusive,
            evidence: AlgorithmicIntegrationVerificationEvidence::None,
            normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            blocker: AlgorithmicIntegrationVerificationBlocker::MissingAntiderivative,
            residual_reason: Some(AlgorithmicIntegrationResidualReason::VerificationInconclusive),
            derivative: None,
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    };

    let variable = candidate.variable.clone();
    let Some(derivative) = differentiate_symbolic_expr(ctx, antiderivative, &variable) else {
        return AlgorithmicIntegrationVerificationReport {
            status: AlgorithmicIntegrationVerificationStatus::Inconclusive,
            evidence: AlgorithmicIntegrationVerificationEvidence::None,
            normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            blocker: AlgorithmicIntegrationVerificationBlocker::DifferentiationUnavailable,
            residual_reason: Some(AlgorithmicIntegrationResidualReason::VerificationInconclusive),
            derivative: None,
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    };

    if derivative_matches_integrand(ctx, derivative, candidate.integrand) {
        let status = if candidate.required_conditions.is_empty() {
            AlgorithmicIntegrationVerificationStatus::Verified
        } else {
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        };
        return AlgorithmicIntegrationVerificationReport {
            status,
            evidence: AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation,
            normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: None,
            derivative: Some(derivative),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    }

    if method_specific_derivative_matches_integrand(ctx, candidate, derivative) {
        let status = if candidate.required_conditions.is_empty() {
            AlgorithmicIntegrationVerificationStatus::Verified
        } else {
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        };
        return AlgorithmicIntegrationVerificationReport {
            status,
            evidence: AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation,
            normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: None,
            derivative: Some(derivative),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    }

    let normalization_attempt = normalize_backend_verification_expr_to_match(
        ctx,
        derivative,
        candidate.integrand,
        &candidate.variable,
        &candidate.required_conditions,
    );

    if let Some(normalization_reason) = normalization_attempt.matched_reason {
        let status = if candidate.required_conditions.is_empty() {
            AlgorithmicIntegrationVerificationStatus::Verified
        } else {
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        };
        return AlgorithmicIntegrationVerificationReport {
            status,
            evidence: AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation,
            normalization_reason,
            verification_normalization_passes_used: normalization_attempt.passes_used,
            blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: None,
            derivative: Some(derivative),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    }

    if algebraic_rational_zero_test(
        ctx,
        derivative,
        candidate.integrand,
        &candidate.variable,
        &candidate.required_conditions,
    ) == Some(true)
    {
        let status = if candidate.required_conditions.is_empty() {
            AlgorithmicIntegrationVerificationStatus::Verified
        } else {
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        };
        return AlgorithmicIntegrationVerificationReport {
            status,
            evidence: AlgorithmicIntegrationVerificationEvidence::AlgebraicZeroTest,
            normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: normalization_attempt.passes_used,
            blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: None,
            derivative: Some(derivative),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
        };
    }

    let verification_residual = ctx.add(Expr::Sub(derivative, candidate.integrand));
    let verification_residual_kind =
        classify_backend_verification_residual(ctx, verification_residual, &candidate.variable);
    let verification_residual_signature = classify_backend_verification_residual_signature(
        ctx,
        verification_residual,
        &candidate.variable,
    );
    AlgorithmicIntegrationVerificationReport {
        status: AlgorithmicIntegrationVerificationStatus::Failed,
        evidence: AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation,
        normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason::None,
        verification_normalization_passes_used: normalization_attempt.passes_used,
        blocker: AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch,
        residual_reason: Some(AlgorithmicIntegrationResidualReason::VerificationFailed),
        derivative: Some(derivative),
        verification_residual: Some(verification_residual),
        verification_residual_kind: Some(verification_residual_kind),
        verification_residual_signature: Some(verification_residual_signature),
    }
}

fn derivative_matches_integrand(ctx: &Context, derivative: ExprId, integrand: ExprId) -> bool {
    derivative == integrand
        || SemanticEqualityChecker::new(ctx).are_equal(derivative, integrand)
        || exprs_equivalent(ctx, derivative, integrand)
}

fn method_specific_derivative_matches_integrand(
    ctx: &mut Context,
    candidate: &AlgorithmicIntegrationCandidate,
    derivative: ExprId,
) -> bool {
    match candidate.method {
        AlgorithmicIntegrationMethod::Rational => rational_affine_quotient_derivative_matches(
            ctx,
            derivative,
            candidate.integrand,
            &candidate.variable,
            &candidate.required_conditions,
        ),
        AlgorithmicIntegrationMethod::Hermite => hermite_conjugate_reciprocal_derivative_matches(
            ctx,
            derivative,
            candidate.integrand,
            &candidate.variable,
            &candidate.required_conditions,
        ),
        _ => false,
    }
}

fn hermite_conjugate_reciprocal_derivative_matches(
    ctx: &mut Context,
    derivative: ExprId,
    integrand: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> bool {
    if let Some(combined) = normalize_hermite_conjugate_reciprocal_derivative_expr(
        ctx,
        derivative,
        variable,
        required_conditions,
        0,
    ) {
        if derivative_matches_integrand(ctx, combined, integrand) {
            return true;
        }
        let cleanup_attempt = normalize_backend_verification_expr_to_match(
            ctx,
            combined,
            integrand,
            variable,
            required_conditions,
        );
        if matches!(
            cleanup_attempt.matched_reason,
            Some(
                AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation
                    | AlgorithmicIntegrationVerificationNormalizationReason::QuotientNumericFactorCancellation
                    | AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
                    | AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
            )
        ) {
            return true;
        }
    }

    let attempt = normalize_backend_verification_expr_to_match(
        ctx,
        derivative,
        integrand,
        variable,
        required_conditions,
    );
    matches!(
        attempt.matched_reason,
        Some(
            AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference
                | AlgorithmicIntegrationVerificationNormalizationReason::QuotientNumericFactorCancellation
                | AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
                | AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
        )
    )
}

fn normalize_hermite_conjugate_reciprocal_derivative_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
    depth: usize,
) -> Option<ExprId> {
    if depth >= BACKEND_VERIFICATION_NORMALIZE_DEPTH {
        return None;
    }

    if let Some(combined) = normalize_backend_conjugate_reciprocal_expr(ctx, expr, variable) {
        return Some(combined);
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let normalized_left = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                left,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(left);
            let normalized_left = method_specific_verification_term(
                ctx,
                normalized_left,
                variable,
                required_conditions,
            );
            let normalized_right = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                right,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(right);
            let normalized_right = method_specific_verification_term(
                ctx,
                normalized_right,
                variable,
                required_conditions,
            );
            if let Some(combined) = normalize_backend_same_denominator_sum(
                ctx,
                normalized_left,
                normalized_right,
                BackendVerificationScope {
                    variable,
                    required_conditions,
                    depth,
                    in_power_exponent: false,
                },
                AlgorithmicIntegrationVerificationNormalizationReason::SameDenominatorNumeratorCancellation,
            ) {
                return Some(combined.expr);
            }
            (normalized_left != left || normalized_right != right)
                .then(|| ctx.add(Expr::Add(normalized_left, normalized_right)))
        }
        Expr::Sub(left, right) => {
            let normalized_left = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                left,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(left);
            let normalized_right = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                right,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(right);
            if let Some(combined) = normalize_backend_conjugate_reciprocal_difference(
                ctx,
                normalized_left,
                normalized_right,
                variable,
            ) {
                return Some(combined);
            }
            (normalized_left != left || normalized_right != right)
                .then(|| ctx.add(Expr::Sub(normalized_left, normalized_right)))
        }
        Expr::Mul(left, right) => normalize_backend_scaled_conjugate_reciprocal_product(
            ctx,
            left,
            right,
            variable,
            required_conditions,
        )
        .or_else(|| {
            normalize_backend_fraction_product_conjugate_reciprocal_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            )
        })
        .or_else(|| {
            let normalized_left = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                left,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(left);
            let normalized_right = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                right,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(right);
            if normalized_left != left || normalized_right != right {
                if let Some(combined) = normalize_backend_scaled_conjugate_reciprocal_product(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                    required_conditions,
                ) {
                    return Some(combined);
                }
                if let Some(combined) =
                    normalize_backend_fraction_product_conjugate_reciprocal_quotient(
                        ctx,
                        normalized_left,
                        normalized_right,
                        variable,
                        required_conditions,
                    )
                {
                    return Some(combined);
                }
            }
            let rebuilt_product = ctx.add(Expr::Mul(normalized_left, normalized_right));
            let normalized_product = method_specific_verification_term(
                ctx,
                rebuilt_product,
                variable,
                required_conditions,
            );
            if normalized_product != rebuilt_product {
                return Some(normalized_product);
            }
            (normalized_left != left || normalized_right != right).then_some(rebuilt_product)
        }),
        Expr::Div(left, right) => normalize_backend_conjugate_reciprocal_quotient(
            ctx,
            left,
            right,
            variable,
            required_conditions,
        )
        .or_else(|| {
            let normalized_left = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                left,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(left);
            let normalized_right = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                right,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(right);
            if normalized_left != left || normalized_right != right {
                if let Some(combined) = normalize_backend_conjugate_reciprocal_quotient(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                    required_conditions,
                ) {
                    return Some(combined);
                }
            }
            let rebuilt_quotient = ctx.add(Expr::Div(normalized_left, normalized_right));
            let normalized_quotient = method_specific_verification_term(
                ctx,
                rebuilt_quotient,
                variable,
                required_conditions,
            );
            if normalized_quotient != rebuilt_quotient {
                return Some(normalized_quotient);
            }
            (normalized_left != left || normalized_right != right).then_some(rebuilt_quotient)
        }),
        Expr::Pow(base, exponent) => {
            let normalized_base = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                base,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(base);
            let normalized_exponent = normalize_hermite_conjugate_reciprocal_derivative_expr(
                ctx,
                exponent,
                variable,
                required_conditions,
                depth + 1,
            )
            .unwrap_or(exponent);
            if is_one(ctx, normalized_exponent)
                || backend_numeric_constant_value(ctx, normalized_exponent, 0)
                    .map(|value| value.is_one())
                    .unwrap_or(false)
            {
                return Some(normalized_base);
            }
            (normalized_base != base || normalized_exponent != exponent)
                .then(|| ctx.add(Expr::Pow(normalized_base, normalized_exponent)))
        }
        _ => None,
    }
}

fn rational_affine_quotient_derivative_matches(
    ctx: &mut Context,
    derivative: ExprId,
    integrand: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> bool {
    let Ok(parts) = affine_denominator_linear_numerator_parts(ctx, integrand, variable) else {
        return false;
    };
    let has_quotient_remainder = !is_zero(ctx, parts.quotient_coefficient);

    if has_quotient_remainder {
        let combined = match ctx.get(derivative).clone() {
            Expr::Add(left, right) => {
                let normalized_left =
                    method_specific_verification_term(ctx, left, variable, required_conditions);
                let normalized_right =
                    method_specific_verification_term(ctx, right, variable, required_conditions);
                normalize_backend_affine_quotient_remainder_sum(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                )
            }
            Expr::Sub(left, right) => {
                let normalized_left =
                    method_specific_verification_term(ctx, left, variable, required_conditions);
                let normalized_right =
                    method_specific_verification_term(ctx, right, variable, required_conditions);
                normalize_backend_affine_quotient_remainder_difference(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                )
            }
            _ => None,
        };

        if combined
            .map(|combined| derivative_matches_integrand(ctx, combined, integrand))
            .unwrap_or(false)
        {
            return true;
        }
    }

    let attempt = normalize_backend_verification_expr_to_match(
        ctx,
        derivative,
        integrand,
        variable,
        required_conditions,
    );
    matches!(
        attempt.matched_reason,
        Some(
            AlgorithmicIntegrationVerificationNormalizationReason::PowerOneElision
                | AlgorithmicIntegrationVerificationNormalizationReason::NumericScaledQuotient
                | AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
        )
    )
}

fn method_specific_verification_term(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => normalize_backend_affine_slope_quotient_product(
            ctx,
            left,
            right,
            variable,
            required_conditions,
        )
        .or_else(|| normalize_backend_numeric_scaled_quotient(ctx, left, right))
        .or_else(|| normalize_backend_symbolic_scaled_quotient(ctx, left, right, variable))
        .or_else(|| normalize_backend_fraction_product_quotient(ctx, left, right))
        .unwrap_or(expr),
        Expr::Div(left, right) => normalize_backend_affine_denominator_common_factor_quotient(
            ctx,
            left,
            right,
            variable,
            required_conditions,
        )
        .or_else(|| {
            normalize_backend_scaled_arctan_radius_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            )
        })
        .or_else(|| normalize_backend_quotient_numeric_factor_cancellation(ctx, left, right))
        .or_else(|| {
            normalize_backend_quotient_symbolic_factor_cancellation(ctx, left, right, variable)
        })
        .unwrap_or(expr),
        _ => expr,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BackendVerificationNormalizationMatchAttempt {
    pub(super) matched_reason: Option<AlgorithmicIntegrationVerificationNormalizationReason>,
    pub(super) passes_used: usize,
}

pub(super) fn normalize_backend_verification_expr_to_match(
    ctx: &mut Context,
    derivative: ExprId,
    integrand: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> BackendVerificationNormalizationMatchAttempt {
    let mut current = derivative;
    let mut passes_used = 0;
    for _ in 0..BACKEND_VERIFICATION_NORMALIZE_PASSES {
        let Some(normalized) =
            normalize_backend_verification_expr(ctx, current, variable, required_conditions)
        else {
            return BackendVerificationNormalizationMatchAttempt {
                matched_reason: None,
                passes_used,
            };
        };
        current = normalized.expr;
        let reason = normalized.reason;
        passes_used += 1;
        if derivative_matches_integrand(ctx, current, integrand) {
            return BackendVerificationNormalizationMatchAttempt {
                matched_reason: Some(reason),
                passes_used,
            };
        }
    }
    BackendVerificationNormalizationMatchAttempt {
        matched_reason: None,
        passes_used,
    }
}

fn classify_backend_verification_residual(
    ctx: &mut Context,
    residual: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationVerificationResidualKind {
    let zero = ctx.num(0);
    if is_zero(ctx, residual)
        || SemanticEqualityChecker::new(ctx).are_equal(residual, zero)
        || exprs_equivalent(ctx, residual, zero)
    {
        return AlgorithmicIntegrationVerificationResidualKind::EquivalentZero;
    }
    if contains_named_var(ctx, residual, variable) {
        AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable
    } else {
        AlgorithmicIntegrationVerificationResidualKind::VariableFree
    }
}

fn classify_backend_verification_residual_signature(
    ctx: &mut Context,
    residual: ExprId,
    variable: &str,
) -> AlgorithmicIntegrationVerificationResidualSignature {
    let zero = ctx.num(0);
    if is_zero(ctx, residual)
        || SemanticEqualityChecker::new(ctx).are_equal(residual, zero)
        || exprs_equivalent(ctx, residual, zero)
    {
        return AlgorithmicIntegrationVerificationResidualSignature::EquivalentZero;
    }
    if !contains_named_var(ctx, residual, variable) {
        return AlgorithmicIntegrationVerificationResidualSignature::VariableFreeConstant;
    }
    if is_backend_affine_in_variable(ctx, residual, variable, 0) {
        return AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable;
    }
    if contains_backend_function_of_variable(ctx, residual, variable, 0) {
        return AlgorithmicIntegrationVerificationResidualSignature::FunctionOfVariable;
    }
    AlgorithmicIntegrationVerificationResidualSignature::VariableDependentOther
}

fn is_backend_affine_in_variable(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
) -> bool {
    if depth >= BACKEND_RESIDUAL_SIGNATURE_DEPTH {
        return false;
    }

    match ctx.get(expr) {
        Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == variable,
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            is_backend_affine_in_variable(ctx, *left, variable, depth + 1)
                && is_backend_affine_in_variable(ctx, *right, variable, depth + 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            is_backend_affine_in_variable(ctx, *inner, variable, depth + 1)
        }
        Expr::Mul(left, right) => {
            (!contains_named_var(ctx, *left, variable)
                && is_backend_affine_in_variable(ctx, *right, variable, depth + 1))
                || (!contains_named_var(ctx, *right, variable)
                    && is_backend_affine_in_variable(ctx, *left, variable, depth + 1))
        }
        Expr::Div(numerator, denominator) => {
            !contains_named_var(ctx, *denominator, variable)
                && is_backend_affine_in_variable(ctx, *numerator, variable, depth + 1)
        }
        _ => false,
    }
}

fn contains_backend_function_of_variable(
    ctx: &Context,
    expr: ExprId,
    variable: &str,
    depth: usize,
) -> bool {
    if depth >= BACKEND_RESIDUAL_SIGNATURE_DEPTH {
        return false;
    }

    match ctx.get(expr) {
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| contains_named_var(ctx, *arg, variable)),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            contains_backend_function_of_variable(ctx, *left, variable, depth + 1)
                || contains_backend_function_of_variable(ctx, *right, variable, depth + 1)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_backend_function_of_variable(ctx, *inner, variable, depth + 1)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|entry| contains_backend_function_of_variable(ctx, *entry, variable, depth + 1)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}
