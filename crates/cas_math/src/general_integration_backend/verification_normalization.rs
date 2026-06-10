//! Bounded structural normalization used by verification to match generated derivatives. Expected growth zone for new families.

use super::methods::*;
use super::*;

use crate::expr_domain::exprs_equivalent;
use crate::expr_predicates::contains_named_var;
use crate::semantic_equality::SemanticEqualityChecker;
use cas_ast::{ConditionPredicate, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BackendVerificationNormalization {
    pub(super) expr: ExprId,
    pub(super) reason: AlgorithmicIntegrationVerificationNormalizationReason,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BackendVerificationScope<'a> {
    pub(super) variable: &'a str,
    pub(super) required_conditions: &'a [ConditionPredicate],
    pub(super) depth: usize,
    pub(super) in_power_exponent: bool,
}

impl<'a> BackendVerificationScope<'a> {
    fn child(self) -> Self {
        Self {
            depth: self.depth + 1,
            ..self
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BackendAffineSlope {
    Numeric(BigRational),
    Symbolic(ExprId),
}

impl BackendAffineSlope {
    pub(super) fn required_condition(&self) -> Option<ConditionPredicate> {
        match self {
            BackendAffineSlope::Numeric(_) => None,
            BackendAffineSlope::Symbolic(expr) => Some(ConditionPredicate::NonZero(*expr)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BackendRadiusSquareValue {
    Numeric(BigRational),
    ConditionalSymbolic(ExprId),
}

impl BackendRadiusSquareValue {
    pub(super) fn expr(&self, ctx: &mut Context) -> ExprId {
        match self {
            BackendRadiusSquareValue::Numeric(value) => ctx.add(Expr::Number(value.clone())),
            BackendRadiusSquareValue::ConditionalSymbolic(expr) => *expr,
        }
    }
}

pub(super) fn normalize_backend_verification_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<BackendVerificationNormalization> {
    normalize_backend_verification_expr_inner(ctx, expr, variable, required_conditions, 0, false)
        .filter(|normalized| normalized.expr != expr)
}

fn normalize_backend_verification_expr_inner(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
    depth: usize,
    in_power_exponent: bool,
) -> Option<BackendVerificationNormalization> {
    if depth >= BACKEND_VERIFICATION_NORMALIZE_DEPTH {
        return None;
    }

    let scope = BackendVerificationScope {
        variable,
        required_conditions,
        depth,
        in_power_exponent,
    };

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if let Some(combined) =
                normalize_backend_conjugate_reciprocal_addition(ctx, left, right, variable)
            {
                let reason =
                    AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference;
                return Some(BackendVerificationNormalization {
                    expr: combined,
                    reason,
                });
            }
            if let Some(combined) =
                normalize_backend_affine_quotient_remainder_sum(ctx, left, right, variable)
            {
                let reason =
                    AlgorithmicIntegrationVerificationNormalizationReason::AffineQuotientRemainderSum;
                return Some(BackendVerificationNormalization {
                    expr: combined,
                    reason,
                });
            }
            let normalized =
                normalize_backend_verification_binary(ctx, left, right, scope, Expr::Add)?;
            if let Expr::Add(normalized_left, normalized_right) = ctx.get(normalized.expr).clone() {
                if let Some(combined) = normalize_backend_conjugate_reciprocal_addition(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                ) {
                    let reason =
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference;
                    return Some(BackendVerificationNormalization {
                        expr: combined,
                        reason,
                    });
                }
                if let Some(combined) = normalize_backend_affine_quotient_remainder_sum(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                ) {
                    let reason =
                        AlgorithmicIntegrationVerificationNormalizationReason::AffineQuotientRemainderSum;
                    return Some(BackendVerificationNormalization {
                        expr: combined,
                        reason,
                    });
                }
                if let Some(combined) = normalize_backend_same_denominator_sum(
                    ctx,
                    normalized_left,
                    normalized_right,
                    scope.child(),
                    normalized.reason.clone(),
                ) {
                    return Some(BackendVerificationNormalization {
                        expr: combined.expr,
                        reason: combined.reason,
                    });
                }
            }
            Some(normalized)
        }
        Expr::Sub(original_left, original_right) => {
            let left_normalization = normalize_backend_verification_expr_inner(
                ctx,
                original_left,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            );
            let left = left_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(original_left);
            let right_normalization = normalize_backend_verification_expr_inner(
                ctx,
                original_right,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            );
            let right = right_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(original_right);

            if let Some(combined) =
                normalize_backend_conjugate_reciprocal_difference(ctx, left, right, variable)
            {
                return Some(BackendVerificationNormalization {
                    expr: combined,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                });
            }
            if let Some(combined) =
                normalize_backend_affine_quotient_remainder_difference(ctx, left, right, variable)
            {
                return Some(BackendVerificationNormalization {
                    expr: combined,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::AffineQuotientRemainderSum,
                });
            }

            if in_power_exponent {
                if let (Expr::Number(left_value), Expr::Number(right_value)) =
                    (ctx.get(left), ctx.get(right))
                {
                    let reduced = left_value.clone() - right_value.clone();
                    return Some(BackendVerificationNormalization {
                        expr: ctx.add(Expr::Number(reduced)),
                        reason:
                            AlgorithmicIntegrationVerificationNormalizationReason::ExponentNumericSubtraction,
                    });
                }
            }

            if left != original_left || right != original_right {
                Some(BackendVerificationNormalization {
                    expr: ctx.add(Expr::Sub(left, right)),
                    reason: left_normalization
                        .or(right_normalization)
                        .expect("changed child normalization")
                        .reason,
                })
            } else {
                None
            }
        }
        Expr::Mul(left, right) => {
            if let Some(normalized) =
                normalize_backend_fraction_product_conjugate_reciprocal_quotient(
                    ctx,
                    left,
                    right,
                    variable,
                    required_conditions,
                )
            {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                });
            }
            let left_normalization = normalize_backend_verification_expr_inner(
                ctx,
                left,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            );
            let normalized_left = left_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(left);
            let right_normalization = normalize_backend_verification_expr_inner(
                ctx,
                right,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            );
            let normalized_right = right_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(right);

            if let Some(scaled_arctan_normalized) =
                normalize_backend_fraction_product_scaled_arctan_radius_quotient(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                    required_conditions,
                )
            {
                return Some(BackendVerificationNormalization {
                    expr: scaled_arctan_normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient,
                });
            }

            if let Some(cancelled_quotient) = normalize_backend_quotient_numeric_factor_cancellation(
                ctx,
                normalized_left,
                normalized_right,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: cancelled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::QuotientNumericFactorCancellation,
                });
            }

            if let Some(scaled_conjugate) = normalize_backend_scaled_conjugate_reciprocal_product(
                ctx,
                normalized_left,
                normalized_right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: scaled_conjugate,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                });
            }

            if let Some(cancelled_quotient) =
                normalize_backend_quotient_symbolic_factor_cancellation(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                )
            {
                return Some(BackendVerificationNormalization {
                    expr: cancelled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
                });
            }

            if let Some(scaled_quotient) = normalize_backend_affine_slope_quotient_product(
                ctx,
                normalized_left,
                normalized_right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: scaled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
                });
            }

            if let Some(scaled_quotient) =
                normalize_backend_numeric_scaled_quotient(ctx, normalized_left, normalized_right)
            {
                return Some(BackendVerificationNormalization {
                    expr: scaled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::NumericScaledQuotient,
                });
            }

            if let Some(scaled_quotient) = normalize_backend_symbolic_scaled_quotient(
                ctx,
                normalized_left,
                normalized_right,
                variable,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: scaled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient,
                });
            }

            if let Some(normalized) =
                normalize_backend_fraction_product_quotient(ctx, normalized_left, normalized_right)
            {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::NestedQuotientDenominatorProduct,
                });
            }

            if normalized_left != left || normalized_right != right {
                Some(BackendVerificationNormalization {
                    expr: ctx.add(Expr::Mul(normalized_left, normalized_right)),
                    reason: left_normalization
                        .or(right_normalization)
                        .expect("changed child normalization")
                        .reason,
                })
            } else {
                None
            }
        }
        Expr::Div(left, right) => {
            if let Some(normalized) = normalize_backend_affine_denominator_common_factor_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation,
                });
            }
            if let Some(normalized) = normalize_backend_conjugate_reciprocal_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                });
            }
            if let Some(normalized) =
                normalize_backend_conjugate_pole_product_denominator(ctx, left, right, variable)
            {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                });
            }
            if let Some(flattened) =
                normalize_backend_nested_quotient_denominator_product(ctx, left, right)
            {
                return Some(BackendVerificationNormalization {
                    expr: flattened,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::NestedQuotientDenominatorProduct,
                });
            }
            if let Some(normalized) = normalize_backend_scaled_arctan_radius_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: normalized,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient,
                });
            }
            if let Some(cancelled_quotient) = normalize_backend_common_factor_quotient(
                ctx,
                left,
                right,
                variable,
                required_conditions,
            ) {
                return Some(BackendVerificationNormalization {
                    expr: cancelled_quotient,
                    reason:
                        AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation,
                });
            }
            let normalized =
                normalize_backend_verification_binary(ctx, left, right, scope, Expr::Div)?;
            if let Expr::Div(normalized_left, normalized_right) = ctx.get(normalized.expr).clone() {
                if let Some(normalized) =
                    normalize_backend_affine_denominator_common_factor_quotient(
                        ctx,
                        normalized_left,
                        normalized_right,
                        variable,
                        required_conditions,
                    )
                {
                    return Some(BackendVerificationNormalization {
                        expr: normalized,
                        reason:
                            AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation,
                    });
                }
                if let Some(normalized) = normalize_backend_conjugate_pole_product_denominator(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                ) {
                    return Some(BackendVerificationNormalization {
                        expr: normalized,
                        reason:
                            AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference,
                    });
                }
                if let Some(scaled_arctan_normalized) =
                    normalize_backend_scaled_arctan_radius_quotient(
                        ctx,
                        normalized_left,
                        normalized_right,
                        variable,
                        required_conditions,
                    )
                {
                    return Some(BackendVerificationNormalization {
                        expr: scaled_arctan_normalized,
                        reason:
                            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient,
                    });
                }
                if let Some(cancelled_quotient) = normalize_backend_common_factor_quotient(
                    ctx,
                    normalized_left,
                    normalized_right,
                    variable,
                    required_conditions,
                ) {
                    return Some(BackendVerificationNormalization {
                        expr: cancelled_quotient,
                        reason:
                            AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation,
                    });
                }
            }
            Some(normalized)
        }
        Expr::Pow(base, exponent) => {
            let base_normalization = normalize_backend_verification_expr_inner(
                ctx,
                base,
                variable,
                required_conditions,
                depth + 1,
                false,
            );
            let normalized_base = base_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(base);
            let exponent_normalization = normalize_backend_verification_expr_inner(
                ctx,
                exponent,
                variable,
                required_conditions,
                depth + 1,
                true,
            );
            let normalized_exponent = exponent_normalization
                .as_ref()
                .map(|normalization| normalization.expr)
                .unwrap_or(exponent);

            if is_one(ctx, normalized_exponent)
                || backend_numeric_constant_value(ctx, normalized_exponent, 0)
                    .map(|value| value.is_one())
                    .unwrap_or(false)
            {
                return Some(BackendVerificationNormalization {
                    expr: normalized_base,
                    reason: AlgorithmicIntegrationVerificationNormalizationReason::PowerOneElision,
                });
            }

            if normalized_base != base || normalized_exponent != exponent {
                Some(BackendVerificationNormalization {
                    expr: ctx.add(Expr::Pow(normalized_base, normalized_exponent)),
                    reason: base_normalization
                        .or(exponent_normalization)
                        .expect("changed child normalization")
                        .reason,
                })
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let normalized_inner = normalize_backend_verification_expr_inner(
                ctx,
                inner,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            )?;
            Some(BackendVerificationNormalization {
                expr: ctx.add(Expr::Neg(normalized_inner.expr)),
                reason: normalized_inner.reason,
            })
        }
        Expr::Function(fn_id, args) => {
            let mut changed = false;
            let mut reason = None;
            let normalized_args = args
                .into_iter()
                .map(|arg| {
                    if let Some(normalization) = normalize_backend_verification_expr_inner(
                        ctx,
                        arg,
                        variable,
                        required_conditions,
                        depth + 1,
                        false,
                    ) {
                        changed = true;
                        reason.get_or_insert_with(|| normalization.reason.clone());
                        normalization.expr
                    } else {
                        arg
                    }
                })
                .collect::<Vec<_>>();
            changed.then(|| BackendVerificationNormalization {
                expr: ctx.add(Expr::Function(fn_id, normalized_args)),
                reason: reason.expect("changed function argument normalization"),
            })
        }
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut reason = None;
            let normalized_data = data
                .into_iter()
                .map(|arg| {
                    if let Some(normalization) = normalize_backend_verification_expr_inner(
                        ctx,
                        arg,
                        variable,
                        required_conditions,
                        depth + 1,
                        false,
                    ) {
                        changed = true;
                        reason.get_or_insert_with(|| normalization.reason.clone());
                        normalization.expr
                    } else {
                        arg
                    }
                })
                .collect::<Vec<_>>();
            changed.then(|| BackendVerificationNormalization {
                expr: ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: normalized_data,
                }),
                reason: reason.expect("changed matrix entry normalization"),
            })
        }
        Expr::Hold(inner) => {
            let normalized_inner = normalize_backend_verification_expr_inner(
                ctx,
                inner,
                variable,
                required_conditions,
                depth + 1,
                in_power_exponent,
            )?;
            Some(BackendVerificationNormalization {
                expr: ctx.add(Expr::Hold(normalized_inner.expr)),
                reason: normalized_inner.reason,
            })
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => None,
    }
}

fn normalize_backend_verification_binary(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    scope: BackendVerificationScope<'_>,
    build: fn(ExprId, ExprId) -> Expr,
) -> Option<BackendVerificationNormalization> {
    let child_scope = scope.child();
    let left_normalization = normalize_backend_verification_expr_inner(
        ctx,
        left,
        child_scope.variable,
        child_scope.required_conditions,
        child_scope.depth,
        child_scope.in_power_exponent,
    );
    let normalized_left = left_normalization
        .as_ref()
        .map(|normalization| normalization.expr)
        .unwrap_or(left);
    let right_normalization = normalize_backend_verification_expr_inner(
        ctx,
        right,
        child_scope.variable,
        child_scope.required_conditions,
        child_scope.depth,
        child_scope.in_power_exponent,
    );
    let normalized_right = right_normalization
        .as_ref()
        .map(|normalization| normalization.expr)
        .unwrap_or(right);

    if normalized_left != left || normalized_right != right {
        Some(BackendVerificationNormalization {
            expr: ctx.add(build(normalized_left, normalized_right)),
            reason: left_normalization
                .or(right_normalization)
                .expect("changed child normalization")
                .reason,
        })
    } else {
        None
    }
}

pub(super) fn normalize_backend_numeric_scaled_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    normalize_backend_numeric_scaled_quotient_ordered(ctx, left, right)
        .or_else(|| normalize_backend_numeric_scaled_quotient_ordered(ctx, right, left))
}

pub(super) fn normalize_backend_symbolic_scaled_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    normalize_backend_symbolic_scaled_quotient_ordered(ctx, left, right, variable)
        .or_else(|| normalize_backend_symbolic_scaled_quotient_ordered(ctx, right, left, variable))
}

pub(super) fn normalize_backend_affine_slope_quotient_product(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    normalize_backend_affine_slope_quotient_product_ordered(
        ctx,
        left,
        right,
        variable,
        required_conditions,
    )
    .or_else(|| {
        normalize_backend_affine_slope_quotient_product_ordered(
            ctx,
            right,
            left,
            variable,
            required_conditions,
        )
    })
}

fn normalize_backend_affine_slope_quotient_product_ordered(
    ctx: &mut Context,
    slope_quotient: ExprId,
    coefficient_quotient: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let Expr::Div(slope_numerator, affine_denominator) = ctx.get(slope_quotient).clone() else {
        return None;
    };
    let Expr::Div(coefficient, slope_denominator) = ctx.get(coefficient_quotient).clone() else {
        return None;
    };
    let affine_slope = match affine_denominator_slope(ctx, affine_denominator, variable)? {
        BackendAffineSlope::Symbolic(slope) => slope,
        BackendAffineSlope::Numeric(_) => return None,
    };
    if !backend_factors_match(ctx, slope_numerator, affine_slope)
        || !backend_factors_match(ctx, slope_denominator, affine_slope)
        || !backend_factor_has_nonzero_evidence(ctx, affine_slope, variable, required_conditions)
    {
        return None;
    }
    let slope = BackendAffineSlope::Symbolic(affine_slope);
    if !is_supported_backend_linear_coefficient_for_affine_slope(ctx, coefficient, variable, &slope)
    {
        return None;
    }

    Some(ctx.add(Expr::Div(coefficient, affine_denominator)))
}

fn normalize_backend_nested_quotient_denominator_product(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<ExprId> {
    let Expr::Div(inner_numerator, inner_denominator) = ctx.get(numerator).clone() else {
        return None;
    };
    let combined_denominator = build_backend_product(ctx, inner_denominator, denominator);
    Some(ctx.add(Expr::Div(inner_numerator, combined_denominator)))
}

pub(super) fn normalize_backend_common_factor_quotient(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let mut numerator_factors = backend_mul_factors(ctx, numerator);
    let mut denominator_factors = backend_mul_factors(ctx, denominator);
    let mut cancelled_any = false;
    let mut index = 0usize;
    while index < numerator_factors.len() {
        let factor = numerator_factors[index];
        let Some(denominator_index) = denominator_factors
            .iter()
            .position(|denominator_factor| backend_factors_match(ctx, factor, *denominator_factor))
        else {
            index += 1;
            continue;
        };
        if !backend_factor_has_nonzero_evidence(ctx, factor, variable, required_conditions) {
            index += 1;
            continue;
        }

        numerator_factors.remove(index);
        denominator_factors.remove(denominator_index);
        cancelled_any = true;
    }

    if !cancelled_any {
        return None;
    }

    let normalized_numerator =
        build_backend_factor_product_external_first(ctx, numerator_factors, variable);
    let normalized_denominator = build_backend_factor_product(ctx, denominator_factors);
    if is_one(ctx, normalized_denominator) {
        Some(normalized_numerator)
    } else {
        Some(ctx.add(Expr::Div(normalized_numerator, normalized_denominator)))
    }
}

fn build_backend_factor_product_external_first(
    ctx: &mut Context,
    factors: Vec<ExprId>,
    variable: &str,
) -> ExprId {
    let (mut external_factors, variable_factors): (Vec<_>, Vec<_>) = factors
        .into_iter()
        .partition(|factor| !contains_named_var(ctx, *factor, variable));
    external_factors.extend(variable_factors);
    build_backend_factor_product(ctx, external_factors)
}

fn backend_factors_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right || SemanticEqualityChecker::new(ctx).are_equal(left, right)
}

fn backend_factor_has_nonzero_evidence(
    ctx: &Context,
    factor: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> bool {
    if numeric_value(ctx, factor)
        .map(|value| !value.is_zero())
        .unwrap_or(false)
    {
        return true;
    }
    required_conditions.iter().any(|condition| match condition {
        ConditionPredicate::NonZero(condition_expr)
        | ConditionPredicate::Positive(condition_expr) => {
            backend_factors_match(ctx, factor, *condition_expr)
        }
        _ => false,
    }) && !contains_named_var(ctx, factor, variable)
}

pub(super) fn normalize_backend_scaled_arctan_radius_quotient(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let denominator_factors = backend_mul_factors(ctx, denominator);
    let (scaled_square_index, (variable_expr, radius_expr, radius_square_value)) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(index, factor)| {
                one_plus_scaled_variable_square_parts(ctx, *factor, variable, required_conditions)
                    .map(|parts| (index, parts))
            })?;

    let mut numerator = numerator;
    let mut denominator_numeric_product = BigRational::one();
    let mut denominator_radius_factor_count = 0usize;
    for (index, factor) in denominator_factors.into_iter().enumerate() {
        if index == scaled_square_index {
            continue;
        }

        if let Some(factor_value) = numeric_product_value(ctx, factor) {
            denominator_numeric_product *= factor_value;
        } else if factor == radius_expr
            || SemanticEqualityChecker::new(ctx).are_equal(factor, radius_expr)
        {
            denominator_radius_factor_count += 1;
        } else if is_supported_external_coefficient(ctx, factor, variable) {
            numerator = strip_backend_exact_factor(ctx, numerator, factor, variable)?;
        } else {
            return None;
        }
    }

    let numerator_value = BackendCoefficientProduct::from_expr(ctx, numerator, variable)?;
    if numerator_value.is_zero() {
        return Some(ctx.num(0));
    }

    if !denominator_radius_factor_count.is_multiple_of(2) {
        return None;
    }
    if denominator_numeric_product.is_zero() {
        return None;
    }
    let denominator_radius_square_pair_count = denominator_radius_factor_count / 2;
    let normalized_numerator = match &radius_square_value {
        BackendRadiusSquareValue::Numeric(radius_square_value) => {
            for _ in 0..denominator_radius_square_pair_count {
                denominator_numeric_product *= radius_square_value.clone();
            }
            let scale = radius_square_value.clone() / denominator_numeric_product;
            numerator_value.scale_numeric(ctx, scale)
        }
        BackendRadiusSquareValue::ConditionalSymbolic(radius_square_expr) => {
            if denominator_radius_square_pair_count > 1 {
                return None;
            }
            let scale = BigRational::one() / denominator_numeric_product;
            let scaled_numerator = numerator_value.scale_numeric(ctx, scale);
            if denominator_radius_square_pair_count == 0 {
                build_backend_product(ctx, scaled_numerator, *radius_square_expr)
            } else {
                scaled_numerator
            }
        }
    };

    let two = ctx.num(2);
    let variable_square = ctx.add(Expr::Pow(variable_expr, two));
    let radius_square = radius_square_value.expr(ctx);
    let normalized_denominator = build_backend_sum(ctx, variable_square, radius_square);
    Some(ctx.add(Expr::Div(normalized_numerator, normalized_denominator)))
}

fn normalize_backend_fraction_product_scaled_arctan_radius_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_backend_fraction_factors(
        ctx,
        left,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    collect_backend_fraction_factors(
        ctx,
        right,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    if denominator_factors.is_empty() {
        return None;
    }

    let numerator = build_backend_factor_product(ctx, numerator_factors);
    let denominator = build_backend_factor_product(ctx, denominator_factors);
    normalize_backend_scaled_arctan_radius_quotient(
        ctx,
        numerator,
        denominator,
        variable,
        required_conditions,
    )
}

pub(super) fn normalize_backend_fraction_product_conjugate_reciprocal_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_backend_fraction_factors(
        ctx,
        left,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    collect_backend_fraction_factors(
        ctx,
        right,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    normalize_backend_conjugate_reciprocal_fraction_factors(
        ctx,
        numerator_factors,
        denominator_factors,
        variable,
        required_conditions,
    )
}

pub(super) fn normalize_backend_fraction_product_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_backend_fraction_factors(
        ctx,
        left,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    collect_backend_fraction_factors(
        ctx,
        right,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );
    if denominator_factors.is_empty() {
        return None;
    }

    let numerator = build_backend_factor_product(ctx, numerator_factors);
    let denominator = build_backend_factor_product(ctx, denominator_factors);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn collect_backend_fraction_factors(
    ctx: &Context,
    expr: ExprId,
    in_denominator: bool,
    numerator_factors: &mut Vec<ExprId>,
    denominator_factors: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_backend_fraction_factors(
                ctx,
                *left,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
            collect_backend_fraction_factors(
                ctx,
                *right,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
        }
        Expr::Div(numerator, denominator) => {
            collect_backend_fraction_factors(
                ctx,
                *numerator,
                in_denominator,
                numerator_factors,
                denominator_factors,
            );
            collect_backend_fraction_factors(
                ctx,
                *denominator,
                !in_denominator,
                numerator_factors,
                denominator_factors,
            );
        }
        _ if in_denominator => denominator_factors.push(expr),
        _ => numerator_factors.push(expr),
    }
}

pub(super) fn build_backend_factor_product(ctx: &mut Context, factors: Vec<ExprId>) -> ExprId {
    let mut numeric_product = BigRational::one();
    let mut non_numeric_factors = Vec::new();
    for factor in factors {
        if let Some(value) = numeric_value(ctx, factor) {
            numeric_product *= value;
        } else {
            non_numeric_factors.push(factor);
        }
    }

    if numeric_product.is_zero() {
        return ctx.num(0);
    }
    if !numeric_product.is_one() {
        non_numeric_factors.insert(0, ctx.add(Expr::Number(numeric_product)));
    }

    non_numeric_factors
        .into_iter()
        .fold(ctx.num(1), |product, factor| {
            build_backend_product(ctx, product, factor)
        })
}

fn one_plus_scaled_variable_square_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<(ExprId, ExprId, BackendRadiusSquareValue)> {
    match ctx.get(expr) {
        Expr::Add(left, right) if is_one(ctx, *left) => {
            scaled_variable_square_parts(ctx, *right, variable, required_conditions)
        }
        Expr::Add(left, right) if is_one(ctx, *right) => {
            scaled_variable_square_parts(ctx, *left, variable, required_conditions)
        }
        _ => None,
    }
}

fn scaled_variable_square_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<(ExprId, ExprId, BackendRadiusSquareValue)> {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exponent) if is_two(ctx, exponent) => match ctx.get(base).clone() {
            Expr::Div(numerator, denominator) => {
                let (variable_expr, _) = affine_variable_expr(ctx, numerator, variable)?;
                let radius_square_value =
                    positive_radius_square_value(ctx, denominator, variable, required_conditions)?;
                Some((variable_expr, denominator, radius_square_value))
            }
            _ => None,
        },
        _ => None,
    }
}

pub(super) fn backend_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            let mut factors = backend_mul_factors(ctx, *left);
            factors.extend(backend_mul_factors(ctx, *right));
            factors
        }
        _ => vec![expr],
    }
}

fn numeric_product_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
        Expr::Mul(left, right) => {
            let left_value = numeric_product_value(ctx, *left)?;
            let right_value = numeric_product_value(ctx, *right)?;
            Some(left_value * right_value)
        }
        Expr::Div(left, right) => {
            let left_value = numeric_product_value(ctx, *left)?;
            let right_value = numeric_product_value(ctx, *right)?;
            (!right_value.is_zero()).then_some(left_value / right_value)
        }
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BackendCoefficientProduct {
    Numeric(BigRational),
    External(ExprId),
}

impl BackendCoefficientProduct {
    fn from_expr(ctx: &Context, expr: ExprId, variable: &str) -> Option<Self> {
        if let Some(value) = numeric_product_value(ctx, expr) {
            return Some(Self::Numeric(value));
        }
        if is_supported_external_coefficient(ctx, expr, variable) {
            return Some(Self::External(expr));
        }
        None
    }

    fn is_zero(&self) -> bool {
        matches!(self, Self::Numeric(value) if value.is_zero())
    }

    fn scale_numeric(self, ctx: &mut Context, scale: BigRational) -> ExprId {
        match self {
            Self::Numeric(value) => ctx.add(Expr::Number(value * scale)),
            Self::External(expr) => multiply_backend_numeric_coefficient(ctx, scale, expr),
        }
    }
}

pub(super) fn normalize_backend_quotient_numeric_factor_cancellation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    normalize_backend_quotient_numeric_factor_cancellation_ordered(ctx, left, right).or_else(|| {
        normalize_backend_quotient_numeric_factor_cancellation_ordered(ctx, right, left)
    })
}

pub(super) fn normalize_backend_quotient_symbolic_factor_cancellation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    normalize_backend_quotient_symbolic_factor_cancellation_ordered(ctx, left, right, variable)
        .or_else(|| {
            normalize_backend_quotient_symbolic_factor_cancellation_ordered(
                ctx, right, left, variable,
            )
        })
}

fn normalize_backend_quotient_numeric_factor_cancellation_ordered(
    ctx: &mut Context,
    scale_quotient: ExprId,
    scaled_quotient: ExprId,
) -> Option<ExprId> {
    let Expr::Div(scale_numerator, scale_denominator) = ctx.get(scale_quotient).clone() else {
        return None;
    };
    let scale_denominator_value = numeric_value(ctx, scale_denominator)?;
    if scale_denominator_value.is_zero() {
        return None;
    }

    let Expr::Div(numerator, denominator) = ctx.get(scaled_quotient).clone() else {
        return None;
    };
    let (numerator_coefficient, numerator_core) = split_backend_numeric_factor(ctx, numerator)?;
    if numerator_coefficient.is_zero() {
        return None;
    }

    let remaining_coefficient = numerator_coefficient / scale_denominator_value;
    let combined_numerator = build_backend_product(ctx, scale_numerator, numerator_core);
    let scaled_numerator =
        multiply_backend_numeric_coefficient(ctx, remaining_coefficient, combined_numerator);
    Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
}

fn normalize_backend_quotient_symbolic_factor_cancellation_ordered(
    ctx: &mut Context,
    scale_quotient: ExprId,
    scaled_quotient: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let Expr::Div(scale_numerator, scale_denominator) = ctx.get(scale_quotient).clone() else {
        return None;
    };
    if contains_named_var(ctx, scale_denominator, variable) {
        return None;
    }

    let Expr::Div(numerator, denominator) = ctx.get(scaled_quotient).clone() else {
        return None;
    };
    let remaining_numerator =
        strip_backend_exact_factor(ctx, numerator, scale_denominator, variable)?;
    let scaled_numerator = build_backend_product(ctx, scale_numerator, remaining_numerator);
    Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
}

fn normalize_backend_numeric_scaled_quotient_ordered(
    ctx: &mut Context,
    scale: ExprId,
    quotient: ExprId,
) -> Option<ExprId> {
    let coefficient = numeric_value(ctx, scale)?;
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };

    let scaled_numerator = multiply_backend_numeric_coefficient(ctx, coefficient, numerator);
    Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
}

fn normalize_backend_symbolic_scaled_quotient_ordered(
    ctx: &mut Context,
    scale: ExprId,
    quotient: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if numeric_value(ctx, scale).is_some() || contains_named_var(ctx, scale, variable) {
        return None;
    }

    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };

    let scaled_numerator = build_backend_product(ctx, scale, numerator);
    Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
}

pub(super) fn strip_backend_exact_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if expr == factor || SemanticEqualityChecker::new(ctx).are_equal(expr, factor) {
        return Some(ctx.num(1));
    }

    let mut factors = backend_mul_factors(ctx, expr);
    if factors.len() > 1 {
        if let Some(index) = factors
            .iter()
            .position(|candidate| backend_factors_match(ctx, *candidate, factor))
        {
            factors.remove(index);
            return Some(build_backend_factor_product(ctx, factors));
        }
    }

    match ctx.get(expr).clone() {
        Expr::Pow(base, exponent)
            if is_two(ctx, exponent)
                && (base == factor
                    || SemanticEqualityChecker::new(ctx).are_equal(base, factor)) =>
        {
            Some(base)
        }
        Expr::Mul(left, right) => {
            if left == factor || SemanticEqualityChecker::new(ctx).are_equal(left, factor) {
                return Some(right);
            }
            if right == factor || SemanticEqualityChecker::new(ctx).are_equal(right, factor) {
                return Some(left);
            }
            None
        }
        _ if !contains_named_var(ctx, factor, variable) => None,
        _ => None,
    }
}

pub(super) fn multiply_backend_numeric_coefficient(
    ctx: &mut Context,
    coefficient: BigRational,
    expr: ExprId,
) -> ExprId {
    if coefficient.is_zero() {
        return ctx.add(Expr::Number(BigRational::zero()));
    }
    if coefficient.is_one() {
        return expr;
    }

    match ctx.get(expr).clone() {
        Expr::Number(value) => ctx.add(Expr::Number(coefficient * value)),
        Expr::Mul(left, right) => {
            if let Some(left_value) = numeric_value(ctx, left) {
                return build_backend_numeric_product(ctx, coefficient * left_value, right);
            }
            if let Some(right_value) = numeric_value(ctx, right) {
                return build_backend_numeric_product(ctx, coefficient * right_value, left);
            }
            build_backend_numeric_product(ctx, coefficient, expr)
        }
        _ => build_backend_numeric_product(ctx, coefficient, expr),
    }
}

fn split_backend_numeric_factor(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(value) = numeric_value(ctx, *left) {
                return Some((value, *right));
            }
            if let Some(value) = numeric_value(ctx, *right) {
                return Some((value, *left));
            }
            None
        }
        _ => None,
    }
}

pub(super) fn build_backend_product(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) || is_zero(ctx, right) {
        ctx.num(0)
    } else if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        ctx.add(Expr::Mul(left, right))
    }
}

pub(super) fn build_backend_sum(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

pub(super) fn normalize_backend_affine_quotient_remainder_sum(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    normalize_backend_affine_quotient_remainder_sum_ordered(ctx, left, right, variable).or_else(
        || normalize_backend_affine_quotient_remainder_sum_ordered(ctx, right, left, variable),
    )
}

pub(super) fn normalize_backend_affine_quotient_remainder_difference(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if let Some(negated_right_quotient) = negate_backend_quotient(ctx, right) {
        if let Some(combined) = normalize_backend_affine_quotient_remainder_sum_ordered(
            ctx,
            left,
            negated_right_quotient,
            variable,
        ) {
            return Some(combined);
        }
    }

    if matches!(ctx.get(left), Expr::Div(_, _)) {
        let negated_right = negate_backend_expr(ctx, right);
        return normalize_backend_affine_quotient_remainder_sum_ordered(
            ctx,
            negated_right,
            left,
            variable,
        );
    }

    None
}

fn negate_backend_quotient(ctx: &mut Context, quotient: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };
    let negated_numerator = negate_backend_expr(ctx, numerator);
    Some(ctx.add(Expr::Div(negated_numerator, denominator)))
}

fn normalize_backend_affine_quotient_remainder_sum_ordered(
    ctx: &mut Context,
    coefficient: ExprId,
    quotient: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
        return None;
    };
    let denominator_slope = affine_denominator_slope(ctx, denominator, variable)?;
    if !is_supported_backend_linear_coefficient_for_affine_slope(
        ctx,
        coefficient,
        variable,
        &denominator_slope,
    ) || contains_named_var(ctx, coefficient, variable)
    {
        return None;
    }

    if let Some(combined_numerator) = normalize_backend_affine_quotient_remainder_numerator(
        ctx,
        coefficient,
        numerator,
        denominator,
        &denominator_slope,
        variable,
    ) {
        return Some(ctx.add(Expr::Div(combined_numerator, denominator)));
    }

    let scaled_coefficient = build_backend_product(ctx, coefficient, denominator);
    let combined_numerator = build_backend_sum(ctx, scaled_coefficient, numerator);
    Some(ctx.add(Expr::Div(combined_numerator, denominator)))
}

pub(super) fn normalize_backend_affine_denominator_common_factor_quotient(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let (slope, intercept) = backend_affine_linear_terms(ctx, denominator, variable)?;
    if let Some(factor) = backend_affine_denominator_numeric_factor(ctx, slope, intercept) {
        let normalized_numerator = divide_backend_expr_by_numeric_factor(ctx, numerator, &factor)?;
        let normalized_slope = divide_backend_expr_by_numeric_factor(ctx, slope, &factor)?;
        let normalized_intercept = divide_backend_expr_by_numeric_factor(ctx, intercept, &factor)?;
        let normalized_denominator =
            build_backend_affine_expr(ctx, normalized_slope, normalized_intercept, variable);
        return Some(ctx.add(Expr::Div(normalized_numerator, normalized_denominator)));
    }

    let factor = backend_affine_denominator_symbolic_factor(
        ctx,
        slope,
        intercept,
        variable,
        required_conditions,
    )?;
    let normalized_numerator = divide_backend_expr_by_symbolic_factor(ctx, numerator, factor)?;
    let normalized_slope = strip_backend_exact_factor(ctx, slope, factor, variable)?;
    let normalized_intercept = strip_backend_exact_factor(ctx, intercept, factor, variable)?;
    let normalized_denominator =
        build_backend_affine_expr(ctx, normalized_slope, normalized_intercept, variable);
    Some(ctx.add(Expr::Div(normalized_numerator, normalized_denominator)))
}

fn backend_affine_denominator_numeric_factor(
    ctx: &Context,
    slope: ExprId,
    intercept: ExprId,
) -> Option<BigRational> {
    let slope_factor = backend_numeric_coefficient(ctx, slope)?;
    let intercept_factor = backend_numeric_coefficient(ctx, intercept)?;
    if slope_factor.is_zero() || intercept_factor.is_zero() {
        return None;
    }

    let factor = intercept_factor.abs();
    (!factor.is_one()).then_some(factor)
}

fn backend_numeric_coefficient(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    numeric_value(ctx, expr)
        .or_else(|| split_backend_numeric_factor(ctx, expr).map(|(value, _)| value))
}

fn divide_backend_expr_by_numeric_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: &BigRational,
) -> Option<ExprId> {
    if factor.is_zero() {
        return None;
    }
    if let Some(value) = numeric_value(ctx, expr) {
        return Some(ctx.add(Expr::Number(value / factor.clone())));
    }
    let (coefficient, core) = split_backend_numeric_factor(ctx, expr)?;
    Some(multiply_backend_numeric_coefficient(
        ctx,
        coefficient / factor.clone(),
        core,
    ))
}

fn backend_affine_denominator_symbolic_factor(
    ctx: &mut Context,
    slope: ExprId,
    intercept: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    backend_mul_factors(ctx, intercept)
        .into_iter()
        .find(|factor| {
            numeric_value(ctx, *factor).is_none()
                && !contains_named_var(ctx, *factor, variable)
                && backend_factor_has_nonzero_evidence(ctx, *factor, variable, required_conditions)
                && strip_backend_exact_factor(ctx, slope, *factor, variable).is_some()
                && strip_backend_exact_factor(ctx, intercept, *factor, variable).is_some()
        })
}

fn divide_backend_expr_by_symbolic_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
) -> Option<ExprId> {
    if expr == factor || SemanticEqualityChecker::new(ctx).are_equal(expr, factor) {
        return Some(ctx.num(1));
    }
    if let Some(stripped) = strip_backend_exact_factor(ctx, expr, factor, "") {
        return Some(stripped);
    }
    Some(ctx.add(Expr::Div(expr, factor)))
}

fn build_backend_affine_expr(
    ctx: &mut Context,
    slope: ExprId,
    intercept: ExprId,
    variable: &str,
) -> ExprId {
    let variable_expr = ctx.var(variable);
    let variable_term = build_backend_product(ctx, slope, variable_expr);
    build_backend_sum(ctx, variable_term, intercept)
}

fn normalize_backend_affine_quotient_remainder_numerator(
    ctx: &mut Context,
    quotient_coefficient: ExprId,
    remainder: ExprId,
    denominator: ExprId,
    denominator_slope: &BackendAffineSlope,
    variable: &str,
) -> Option<ExprId> {
    let (_, denominator_intercept) = backend_affine_linear_terms(ctx, denominator, variable)?;
    let numerator_intercept = backend_affine_remainder_intercept(
        ctx,
        remainder,
        quotient_coefficient,
        denominator_intercept,
        variable,
    )?;
    let variable_coefficient =
        multiply_backend_coefficient_by_affine_slope(ctx, quotient_coefficient, denominator_slope)?;
    let variable_expr = ctx.var(variable);
    let variable_term = build_backend_product(ctx, variable_coefficient, variable_expr);
    Some(build_backend_sum(ctx, numerator_intercept, variable_term))
}

fn backend_affine_remainder_intercept(
    ctx: &mut Context,
    remainder: ExprId,
    quotient_coefficient: ExprId,
    denominator_intercept: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let scaled_denominator_intercept =
        build_backend_product(ctx, quotient_coefficient, denominator_intercept);
    let normalized_scaled_denominator_intercept = normalize_backend_symbolic_scaled_quotient(
        ctx,
        quotient_coefficient,
        denominator_intercept,
        variable,
    )
    .unwrap_or(scaled_denominator_intercept);

    if let Expr::Neg(subtrahend) = ctx.get(remainder).clone() {
        return (backend_factors_match(ctx, subtrahend, scaled_denominator_intercept)
            || backend_factors_match(ctx, subtrahend, normalized_scaled_denominator_intercept)
            || exprs_equivalent(ctx, subtrahend, scaled_denominator_intercept)
            || exprs_equivalent(ctx, subtrahend, normalized_scaled_denominator_intercept))
        .then(|| ctx.num(0));
    }

    let Expr::Sub(intercept, subtrahend) = ctx.get(remainder).clone() else {
        return None;
    };
    (backend_factors_match(ctx, subtrahend, scaled_denominator_intercept)
        || backend_factors_match(ctx, subtrahend, normalized_scaled_denominator_intercept)
        || exprs_equivalent(ctx, subtrahend, scaled_denominator_intercept)
        || exprs_equivalent(ctx, subtrahend, normalized_scaled_denominator_intercept))
    .then_some(intercept)
}

fn multiply_backend_coefficient_by_affine_slope(
    ctx: &mut Context,
    coefficient: ExprId,
    slope: &BackendAffineSlope,
) -> Option<ExprId> {
    match slope {
        BackendAffineSlope::Numeric(value) => Some(multiply_backend_numeric_coefficient(
            ctx,
            value.clone(),
            coefficient,
        )),
        BackendAffineSlope::Symbolic(slope_expr) => match ctx.get(coefficient).clone() {
            Expr::Div(numerator, denominator)
                if backend_factors_match(ctx, denominator, *slope_expr) =>
            {
                Some(numerator)
            }
            _ => Some(build_backend_product(ctx, coefficient, *slope_expr)),
        },
    }
}

pub(super) fn normalize_backend_same_denominator_sum(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    scope: BackendVerificationScope<'_>,
    fallback_reason: AlgorithmicIntegrationVerificationNormalizationReason,
) -> Option<BackendVerificationNormalization> {
    let Expr::Div(left_numerator, left_denominator) = ctx.get(left).clone() else {
        return None;
    };
    let Expr::Div(right_numerator, right_denominator) = ctx.get(right).clone() else {
        return None;
    };
    if left_denominator != right_denominator
        && !SemanticEqualityChecker::new(ctx).are_equal(left_denominator, right_denominator)
    {
        return None;
    }

    let numerator = build_backend_sum(ctx, left_numerator, right_numerator);
    if let Some(normalized_numerator) = normalize_backend_verification_expr_inner(
        ctx,
        numerator,
        scope.variable,
        scope.required_conditions,
        scope.depth + 1,
        scope.in_power_exponent,
    ) {
        return Some(BackendVerificationNormalization {
            expr: ctx.add(Expr::Div(normalized_numerator.expr, left_denominator)),
            reason:
                AlgorithmicIntegrationVerificationNormalizationReason::SameDenominatorNumeratorCancellation,
        });
    }

    Some(BackendVerificationNormalization {
        expr: ctx.add(Expr::Div(numerator, left_denominator)),
        reason: fallback_reason,
    })
}

pub(super) fn normalize_backend_conjugate_reciprocal_difference(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let (left_numerator, left_denominator) = backend_quotient_parts(ctx, left)?;
    let (right_numerator, right_denominator) = backend_quotient_parts(ctx, right)?;
    if !backend_factors_match(ctx, left_numerator, right_numerator) {
        return None;
    }
    let (_, radius) =
        conjugate_affine_pole_pair(ctx, left_denominator, right_denominator, variable)?;
    let doubled_radius =
        build_backend_numeric_product(ctx, BigRational::from_integer(2.into()), radius);
    let numerator = build_backend_product(ctx, left_numerator, doubled_radius);
    let denominator = build_backend_product(ctx, left_denominator, right_denominator);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn normalize_backend_conjugate_reciprocal_addition(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    match ctx.get(right) {
        Expr::Neg(inner) => {
            normalize_backend_conjugate_reciprocal_difference(ctx, left, *inner, variable)
        }
        _ => None,
    }
}

pub(super) fn normalize_backend_conjugate_reciprocal_expr(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            normalize_backend_conjugate_reciprocal_addition(ctx, *left, *right, variable)
        }
        Expr::Sub(left, right) => {
            normalize_backend_conjugate_reciprocal_difference(ctx, *left, *right, variable)
        }
        _ => None,
    }
}

pub(super) fn normalize_backend_scaled_conjugate_reciprocal_product(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    normalize_backend_scaled_conjugate_reciprocal_product_ordered(
        ctx,
        left,
        right,
        variable,
        required_conditions,
    )
    .or_else(|| {
        normalize_backend_scaled_conjugate_reciprocal_product_ordered(
            ctx,
            right,
            left,
            variable,
            required_conditions,
        )
    })
}

fn normalize_backend_scaled_conjugate_reciprocal_product_ordered(
    ctx: &mut Context,
    scale: ExprId,
    reciprocal_difference: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    if contains_named_var(ctx, scale, variable) {
        return None;
    }
    let combined =
        normalize_backend_conjugate_reciprocal_expr(ctx, reciprocal_difference, variable)?;
    let Expr::Div(combined_numerator, combined_denominator) = ctx.get(combined).clone() else {
        return None;
    };
    let normalized_combined = normalize_backend_conjugate_pole_product_denominator(
        ctx,
        combined_numerator,
        combined_denominator,
        variable,
    )?;
    let Expr::Div(numerator, denominator) = ctx.get(normalized_combined).clone() else {
        return None;
    };

    match ctx.get(scale).clone() {
        Expr::Div(scale_numerator, scale_denominator) => {
            let scaled_numerator = build_backend_product(ctx, scale_numerator, numerator);
            let scaled_denominator = build_backend_product(ctx, scale_denominator, denominator);
            normalize_backend_common_factor_quotient(
                ctx,
                scaled_numerator,
                scaled_denominator,
                variable,
                required_conditions,
            )
            .or_else(|| Some(ctx.add(Expr::Div(scaled_numerator, scaled_denominator))))
        }
        _ => {
            let scaled_numerator = build_backend_product(ctx, scale, numerator);
            Some(ctx.add(Expr::Div(scaled_numerator, denominator)))
        }
    }
}

pub(super) fn normalize_backend_conjugate_reciprocal_quotient(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    normalize_backend_conjugate_reciprocal_fraction_factors(
        ctx,
        backend_mul_factors(ctx, numerator),
        backend_mul_factors(ctx, denominator),
        variable,
        required_conditions,
    )
}

fn normalize_backend_conjugate_reciprocal_fraction_factors(
    ctx: &mut Context,
    mut numerator_factors: Vec<ExprId>,
    denominator_factors: Vec<ExprId>,
    variable: &str,
    required_conditions: &[ConditionPredicate],
) -> Option<ExprId> {
    let (reciprocal_index, combined) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(index, factor)| {
                normalize_backend_conjugate_reciprocal_expr(ctx, *factor, variable)
                    .map(|combined| (index, combined))
            })?;
    numerator_factors.remove(reciprocal_index);

    let Expr::Div(combined_numerator, combined_denominator) = ctx.get(combined).clone() else {
        return None;
    };
    let mut scaled_numerator_factors = backend_mul_factors(ctx, combined_numerator);
    scaled_numerator_factors.extend(numerator_factors);
    let scaled_numerator = build_backend_factor_product(ctx, scaled_numerator_factors);

    let mut scaled_denominator_factors = backend_mul_factors(ctx, combined_denominator);
    scaled_denominator_factors.extend(denominator_factors);
    let scaled_denominator = build_backend_factor_product(ctx, scaled_denominator_factors);

    let normalized_quotient = normalize_backend_common_factor_quotient(
        ctx,
        scaled_numerator,
        scaled_denominator,
        variable,
        required_conditions,
    )
    .unwrap_or_else(|| ctx.add(Expr::Div(scaled_numerator, scaled_denominator)));

    match ctx.get(normalized_quotient).clone() {
        Expr::Div(normalized_numerator, normalized_denominator) => {
            normalize_backend_conjugate_pole_product_denominator(
                ctx,
                normalized_numerator,
                normalized_denominator,
                variable,
            )
            .or(Some(normalized_quotient))
        }
        _ => Some(normalized_quotient),
    }
}

fn normalize_backend_conjugate_pole_product_denominator(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    variable: &str,
) -> Option<ExprId> {
    let factors = backend_mul_factors(ctx, denominator);
    if factors.len() != 2 {
        return None;
    }
    let (variable_expr, radius) = conjugate_affine_pole_pair(ctx, factors[0], factors[1], variable)
        .or_else(|| conjugate_affine_pole_pair(ctx, factors[1], factors[0], variable))?;
    let two = ctx.num(2);
    let variable_square = ctx.add(Expr::Pow(variable_expr, two));
    let radius_square = ctx.add(Expr::Pow(radius, two));
    let normalized_denominator = build_backend_difference(ctx, variable_square, radius_square);
    Some(ctx.add(Expr::Div(numerator, normalized_denominator)))
}

fn backend_quotient_parts(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => Some((*numerator, *denominator)),
        _ => None,
    }
}

fn conjugate_affine_pole_pair(
    ctx: &mut Context,
    minus_expr: ExprId,
    plus_expr: ExprId,
    variable: &str,
) -> Option<(ExprId, ExprId)> {
    let (variable_expr, radius) = match ctx.get(minus_expr) {
        Expr::Sub(variable_expr, radius) => (*variable_expr, *radius),
        _ => return None,
    };
    if !is_affine_variable_expr(ctx, variable_expr, variable) {
        return None;
    }
    if !is_supported_external_coefficient(ctx, radius, variable) {
        return None;
    }
    if affine_plus_radius_matches(ctx, plus_expr, variable_expr, radius) {
        Some((variable_expr, radius))
    } else {
        None
    }
}

fn is_affine_variable_expr(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    if is_variable(ctx, expr, variable) {
        return true;
    }
    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            (contains_named_var(ctx, *left, variable) && !contains_named_var(ctx, *right, variable))
                || (contains_named_var(ctx, *right, variable)
                    && !contains_named_var(ctx, *left, variable))
        }
        Expr::Mul(_, _) => contains_named_var(ctx, expr, variable),
        _ => false,
    }
}

fn affine_plus_radius_matches(
    ctx: &mut Context,
    expr: ExprId,
    variable_expr: ExprId,
    radius: ExprId,
) -> bool {
    if let Expr::Add(left, right) = ctx.get(expr) {
        if (backend_factors_match(ctx, *left, variable_expr)
            && backend_factors_match(ctx, *right, radius))
            || (backend_factors_match(ctx, *right, variable_expr)
                && backend_factors_match(ctx, *left, radius))
        {
            return true;
        }
    }

    let expected = build_backend_sum(ctx, variable_expr, radius);
    expr == expected
        || SemanticEqualityChecker::new(ctx).are_equal(expr, expected)
        || exprs_equivalent(ctx, expr, expected)
}

pub(super) fn build_backend_difference(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        negate_backend_expr(ctx, right)
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

pub(super) fn build_backend_difference_canceling_sum_term(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    if let Some(remainder) = remove_matching_backend_additive_term(ctx, left, right) {
        return remainder;
    }
    build_backend_difference(ctx, left, right)
}

fn remove_matching_backend_additive_term(
    ctx: &mut Context,
    expr: ExprId,
    target: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if backend_commutative_product_factors_match(ctx, left, target) {
                return Some(right);
            }
            if backend_commutative_product_factors_match(ctx, right, target) {
                return Some(left);
            }
            if let Some(reduced_left) = remove_matching_backend_additive_term(ctx, left, target) {
                return Some(build_backend_sum(ctx, reduced_left, right));
            }
            if let Some(reduced_right) = remove_matching_backend_additive_term(ctx, right, target) {
                return Some(build_backend_sum(ctx, left, reduced_right));
            }
            None
        }
        _ => None,
    }
}

fn backend_commutative_product_factors_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if backend_factors_match(ctx, left, right) {
        return true;
    }

    let left_factors = backend_mul_factors(ctx, left);
    let mut right_factors = backend_mul_factors(ctx, right);
    if left_factors.len() != right_factors.len() || left_factors.len() <= 1 {
        return false;
    }

    for left_factor in left_factors {
        let Some(index) = right_factors
            .iter()
            .position(|right_factor| backend_factors_match(ctx, left_factor, *right_factor))
        else {
            return false;
        };
        right_factors.remove(index);
    }
    true
}

pub(super) fn negate_backend_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    if let Some(value) = numeric_value(ctx, expr) {
        return ctx.add(Expr::Number(-value));
    }
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn build_backend_numeric_product(
    ctx: &mut Context,
    coefficient: BigRational,
    expr: ExprId,
) -> ExprId {
    if coefficient.is_zero() {
        ctx.add(Expr::Number(BigRational::zero()))
    } else if coefficient.is_one() {
        expr
    } else {
        let coefficient_expr = ctx.add(Expr::Number(coefficient));
        ctx.add(Expr::Mul(coefficient_expr, expr))
    }
}
