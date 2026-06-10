//! Result contract for backend candidates: methods, statuses, reasons, and policies.

use super::*;

use cas_ast::{ConditionPredicate, ExprId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationMethod {
    Rational,
    Hermite,
    HeurischProbe,
    TableReused,
    Unsupported,
}

impl AlgorithmicIntegrationMethod {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationMethod::Rational => "rational",
            AlgorithmicIntegrationMethod::Hermite => "hermite",
            AlgorithmicIntegrationMethod::HeurischProbe => "heurisch_probe",
            AlgorithmicIntegrationMethod::TableReused => "table_reused",
            AlgorithmicIntegrationMethod::Unsupported => "unsupported",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationStatus {
    Verified,
    VerifiedUnderConditions,
    Inconclusive,
    Failed,
    NotAttempted,
}

impl AlgorithmicIntegrationVerificationStatus {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationStatus::Verified => "verified",
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions => {
                "verified_under_conditions"
            }
            AlgorithmicIntegrationVerificationStatus::Inconclusive => "inconclusive",
            AlgorithmicIntegrationVerificationStatus::Failed => "failed",
            AlgorithmicIntegrationVerificationStatus::NotAttempted => "not_attempted",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationEvidence {
    None,
    Preverified,
    DirectDifferentiation,
    MethodSpecificDifferentiation,
    NormalizedDifferentiation,
    FailedDifferentiation,
}

impl AlgorithmicIntegrationVerificationEvidence {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationEvidence::None => "none",
            AlgorithmicIntegrationVerificationEvidence::Preverified => "preverified",
            AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation => {
                "direct_differentiation"
            }
            AlgorithmicIntegrationVerificationEvidence::MethodSpecificDifferentiation => {
                "method_specific_differentiation"
            }
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation => {
                "normalized_differentiation"
            }
            AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation => {
                "failed_differentiation"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationNormalizationReason {
    None,
    ExponentNumericSubtraction,
    NumericScaledQuotient,
    SymbolicScaledQuotient,
    ScaledArctanRadiusQuotient,
    PowerOneElision,
    QuotientNumericFactorCancellation,
    QuotientCommonFactorCancellation,
    SameDenominatorNumeratorCancellation,
    AffineQuotientRemainderSum,
    ConjugateReciprocalDifference,
    NestedQuotientDenominatorProduct,
}

impl AlgorithmicIntegrationVerificationNormalizationReason {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationNormalizationReason::None => "none",
            AlgorithmicIntegrationVerificationNormalizationReason::ExponentNumericSubtraction => {
                "exponent_numeric_subtraction"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::NumericScaledQuotient => {
                "numeric_scaled_quotient"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient => {
                "symbolic_scaled_quotient"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient => {
                "scaled_arctan_radius_quotient"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::PowerOneElision => {
                "power_one_elision"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::QuotientNumericFactorCancellation => {
                "quotient_numeric_factor_cancellation"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::QuotientCommonFactorCancellation => {
                "quotient_common_factor_cancellation"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::SameDenominatorNumeratorCancellation => {
                "same_denominator_numerator_cancellation"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::AffineQuotientRemainderSum => {
                "affine_quotient_remainder_sum"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::ConjugateReciprocalDifference => {
                "conjugate_reciprocal_difference"
            }
            AlgorithmicIntegrationVerificationNormalizationReason::NestedQuotientDenominatorProduct => {
                "nested_quotient_denominator_product"
            }
        }
    }

    pub fn is_present(&self) -> bool {
        !matches!(
            self,
            AlgorithmicIntegrationVerificationNormalizationReason::None
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationBlocker {
    None,
    MissingAntiderivative,
    DifferentiationUnavailable,
    BudgetExceeded,
    DerivativeMismatch,
}

impl AlgorithmicIntegrationVerificationBlocker {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationBlocker::None => "none",
            AlgorithmicIntegrationVerificationBlocker::MissingAntiderivative => {
                "missing_antiderivative"
            }
            AlgorithmicIntegrationVerificationBlocker::DifferentiationUnavailable => {
                "differentiation_unavailable"
            }
            AlgorithmicIntegrationVerificationBlocker::BudgetExceeded => "budget_exceeded",
            AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch => "derivative_mismatch",
        }
    }

    pub fn is_present(&self) -> bool {
        !matches!(self, AlgorithmicIntegrationVerificationBlocker::None)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationResidualKind {
    EquivalentZero,
    VariableFree,
    DependsOnVariable,
}

impl AlgorithmicIntegrationVerificationResidualKind {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationResidualKind::EquivalentZero => "equivalent_zero",
            AlgorithmicIntegrationVerificationResidualKind::VariableFree => "variable_free",
            AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable => {
                "depends_on_variable"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationResidualSignature {
    EquivalentZero,
    VariableFreeConstant,
    AffineInVariable,
    FunctionOfVariable,
    VariableDependentOther,
}

impl AlgorithmicIntegrationVerificationResidualSignature {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationVerificationResidualSignature::EquivalentZero => {
                "equivalent_zero"
            }
            AlgorithmicIntegrationVerificationResidualSignature::VariableFreeConstant => {
                "variable_free_constant"
            }
            AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable => {
                "affine_in_variable"
            }
            AlgorithmicIntegrationVerificationResidualSignature::FunctionOfVariable => {
                "function_of_variable"
            }
            AlgorithmicIntegrationVerificationResidualSignature::VariableDependentOther => {
                "variable_dependent_other"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationResidualReason {
    UnsupportedMethod,
    DomainPolicyMissing,
    BranchPolicyMissing,
    VerificationFailed,
    VerificationInconclusive,
    BudgetExceeded,
    DisabledByMode,
}

impl AlgorithmicIntegrationResidualReason {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationResidualReason::UnsupportedMethod => "unsupported_method",
            AlgorithmicIntegrationResidualReason::DomainPolicyMissing => "domain_policy_missing",
            AlgorithmicIntegrationResidualReason::BranchPolicyMissing => "branch_policy_missing",
            AlgorithmicIntegrationResidualReason::VerificationFailed => "verification_failed",
            AlgorithmicIntegrationResidualReason::VerificationInconclusive => {
                "verification_inconclusive"
            }
            AlgorithmicIntegrationResidualReason::BudgetExceeded => "budget_exceeded",
            AlgorithmicIntegrationResidualReason::DisabledByMode => "disabled_by_mode",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationFailureClass {
    UnsupportedMethod,
    DomainPolicyMissing,
    BranchPolicyMissing,
    DisabledByMode,
    AssumptionPolicyMissing,
    DiagnosticTraceOnly,
    ConstantPolicyMissing,
    MissingAntiderivative,
    DifferentiationUnavailable,
    BudgetExceeded,
    VerificationInconclusive,
    RejectedUnverified,
    DerivativeMismatchUnclassified,
    ResidualEquivalentZero,
    ResidualVariableFreeConstant,
    ResidualAffineInVariable,
    ResidualFunctionOfVariable,
    ResidualVariableDependentOther,
}

impl AlgorithmicIntegrationFailureClass {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationFailureClass::UnsupportedMethod => "unsupported_method",
            AlgorithmicIntegrationFailureClass::DomainPolicyMissing => "domain_policy_missing",
            AlgorithmicIntegrationFailureClass::BranchPolicyMissing => "branch_policy_missing",
            AlgorithmicIntegrationFailureClass::DisabledByMode => "disabled_by_mode",
            AlgorithmicIntegrationFailureClass::AssumptionPolicyMissing => {
                "assumption_policy_missing"
            }
            AlgorithmicIntegrationFailureClass::DiagnosticTraceOnly => "diagnostic_trace_only",
            AlgorithmicIntegrationFailureClass::ConstantPolicyMissing => "constant_policy_missing",
            AlgorithmicIntegrationFailureClass::MissingAntiderivative => "missing_antiderivative",
            AlgorithmicIntegrationFailureClass::DifferentiationUnavailable => {
                "differentiation_unavailable"
            }
            AlgorithmicIntegrationFailureClass::BudgetExceeded => "budget_exceeded",
            AlgorithmicIntegrationFailureClass::VerificationInconclusive => {
                "verification_inconclusive"
            }
            AlgorithmicIntegrationFailureClass::RejectedUnverified => "rejected_unverified",
            AlgorithmicIntegrationFailureClass::DerivativeMismatchUnclassified => {
                "derivative_mismatch_unclassified"
            }
            AlgorithmicIntegrationFailureClass::ResidualEquivalentZero => {
                "residual_equivalent_zero"
            }
            AlgorithmicIntegrationFailureClass::ResidualVariableFreeConstant => {
                "residual_variable_free_constant"
            }
            AlgorithmicIntegrationFailureClass::ResidualAffineInVariable => {
                "residual_affine_in_variable"
            }
            AlgorithmicIntegrationFailureClass::ResidualFunctionOfVariable => {
                "residual_function_of_variable"
            }
            AlgorithmicIntegrationFailureClass::ResidualVariableDependentOther => {
                "residual_variable_dependent_other"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationTraceLevel {
    EducationalFull,
    AlgorithmicSummary,
    DiagnosticOnly,
}

impl AlgorithmicIntegrationTraceLevel {
    pub fn metric_label(&self) -> &'static str {
        match self {
            AlgorithmicIntegrationTraceLevel::EducationalFull => "educational_full",
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary => "algorithmic_summary",
            AlgorithmicIntegrationTraceLevel::DiagnosticOnly => "diagnostic_only",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntegrationConstantPolicy {
    Unspecified,
    ArbitraryConstantOmitted,
    ComponentLocalConstant,
    ConditionalConstant,
}

impl IntegrationConstantPolicy {
    pub fn metric_label(&self) -> &'static str {
        match self {
            IntegrationConstantPolicy::Unspecified => "unspecified",
            IntegrationConstantPolicy::ArbitraryConstantOmitted => "arbitrary_constant_omitted",
            IntegrationConstantPolicy::ComponentLocalConstant => "component_local_constant",
            IntegrationConstantPolicy::ConditionalConstant => "conditional_constant",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationDomainPolicy {
    Unspecified,
    UnconditionalReal,
    RealWithRequiredConditions,
    DomainPolicyMissing,
    BranchPolicyMissing,
    AssumptionPolicyMissing,
}

impl AlgorithmicIntegrationDomainPolicy {
    pub fn metric_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationDomainPolicy::Unspecified => "unspecified",
            AlgorithmicIntegrationDomainPolicy::UnconditionalReal => "unconditional_real",
            AlgorithmicIntegrationDomainPolicy::RealWithRequiredConditions => {
                "real_with_required_conditions"
            }
            AlgorithmicIntegrationDomainPolicy::DomainPolicyMissing => "domain_policy_missing",
            AlgorithmicIntegrationDomainPolicy::BranchPolicyMissing => "branch_policy_missing",
            AlgorithmicIntegrationDomainPolicy::AssumptionPolicyMissing => {
                "assumption_policy_missing"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationPublicationStatus {
    Accepted,
    RejectedNoAntiderivative,
    RejectedResidualReason,
    RejectedAssumptions,
    RejectedDiagnosticTrace,
    RejectedUnspecifiedConstant,
    RejectedUnverified,
}

impl AlgorithmicIntegrationPublicationStatus {
    pub fn metric_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationPublicationStatus::Accepted => "accepted",
            AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative => {
                "rejected_no_antiderivative"
            }
            AlgorithmicIntegrationPublicationStatus::RejectedResidualReason => {
                "rejected_residual_reason"
            }
            AlgorithmicIntegrationPublicationStatus::RejectedAssumptions => "rejected_assumptions",
            AlgorithmicIntegrationPublicationStatus::RejectedDiagnosticTrace => {
                "rejected_diagnostic_trace"
            }
            AlgorithmicIntegrationPublicationStatus::RejectedUnspecifiedConstant => {
                "rejected_unspecified_constant"
            }
            AlgorithmicIntegrationPublicationStatus::RejectedUnverified => "rejected_unverified",
        }
    }

    pub fn is_accepted(self) -> bool {
        matches!(self, AlgorithmicIntegrationPublicationStatus::Accepted)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationFallbackStatus {
    Eligible,
    BlockedByCandidatePolicy,
    BlockedByMode,
}

impl AlgorithmicIntegrationFallbackStatus {
    pub fn metric_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationFallbackStatus::Eligible => "eligible",
            AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy => {
                "blocked_by_candidate_policy"
            }
            AlgorithmicIntegrationFallbackStatus::BlockedByMode => "blocked_by_mode",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationVerificationOutcome {
    Verified {
        derivative: ExprId,
        evidence: AlgorithmicIntegrationVerificationEvidence,
    },
    Failed {
        derivative: ExprId,
    },
    Inconclusive {
        reason: AlgorithmicIntegrationResidualReason,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmicIntegrationVerificationReport {
    pub status: AlgorithmicIntegrationVerificationStatus,
    pub evidence: AlgorithmicIntegrationVerificationEvidence,
    pub normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason,
    pub verification_normalization_passes_used: usize,
    pub blocker: AlgorithmicIntegrationVerificationBlocker,
    pub residual_reason: Option<AlgorithmicIntegrationResidualReason>,
    pub derivative: Option<ExprId>,
    pub verification_residual: Option<ExprId>,
    pub verification_residual_kind: Option<AlgorithmicIntegrationVerificationResidualKind>,
    pub verification_residual_signature:
        Option<AlgorithmicIntegrationVerificationResidualSignature>,
}

impl AlgorithmicIntegrationVerificationReport {
    pub fn apply_to_candidate(&self, candidate: &mut AlgorithmicIntegrationCandidate) {
        candidate.verification_status = self.status.clone();
        candidate.verification_evidence = self.evidence.clone();
        candidate.verification_normalization_reason = self.normalization_reason.clone();
        candidate.verification_normalization_passes_used =
            self.verification_normalization_passes_used;
        candidate.verification_blocker = self.blocker.clone();
        candidate.residual_reason = self.residual_reason.clone();
        candidate.verification_residual = self.verification_residual;
        candidate.verification_residual_kind = self.verification_residual_kind.clone();
        candidate.verification_residual_signature = self.verification_residual_signature.clone();
    }

    pub(super) fn into_outcome(self) -> AlgorithmicIntegrationVerificationOutcome {
        match self.status {
            AlgorithmicIntegrationVerificationStatus::Verified
            | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions => {
                AlgorithmicIntegrationVerificationOutcome::Verified {
                    derivative: self
                        .derivative
                        .expect("verified integration report includes derivative"),
                    evidence: self.evidence,
                }
            }
            AlgorithmicIntegrationVerificationStatus::Failed => {
                AlgorithmicIntegrationVerificationOutcome::Failed {
                    derivative: self
                        .derivative
                        .expect("failed integration report includes derivative"),
                }
            }
            AlgorithmicIntegrationVerificationStatus::Inconclusive
            | AlgorithmicIntegrationVerificationStatus::NotAttempted => {
                AlgorithmicIntegrationVerificationOutcome::Inconclusive {
                    reason: self
                        .residual_reason
                        .unwrap_or(AlgorithmicIntegrationResidualReason::VerificationInconclusive),
                }
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmicIntegrationCandidate {
    pub integrand: ExprId,
    pub variable: String,
    pub antiderivative: Option<ExprId>,
    pub method: AlgorithmicIntegrationMethod,
    pub assumptions: Vec<ExprId>,
    pub required_conditions: Vec<ConditionPredicate>,
    pub constant_policy: IntegrationConstantPolicy,
    pub verification_status: AlgorithmicIntegrationVerificationStatus,
    pub verification_evidence: AlgorithmicIntegrationVerificationEvidence,
    pub verification_normalization_reason: AlgorithmicIntegrationVerificationNormalizationReason,
    pub verification_normalization_passes_used: usize,
    pub verification_blocker: AlgorithmicIntegrationVerificationBlocker,
    pub residual_reason: Option<AlgorithmicIntegrationResidualReason>,
    pub verification_residual: Option<ExprId>,
    pub verification_residual_kind: Option<AlgorithmicIntegrationVerificationResidualKind>,
    pub verification_residual_signature:
        Option<AlgorithmicIntegrationVerificationResidualSignature>,
    pub trace_level: AlgorithmicIntegrationTraceLevel,
    pub method_probe_attempts: Vec<AlgorithmicIntegrationMethod>,
    pub method_probe_no_match_reasons: Vec<(
        AlgorithmicIntegrationMethod,
        AlgorithmicIntegrationProbeNoMatchReason,
    )>,
    pub method_probe_budget_limit: usize,
    pub verification_check_budget_limit: usize,
    pub method_probes_used: usize,
    pub verification_checks_used: usize,
}

impl AlgorithmicIntegrationCandidate {
    pub fn disabled(integrand: ExprId, variable: impl Into<String>) -> Self {
        Self::without_antiderivative(
            integrand,
            variable,
            AlgorithmicIntegrationMethod::Unsupported,
            AlgorithmicIntegrationVerificationStatus::NotAttempted,
            AlgorithmicIntegrationResidualReason::DisabledByMode,
        )
    }

    pub fn unsupported(integrand: ExprId, variable: impl Into<String>) -> Self {
        Self::without_antiderivative(
            integrand,
            variable,
            AlgorithmicIntegrationMethod::Unsupported,
            AlgorithmicIntegrationVerificationStatus::NotAttempted,
            AlgorithmicIntegrationResidualReason::UnsupportedMethod,
        )
    }

    pub fn budget_exceeded(integrand: ExprId, variable: impl Into<String>) -> Self {
        Self::without_antiderivative(
            integrand,
            variable,
            AlgorithmicIntegrationMethod::Unsupported,
            AlgorithmicIntegrationVerificationStatus::Inconclusive,
            AlgorithmicIntegrationResidualReason::BudgetExceeded,
        )
    }

    pub fn verified(
        integrand: ExprId,
        variable: impl Into<String>,
        antiderivative: ExprId,
        method: AlgorithmicIntegrationMethod,
    ) -> Self {
        Self {
            integrand,
            variable: variable.into(),
            antiderivative: Some(antiderivative),
            method,
            assumptions: Vec::new(),
            required_conditions: Vec::new(),
            constant_policy: IntegrationConstantPolicy::ArbitraryConstantOmitted,
            verification_status: AlgorithmicIntegrationVerificationStatus::Verified,
            verification_evidence: AlgorithmicIntegrationVerificationEvidence::Preverified,
            verification_normalization_reason:
                AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            verification_blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: None,
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
            trace_level: AlgorithmicIntegrationTraceLevel::AlgorithmicSummary,
            method_probe_attempts: Vec::new(),
            method_probe_no_match_reasons: Vec::new(),
            method_probe_budget_limit: 0,
            verification_check_budget_limit: 0,
            method_probes_used: 0,
            verification_checks_used: 0,
        }
    }

    pub fn verified_table_reused(
        integrand: ExprId,
        variable: impl Into<String>,
        antiderivative: ExprId,
    ) -> Self {
        Self::verified(
            integrand,
            variable,
            antiderivative,
            AlgorithmicIntegrationMethod::TableReused,
        )
    }

    pub fn unverified(
        integrand: ExprId,
        variable: impl Into<String>,
        antiderivative: ExprId,
        method: AlgorithmicIntegrationMethod,
    ) -> Self {
        Self {
            integrand,
            variable: variable.into(),
            antiderivative: Some(antiderivative),
            method,
            assumptions: Vec::new(),
            required_conditions: Vec::new(),
            constant_policy: IntegrationConstantPolicy::ArbitraryConstantOmitted,
            verification_status: AlgorithmicIntegrationVerificationStatus::NotAttempted,
            verification_evidence: AlgorithmicIntegrationVerificationEvidence::None,
            verification_normalization_reason:
                AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            verification_blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: Some(AlgorithmicIntegrationResidualReason::VerificationInconclusive),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
            trace_level: AlgorithmicIntegrationTraceLevel::AlgorithmicSummary,
            method_probe_attempts: Vec::new(),
            method_probe_no_match_reasons: Vec::new(),
            method_probe_budget_limit: 0,
            verification_check_budget_limit: 0,
            method_probes_used: 0,
            verification_checks_used: 0,
        }
    }

    pub fn unverified_table_reused(
        integrand: ExprId,
        variable: impl Into<String>,
        antiderivative: ExprId,
    ) -> Self {
        Self::unverified(
            integrand,
            variable,
            antiderivative,
            AlgorithmicIntegrationMethod::TableReused,
        )
    }

    pub fn publication_status(&self) -> AlgorithmicIntegrationPublicationStatus {
        if self.antiderivative.is_none() {
            return AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative;
        }
        if self.residual_reason.is_some() {
            return AlgorithmicIntegrationPublicationStatus::RejectedResidualReason;
        }
        if !self.assumptions.is_empty() {
            return AlgorithmicIntegrationPublicationStatus::RejectedAssumptions;
        }
        if matches!(
            self.trace_level,
            AlgorithmicIntegrationTraceLevel::DiagnosticOnly
        ) {
            return AlgorithmicIntegrationPublicationStatus::RejectedDiagnosticTrace;
        }
        if matches!(self.constant_policy, IntegrationConstantPolicy::Unspecified) {
            return AlgorithmicIntegrationPublicationStatus::RejectedUnspecifiedConstant;
        }
        if !matches!(
            self.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
                | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        ) {
            return AlgorithmicIntegrationPublicationStatus::RejectedUnverified;
        }

        AlgorithmicIntegrationPublicationStatus::Accepted
    }

    pub fn is_publicly_acceptable(&self) -> bool {
        self.publication_status().is_accepted()
    }

    pub fn domain_policy(&self) -> AlgorithmicIntegrationDomainPolicy {
        if self.antiderivative.is_none() {
            return AlgorithmicIntegrationDomainPolicy::Unspecified;
        }
        if matches!(
            self.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::DomainPolicyMissing)
        ) {
            return AlgorithmicIntegrationDomainPolicy::DomainPolicyMissing;
        }
        if matches!(
            self.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::BranchPolicyMissing)
        ) {
            return AlgorithmicIntegrationDomainPolicy::BranchPolicyMissing;
        }
        if !self.assumptions.is_empty() {
            return AlgorithmicIntegrationDomainPolicy::AssumptionPolicyMissing;
        }
        if self.required_conditions.is_empty() {
            AlgorithmicIntegrationDomainPolicy::UnconditionalReal
        } else {
            AlgorithmicIntegrationDomainPolicy::RealWithRequiredConditions
        }
    }

    pub fn public_antiderivative(&self) -> Option<ExprId> {
        self.is_publicly_acceptable()
            .then_some(self.antiderivative?)
    }

    pub fn failure_class(&self) -> Option<AlgorithmicIntegrationFailureClass> {
        if self.is_publicly_acceptable() {
            return None;
        }

        if let Some(signature) = &self.verification_residual_signature {
            return Some(match signature {
                AlgorithmicIntegrationVerificationResidualSignature::EquivalentZero => {
                    AlgorithmicIntegrationFailureClass::ResidualEquivalentZero
                }
                AlgorithmicIntegrationVerificationResidualSignature::VariableFreeConstant => {
                    AlgorithmicIntegrationFailureClass::ResidualVariableFreeConstant
                }
                AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable => {
                    AlgorithmicIntegrationFailureClass::ResidualAffineInVariable
                }
                AlgorithmicIntegrationVerificationResidualSignature::FunctionOfVariable => {
                    AlgorithmicIntegrationFailureClass::ResidualFunctionOfVariable
                }
                AlgorithmicIntegrationVerificationResidualSignature::VariableDependentOther => {
                    AlgorithmicIntegrationFailureClass::ResidualVariableDependentOther
                }
            });
        }

        match self.verification_blocker {
            AlgorithmicIntegrationVerificationBlocker::None => {}
            AlgorithmicIntegrationVerificationBlocker::MissingAntiderivative => {
                return Some(AlgorithmicIntegrationFailureClass::MissingAntiderivative);
            }
            AlgorithmicIntegrationVerificationBlocker::DifferentiationUnavailable => {
                return Some(AlgorithmicIntegrationFailureClass::DifferentiationUnavailable);
            }
            AlgorithmicIntegrationVerificationBlocker::BudgetExceeded => {
                return Some(AlgorithmicIntegrationFailureClass::BudgetExceeded);
            }
            AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch => {
                return Some(AlgorithmicIntegrationFailureClass::DerivativeMismatchUnclassified);
            }
        }

        if let Some(reason) = &self.residual_reason {
            return Some(match reason {
                AlgorithmicIntegrationResidualReason::UnsupportedMethod => {
                    AlgorithmicIntegrationFailureClass::UnsupportedMethod
                }
                AlgorithmicIntegrationResidualReason::DomainPolicyMissing => {
                    AlgorithmicIntegrationFailureClass::DomainPolicyMissing
                }
                AlgorithmicIntegrationResidualReason::BranchPolicyMissing => {
                    AlgorithmicIntegrationFailureClass::BranchPolicyMissing
                }
                AlgorithmicIntegrationResidualReason::VerificationFailed => {
                    AlgorithmicIntegrationFailureClass::DerivativeMismatchUnclassified
                }
                AlgorithmicIntegrationResidualReason::VerificationInconclusive => {
                    AlgorithmicIntegrationFailureClass::VerificationInconclusive
                }
                AlgorithmicIntegrationResidualReason::BudgetExceeded => {
                    AlgorithmicIntegrationFailureClass::BudgetExceeded
                }
                AlgorithmicIntegrationResidualReason::DisabledByMode => {
                    AlgorithmicIntegrationFailureClass::DisabledByMode
                }
            });
        }

        Some(match self.publication_status() {
            AlgorithmicIntegrationPublicationStatus::Accepted => return None,
            AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative => {
                AlgorithmicIntegrationFailureClass::MissingAntiderivative
            }
            AlgorithmicIntegrationPublicationStatus::RejectedResidualReason => {
                AlgorithmicIntegrationFailureClass::VerificationInconclusive
            }
            AlgorithmicIntegrationPublicationStatus::RejectedAssumptions => {
                AlgorithmicIntegrationFailureClass::AssumptionPolicyMissing
            }
            AlgorithmicIntegrationPublicationStatus::RejectedDiagnosticTrace => {
                AlgorithmicIntegrationFailureClass::DiagnosticTraceOnly
            }
            AlgorithmicIntegrationPublicationStatus::RejectedUnspecifiedConstant => {
                AlgorithmicIntegrationFailureClass::ConstantPolicyMissing
            }
            AlgorithmicIntegrationPublicationStatus::RejectedUnverified => {
                AlgorithmicIntegrationFailureClass::RejectedUnverified
            }
        })
    }

    pub fn fallback_status(
        &self,
        config: AlgorithmicIntegrationBackendConfig,
    ) -> AlgorithmicIntegrationFallbackStatus {
        if !self.publication_status().is_accepted() {
            return AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy;
        }
        if !config.mode.permits_public_fallback() {
            return AlgorithmicIntegrationFallbackStatus::BlockedByMode;
        }

        AlgorithmicIntegrationFallbackStatus::Eligible
    }

    pub fn fallback_antiderivative(
        &self,
        config: AlgorithmicIntegrationBackendConfig,
    ) -> Option<ExprId> {
        matches!(
            self.fallback_status(config),
            AlgorithmicIntegrationFallbackStatus::Eligible
        )
        .then_some(self.antiderivative?)
    }

    fn without_antiderivative(
        integrand: ExprId,
        variable: impl Into<String>,
        method: AlgorithmicIntegrationMethod,
        verification_status: AlgorithmicIntegrationVerificationStatus,
        residual_reason: AlgorithmicIntegrationResidualReason,
    ) -> Self {
        Self {
            integrand,
            variable: variable.into(),
            antiderivative: None,
            method,
            assumptions: Vec::new(),
            required_conditions: Vec::new(),
            constant_policy: IntegrationConstantPolicy::Unspecified,
            verification_status,
            verification_evidence: AlgorithmicIntegrationVerificationEvidence::None,
            verification_normalization_reason:
                AlgorithmicIntegrationVerificationNormalizationReason::None,
            verification_normalization_passes_used: 0,
            verification_blocker: AlgorithmicIntegrationVerificationBlocker::None,
            residual_reason: Some(residual_reason),
            verification_residual: None,
            verification_residual_kind: None,
            verification_residual_signature: None,
            trace_level: AlgorithmicIntegrationTraceLevel::DiagnosticOnly,
            method_probe_attempts: Vec::new(),
            method_probe_no_match_reasons: Vec::new(),
            method_probe_budget_limit: 0,
            verification_check_budget_limit: 0,
            method_probes_used: 0,
            verification_checks_used: 0,
        }
    }

    pub(super) fn mark_budget_exceeded(&mut self) {
        self.verification_status = AlgorithmicIntegrationVerificationStatus::Inconclusive;
        self.verification_evidence = AlgorithmicIntegrationVerificationEvidence::None;
        self.verification_normalization_reason =
            AlgorithmicIntegrationVerificationNormalizationReason::None;
        self.verification_normalization_passes_used = 0;
        self.verification_blocker = AlgorithmicIntegrationVerificationBlocker::BudgetExceeded;
        self.residual_reason = Some(AlgorithmicIntegrationResidualReason::BudgetExceeded);
        self.verification_residual = None;
        self.verification_residual_kind = None;
        self.verification_residual_signature = None;
    }

    pub(super) fn record_probe_usage(&mut self, probe_runner: &AlgorithmicIntegrationProbeRunner) {
        self.method_probe_attempts = probe_runner.method_probe_attempts().to_vec();
        self.method_probe_no_match_reasons = probe_runner.method_probe_no_match_reasons().to_vec();
        self.method_probe_budget_limit = probe_runner.method_probe_budget_limit();
        self.verification_check_budget_limit = probe_runner.verification_check_budget_limit();
        self.method_probes_used = probe_runner.method_probes_used();
        self.verification_checks_used = probe_runner.verification_checks_used();
    }
}
