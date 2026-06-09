//! Boundary types for the general algorithmic integration backend.
//!
//! This module is intentionally behavior-light: it defines the result contract
//! that a broader integration backend must satisfy before any candidate can be
//! consumed by public integration routes.

use crate::expr_domain::exprs_equivalent;
use crate::expr_predicates::contains_named_var;
use crate::semantic_equality::SemanticEqualityChecker;
use crate::symbolic_differentiation_support::differentiate_symbolic_expr;
use cas_ast::{BuiltinFn, ConditionPredicate, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

const BACKEND_VERIFICATION_NORMALIZE_DEPTH: usize = 32;
const BACKEND_VERIFICATION_NORMALIZE_PASSES: usize = 4;
const BACKEND_RESIDUAL_SIGNATURE_DEPTH: usize = 48;
const BACKEND_EXTERNAL_COEFFICIENT_DEPTH: usize = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AlgorithmicIntegrationBackendMode {
    #[default]
    Disabled,
    DiagnosticOnly,
    ResidualFallback,
}

impl AlgorithmicIntegrationBackendMode {
    pub fn metric_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationBackendMode::Disabled => "disabled",
            AlgorithmicIntegrationBackendMode::DiagnosticOnly => "diagnostic_only",
            AlgorithmicIntegrationBackendMode::ResidualFallback => "residual_fallback",
        }
    }

    pub fn attempts_backend(self) -> bool {
        !matches!(self, AlgorithmicIntegrationBackendMode::Disabled)
    }

    pub fn permits_public_fallback(self) -> bool {
        matches!(self, AlgorithmicIntegrationBackendMode::ResidualFallback)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlgorithmicIntegrationBackendBudget {
    pub max_method_probes: usize,
    pub max_verification_checks: usize,
}

impl AlgorithmicIntegrationBackendBudget {
    pub const fn new(max_method_probes: usize, max_verification_checks: usize) -> Self {
        Self {
            max_method_probes,
            max_verification_checks,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            max_method_probes: 0,
            max_verification_checks: 0,
        }
    }

    pub const fn single_probe() -> Self {
        Self {
            max_method_probes: 1,
            max_verification_checks: 1,
        }
    }

    pub const fn single_probe_without_verification() -> Self {
        Self {
            max_method_probes: 1,
            max_verification_checks: 0,
        }
    }

    pub const fn two_probes() -> Self {
        Self {
            max_method_probes: 2,
            max_verification_checks: 1,
        }
    }

    pub const fn three_probes() -> Self {
        Self {
            max_method_probes: 3,
            max_verification_checks: 1,
        }
    }

    pub fn allows_method_probe(self) -> bool {
        self.max_method_probes > 0
    }

    pub fn allows_verification_check(self) -> bool {
        self.max_verification_checks > 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlgorithmicIntegrationBackendConfig {
    pub mode: AlgorithmicIntegrationBackendMode,
    pub budget: AlgorithmicIntegrationBackendBudget,
}

impl AlgorithmicIntegrationBackendConfig {
    pub fn disabled() -> Self {
        Self {
            mode: AlgorithmicIntegrationBackendMode::Disabled,
            budget: AlgorithmicIntegrationBackendBudget::disabled(),
        }
    }

    pub fn diagnostic_only() -> Self {
        Self {
            mode: AlgorithmicIntegrationBackendMode::DiagnosticOnly,
            budget: AlgorithmicIntegrationBackendBudget::three_probes(),
        }
    }

    pub fn residual_fallback() -> Self {
        Self {
            mode: AlgorithmicIntegrationBackendMode::ResidualFallback,
            budget: AlgorithmicIntegrationBackendBudget::three_probes(),
        }
    }

    pub fn with_budget(mut self, budget: AlgorithmicIntegrationBackendBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl Default for AlgorithmicIntegrationBackendConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AlgorithmicIntegrationProbeRunner {
    remaining_method_probes: usize,
    remaining_verification_checks: usize,
    method_probe_attempts: Vec<AlgorithmicIntegrationMethod>,
    method_probe_no_match_reasons: Vec<(
        AlgorithmicIntegrationMethod,
        AlgorithmicIntegrationProbeNoMatchReason,
    )>,
    method_probes_used: usize,
    verification_checks_used: usize,
    method_budget_exhausted: bool,
    verification_budget_exhausted: bool,
}

impl AlgorithmicIntegrationProbeRunner {
    pub fn new(budget: AlgorithmicIntegrationBackendBudget) -> Self {
        Self {
            remaining_method_probes: budget.max_method_probes,
            remaining_verification_checks: budget.max_verification_checks,
            method_probe_attempts: Vec::new(),
            method_probe_no_match_reasons: Vec::new(),
            method_probes_used: 0,
            verification_checks_used: 0,
            method_budget_exhausted: false,
            verification_budget_exhausted: false,
        }
    }

    pub fn remaining_method_probes(&self) -> usize {
        self.remaining_method_probes
    }

    pub fn remaining_verification_checks(&self) -> usize {
        self.remaining_verification_checks
    }

    pub fn method_probes_used(&self) -> usize {
        self.method_probes_used
    }

    pub fn verification_checks_used(&self) -> usize {
        self.verification_checks_used
    }

    pub fn method_probe_attempts(&self) -> &[AlgorithmicIntegrationMethod] {
        &self.method_probe_attempts
    }

    pub fn method_probe_no_match_reasons(
        &self,
    ) -> &[(
        AlgorithmicIntegrationMethod,
        AlgorithmicIntegrationProbeNoMatchReason,
    )] {
        &self.method_probe_no_match_reasons
    }

    pub fn method_budget_exhausted(&self) -> bool {
        self.method_budget_exhausted
    }

    pub fn verification_budget_exhausted(&self) -> bool {
        self.verification_budget_exhausted
    }

    pub fn try_method_probe<F>(
        &mut self,
        method: AlgorithmicIntegrationMethod,
        probe: F,
    ) -> Option<AlgorithmicIntegrationCandidate>
    where
        F: FnOnce(&mut Self) -> AlgorithmicIntegrationProbeResult,
    {
        if self.remaining_method_probes == 0 {
            self.method_budget_exhausted = true;
            return None;
        }

        self.method_probe_attempts.push(method);
        self.remaining_method_probes -= 1;
        self.method_probes_used += 1;
        match probe(self) {
            AlgorithmicIntegrationProbeResult::Candidate(candidate) => Some(candidate),
            AlgorithmicIntegrationProbeResult::NoMatch(reason) => {
                self.method_probe_no_match_reasons.push((method, reason));
                None
            }
        }
    }

    pub fn try_verification_check(&mut self) -> bool {
        if self.remaining_verification_checks == 0 {
            self.verification_budget_exhausted = true;
            return false;
        }

        self.remaining_verification_checks -= 1;
        self.verification_checks_used += 1;
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationProbeResult {
    Candidate(AlgorithmicIntegrationCandidate),
    NoMatch(AlgorithmicIntegrationProbeNoMatchReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmicIntegrationProbeNoMatchReason {
    ShapeMismatch,
    DenominatorPolicyMismatch,
    NumeratorPolicyMismatch,
    NumeratorDerivativeMismatch,
    RadiusPolicyMismatch,
}

impl AlgorithmicIntegrationProbeNoMatchReason {
    pub fn metric_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch => "shape_mismatch",
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch => {
                "denominator_policy_mismatch"
            }
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch => {
                "numerator_policy_mismatch"
            }
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorDerivativeMismatch => {
                "numerator_derivative_mismatch"
            }
            AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch => {
                "radius_policy_mismatch"
            }
        }
    }
}

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

    fn into_outcome(self) -> AlgorithmicIntegrationVerificationOutcome {
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
            method_probes_used: 0,
            verification_checks_used: 0,
        }
    }

    fn mark_budget_exceeded(&mut self) {
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

    fn record_probe_usage(&mut self, probe_runner: &AlgorithmicIntegrationProbeRunner) {
        self.method_probe_attempts = probe_runner.method_probe_attempts().to_vec();
        self.method_probe_no_match_reasons = probe_runner.method_probe_no_match_reasons().to_vec();
        self.method_probes_used = probe_runner.method_probes_used();
        self.verification_checks_used = probe_runner.verification_checks_used();
    }
}

pub fn verify_antiderivative_by_differentiation(
    ctx: &mut Context,
    candidate: &mut AlgorithmicIntegrationCandidate,
) -> AlgorithmicIntegrationVerificationOutcome {
    let report = antiderivative_verification_report(ctx, candidate);
    report.apply_to_candidate(candidate);
    report.into_outcome()
}

pub fn antiderivative_verification_report(
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct BackendVerificationNormalizationMatchAttempt {
    matched_reason: Option<AlgorithmicIntegrationVerificationNormalizationReason>,
    passes_used: usize,
}

fn normalize_backend_verification_expr_to_match(
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct BackendVerificationNormalization {
    expr: ExprId,
    reason: AlgorithmicIntegrationVerificationNormalizationReason,
}

#[derive(Clone, Copy, Debug)]
struct BackendVerificationScope<'a> {
    variable: &'a str,
    required_conditions: &'a [ConditionPredicate],
    depth: usize,
    in_power_exponent: bool,
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
enum BackendAffineSlope {
    Numeric(BigRational),
    Symbolic(ExprId),
}

impl BackendAffineSlope {
    fn required_condition(&self) -> Option<ConditionPredicate> {
        match self {
            BackendAffineSlope::Numeric(_) => None,
            BackendAffineSlope::Symbolic(expr) => Some(ConditionPredicate::NonZero(*expr)),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BackendRadiusSquareValue {
    Numeric(BigRational),
    ConditionalSymbolic(ExprId),
}

impl BackendRadiusSquareValue {
    fn expr(&self, ctx: &mut Context) -> ExprId {
        match self {
            BackendRadiusSquareValue::Numeric(value) => ctx.add(Expr::Number(value.clone())),
            BackendRadiusSquareValue::ConditionalSymbolic(expr) => *expr,
        }
    }
}

fn normalize_backend_verification_expr(
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
            let normalized =
                normalize_backend_verification_binary(ctx, left, right, scope, Expr::Add)?;
            if let Expr::Add(normalized_left, normalized_right) = ctx.get(normalized.expr).clone() {
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

            if is_one(ctx, normalized_exponent) {
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

fn normalize_backend_numeric_scaled_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    normalize_backend_numeric_scaled_quotient_ordered(ctx, left, right)
        .or_else(|| normalize_backend_numeric_scaled_quotient_ordered(ctx, right, left))
}

fn normalize_backend_symbolic_scaled_quotient(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    variable: &str,
) -> Option<ExprId> {
    normalize_backend_symbolic_scaled_quotient_ordered(ctx, left, right, variable)
        .or_else(|| normalize_backend_symbolic_scaled_quotient_ordered(ctx, right, left, variable))
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

fn normalize_backend_common_factor_quotient(
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

fn normalize_backend_scaled_arctan_radius_quotient(
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

fn build_backend_factor_product(ctx: &mut Context, factors: Vec<ExprId>) -> ExprId {
    factors.into_iter().fold(ctx.num(1), |product, factor| {
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

fn backend_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
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

fn normalize_backend_quotient_numeric_factor_cancellation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    normalize_backend_quotient_numeric_factor_cancellation_ordered(ctx, left, right).or_else(|| {
        normalize_backend_quotient_numeric_factor_cancellation_ordered(ctx, right, left)
    })
}

fn normalize_backend_quotient_symbolic_factor_cancellation(
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

fn strip_backend_exact_factor(
    ctx: &mut Context,
    expr: ExprId,
    factor: ExprId,
    variable: &str,
) -> Option<ExprId> {
    if expr == factor || SemanticEqualityChecker::new(ctx).are_equal(expr, factor) {
        return Some(ctx.num(1));
    }

    match ctx.get(expr).clone() {
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

fn multiply_backend_numeric_coefficient(
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

fn build_backend_product(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_one(ctx, left) {
        right
    } else if is_one(ctx, right) {
        left
    } else {
        ctx.add(Expr::Mul(left, right))
    }
}

fn build_backend_sum(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, left) {
        right
    } else if is_zero(ctx, right) {
        left
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

fn normalize_backend_same_denominator_sum(
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

fn build_backend_difference(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_zero(ctx, right) {
        left
    } else if is_zero(ctx, left) {
        negate_backend_expr(ctx, right)
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

fn negate_backend_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
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

fn try_rational_reciprocal_affine_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
    let (numerator, denominator, denominator_slope) =
        match scaled_affine_reciprocal_parts(ctx, integrand, variable) {
            Ok(parts) => parts,
            Err(reason) => return AlgorithmicIntegrationProbeResult::NoMatch(reason),
        };

    let abs_denominator = ctx.call_builtin(BuiltinFn::Abs, vec![denominator]);
    let log_denominator = ctx.call_builtin(BuiltinFn::Ln, vec![abs_denominator]);
    let antiderivative_scale =
        divide_backend_coefficient_by_slope(ctx, numerator, &denominator_slope);
    let antiderivative = if is_one(ctx, antiderivative_scale) {
        log_denominator
    } else {
        build_backend_product(ctx, antiderivative_scale, log_denominator)
    };
    let mut candidate = AlgorithmicIntegrationCandidate::unverified(
        integrand,
        variable,
        antiderivative,
        AlgorithmicIntegrationMethod::Rational,
    );
    candidate
        .required_conditions
        .push(ConditionPredicate::NonZero(denominator));
    if let Some(condition) = denominator_slope.required_condition() {
        candidate.required_conditions.push(condition);
    }
    if !probe_runner.try_verification_check() {
        candidate.mark_budget_exceeded();
        return AlgorithmicIntegrationProbeResult::Candidate(candidate);
    }
    verify_antiderivative_by_differentiation(ctx, &mut candidate);

    AlgorithmicIntegrationProbeResult::Candidate(candidate)
}

fn try_hermite_positive_quadratic_log_derivative_probe(
    ctx: &mut Context,
    integrand: ExprId,
    variable: &str,
    probe_runner: &mut AlgorithmicIntegrationProbeRunner,
) -> AlgorithmicIntegrationProbeResult {
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
    if let Some(condition) = radius_condition {
        candidate.required_conditions.push(condition);
    }
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

fn scaled_affine_reciprocal_parts(
    ctx: &mut Context,
    expr: ExprId,
    variable: &str,
) -> Result<(ExprId, ExprId, BackendAffineSlope), AlgorithmicIntegrationProbeNoMatchReason> {
    match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => {
            if !is_supported_scaled_affine_reciprocal_numerator(ctx, numerator, variable) {
                return Err(AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch);
            }
            let Some(denominator_slope) = affine_denominator_slope(ctx, denominator, variable)
            else {
                return Err(AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch);
            };
            Ok((numerator, denominator, denominator_slope))
        }
        _ => Err(AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch),
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
    is_zero(ctx, expr)
        || numeric_value(ctx, expr)
            .map(|value| !value.is_zero())
            .unwrap_or(false)
        || is_supported_external_coefficient(ctx, expr, variable)
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
            let (variable_expr, variable_slope, _, required_condition) =
                positive_shifted_quadratic_denominator_parts(ctx, denominator, variable)?;
            let coefficient =
                affine_variable_coefficient_expr(ctx, numerator, variable_expr, variable)?;
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
            if !is_supported_backend_linear_coefficient(ctx, variable_coefficient, variable)
                || !is_supported_backend_linear_coefficient(ctx, constant_term, variable)
            {
                return None;
            }
            let radius = positive_radius_expr(ctx, radius_square, &required_condition)?;
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

fn positive_shifted_quadratic_denominator_parts(
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
            None
        }
        _ => None,
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

fn affine_variable_expr(
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

fn linear_numerator_decomposition_terms(
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

    match ctx.get(expr).clone() {
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
    }
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
    ctx.add(Expr::Div(coefficient, divisor))
}

fn is_symbolic_external_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) != variable)
        && !contains_named_var(ctx, expr, variable)
}

fn is_supported_external_coefficient(ctx: &Context, expr: ExprId, variable: &str) -> bool {
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

fn affine_denominator_slope(
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

fn is_variable(ctx: &Context, expr: ExprId, variable: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == variable)
}

fn is_numeric_constant(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_))
}

fn numeric_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(value) => Some(value.clone()),
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

fn positive_radius_expr(
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

fn positive_radius_square_value(
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

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_one())
}

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.is_zero())
}

fn is_two(ctx: &Context, expr: ExprId) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Instant;

    #[test]
    fn default_config_disables_backend() {
        assert_eq!(
            AlgorithmicIntegrationBackendConfig::default(),
            AlgorithmicIntegrationBackendConfig::disabled()
        );
    }

    #[test]
    fn probe_runner_consumes_method_and_verification_budget() {
        let mut runner = AlgorithmicIntegrationProbeRunner::new(
            AlgorithmicIntegrationBackendBudget::single_probe(),
        );
        let mut first_probe_ran = false;

        assert!(runner
            .try_method_probe(AlgorithmicIntegrationMethod::Rational, |probe_runner| {
                first_probe_ran = true;
                assert_eq!(probe_runner.remaining_method_probes(), 0);
                assert!(probe_runner.try_verification_check());
                AlgorithmicIntegrationProbeResult::NoMatch(
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                )
            },)
            .is_none());

        assert!(first_probe_ran);
        assert_eq!(runner.remaining_method_probes(), 0);
        assert_eq!(runner.remaining_verification_checks(), 0);
        assert_eq!(
            runner.method_probe_attempts(),
            &[AlgorithmicIntegrationMethod::Rational]
        );
        assert_eq!(
            runner.method_probe_no_match_reasons(),
            &[(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
            )]
        );
        assert_eq!(runner.method_probes_used(), 1);
        assert_eq!(runner.verification_checks_used(), 1);
        assert!(!runner.method_budget_exhausted());
        assert!(!runner.verification_budget_exhausted());

        let mut second_probe_ran = false;
        assert!(runner
            .try_method_probe(AlgorithmicIntegrationMethod::Hermite, |_| {
                second_probe_ran = true;
                AlgorithmicIntegrationProbeResult::NoMatch(
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                )
            },)
            .is_none());

        assert!(!second_probe_ran);
        assert_eq!(
            runner.method_probe_attempts(),
            &[AlgorithmicIntegrationMethod::Rational]
        );
        assert_eq!(
            runner.method_probe_no_match_reasons(),
            &[(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch
            )]
        );
        assert_eq!(runner.method_probes_used(), 1);
        assert_eq!(runner.verification_checks_used(), 1);
        assert!(runner.method_budget_exhausted());
    }

    #[test]
    fn disabled_backend_candidate_is_diagnostic_only() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::default(),
        );

        assert_eq!(candidate.integrand, integrand);
        assert_eq!(candidate.variable, "x");
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::NotAttempted
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::DisabledByMode)
        );
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::DiagnosticOnly
        );
        assert_eq!(candidate.method_probes_used, 0);
        assert_eq!(candidate.verification_checks_used, 0);
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(
            candidate.publication_status(),
            AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative
        );
        assert_eq!(
            candidate.fallback_status(AlgorithmicIntegrationBackendConfig::default()),
            AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
        );
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn diagnostic_backend_respects_method_probe_budget() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);
        let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::disabled());

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Inconclusive
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
        );
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(
            candidate.publication_status(),
            AlgorithmicIntegrationPublicationStatus::RejectedNoAntiderivative
        );
        assert_eq!(
            candidate.fallback_status(config),
            AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
        );
        assert_eq!(candidate.public_antiderivative(), None);
        assert_eq!(candidate.fallback_antiderivative(config), None);
    }

    #[test]
    fn diagnostic_empty_backend_reports_unsupported_without_public_candidate() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn diagnostic_rational_probe_verifies_but_is_not_fallback_consumable() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected reciprocal integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.antiderivative.is_some());
        assert!(candidate.is_publicly_acceptable());
        assert_eq!(
            candidate.publication_status(),
            AlgorithmicIntegrationPublicationStatus::Accepted
        );
        assert_eq!(
            candidate.fallback_status(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
            AlgorithmicIntegrationFallbackStatus::BlockedByMode
        );
        assert!(candidate.public_antiderivative().is_some());
        assert_eq!(
            candidate
                .fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
            None
        );
    }

    #[test]
    fn residual_fallback_mode_can_consume_verified_conditioned_candidate() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
        let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.publication_status(),
            AlgorithmicIntegrationPublicationStatus::Accepted
        );
        assert_eq!(
            candidate.fallback_status(config),
            AlgorithmicIntegrationFallbackStatus::Eligible
        );
        assert!(candidate.fallback_antiderivative(config).is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_numeric_scaled_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2/(x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected scaled reciprocal integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_symbolic_scaled_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("a/(x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected symbolic scaled reciprocal integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_product_symbolic_scaled_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*a/(x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected product symbolic scaled reciprocal integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_numeric_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(2*x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected numeric slope reciprocal integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_negative_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(1-2*x)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_symbolic_scaled_numeric_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("a/(2*x+1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_symbolic_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(a*x+b)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected symbolic slope reciprocal integrand"),
        };
        let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
            .expect("symbolic affine slope")
        {
            BackendAffineSlope::Symbolic(slope) => slope,
            BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![
                ConditionPredicate::NonZero(denominator),
                ConditionPredicate::NonZero(slope),
            ]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_product_symbolic_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(2*a*x+b)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected product symbolic slope reciprocal integrand"),
        };
        let slope = match affine_denominator_slope(&mut ctx, denominator, "x")
            .expect("product symbolic affine slope")
        {
            BackendAffineSlope::Symbolic(slope) => slope,
            BackendAffineSlope::Numeric(_) => panic!("expected symbolic slope"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient
        );
        assert_eq!(
            candidate.required_conditions,
            vec![
                ConditionPredicate::NonZero(denominator),
                ConditionPredicate::NonZero(slope),
            ]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_external_scaled_symbolic_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("c/(a*x+b)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 2);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_external_scaled_product_symbolic_slope_affine_reciprocal()
    {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("c/(2*a*x+b)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 2);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_verifies_negative_symbolic_slope_affine_reciprocal() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(b-a*x)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 2);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_rational_probe_rejects_variable_dependent_reciprocal_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x/(x+1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(
            candidate.method_probe_no_match_reasons.first(),
            Some(&(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            ))
        );
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn diagnostic_rational_probe_rejects_variable_dependent_product_slope() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(2*x*x+b)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(
            candidate.method_probe_no_match_reasons.first(),
            Some(&(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            ))
        );
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn diagnostic_rational_probe_respects_verification_budget() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x+1)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected reciprocal integrand"),
        };
        let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::single_probe_without_verification());

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Rational);
        assert!(candidate.antiderivative.is_some());
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Inconclusive
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
        );
        assert_eq!(
            candidate.verification_blocker,
            AlgorithmicIntegrationVerificationBlocker::BudgetExceeded
        );
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 0);
        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(
            candidate.publication_status(),
            AlgorithmicIntegrationPublicationStatus::RejectedResidualReason
        );
        assert_eq!(
            candidate.fallback_status(config),
            AlgorithmicIntegrationFallbackStatus::BlockedByCandidatePolicy
        );
        assert_eq!(candidate.public_antiderivative(), None);
        assert_eq!(candidate.fallback_antiderivative(config), None);
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_after_rational_probe_miss() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::PowerOneElision
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
        assert_eq!(
            candidate
                .fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
            None
        );
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_positive_numeric_quadratic_shift() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_scaled_positive_quadratic_log_derivative() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("3*x/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_external_scale_log_derivative() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("a*x/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_accepts_symbolic_scale_after_variable() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x*a/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_unit_positive_quadratic_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(x+1)/(x^2+1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_scaled_unit_positive_quadratic_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(2*x+3)/(x^2+1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_exact_square_radius_positive_quadratic_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(x+1)/(x^2+4)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_scaled_exact_square_radius_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(2*x+3)/(x^2+4)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_mixed_coefficients_with_verifier_policy() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(a*x+b)/(x^2+4)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_constant_over_exact_square_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+4)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_scaled_constant_over_exact_square_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("3/(x^2+4)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_constant_over_nonexact_numeric_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_scaled_constant_over_nonexact_numeric_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("3/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_nonexact_radius_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(x+1)/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.required_conditions.is_empty());
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_log_derivative() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_constant_over_symbolic_positive_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_constant_over_symbolic_positive_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("c/(x^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(x+1)/(x^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::NumericScaledQuotient
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_positive_shift_external_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(b*x+c)/(x^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_shifted_affine_symbolic_constant_over_positive_radius() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("c/((x+b)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_shifted_affine_external_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(m*(x+b)+c)/((x+b)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_numeric_slope_shifted_affine_constant_over_positive_radius(
    ) {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/((2*x+b)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_negative_numeric_slope_shifted_affine_constant() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/((b-2*x)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(candidate.residual_reason, None);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_numeric_slope_shifted_affine_external_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand =
            cas_parser::parse("(m*(2*x+b)+c)/((2*x+b)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::Positive(radius_square)]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            )]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_slope_shifted_affine_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("integrand");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            candidate.verification_normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::SymbolicScaledQuotient
        );
        assert_eq!(
            candidate.required_conditions,
            vec![
                ConditionPredicate::Positive(radius_square),
                ConditionPredicate::NonZero(slope),
            ]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            ),]
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
    }

    #[test]
    fn backend_verifier_cancels_generated_symbolic_slope_quotient_factors() {
        let mut ctx = Context::new();
        let quotient =
            cas_parser::parse("((s*x+b)*m*s*2)/(s*2)", &mut ctx).expect("generated quotient");
        let expected = cas_parser::parse("m*(s*x+b)", &mut ctx).expect("expected quotient");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");
        let Expr::Div(numerator, denominator) = ctx.get(quotient).clone() else {
            panic!("expected quotient");
        };

        let normalized = normalize_backend_common_factor_quotient(
            &mut ctx,
            numerator,
            denominator,
            "x",
            &[ConditionPredicate::NonZero(slope)],
        )
        .expect("common factor cancellation");

        assert!(
            normalized == expected
                || SemanticEqualityChecker::new(&ctx).are_equal(normalized, expected),
            "expected {}, got {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expected
            },
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: normalized
            }
        );
    }

    #[test]
    fn backend_verifier_normalizes_generated_symbolic_slope_recombined_numerator() {
        let mut ctx = Context::new();
        let numerator =
            cas_parser::parse("((s*x+b)*m*s*2)/(s*2)+c", &mut ctx).expect("generated numerator");
        let expected = cas_parser::parse("m*(s*x+b)+c", &mut ctx).expect("expected numerator");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");

        let normalized = normalize_backend_verification_expr(
            &mut ctx,
            numerator,
            "x",
            &[ConditionPredicate::NonZero(slope)],
        )
        .expect("recombined numerator normalization");

        assert!(
            normalized.expr == expected
                || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
            "expected {}, got {}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expected
            },
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: normalized.expr
            }
        );
    }

    #[test]
    fn backend_verifier_normalizes_generated_symbolic_slope_derivative() {
        let mut ctx = Context::new();
        let derivative = cas_parser::parse(
            "(m*s*(s*x+b)^(2-1)*2)/(s*2*((s*x+b)^2+a)) + (c*s)/(sqrt(a)*s*(((s*x+b)/sqrt(a))^2+1)*sqrt(a))",
            &mut ctx,
        )
        .expect("generated derivative");
        let expected = cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx)
            .expect("expected derivative");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");

        let normalized = normalize_backend_verification_expr(
            &mut ctx,
            derivative,
            "x",
            &[
                ConditionPredicate::Positive(radius_square),
                ConditionPredicate::NonZero(slope),
            ],
        )
        .expect("generated derivative normalization");

        assert!(
            normalized.expr == expected
                || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
            "expected {}, got {} [{:?}]",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expected
            },
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: normalized.expr
            },
            normalized.reason
        );
    }

    #[test]
    fn backend_verifier_normalizes_generated_symbolic_slope_intermediate_quotient() {
        let mut ctx = Context::new();
        let intermediate = cas_parser::parse("(((s*x+b)*m*s*2)/(s*2)+c)/((s*x+b)^2+a)", &mut ctx)
            .expect("intermediate quotient");
        let expected =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("expected quotient");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");

        let normalized = normalize_backend_verification_expr(
            &mut ctx,
            intermediate,
            "x",
            &[
                ConditionPredicate::Positive(radius_square),
                ConditionPredicate::NonZero(slope),
            ],
        )
        .expect("intermediate quotient normalization");

        assert!(
            normalized.expr == expected
                || SemanticEqualityChecker::new(&ctx).are_equal(normalized.expr, expected),
            "expected {}, got {} [{:?}]",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: expected
            },
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: normalized.expr
            },
            normalized.reason
        );
    }

    #[test]
    fn backend_verifier_fixed_point_normalizes_generated_symbolic_slope_derivative_ast() {
        let mut ctx = Context::new();
        let integrand =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse(
            "(c*arctan((s*x+b)/sqrt(a)))/(sqrt(a)*s) + (m*ln((s*x+b)^2+a))/(s*2)",
            &mut ctx,
        )
        .expect("antiderivative");
        let radius_square = cas_parser::parse("a", &mut ctx).expect("radius square");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");
        let derivative =
            differentiate_symbolic_expr(&mut ctx, antiderivative, "x").expect("derivative");

        let matched = normalize_backend_verification_expr_to_match(
            &mut ctx,
            derivative,
            integrand,
            "x",
            &[
                ConditionPredicate::Positive(radius_square),
                ConditionPredicate::NonZero(slope),
            ],
        );

        assert!(matched.matched_reason.is_some());
        assert!(matched.passes_used > 1);
    }

    #[test]
    fn diagnostic_hermite_probe_rejects_variable_dependent_affine_square_slope() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/((x*x+b)^2+a)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![
                (
                    AlgorithmicIntegrationMethod::Rational,
                    AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::Hermite,
                    AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::HeurischProbe,
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                ),
            ]
        );
        assert!(!candidate.is_publicly_acceptable());
    }

    #[test]
    fn verification_report_requires_positive_condition_for_symbolic_radius_square() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+a)", &mut ctx).expect("integrand");
        let generated_candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let antiderivative = generated_candidate
            .antiderivative
            .expect("generated antiderivative");
        let mut unconditioned_candidate = AlgorithmicIntegrationCandidate::unverified(
            integrand,
            "x",
            antiderivative,
            AlgorithmicIntegrationMethod::Hermite,
        );

        let unconditioned_report =
            antiderivative_verification_report(&mut ctx, &unconditioned_candidate);

        assert_ne!(
            unconditioned_report.status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_ne!(
            unconditioned_report.status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );

        unconditioned_candidate.required_conditions =
            generated_candidate.required_conditions.clone();
        let conditioned_report =
            antiderivative_verification_report(&mut ctx, &unconditioned_candidate);

        assert_eq!(
            conditioned_report.status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            conditioned_report.evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(
            conditioned_report.normalization_reason,
            AlgorithmicIntegrationVerificationNormalizationReason::ScaledArctanRadiusQuotient
        );
        assert_eq!(conditioned_report.residual_reason, None);
    }

    #[test]
    fn diagnostic_hermite_probe_rejects_variable_dependent_scale() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x*x/(x^2+2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert_eq!(candidate.method_probes_used, 3);
        assert_eq!(candidate.verification_checks_used, 0);
        assert!(!candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_rejects_indefinite_quadratic_denominator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2-1)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::NotAttempted
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![
                (
                    AlgorithmicIntegrationMethod::Rational,
                    AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::Hermite,
                    AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::HeurischProbe,
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                ),
            ]
        );
        assert!(!candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_square_radius_log_derivative() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2+a^2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert!(candidate.antiderivative.is_some());
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert!(matches!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
                | AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        ));
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 1);
        assert!(matches!(
            candidate.required_conditions[0],
            ConditionPredicate::NonZero(_)
        ));
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            ),]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_square_radius_arctan_branch() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+a^2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert!(candidate.antiderivative.is_some());
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert!(matches!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
                | AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        ));
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 1);
        assert!(matches!(
            candidate.required_conditions[0],
            ConditionPredicate::NonZero(_)
        ));
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
            ),]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_square_radius_mixed_numerator() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("(x+1)/(x^2+a^2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.required_conditions.len(), 1);
        assert!(matches!(
            candidate.required_conditions[0],
            ConditionPredicate::NonZero(_)
        ));
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_verifies_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator(
    ) {
        let mut ctx = Context::new();
        let integrand =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a^2)", &mut ctx).expect("integrand");
        let radius = cas_parser::parse("a", &mut ctx).expect("radius");
        let slope = cas_parser::parse("s", &mut ctx).expect("slope");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Hermite);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::NormalizedDifferentiation
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(
            candidate.required_conditions,
            vec![
                ConditionPredicate::NonZero(radius),
                ConditionPredicate::NonZero(slope),
            ]
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![(
                AlgorithmicIntegrationMethod::Rational,
                AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch,
            ),]
        );
        assert_eq!(candidate.method_probes_used, 2);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_keeps_indefinite_symbolic_square_denominator_residual() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2-a^2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![
                (
                    AlgorithmicIntegrationMethod::Rational,
                    AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::Hermite,
                    AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::HeurischProbe,
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                ),
            ]
        );
        assert_eq!(candidate.method_probes_used, 3);
        assert_eq!(candidate.verification_checks_used, 0);
        assert!(!candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_keeps_symbolic_slope_indefinite_square_denominator_residual() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/((s*x+b)^2-a^2)", &mut ctx).expect("integrand");

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::UnsupportedMethod)
        );
        assert_eq!(
            candidate.method_probe_no_match_reasons,
            vec![
                (
                    AlgorithmicIntegrationMethod::Rational,
                    AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::Hermite,
                    AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch,
                ),
                (
                    AlgorithmicIntegrationMethod::HeurischProbe,
                    AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch,
                ),
            ]
        );
        assert_eq!(candidate.method_probes_used, 3);
        assert_eq!(candidate.verification_checks_used, 0);
        assert!(!candidate.is_publicly_acceptable());
    }

    #[test]
    fn diagnostic_hermite_probe_requires_second_method_probe_budget() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("integrand");
        let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::single_probe());

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Inconclusive
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::None
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
        );
        assert_eq!(candidate.method_probes_used, 1);
        assert_eq!(candidate.verification_checks_used, 0);
        assert_eq!(candidate.public_antiderivative(), None);
        assert_eq!(candidate.fallback_antiderivative(config), None);
    }

    #[test]
    fn diagnostic_heurisch_probe_verifies_after_two_probe_misses() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
        let denominator = match ctx.get(integrand) {
            Expr::Div(_, denominator) => *denominator,
            _ => panic!("expected quotient integrand"),
        };

        let candidate = try_algorithmic_integration_backend(
            &mut ctx,
            integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );

        assert_eq!(
            candidate.method,
            AlgorithmicIntegrationMethod::HeurischProbe
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(
            candidate.verification_blocker,
            AlgorithmicIntegrationVerificationBlocker::None
        );
        assert_eq!(
            candidate.required_conditions,
            vec![ConditionPredicate::NonZero(denominator)]
        );
        assert_eq!(
            candidate.trace_level,
            AlgorithmicIntegrationTraceLevel::AlgorithmicSummary
        );
        assert_eq!(candidate.method_probes_used, 3);
        assert_eq!(candidate.verification_checks_used, 1);
        assert!(candidate.public_antiderivative().is_some());
        assert_eq!(
            candidate
                .fallback_antiderivative(AlgorithmicIntegrationBackendConfig::diagnostic_only()),
            None
        );
    }

    #[test]
    fn diagnostic_heurisch_probe_respects_method_probe_budget() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
        let config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::single_probe());

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(candidate.method, AlgorithmicIntegrationMethod::Unsupported);
        assert_eq!(candidate.antiderivative, None);
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Inconclusive
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::BudgetExceeded)
        );
        assert_eq!(candidate.public_antiderivative(), None);
        assert_eq!(candidate.fallback_antiderivative(config), None);
    }

    #[test]
    fn residual_fallback_mode_can_consume_verified_heurisch_candidate() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("integrand");
        let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

        let candidate = try_algorithmic_integration_backend(&mut ctx, integrand, "x", config);

        assert_eq!(
            candidate.method,
            AlgorithmicIntegrationMethod::HeurischProbe
        );
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
        );
        assert!(candidate.fallback_antiderivative(config).is_some());
    }

    #[test]
    fn public_acceptance_requires_verified_antiderivative_without_residual() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);
        let antiderivative = ctx.var("x");

        let candidate =
            AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);

        assert!(candidate.is_publicly_acceptable());
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::Preverified
        );
        assert_eq!(candidate.public_antiderivative(), Some(antiderivative));
    }

    #[test]
    fn public_acceptance_requires_consumable_trace_and_constant_policy() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);
        let antiderivative = ctx.var("x");
        let config = AlgorithmicIntegrationBackendConfig::residual_fallback();

        let mut diagnostic_trace_candidate =
            AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
        diagnostic_trace_candidate.trace_level = AlgorithmicIntegrationTraceLevel::DiagnosticOnly;

        assert!(!diagnostic_trace_candidate.is_publicly_acceptable());
        assert_eq!(diagnostic_trace_candidate.public_antiderivative(), None);
        assert_eq!(
            diagnostic_trace_candidate.fallback_antiderivative(config),
            None
        );

        let mut unspecified_constant_candidate =
            AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
        unspecified_constant_candidate.constant_policy = IntegrationConstantPolicy::Unspecified;

        assert!(!unspecified_constant_candidate.is_publicly_acceptable());
        assert_eq!(unspecified_constant_candidate.public_antiderivative(), None);
        assert_eq!(
            unspecified_constant_candidate.fallback_antiderivative(config),
            None
        );

        let mut raw_assumption_candidate =
            AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
        raw_assumption_candidate.assumptions.push(ctx.var("a"));

        assert!(!raw_assumption_candidate.is_publicly_acceptable());
        assert_eq!(raw_assumption_candidate.public_antiderivative(), None);
        assert_eq!(
            raw_assumption_candidate.fallback_antiderivative(config),
            None
        );
    }

    #[test]
    fn residual_blocks_public_candidate_even_when_antiderivative_exists() {
        let mut ctx = Context::new();
        let integrand = ctx.num(1);
        let antiderivative = ctx.var("x");
        let mut candidate =
            AlgorithmicIntegrationCandidate::verified_table_reused(integrand, "x", antiderivative);
        candidate.residual_reason = Some(AlgorithmicIntegrationResidualReason::DomainPolicyMissing);

        assert!(!candidate.is_publicly_acceptable());
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn unverified_candidate_is_blocked_until_direct_diff_verification() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
        let mut candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        assert!(!candidate.is_publicly_acceptable());

        let outcome = verify_antiderivative_by_differentiation(&mut ctx, &mut candidate);

        assert!(matches!(
            outcome,
            AlgorithmicIntegrationVerificationOutcome::Verified { .. }
        ));
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
        assert_eq!(candidate.public_antiderivative(), Some(antiderivative));
    }

    #[test]
    fn verification_report_does_not_mutate_candidate_until_applied() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
        let mut candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        let report = antiderivative_verification_report(&mut ctx, &candidate);

        assert_eq!(
            report.status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(
            report.evidence,
            AlgorithmicIntegrationVerificationEvidence::DirectDifferentiation
        );
        assert_eq!(
            report.blocker,
            AlgorithmicIntegrationVerificationBlocker::None
        );
        assert_eq!(report.residual_reason, None);
        assert!(report.derivative.is_some());
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::NotAttempted
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::VerificationInconclusive)
        );

        report.apply_to_candidate(&mut candidate);

        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Verified
        );
        assert_eq!(candidate.residual_reason, None);
    }

    #[test]
    fn direct_diff_verification_rejects_mismatched_candidate() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
        let mut candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        let outcome = verify_antiderivative_by_differentiation(&mut ctx, &mut candidate);

        assert!(matches!(
            outcome,
            AlgorithmicIntegrationVerificationOutcome::Failed { .. }
        ));
        assert_eq!(
            candidate.verification_status,
            AlgorithmicIntegrationVerificationStatus::Failed
        );
        assert_eq!(
            candidate.verification_evidence,
            AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation
        );
        assert_eq!(
            candidate.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::VerificationFailed)
        );
        assert!(candidate.verification_residual.is_some());
        assert_eq!(
            candidate.verification_residual_kind,
            Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
        );
        assert_eq!(
            candidate.verification_residual_signature,
            Some(AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable)
        );
        assert_eq!(
            candidate.verification_blocker,
            AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch
        );
        assert_eq!(
            candidate.failure_class(),
            Some(AlgorithmicIntegrationFailureClass::ResidualAffineInVariable)
        );
        assert_eq!(candidate.public_antiderivative(), None);
    }

    #[test]
    fn failure_class_covers_publication_policy_rejections() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("x^2/2", &mut ctx).expect("antiderivative");
        let assumption = cas_parser::parse("a > 0", &mut ctx).unwrap_or(antiderivative);
        let mut candidate = AlgorithmicIntegrationCandidate::verified(
            integrand,
            "x",
            antiderivative,
            AlgorithmicIntegrationMethod::Rational,
        );

        candidate.assumptions.push(assumption);
        assert_eq!(
            candidate.failure_class(),
            Some(AlgorithmicIntegrationFailureClass::AssumptionPolicyMissing)
        );

        candidate.assumptions.clear();
        candidate.trace_level = AlgorithmicIntegrationTraceLevel::DiagnosticOnly;
        assert_eq!(
            candidate.failure_class(),
            Some(AlgorithmicIntegrationFailureClass::DiagnosticTraceOnly)
        );

        candidate.trace_level = AlgorithmicIntegrationTraceLevel::AlgorithmicSummary;
        candidate.constant_policy = IntegrationConstantPolicy::Unspecified;
        assert_eq!(
            candidate.failure_class(),
            Some(AlgorithmicIntegrationFailureClass::ConstantPolicyMissing)
        );

        candidate.constant_policy = IntegrationConstantPolicy::ArbitraryConstantOmitted;
        candidate.verification_status = AlgorithmicIntegrationVerificationStatus::NotAttempted;
        assert_eq!(
            candidate.failure_class(),
            Some(AlgorithmicIntegrationFailureClass::RejectedUnverified)
        );
    }

    #[test]
    fn verification_report_preserves_derivative_mismatch_blocker() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("x", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
        let candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        let report = antiderivative_verification_report(&mut ctx, &candidate);

        assert_eq!(
            report.status,
            AlgorithmicIntegrationVerificationStatus::Failed
        );
        assert_eq!(
            report.evidence,
            AlgorithmicIntegrationVerificationEvidence::FailedDifferentiation
        );
        assert_eq!(
            report.blocker,
            AlgorithmicIntegrationVerificationBlocker::DerivativeMismatch
        );
        assert_eq!(
            report.residual_reason,
            Some(AlgorithmicIntegrationResidualReason::VerificationFailed)
        );
        assert!(report.derivative.is_some());
        assert!(report.verification_residual.is_some());
        assert_eq!(
            report.verification_residual_kind,
            Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
        );
        assert_eq!(
            report.verification_residual_signature,
            Some(AlgorithmicIntegrationVerificationResidualSignature::AffineInVariable)
        );
    }

    #[test]
    fn verification_report_classifies_variable_free_residual_mismatch() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("0", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
        let candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        let report = antiderivative_verification_report(&mut ctx, &candidate);

        assert_eq!(
            report.status,
            AlgorithmicIntegrationVerificationStatus::Failed
        );
        assert_eq!(
            report.verification_residual_kind,
            Some(AlgorithmicIntegrationVerificationResidualKind::VariableFree)
        );
        assert_eq!(
            report.verification_residual_signature,
            Some(AlgorithmicIntegrationVerificationResidualSignature::VariableFreeConstant)
        );
    }

    #[test]
    fn verification_report_classifies_function_residual_signature() {
        let mut ctx = Context::new();
        let integrand = cas_parser::parse("sin(x)", &mut ctx).expect("integrand");
        let antiderivative = cas_parser::parse("x", &mut ctx).expect("antiderivative");
        let candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            integrand,
            "x",
            antiderivative,
        );

        let report = antiderivative_verification_report(&mut ctx, &candidate);

        assert_eq!(
            report.status,
            AlgorithmicIntegrationVerificationStatus::Failed
        );
        assert_eq!(
            report.verification_residual_kind,
            Some(AlgorithmicIntegrationVerificationResidualKind::DependsOnVariable)
        );
        assert_eq!(
            report.verification_residual_signature,
            Some(AlgorithmicIntegrationVerificationResidualSignature::FunctionOfVariable)
        );
    }

    #[test]
    fn backend_observability_reports_boundary_metrics() {
        let mut ctx = Context::new();
        let disabled_integrand = ctx.num(1);
        let disabled = try_algorithmic_integration_backend(
            &mut ctx,
            disabled_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::default(),
        );
        let unsupported_integrand = ctx.num(1);
        let unsupported = try_algorithmic_integration_backend(
            &mut ctx,
            unsupported_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let method_limited_integrand = ctx.num(1);
        let method_limited_config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::disabled());
        let method_limited = try_algorithmic_integration_backend(
            &mut ctx,
            method_limited_integrand,
            "x",
            method_limited_config,
        );
        let rational_integrand =
            cas_parser::parse("1/(x+1)", &mut ctx).expect("rational integrand");
        let diagnostic_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let scaled_rational_integrand =
            cas_parser::parse("2/(x+1)", &mut ctx).expect("scaled rational integrand");
        let diagnostic_scaled_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            scaled_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let symbolic_scaled_rational_integrand =
            cas_parser::parse("a/(x+1)", &mut ctx).expect("symbolic scaled rational integrand");
        let diagnostic_symbolic_scaled_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_scaled_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let numeric_slope_rational_integrand =
            cas_parser::parse("1/(2*x+1)", &mut ctx).expect("numeric slope rational integrand");
        let diagnostic_numeric_slope_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            numeric_slope_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let symbolic_slope_rational_integrand =
            cas_parser::parse("1/(a*x+b)", &mut ctx).expect("symbolic slope rational integrand");
        let diagnostic_symbolic_slope_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_slope_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let product_symbolic_slope_rational_integrand = cas_parser::parse("1/(2*a*x+b)", &mut ctx)
            .expect("product symbolic slope rational integrand");
        let diagnostic_product_symbolic_slope_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            product_symbolic_slope_rational_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let hermite_integrand =
            cas_parser::parse("2*x/(x^2+1)", &mut ctx).expect("hermite integrand");
        let diagnostic_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let shifted_hermite_integrand =
            cas_parser::parse("2*x/(x^2+2)", &mut ctx).expect("shifted hermite integrand");
        let diagnostic_shifted_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            shifted_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let scaled_hermite_integrand =
            cas_parser::parse("3*x/(x^2+2)", &mut ctx).expect("scaled hermite integrand");
        let diagnostic_scaled_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            scaled_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let symbolic_scaled_hermite_integrand =
            cas_parser::parse("a*x/(x^2+2)", &mut ctx).expect("symbolic scaled hermite integrand");
        let diagnostic_symbolic_scaled_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_scaled_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let unit_mixed_numerator_hermite_integrand =
            cas_parser::parse("(x+1)/(x^2+1)", &mut ctx).expect("unit mixed numerator hermite");
        let diagnostic_unit_mixed_numerator_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            unit_mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let square_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(x+1)/(x^2+4)", &mut ctx)
                .expect("square radius mixed numerator hermite");
        let diagnostic_square_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                square_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let square_radius_constant_hermite_integrand =
            cas_parser::parse("1/(x^2+4)", &mut ctx).expect("square radius constant hermite");
        let diagnostic_square_radius_constant_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            square_radius_constant_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let mixed_numerator_hermite_integrand =
            cas_parser::parse("(x+1)/(x^2+2)", &mut ctx).expect("mixed numerator hermite gap");
        let diagnostic_mixed_numerator_hermite_probe = try_algorithmic_integration_backend(
            &mut ctx,
            mixed_numerator_hermite_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let symbolic_positive_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(x+1)/(x^2+a)", &mut ctx)
                .expect("symbolic positive radius mixed numerator hermite");
        let diagnostic_symbolic_positive_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_positive_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(b*x+c)/(x^2+a)", &mut ctx)
                .expect("symbolic external positive radius mixed numerator hermite");
        let diagnostic_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(m*(x+b)+c)/((x+b)^2+a)", &mut ctx)
                .expect("shifted affine symbolic external positive radius mixed numerator hermite");
        let diagnostic_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(m*(2*x+b)+c)/((2*x+b)^2+a)", &mut ctx)
                .expect("numeric slope shifted affine symbolic external positive radius mixed numerator hermite");
        let diagnostic_numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx)
                .expect("symbolic slope shifted affine symbolic external positive radius mixed numerator hermite");
        let diagnostic_symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let symbolic_square_radius_log_derivative_integrand =
            cas_parser::parse("2*x/(x^2+a^2)", &mut ctx)
                .expect("symbolic square radius log derivative");
        let diagnostic_symbolic_square_radius_log_derivative_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_square_radius_log_derivative_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let symbolic_square_radius_arctan_integrand =
            cas_parser::parse("1/(x^2+a^2)", &mut ctx).expect("symbolic square radius arctan");
        let diagnostic_symbolic_square_radius_arctan_probe = try_algorithmic_integration_backend(
            &mut ctx,
            symbolic_square_radius_arctan_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_integrand =
            cas_parser::parse("(m*(s*x+b)+c)/((s*x+b)^2+a^2)", &mut ctx)
                .expect("symbolic slope shifted symbolic square radius mixed numerator");
        let diagnostic_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let indefinite_symbolic_square_denominator_policy_gap_integrand =
            cas_parser::parse("1/(x^2-a^2)", &mut ctx)
                .expect("indefinite symbolic square denominator policy gap");
        let diagnostic_indefinite_symbolic_square_denominator_policy_gap_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                indefinite_symbolic_square_denominator_policy_gap_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let symbolic_slope_indefinite_square_denominator_policy_gap_integrand =
            cas_parser::parse("1/((s*x+b)^2-a^2)", &mut ctx)
                .expect("symbolic slope indefinite square denominator policy gap");
        let diagnostic_symbolic_slope_indefinite_square_denominator_policy_gap_probe =
            try_algorithmic_integration_backend(
                &mut ctx,
                symbolic_slope_indefinite_square_denominator_policy_gap_integrand,
                "x",
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
            );
        let heurisch_integrand =
            cas_parser::parse("cos(x)/sin(x)", &mut ctx).expect("heurisch integrand");
        let diagnostic_heurisch_probe = try_algorithmic_integration_backend(
            &mut ctx,
            heurisch_integrand,
            "x",
            AlgorithmicIntegrationBackendConfig::diagnostic_only(),
        );
        let fallback_rational_integrand =
            cas_parser::parse("1/(x+1)", &mut ctx).expect("fallback rational integrand");
        let fallback_config = AlgorithmicIntegrationBackendConfig::residual_fallback();
        let fallback_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            fallback_rational_integrand,
            "x",
            fallback_config,
        );
        let budget_limited_integrand =
            cas_parser::parse("1/(x+1)", &mut ctx).expect("budget-limited rational integrand");
        let budget_limited_config = AlgorithmicIntegrationBackendConfig::diagnostic_only()
            .with_budget(AlgorithmicIntegrationBackendBudget::single_probe_without_verification());
        let budget_limited_rational_probe = try_algorithmic_integration_backend(
            &mut ctx,
            budget_limited_integrand,
            "x",
            budget_limited_config,
        );

        let arctan_integrand = cas_parser::parse("1/(x^2+1)", &mut ctx).expect("integrand");
        let arctan_antiderivative =
            cas_parser::parse("arctan(x)", &mut ctx).expect("antiderivative");
        let mut verified_candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            arctan_integrand,
            "x",
            arctan_antiderivative,
        );
        let verification_start = Instant::now();
        verify_antiderivative_by_differentiation(&mut ctx, &mut verified_candidate);
        let verified_elapsed = verification_start.elapsed();

        let mismatch_integrand = cas_parser::parse("x", &mut ctx).expect("mismatch integrand");
        let mismatch_antiderivative =
            cas_parser::parse("x", &mut ctx).expect("mismatch antiderivative");
        let mut rejected_candidate = AlgorithmicIntegrationCandidate::unverified_table_reused(
            mismatch_integrand,
            "x",
            mismatch_antiderivative,
        );
        let rejection_start = Instant::now();
        verify_antiderivative_by_differentiation(&mut ctx, &mut rejected_candidate);
        let rejected_elapsed = rejection_start.elapsed();

        let function_mismatch_integrand =
            cas_parser::parse("sin(x)", &mut ctx).expect("function mismatch integrand");
        let function_mismatch_antiderivative =
            cas_parser::parse("x", &mut ctx).expect("function mismatch antiderivative");
        let mut function_rejected_candidate =
            AlgorithmicIntegrationCandidate::unverified_table_reused(
                function_mismatch_integrand,
                "x",
                function_mismatch_antiderivative,
            );
        let function_rejection_start = Instant::now();
        verify_antiderivative_by_differentiation(&mut ctx, &mut function_rejected_candidate);
        let function_rejected_elapsed = function_rejection_start.elapsed();

        let observed = vec![
            (AlgorithmicIntegrationBackendConfig::default(), disabled),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                unsupported,
            ),
            (method_limited_config, method_limited),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_scaled_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_scaled_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_numeric_slope_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_slope_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_product_symbolic_slope_rational_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_shifted_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_scaled_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_scaled_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_unit_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_square_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_square_radius_constant_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_positive_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_numeric_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_slope_shifted_affine_symbolic_external_positive_radius_mixed_numerator_hermite_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_square_radius_log_derivative_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_square_radius_arctan_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_slope_shifted_symbolic_square_radius_mixed_numerator_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_indefinite_symbolic_square_denominator_policy_gap_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_symbolic_slope_indefinite_square_denominator_policy_gap_probe,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                diagnostic_heurisch_probe,
            ),
            (fallback_config, fallback_rational_probe),
            (budget_limited_config, budget_limited_rational_probe),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                verified_candidate,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                rejected_candidate,
            ),
            (
                AlgorithmicIntegrationBackendConfig::diagnostic_only(),
                function_rejected_candidate,
            ),
        ];

        assert_eq!(
            observed
                .iter()
                .filter(|(_, candidate)| candidate.is_publicly_acceptable())
                .count(),
            25
        );
        assert_eq!(
            observed
                .iter()
                .filter(|(_, candidate)| {
                    candidate.is_publicly_acceptable()
                        && !matches!(
                            candidate.verification_status,
                            AlgorithmicIntegrationVerificationStatus::Verified
                                | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                        )
                })
                .count(),
            0
        );
        assert_eq!(
            observed
                .iter()
                .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
                .count(),
            1
        );
        assert_eq!(
            method_probe_no_match_reason_counts(&observed),
            BTreeMap::from([
                ("hermite/denominator_policy_mismatch".to_string(), 1),
                ("hermite/radius_policy_mismatch".to_string(), 2),
                ("hermite/shape_mismatch".to_string(), 1),
                ("heurisch_probe/shape_mismatch".to_string(), 3),
                ("rational/denominator_policy_mismatch".to_string(), 4),
                ("rational/numerator_policy_mismatch".to_string(), 15),
                ("rational/shape_mismatch".to_string(), 1),
            ])
        );
        assert_eq!(
            failure_class_counts(&observed),
            BTreeMap::from([
                ("budget_exceeded", 2),
                ("disabled_by_mode", 1),
                ("residual_affine_in_variable", 1),
                ("residual_function_of_variable", 1),
                ("unsupported_method", 3),
            ])
        );
        assert_eq!(
            failure_class_by_method(&observed),
            BTreeMap::from([
                ("rational/budget_exceeded".to_string(), 1),
                ("table_reused/residual_affine_in_variable".to_string(), 1,),
                ("table_reused/residual_function_of_variable".to_string(), 1,),
                ("unsupported/budget_exceeded".to_string(), 1),
                ("unsupported/disabled_by_mode".to_string(), 1),
                ("unsupported/unsupported_method".to_string(), 3),
            ])
        );

        let verification_elapsed_ms =
            (verified_elapsed + rejected_elapsed + function_rejected_elapsed).as_secs_f64()
                * 1000.0;
        println!(
            "algorithmic_backend_observability: {{\"attempts\":{},\"public_accepted\":{},\"unverified_public_acceptances\":{},\"fallback_eligible\":{},\"unverified_fallback_acceptances\":{},\"method_probe_budget_exhausted\":{},\"verification_budget_exceeded\":{},\"method_probes_used_total\":{},\"verification_checks_used_total\":{},\"verification_elapsed_ms\":{:.3},\"mode_counts\":{},\"method_counts\":{},\"method_probe_usage_by_method\":{},\"method_probe_attempt_counts\":{},\"method_probe_candidate_counts\":{},\"method_probe_no_match_counts\":{},\"method_probe_no_match_reason_counts\":{},\"verification_check_usage_by_method\":{},\"verification_status_by_method\":{},\"residual_reason_by_method\":{},\"verification_blocker_counts\":{},\"verification_blocker_by_method\":{},\"failure_class_counts\":{},\"failure_class_by_method\":{},\"verification_residual_counts\":{},\"verification_residual_by_method\":{},\"verification_residual_kind_counts\":{},\"verification_residual_kind_by_method\":{},\"verification_residual_signature_counts\":{},\"verification_residual_signature_by_method\":{},\"publication_status_counts\":{},\"publication_status_by_method\":{},\"fallback_status_counts\":{},\"fallback_status_by_method\":{},\"trace_level_counts\":{},\"constant_policy_counts\":{},\"public_trace_level_counts\":{},\"public_constant_policy_counts\":{},\"fallback_trace_level_counts\":{},\"fallback_constant_policy_counts\":{},\"assumption_exprs\":{},\"public_assumption_exprs\":{},\"fallback_assumption_exprs\":{},\"verification_evidence_counts\":{},\"public_verification_evidence_counts\":{},\"fallback_verification_evidence_counts\":{},\"verification_evidence_by_method\":{},\"public_verification_evidence_by_method\":{},\"fallback_verification_evidence_by_method\":{},\"verification_normalization_reason_counts\":{},\"public_verification_normalization_reason_counts\":{},\"fallback_verification_normalization_reason_counts\":{},\"verification_normalization_reason_by_method\":{},\"public_verification_normalization_reason_by_method\":{},\"fallback_verification_normalization_reason_by_method\":{},\"verification_normalization_pass_count_counts\":{},\"public_verification_normalization_pass_count_counts\":{},\"fallback_verification_normalization_pass_count_counts\":{},\"verification_normalization_pass_count_by_method\":{},\"public_verification_normalization_pass_count_by_method\":{},\"fallback_verification_normalization_pass_count_by_method\":{},\"max_verification_normalization_passes\":{},\"public_max_verification_normalization_passes\":{},\"fallback_max_verification_normalization_passes\":{},\"verification_status_counts\":{},\"residual_reason_counts\":{},\"required_condition_counts\":{}}}",
            observed.len(),
            observed
                .iter()
                .filter(|(_, candidate)| candidate.is_publicly_acceptable())
                .count(),
            observed
                .iter()
                .filter(|(_, candidate)| {
                    candidate.is_publicly_acceptable()
                        && !matches!(
                            candidate.verification_status,
                            AlgorithmicIntegrationVerificationStatus::Verified
                                | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                        )
                })
                .count(),
            observed
                .iter()
                .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
                .count(),
            observed
                .iter()
                .filter(|(config, candidate)| {
                    candidate.fallback_antiderivative(*config).is_some()
                        && !matches!(
                            candidate.verification_status,
                            AlgorithmicIntegrationVerificationStatus::Verified
                                | AlgorithmicIntegrationVerificationStatus::VerifiedUnderConditions
                        )
                })
                .count(),
            method_probe_budget_exhausted_count(&observed),
            verification_budget_exceeded_count(&observed),
            method_probes_used_total(&observed),
            verification_checks_used_total(&observed),
            verification_elapsed_ms,
            json_count_map(mode_counts(&observed)),
            json_count_map(method_counts(&observed)),
            json_count_map(method_probe_usage_by_method(&observed)),
            json_count_map(method_probe_attempt_counts(&observed)),
            json_count_map(method_probe_candidate_counts(&observed)),
            json_count_map(method_probe_no_match_counts(&observed)),
            json_string_count_map(method_probe_no_match_reason_counts(&observed)),
            json_count_map(verification_check_usage_by_method(&observed)),
            json_string_count_map(verification_status_by_method(&observed)),
            json_string_count_map(residual_reason_by_method(&observed)),
            json_count_map(verification_blocker_counts(&observed)),
            json_string_count_map(verification_blocker_by_method(&observed)),
            json_count_map(failure_class_counts(&observed)),
            json_string_count_map(failure_class_by_method(&observed)),
            json_count_map(verification_residual_counts(&observed)),
            json_string_count_map(verification_residual_by_method(&observed)),
            json_count_map(verification_residual_kind_counts(&observed)),
            json_string_count_map(verification_residual_kind_by_method(&observed)),
            json_count_map(verification_residual_signature_counts(&observed)),
            json_string_count_map(verification_residual_signature_by_method(&observed)),
            json_count_map(publication_status_counts(&observed)),
            json_string_count_map(publication_status_by_method(&observed)),
            json_count_map(fallback_status_counts(&observed)),
            json_string_count_map(fallback_status_by_method(&observed)),
            json_count_map(trace_level_counts(&observed)),
            json_count_map(constant_policy_counts(&observed)),
            json_count_map(public_trace_level_counts(&observed)),
            json_count_map(public_constant_policy_counts(&observed)),
            json_count_map(fallback_trace_level_counts(&observed)),
            json_count_map(fallback_constant_policy_counts(&observed)),
            assumption_expr_count(&observed),
            public_assumption_expr_count(&observed),
            fallback_assumption_expr_count(&observed),
            json_count_map(verification_evidence_counts(&observed)),
            json_count_map(public_verification_evidence_counts(&observed)),
            json_count_map(fallback_verification_evidence_counts(&observed)),
            json_string_count_map(verification_evidence_by_method(&observed)),
            json_string_count_map(public_verification_evidence_by_method(&observed)),
            json_string_count_map(fallback_verification_evidence_by_method(&observed)),
            json_count_map(verification_normalization_reason_counts(&observed)),
            json_count_map(public_verification_normalization_reason_counts(
                &observed
            )),
            json_count_map(fallback_verification_normalization_reason_counts(
                &observed
            )),
            json_string_count_map(verification_normalization_reason_by_method(&observed)),
            json_string_count_map(public_verification_normalization_reason_by_method(
                &observed
            )),
            json_string_count_map(fallback_verification_normalization_reason_by_method(
                &observed
            )),
            json_string_count_map(verification_normalization_pass_count_counts(&observed)),
            json_string_count_map(public_verification_normalization_pass_count_counts(
                &observed
            )),
            json_string_count_map(fallback_verification_normalization_pass_count_counts(
                &observed
            )),
            json_string_count_map(verification_normalization_pass_count_by_method(&observed)),
            json_string_count_map(public_verification_normalization_pass_count_by_method(
                &observed
            )),
            json_string_count_map(fallback_verification_normalization_pass_count_by_method(
                &observed
            )),
            max_verification_normalization_passes(&observed),
            public_max_verification_normalization_passes(&observed),
            fallback_max_verification_normalization_passes(&observed),
            json_count_map(verification_status_counts(&observed)),
            json_count_map(residual_reason_counts(&observed)),
            json_count_map(required_condition_counts(&observed)),
        );
    }

    fn method_probe_budget_exhausted_count(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(config, candidate)| {
                config.mode.attempts_backend()
                    && config.budget.max_method_probes == 0
                    && candidate.residual_reason.as_ref()
                        == Some(&AlgorithmicIntegrationResidualReason::BudgetExceeded)
            })
            .count()
    }

    fn verification_budget_exceeded_count(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(config, candidate)| {
                config.mode.attempts_backend()
                    && config.budget.max_method_probes > 0
                    && config.budget.max_verification_checks == 0
                    && candidate.antiderivative.is_some()
                    && candidate.residual_reason.as_ref()
                        == Some(&AlgorithmicIntegrationResidualReason::BudgetExceeded)
            })
            .count()
    }

    fn method_probes_used_total(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .map(|(_, candidate)| candidate.method_probes_used)
            .sum()
    }

    fn verification_checks_used_total(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .map(|(_, candidate)| candidate.verification_checks_used)
            .sum()
    }

    fn method_probe_usage_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.method_probes_used > 0 {
                *counts.entry(candidate.method.metric_label()).or_insert(0) +=
                    candidate.method_probes_used;
            }
        }
        counts
    }

    fn method_probe_attempt_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            for method in &candidate.method_probe_attempts {
                *counts.entry(method.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn method_probe_candidate_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if !candidate.method_probe_attempts.is_empty()
                && !matches!(candidate.method, AlgorithmicIntegrationMethod::Unsupported)
            {
                *counts.entry(candidate.method.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn method_probe_no_match_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = method_probe_attempt_counts(observed);
        for (method, candidate_count) in method_probe_candidate_counts(observed) {
            let attempt_count = counts
                .get_mut(method)
                .expect("candidate method was attempted by a method probe");
            *attempt_count = attempt_count
                .checked_sub(candidate_count)
                .expect("candidate count cannot exceed method attempts");
        }
        counts.retain(|_, count| *count > 0);
        counts
    }

    fn method_probe_no_match_reason_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            for (method, reason) in &candidate.method_probe_no_match_reasons {
                let key = format!("{}/{}", method.metric_label(), reason.metric_label());
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_check_usage_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_checks_used > 0 {
                *counts.entry(candidate.method.metric_label()).or_insert(0) +=
                    candidate.verification_checks_used;
            }
        }
        counts
    }

    fn method_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts.entry(candidate.method.metric_label()).or_insert(0) += 1;
        }
        counts
    }

    fn verification_status_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_status.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn residual_reason_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(reason) = &candidate.residual_reason {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    reason.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_blocker_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_blocker.is_present() {
                *counts
                    .entry(candidate.verification_blocker.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_blocker_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_blocker.is_present() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_blocker.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn failure_class_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(failure_class) = candidate.failure_class() {
                *counts.entry(failure_class.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn failure_class_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(failure_class) = candidate.failure_class() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    failure_class.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_residual.is_some() {
                *counts.entry("derivative_minus_integrand").or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_residual.is_some() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    "derivative_minus_integrand"
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_kind_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(kind) = &candidate.verification_residual_kind {
                *counts.entry(kind.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_kind_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(kind) = &candidate.verification_residual_kind {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    kind.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_signature_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(signature) = &candidate.verification_residual_signature {
                *counts.entry(signature.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_residual_signature_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(signature) = &candidate.verification_residual_signature {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    signature.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn publication_status_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.publication_status().metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn publication_status_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.publication_status().metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn fallback_status_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            *counts
                .entry(candidate.fallback_status(*config).metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn fallback_status_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.fallback_status(*config).metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn trace_level_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.trace_level.metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn constant_policy_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.constant_policy.metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn public_trace_level_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                *counts
                    .entry(candidate.trace_level.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn public_constant_policy_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                *counts
                    .entry(candidate.constant_policy.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_trace_level_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                *counts
                    .entry(candidate.trace_level.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_constant_policy_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                *counts
                    .entry(candidate.constant_policy.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn assumption_expr_count(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .map(|(_, candidate)| candidate.assumptions.len())
            .sum()
    }

    fn public_assumption_expr_count(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(_, candidate)| candidate.is_publicly_acceptable())
            .map(|(_, candidate)| candidate.assumptions.len())
            .sum()
    }

    fn fallback_assumption_expr_count(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
            .map(|(_, candidate)| candidate.assumptions.len())
            .sum()
    }

    fn verification_evidence_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.verification_evidence.metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn public_verification_evidence_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                *counts
                    .entry(candidate.verification_evidence.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_evidence_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                *counts
                    .entry(candidate.verification_evidence.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_evidence_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_evidence.metric_label()
            );
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn public_verification_evidence_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_evidence.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_evidence_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_evidence.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_normalization_reason_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_normalization_reason.is_present() {
                *counts
                    .entry(candidate.verification_normalization_reason.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn public_verification_normalization_reason_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable()
                && candidate.verification_normalization_reason.is_present()
            {
                *counts
                    .entry(candidate.verification_normalization_reason.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_normalization_reason_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some()
                && candidate.verification_normalization_reason.is_present()
            {
                *counts
                    .entry(candidate.verification_normalization_reason.metric_label())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_normalization_reason_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.verification_normalization_reason.is_present() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_normalization_reason.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn public_verification_normalization_reason_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable()
                && candidate.verification_normalization_reason.is_present()
            {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_normalization_reason.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_normalization_reason_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some()
                && candidate.verification_normalization_reason.is_present()
            {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_normalization_reason.metric_label()
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_normalization_pass_count_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.verification_normalization_passes_used.to_string())
                .or_insert(0) += 1;
        }
        counts
    }

    fn public_verification_normalization_pass_count_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                *counts
                    .entry(candidate.verification_normalization_passes_used.to_string())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_normalization_pass_count_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                *counts
                    .entry(candidate.verification_normalization_passes_used.to_string())
                    .or_insert(0) += 1;
            }
        }
        counts
    }

    fn verification_normalization_pass_count_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            let key = format!(
                "{}/{}",
                candidate.method.metric_label(),
                candidate.verification_normalization_passes_used
            );
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    fn public_verification_normalization_pass_count_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if candidate.is_publicly_acceptable() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_normalization_passes_used
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn fallback_verification_normalization_pass_count_by_method(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for (config, candidate) in observed {
            if candidate.fallback_antiderivative(*config).is_some() {
                let key = format!(
                    "{}/{}",
                    candidate.method.metric_label(),
                    candidate.verification_normalization_passes_used
                );
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }

    fn max_verification_normalization_passes(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .map(|(_, candidate)| candidate.verification_normalization_passes_used)
            .max()
            .unwrap_or(0)
    }

    fn public_max_verification_normalization_passes(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(_, candidate)| candidate.is_publicly_acceptable())
            .map(|(_, candidate)| candidate.verification_normalization_passes_used)
            .max()
            .unwrap_or(0)
    }

    fn fallback_max_verification_normalization_passes(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> usize {
        observed
            .iter()
            .filter(|(config, candidate)| candidate.fallback_antiderivative(*config).is_some())
            .map(|(_, candidate)| candidate.verification_normalization_passes_used)
            .max()
            .unwrap_or(0)
    }

    fn verification_status_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            *counts
                .entry(candidate.verification_status.metric_label())
                .or_insert(0) += 1;
        }
        counts
    }

    fn residual_reason_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            if let Some(reason) = &candidate.residual_reason {
                *counts.entry(reason.metric_label()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn mode_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (config, _) in observed {
            *counts.entry(config.mode.metric_label()).or_insert(0) += 1;
        }
        counts
    }

    fn required_condition_counts(
        observed: &[(
            AlgorithmicIntegrationBackendConfig,
            AlgorithmicIntegrationCandidate,
        )],
    ) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for (_, candidate) in observed {
            for condition in &candidate.required_conditions {
                let label = match condition {
                    ConditionPredicate::NonZero(_) => "nonzero",
                    ConditionPredicate::Positive(_) => "positive",
                    ConditionPredicate::NonNegative(_) => "nonnegative",
                    ConditionPredicate::LowerBound { .. } => "lower_bound",
                    ConditionPredicate::Defined(_) => "defined",
                    ConditionPredicate::InvTrigPrincipalRange { .. } => "inv_trig_principal_range",
                    ConditionPredicate::EqZero(_) => "eq_zero",
                    ConditionPredicate::EqOne(_) => "eq_one",
                };
                *counts.entry(label).or_insert(0) += 1;
            }
        }
        counts
    }

    fn json_count_map(counts: BTreeMap<&'static str, usize>) -> String {
        let entries = counts
            .into_iter()
            .map(|(key, value)| format!("\"{key}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{entries}}}")
    }

    fn json_string_count_map(counts: BTreeMap<String, usize>) -> String {
        let entries = counts
            .into_iter()
            .map(|(key, value)| format!("\"{key}\":{value}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("{{{entries}}}")
    }
}
