//! Backend modes, budgets, configuration, and probe accounting.

use super::*;

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
    method_probe_budget_limit: usize,
    verification_check_budget_limit: usize,
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
            method_probe_budget_limit: budget.max_method_probes,
            verification_check_budget_limit: budget.max_verification_checks,
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

    pub fn method_probe_budget_limit(&self) -> usize {
        self.method_probe_budget_limit
    }

    pub fn verification_check_budget_limit(&self) -> usize {
        self.verification_check_budget_limit
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

    pub fn class_label(self) -> &'static str {
        match self {
            AlgorithmicIntegrationProbeNoMatchReason::ShapeMismatch => "shape",
            AlgorithmicIntegrationProbeNoMatchReason::DenominatorPolicyMismatch
            | AlgorithmicIntegrationProbeNoMatchReason::NumeratorPolicyMismatch
            | AlgorithmicIntegrationProbeNoMatchReason::RadiusPolicyMismatch => "policy",
            AlgorithmicIntegrationProbeNoMatchReason::NumeratorDerivativeMismatch => {
                "derivative_evidence"
            }
        }
    }
}
