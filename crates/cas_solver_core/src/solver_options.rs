//! Shared solver option model.
//!
//! This type lives in `cas_solver_core` so higher-level crates can reuse a
//! single semantic option shape without duplicating the domain/budget fields.

/// Options for solver operations, containing semantic context.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::value_domain::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::domain_mode::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::assume_scope::AssumeScope,
    /// Budget for conditional branching (anti-explosion)
    pub budget: crate::solve_budget::SolveBudget,
    /// If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative.
    pub detailed_steps: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::value_domain::ValueDomain::RealOnly,
            domain_mode: crate::domain_mode::DomainMode::Generic,
            assume_scope: crate::assume_scope::AssumeScope::Real,
            budget: crate::solve_budget::SolveBudget::default(),
            detailed_steps: true,
        }
    }
}

impl SolverOptions {
    /// Build options from semantic axes and solve budget.
    pub fn from_axes(
        value_domain: crate::value_domain::ValueDomain,
        domain_mode: crate::domain_mode::DomainMode,
        assume_scope: crate::assume_scope::AssumeScope,
        budget: crate::solve_budget::SolveBudget,
    ) -> Self {
        Self {
            value_domain,
            domain_mode,
            assume_scope,
            budget,
            ..Default::default()
        }
    }

    /// Convert domain mode into core solver domain mode kind.
    pub fn core_domain_mode(&self) -> crate::log_domain::DomainModeKind {
        crate::strategy_options::solver_domain_mode_kind(
            matches!(self.domain_mode, crate::domain_mode::DomainMode::Assume),
            matches!(self.domain_mode, crate::domain_mode::DomainMode::Strict),
        )
    }

    /// Returns true when assume-scope allows wildcard assumptions.
    pub fn wildcard_scope(&self) -> bool {
        self.assume_scope == crate::assume_scope::AssumeScope::Wildcard
    }
}
