/// Options for solver operations, containing semantic context.
///
/// `cas_solver` owns this facade type so solver consumers can remain decoupled
/// from `cas_engine` during migration.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::AssumeScope,
    /// Budget for conditional branching (anti-explosion)
    pub budget: cas_solver_core::solve_budget::SolveBudget,
    /// If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative.
    pub detailed_steps: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::ValueDomain::RealOnly,
            domain_mode: crate::DomainMode::Generic,
            assume_scope: crate::AssumeScope::Real,
            budget: cas_solver_core::solve_budget::SolveBudget::default(),
            detailed_steps: true,
        }
    }
}

impl SolverOptions {
    /// Build solver options from eval options.
    pub fn from_eval_options(options: &crate::EvalOptions) -> Self {
        let core = cas_solver_core::solver_options::SolverOptions::from_eval_config(
            options.shared.semantics,
            options.budget,
        );
        Self {
            value_domain: core.value_domain,
            domain_mode: core.domain_mode,
            assume_scope: core.assume_scope,
            budget: core.budget,
            detailed_steps: core.detailed_steps,
        }
    }

    /// Convert domain mode into core solver domain mode kind.
    pub fn core_domain_mode(&self) -> cas_solver_core::log_domain::DomainModeKind {
        cas_solver_core::strategy_options::solver_domain_mode_kind(
            matches!(self.domain_mode, crate::DomainMode::Assume),
            matches!(self.domain_mode, crate::DomainMode::Strict),
        )
    }

    /// Returns true when assume-scope allows wildcard assumptions.
    pub fn wildcard_scope(&self) -> bool {
        self.assume_scope == crate::AssumeScope::Wildcard
    }

    /// Convert facade options to the shared core solver options.
    pub fn to_core(self) -> cas_solver_core::solver_options::SolverOptions {
        cas_solver_core::solver_options::SolverOptions {
            value_domain: self.value_domain,
            domain_mode: self.domain_mode,
            assume_scope: self.assume_scope,
            budget: self.budget,
            detailed_steps: self.detailed_steps,
        }
    }
}

impl From<cas_solver_core::solver_options::SolverOptions> for SolverOptions {
    fn from(value: cas_solver_core::solver_options::SolverOptions) -> Self {
        Self {
            value_domain: value.value_domain,
            domain_mode: value.domain_mode,
            assume_scope: value.assume_scope,
            budget: value.budget,
            detailed_steps: value.detailed_steps,
        }
    }
}

impl From<SolverOptions> for cas_solver_core::solver_options::SolverOptions {
    fn from(value: SolverOptions) -> Self {
        value.to_core()
    }
}
