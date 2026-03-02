use cas_ast::Equation;

/// Options for solver operations, containing semantic context.
///
/// This mirrors the engine-side options while keeping `cas_solver` API
/// independent from `cas_engine::solver` internals.
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
    /// If true, generate detailed step narrative.
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
    /// Build solver options from engine eval options.
    pub fn from_eval_options(options: &crate::EvalOptions) -> Self {
        Self {
            value_domain: options.shared.semantics.value_domain,
            domain_mode: options.shared.semantics.domain_mode,
            assume_scope: options.shared.semantics.assume_scope,
            budget: options.budget,
            ..Default::default()
        }
    }

    /// Convert solver domain mode into core solver domain mode kind.
    pub fn core_domain_mode(&self) -> cas_solver_core::log_domain::DomainModeKind {
        cas_solver_core::log_domain::domain_mode_kind_from_flags(
            matches!(self.domain_mode, crate::DomainMode::Assume),
            matches!(self.domain_mode, crate::DomainMode::Strict),
        )
    }

    /// Returns true when assume-scope allows wildcard assumptions.
    pub fn wildcard_scope(&self) -> bool {
        self.assume_scope == crate::AssumeScope::Wildcard
    }
}

/// Domain environment for solver operations.
pub type SolveDomainEnv = cas_solver_core::domain_env::SolveDomainEnv<crate::ImplicitDomain>;

/// Solver context for recursive and nested solve flows.
pub type SolveCtx = cas_solver_core::solve_context::SolveContext<
    SolveDomainEnv,
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Display-ready solve steps (post-cleanup).
pub type DisplaySolveSteps = cas_solver_core::display_steps::DisplaySteps<SolveStep>;

/// Diagnostics collected during solve operation.
pub type SolveDiagnostics = cas_solver_core::solve_types::SolveDiagnostics<
    crate::ImplicitCondition,
    crate::AssumptionEvent,
    crate::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Educational sub-step for solver derivations.
pub type SolveSubStep =
    cas_solver_core::solve_types::SolveSubStep<Equation, crate::ImportanceLevel>;

/// Top-level solver step entry.
pub type SolveStep =
    cas_solver_core::solve_types::SolveStep<Equation, crate::ImportanceLevel, SolveSubStep>;
