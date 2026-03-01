use cas_ast::Equation;

/// Options for solver operations, containing semantic context.
///
/// This struct passes value domain and domain mode information to the solver,
/// enabling domain-aware decisions like rejecting log operations on negative bases.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: crate::semantics::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: crate::domain::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: crate::semantics::AssumeScope,
    /// V2.0: Budget for conditional branching (anti-explosion)
    pub budget: cas_solver_core::solve_budget::SolveBudget,
    /// V2.9.8: If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative (3 steps for Succinct verbosity).
    pub detailed_steps: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: crate::semantics::ValueDomain::RealOnly,
            domain_mode: crate::domain::DomainMode::Generic,
            assume_scope: crate::semantics::AssumeScope::Real,
            budget: cas_solver_core::solve_budget::SolveBudget::default(),
            detailed_steps: true, // V2.9.8: Default to detailed (Normal/Verbose)
        }
    }
}

impl SolverOptions {
    /// Build solver options from engine eval options.
    pub fn from_eval_options(options: &crate::options::EvalOptions) -> Self {
        Self {
            value_domain: options.shared.semantics.value_domain,
            domain_mode: options.shared.semantics.domain_mode,
            assume_scope: options.shared.semantics.assume_scope,
            budget: options.budget,
            ..Default::default()
        }
    }

    /// Convert engine domain mode into core solver domain mode kind.
    pub fn core_domain_mode(&self) -> cas_solver_core::log_domain::DomainModeKind {
        cas_solver_core::log_domain::domain_mode_kind_from_flags(
            matches!(self.domain_mode, crate::domain::DomainMode::Assume),
            matches!(self.domain_mode, crate::domain::DomainMode::Strict),
        )
    }

    /// Returns true when assume-scope allows wildcard assumptions.
    pub fn wildcard_scope(&self) -> bool {
        self.assume_scope == crate::semantics::AssumeScope::Wildcard
    }
}

/// Domain environment for solver operations.
///
/// Contains the "semantic ground" under which the solver operates:
/// - `required`: Constraints inferred from equation structure (e.g., sqrt(y) -> y >= 0)
///
/// This is passed explicitly rather than via TLS for clean reentrancy and testability.
pub(crate) type SolveDomainEnv =
    cas_solver_core::domain_env::SolveDomainEnv<crate::implicit_domain::ImplicitDomain>;

impl cas_solver_core::domain_env::RequiredDomainSet for crate::implicit_domain::ImplicitDomain {
    fn contains_positive(&self, expr: cas_ast::ExprId) -> bool {
        crate::implicit_domain::ImplicitDomain::contains_positive(self, expr)
    }

    fn contains_nonnegative(&self, expr: cas_ast::ExprId) -> bool {
        crate::implicit_domain::ImplicitDomain::contains_nonnegative(self, expr)
    }

    fn contains_nonzero(&self, expr: cas_ast::ExprId) -> bool {
        crate::implicit_domain::ImplicitDomain::contains_nonzero(self, expr)
    }

    fn to_condition_set(&self) -> cas_ast::ConditionSet {
        crate::implicit_domain::ImplicitDomain::to_condition_set(self)
    }
}

/// Solver context — threaded explicitly through the solve pipeline.
///
/// Holds per-invocation state that was formerly stored in TLS,
/// enabling clean reentrancy for recursive/nested solves.
///
/// Shared sink state is centralized in `cas_solver_core::shared_context`,
/// so recursive sub-solves contribute to one accumulator set.
/// `solve_with_display_steps` creates one, and every recursive
/// `solve_with_ctx_and_options` / `solve_with_options` pushes into it.
pub type SolveCtx = cas_solver_core::solve_context::SolveContext<
    SolveDomainEnv,
    crate::implicit_domain::ImplicitCondition,
    crate::assumptions::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

// =============================================================================
// Type-Safe Step Pipeline (V2.9.8)
// =============================================================================
// These newtypes enforce that renderers only consume post-processed steps.
// This eliminates bifurcation between text/timeline outputs at compile time.

/// Display-ready solve steps after didactic cleanup and narration.
/// All renderers (text, timeline, JSON) consume this type only.
pub type DisplaySolveSteps = cas_solver_core::display_steps::DisplaySteps<SolveStep>;

/// Diagnostics collected during solve operation.
///
/// This is returned alongside solutions to provide transparency about
/// what conditions were required vs assumed during solving.
pub type SolveDiagnostics = cas_solver_core::solve_types::SolveDiagnostics<
    crate::implicit_domain::ImplicitCondition,
    crate::assumptions::AssumptionEvent,
    crate::assumptions::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Educational sub-step for solver derivations (e.g., completing the square)
/// Displayed as indented in REPL and collapsible in timeline.
pub type SolveSubStep =
    cas_solver_core::solve_types::SolveSubStep<Equation, crate::step::ImportanceLevel>;

pub type SolveStep =
    cas_solver_core::solve_types::SolveStep<Equation, crate::step::ImportanceLevel, SolveSubStep>;
