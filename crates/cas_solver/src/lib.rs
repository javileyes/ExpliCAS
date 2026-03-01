//! Solver facade crate.
//!
//! During migration this crate re-exports the solver API from `cas_engine`.

pub mod check;
pub mod json;
pub mod substitute;

/// Backward-compatible facade for former `cas_engine::strategies::substitute_expr` imports.
pub mod strategies {
    pub use cas_ast::substitute_expr_by_id as substitute_expr;
}

pub use cas_engine::error;
pub use cas_engine::normalize_and_dedupe_conditions;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::ConstFoldMode;
pub use cas_engine::ParentContext;
pub use cas_engine::Rule;
pub use cas_engine::{limit, Approach, LimitOptions, PreSimplifyMode};
pub use cas_engine::{
    take_blocked_hints, AssumeScope, AssumptionKey, AssumptionReporting, AutoExpandBinomials,
    BlockedHint, BranchMode, Budget, CasError, ComplexMode, ContextMode, DomainMode, Engine,
    EquivalenceResult, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly,
    ImplicitCondition, PipelineStats, RequiresDisplayLevel, Simplifier, SimplifyOptions, StepsMode,
};
pub use cas_engine::{AssumptionRecord, DomainWarning};
pub use cas_engine::{BranchPolicy, InverseTrigPolicy, ValueDomain};
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
pub use cas_engine::{RequirementDescriptor, SolveSafety};
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use cas_solver_core::verify_stats;
pub use check::{verify_solution, verify_solution_set, VerifyResult, VerifyStatus, VerifySummary};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, substitute_str_to_json, EnvelopeEvalOptions,
};
pub use substitute::{substitute_power_aware, substitute_with_steps, SubstituteOptions};

/// Options for solver operations, containing semantic context.
///
/// This mirrors `cas_engine::solver::SolverOptions` while keeping the public
/// `cas_solver` API decoupled from engine-internal solver modules.
#[derive(Debug, Clone, Copy)]
pub struct SolverOptions {
    /// The value domain (RealOnly or ComplexEnabled)
    pub value_domain: cas_engine::ValueDomain,
    /// The domain mode (Strict, Assume, Generic)
    pub domain_mode: cas_engine::DomainMode,
    /// Scope for assumptions (only active if domain_mode=Assume)
    pub assume_scope: cas_engine::AssumeScope,
    /// Budget for conditional branching (anti-explosion)
    pub budget: cas_solver_core::solve_budget::SolveBudget,
    /// If true, generate detailed step narrative (5 atomic steps).
    /// If false, generate compact narrative (3 steps for Succinct verbosity).
    pub detailed_steps: bool,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            value_domain: cas_engine::ValueDomain::RealOnly,
            domain_mode: cas_engine::DomainMode::Generic,
            assume_scope: cas_engine::AssumeScope::Real,
            budget: cas_solver_core::solve_budget::SolveBudget::default(),
            detailed_steps: true,
        }
    }
}

impl From<SolverOptions> for cas_engine::api::SolverOptions {
    fn from(value: SolverOptions) -> Self {
        Self {
            value_domain: value.value_domain,
            domain_mode: value.domain_mode,
            assume_scope: value.assume_scope,
            budget: value.budget,
            detailed_steps: value.detailed_steps,
        }
    }
}

/// Domain environment for solver operations.
pub type SolveDomainEnv = cas_solver_core::domain_env::SolveDomainEnv<cas_engine::ImplicitDomain>;

/// Solver context for recursive and nested solve flows.
pub type SolveCtx = cas_solver_core::solve_context::SolveContext<
    SolveDomainEnv,
    cas_engine::ImplicitCondition,
    cas_engine::AssumptionEvent,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Diagnostics collected during solve operation.
pub type SolveDiagnostics = cas_solver_core::solve_types::SolveDiagnostics<
    cas_engine::ImplicitCondition,
    cas_engine::AssumptionEvent,
    cas_engine::AssumptionRecord,
    cas_formatter::display_transforms::ScopeTag,
>;

/// Educational sub-step for solver derivations.
pub type SolveSubStep =
    cas_solver_core::solve_types::SolveSubStep<cas_ast::Equation, cas_engine::ImportanceLevel>;

/// Solve step used by didactic and timeline layers.
pub type SolveStep = cas_solver_core::solve_types::SolveStep<
    cas_ast::Equation,
    cas_engine::ImportanceLevel,
    SolveSubStep,
>;

/// Display-ready solve steps after didactic cleanup and narration.
pub type DisplaySolveSteps = cas_solver_core::display_steps::DisplaySteps<SolveStep>;

/// Solve with default options.
#[inline]
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut cas_engine::Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), cas_engine::CasError> {
    cas_engine::api::solve(eq, var, simplifier)
}

/// Solve and return display-safe steps plus diagnostics.
#[inline]
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut cas_engine::Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), cas_engine::CasError> {
    cas_engine::api::solve_with_display_steps(eq, var, simplifier, opts.into())
}

/// Number-theory helpers exposed by the solver facade without pulling engine rule modules.
pub mod number_theory {
    pub use cas_math::number_theory_support::{compute_gcd, explain_gcd, GcdResult};
}

/// Backward-compatible facade for former `cas_engine::expand::*` imports.
pub mod expand {
    pub use cas_engine::{
        eager_eval_expand_calls, estimate_expand_terms, expand, expand_div, expand_mul, expand_pow,
        expand_with_stats,
    };
}

/// Backward-compatible facade for former `cas_engine::helpers::*` imports.
pub mod helpers {
    pub use cas_engine::{is_zero, prove_nonzero, prove_positive};
}

/// Backward-compatible facade for former `cas_engine::engine::*` imports.
pub mod engine {
    pub use cas_engine::{
        eval_f64, eval_f64_checked, Engine, EquivalenceResult, EvalCheckedError,
        EvalCheckedOptions, LoopConfig, Simplifier,
    };
}

/// Backward-compatible facade for former `cas_engine::phase::*` imports.
pub mod phase {
    pub use cas_engine::{
        ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseMask, PhaseStats, PipelineStats,
        SharedSemanticConfig, SimplifyOptions, SimplifyPhase,
    };
}

/// Backward-compatible facade for former `cas_engine::semantics::*` imports.
pub mod semantics {
    pub use cas_engine::{
        AssumeScope, BranchPolicy, EvalConfig, InverseTrigPolicy, NormalFormGoal, ValueDomain,
    };
}

/// Backward-compatible facade for former `cas_engine::rationalize::*` imports.
pub mod rationalize {
    pub use cas_math::rationalize::{
        rationalize_denominator, RationalizeConfig, RationalizeResult,
    };
}

/// Backward-compatible facade for former `cas_engine::telescoping::*` imports.
pub mod telescoping {
    pub use cas_engine::{
        telescope, try_dirichlet_kernel_identity_pub, DirichletKernelResult, TelescopingResult,
        TelescopingStep,
    };
}
