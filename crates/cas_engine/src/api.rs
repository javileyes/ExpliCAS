//! V2.1 Issue #4: Stable Public API
//!
//! This module re-exports the stable public types for external integrators.
//! These types form the contractual API and should maintain backward compatibility.
//!
//! # Stability Guarantee
//!
//! Types in this module are considered stable. Changes to their structure
//! will be documented in CHANGELOG and follow semantic versioning.
//!
//! # Usage
//!
//! ```rust,ignore
//! use cas_engine::api::{SolveResult, SolutionSet, Case, ConditionPredicate, ConditionSet};
//! use cas_engine::api::SolveBudget;
//! ```
//!
//! # Example: Checking Solution Type
//!
//! ```rust,ignore
//! use cas_engine::api::SolutionSet;
//!
//! fn process_solution(sol: &SolutionSet) {
//!     match sol {
//!         SolutionSet::Discrete(solutions) => {
//!             println!("Found {} discrete solutions", solutions.len());
//!         }
//!         SolutionSet::AllReals => {
//!             println!("Solution is all real numbers");
//!         }
//!         SolutionSet::Empty => {
//!             println!("No solution exists");
//!         }
//!         SolutionSet::Conditional(cases) => {
//!             println!("Conditional solution with {} cases", cases.len());
//!             for case in cases {
//!                 // Access case.when (ConditionSet) and case.then (SolveResult)
//!             }
//!         }
//!         _ => {}
//!     }
//! }
//! ```

// =============================================================================
// Stable Types from cas_ast::domain
// =============================================================================

/// Complete result of a solve operation.
/// Contains solutions and optional residual for partially solved equations.
pub use cas_ast::SolveResult;

/// Set of solutions: Discrete, Continuous, Conditional, etc.
pub use cas_ast::SolutionSet;

/// A single case in a conditional solution (guard + result).
pub use cas_ast::Case;

/// A condition predicate (NonZero, Positive, EqZero, etc.).
pub use cas_ast::ConditionPredicate;

/// A conjunction of condition predicates.
pub use cas_ast::ConditionSet;

/// Interval representation for continuous solutions.
pub use cas_ast::Interval;

/// Bound type for intervals (Open or Closed).
pub use cas_ast::BoundType;

// =============================================================================
// Stable Types from cas_engine::solver
// =============================================================================

pub use crate::cancel_runtime::cancel_additive_terms_semantic;
pub use crate::const_fold::{fold_constants, ConstFoldMode, ConstFoldResult};
pub use crate::helpers::{prove_nonzero, prove_positive};
pub use cas_math::evaluator_f64::{
    eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions,
};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
/// Budget for conditional branching in solver.
/// Controls how many branches can be created (anti-explosion).
pub use cas_solver_core::solve_budget::SolveBudget;

/// Options for solver operations including domain and budget.
pub type SolverOptions = cas_solver_core::solver_options::SolverOptions;

// =============================================================================
// Stable Entrypoints
// =============================================================================

/// Solve an equation with default options.
///
/// # Example
/// ```rust,ignore
/// use cas_engine::api::solve;
/// use cas_engine::Engine;
///
/// let mut engine = Engine::new();
/// // ... set up equation ...
/// // let (solution_set, steps) = solve(&equation, "x", &mut engine.simplifier)?;
/// ```
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut crate::engine::Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), crate::error::CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        SolverOptions::default(),
        crate::solve_core_runtime::solve_inner,
    )
}

pub use crate::expand::{expand, expand_with_stats};
pub use crate::implicit_domain::{
    derive_requires_from_equation, domain_delta_check, infer_domain_calls_get,
    infer_domain_calls_reset, infer_implicit_domain,
};
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::assumption_model::{
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionKind, AssumptionRecord,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::branch_policy::BranchPolicy;
pub use cas_solver_core::cancel_common_terms::{cancel_common_additive_terms, CancelResult};
pub use cas_solver_core::const_fold_types::{
    ConstFoldMode as SolverConstFoldMode, ConstFoldResult as SolverConstFoldResult,
};
pub use cas_solver_core::domain_assumption_classification::classify_assumption;
pub use cas_solver_core::domain_condition::{ImplicitCondition, ImplicitDomain};
pub use cas_solver_core::domain_context::DomainContext;
pub use cas_solver_core::domain_facts_model::{FactStrength, Predicate};
pub use cas_solver_core::domain_inference::DomainDelta;
pub use cas_solver_core::domain_normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use cas_solver_core::domain_oracle_model::DomainOracle;
pub use cas_solver_core::domain_proof::Proof;
pub use cas_solver_core::equivalence::EquivalenceResult;

/// Default oracle backed by local predicate proof runtime.
pub struct StandardOracle<'a> {
    inner: cas_solver_core::standard_oracle::StandardOracle<'a>,
}

impl<'a> StandardOracle<'a> {
    /// Create a new oracle with the given context and semantic configuration.
    pub fn new(
        ctx: &'a cas_ast::Context,
        mode: crate::DomainMode,
        value_domain: crate::ValueDomain,
    ) -> Self {
        Self {
            inner: cas_solver_core::standard_oracle::StandardOracle::new(
                ctx,
                mode,
                value_domain,
                cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_proof_simplifier::<crate::engine::Simplifier>,
                cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<crate::engine::Simplifier>,
                cas_solver_core::proof_runtime_bound_runtime::prove_nonnegative_with_runtime_proof_simplifier::<crate::engine::Simplifier>,
            ),
        }
    }

    #[inline]
    pub fn mode(&self) -> crate::DomainMode {
        self.inner.mode()
    }

    #[inline]
    pub fn value_domain(&self) -> crate::ValueDomain {
        self.inner.value_domain()
    }
}

impl cas_solver_core::domain_oracle_model::DomainOracle for StandardOracle<'_> {
    type Decision = cas_solver_core::domain_cancel_decision::CancelDecision;

    fn query(
        &self,
        pred: &cas_solver_core::domain_facts_model::Predicate,
    ) -> cas_solver_core::domain_facts_model::FactStrength {
        self.inner.query(pred)
    }

    fn allows(
        &self,
        pred: &cas_solver_core::domain_facts_model::Predicate,
    ) -> cas_solver_core::domain_cancel_decision::CancelDecision {
        self.inner.allows(pred)
    }
}

/// Solve an equation with explicit options, returning display-ready steps.
///
/// V2.9.8: Type-safe API - returns `DisplaySolveSteps` which guarantees
/// cleanup has been applied. Replaces the old `solve_with_options` for
/// display-facing code.
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut crate::engine::Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), crate::error::CasError> {
    cas_solver_core::solver_entrypoints_bound_runtime::solve_with_display_steps_with_default_runtime_ctx_and_backend_with_state(
        eq,
        var,
        simplifier,
        opts,
        crate::solve_core_runtime::solve_inner,
        crate::collect_assumption_records,
        |simplifier, raw_steps| {
            cas_solver_core::solver_entrypoints_bound_runtime::cleanup_display_solve_steps_with_runtime_state(
                simplifier,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}

/// Display-ready solve steps (post-cleanup).
/// Wrapper type that enforces step processing has been applied.
pub type DisplaySolveSteps = cas_solver_core::solve_runtime_types::RuntimeDisplaySolveSteps;
/// Diagnostics collected during solve operation.
pub type SolveDiagnostics = cas_solver_core::solve_runtime_types::RuntimeSolveDiagnostics;
/// Raw solve step entry (pre-display wrapper).
pub type SolveStep = cas_solver_core::solve_runtime_types::RuntimeSolveStep;
/// Raw solve sub-step entry.
pub type SolveSubStep = cas_solver_core::solve_runtime_types::RuntimeSolveSubStep;
/// Solver helper: check whether an expression contains the target variable.
pub use cas_solver_core::isolation_utils::contains_var;
/// Solver helper: infer variable from free-variable set.
pub use cas_solver_core::solve_infer::infer_solve_variable;
/// Solver verification instrumentation counters.
pub use cas_solver_core::verify_stats;

/// Verify a single candidate solution by substitution.
pub fn verify_solution(
    simplifier: &mut crate::engine::Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solution: cas_ast::ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_runtime_bound_runtime::verify_solution_with_runtime_state_and_runtime_proof_simplifier_with_state(
        simplifier,
        equation,
        var,
        solution,
    )
}
/// Verify an entire solution set by substitution.
pub fn verify_solution_set(
    simplifier: &mut crate::engine::Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_runtime_bound_runtime::verify_solution_set_with_runtime_state_and_runtime_proof_simplifier_with_state(
        simplifier, equation, var, solutions,
    )
}
/// Verification result for a complete solution set.
pub use cas_solver_core::verification::VerifyResult;
/// Solver verification status for one candidate solution.
pub use cas_solver_core::verification::VerifyStatus;
/// Solver verification result summary.
pub use cas_solver_core::verification::VerifySummary;

// =============================================================================
// Display Traits (Human-readable output)
// =============================================================================

/// Display wrapper for expressions with context.
/// Used for human-readable output of expressions.
pub use cas_formatter::DisplayExpr;

/// LaTeX wrapper for expressions with context.
/// Used for mathematical typesetting output.
pub use cas_formatter::LaTeXExpr;
