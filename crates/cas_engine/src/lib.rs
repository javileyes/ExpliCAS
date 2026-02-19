// All clippy lints addressed: code fixes or local #[allow] attributes applied.
// See too_many_arguments allows in: inverse_trig.rs, step.rs
// See arc_with_non_send_sync allows in: profile_cache.rs

pub mod assumptions;
pub(crate) mod best_so_far;
pub mod budget;
pub(crate) mod collect;
pub mod const_fold;
pub(crate) mod cycle_detector;
pub mod cycle_events;
pub mod diagnostics;
pub mod didactic;
pub mod domain;
pub mod domain_facts;
pub mod domain_oracle;
pub mod engine;
pub mod eval;
pub mod eval_step_pipeline;
pub mod expand;
pub mod helpers;
pub mod implicit_domain;
pub mod limits;
pub mod options;
pub mod orchestrator;
pub mod parent_context;
pub mod phase;
pub mod profile_cache;
pub mod profiler;
pub mod rationalize;
pub mod rationalize_policy;
pub(crate) mod recursion_guard;
pub mod rule;
pub mod rules;
pub mod semantics;
pub mod solve_safety;
pub mod solver;
pub mod step;
pub(crate) mod step_optimization;
pub(crate) mod strategies;
pub mod substitute;
pub mod telescoping;
pub mod timeline;

// Property-based numeric tests for rewrite correctness
#[cfg(test)]
mod numeric_property_tests;

/// V2.1 Issue #4: Stable Public API
///
/// This module contains the stable, versioned API for external integrators.
/// Types and functions exported here are guaranteed to maintain backward
/// compatibility following semantic versioning.
pub mod api;

pub mod error;
#[macro_use]
pub mod macros;

pub(crate) use cas_math::build;
pub use cas_math::canonical_forms;
pub use cas_math::expr_nary as nary;
pub use cas_math::factor;
pub use cas_math::pattern_marks;

pub use cas_ast::ordering;
pub use cas_ast::target_kind;
pub(crate) use cas_ast::visitors;
pub use cas_formatter::visualizer;
pub(crate) use cas_math::pattern_scanner;

pub use assumptions::{
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionRecord, AssumptionReporting,
};
pub use budget::{Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats};
pub use cycle_events::{CycleEvent, CycleLevel};
pub use domain::{can_cancel_factor, BlockedHint, CancelDecision, DomainMode, Proof};
pub use engine::{
    eval_f64_checked, strip_all_holds, EquivalenceResult, EvalCheckedError, EvalCheckedOptions,
    Simplifier,
};
pub use error::{CasError, CasResult};
pub use eval::*;
pub use options::AutoExpandBinomials; // V2.15.8: education mode expansion
pub use options::HeuristicPoly; // V2.15.9: smart polynomial simplification
pub use phase::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseStats, PipelineStats, SimplifyOptions,
    SimplifyPhase,
};
pub use rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome, RationalizeReason};
pub use rule::{Rule, SoundnessLabel};
pub use semantics::{
    AssumeScope, BranchPolicy, EvalConfig, InverseTrigPolicy, NormalFormGoal, ValueDomain,
};
pub use step::{DisplayEvalSteps, Step};
pub use visitors::{DepthVisitor, VariableCollector};

// Equation-level primitives (not simplifier rules — used by solver pipeline)
pub use rules::cancel_common_terms::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult,
};
