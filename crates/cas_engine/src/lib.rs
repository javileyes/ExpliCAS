// All clippy lints addressed: code fixes or local #[allow] attributes applied.
// See too_many_arguments allows in: inverse_trig.rs, gcd_zippel_modp.rs (×2), step.rs
// See arc_with_non_send_sync allows in: profile_cache.rs

pub mod assumptions;
pub mod auto_expand_scan;
pub(crate) mod best_so_far;
pub mod budget;
pub mod canonical_forms;
pub mod collect;
pub(crate) mod const_eval;
pub mod const_fold;
pub(crate) mod cycle_detector;
pub mod diagnostics;
pub mod didactic;
pub(crate) mod display_context;
pub mod domain;
pub mod engine;
pub mod env;
pub mod eval;
pub mod eval_step_pipeline;
pub mod expand;
pub(crate) mod factor;
pub mod gcd_zippel_modp;
pub mod helpers;
pub mod implicit_domain;
pub mod limits;
pub(crate) mod matrix;
pub mod modp;
pub(crate) mod mono;
pub mod multinomial_expand;
pub mod multipoly;
pub(crate) mod multipoly_display;
pub mod multipoly_modp;
pub mod options;
pub mod orchestrator;
pub mod ordering;
pub mod parent_context;
pub mod pattern_detection;
pub mod pattern_marks;
pub mod pattern_scanner;
pub mod phase;
pub(crate) mod poly_lowering;
pub(crate) mod poly_modp_conv;
pub(crate) mod poly_result;
pub mod poly_store;
pub(crate) mod polynomial;
pub mod profile_cache;
pub mod profiler;
pub mod rationalize;
pub mod rationalize_policy;
pub(crate) mod recursion_guard;
pub mod rule;
pub mod rules;
pub(crate) mod semantic_equality;
pub mod semantics;
pub mod session;
pub mod session_snapshot;
pub mod session_state;
pub mod solve_safety;
pub mod solver;
pub mod step;
pub(crate) mod step_optimization;
pub(crate) mod strategies;
pub mod substitute;
pub mod target_kind;
pub mod telescoping;
pub mod timeline;
pub(crate) mod unipoly_modp;
pub mod visualizer;

// Property-based numeric tests for rewrite correctness
#[cfg(test)]
mod numeric_property_tests;

/// V2.1 Issue #4: Stable Public API
///
/// This module contains the stable, versioned API for external integrators.
/// Types and functions exported here are guaranteed to maintain backward
/// compatibility following semantic versioning.
pub mod api;

pub(crate) mod build;
pub mod error;
pub mod json; // Canonical JSON API types (PR-B)
pub mod nary;
pub mod visitors;
#[macro_use]
pub mod macros;

pub use budget::{Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats};
pub use engine::{
    eval_f64_checked, strip_all_holds, EquivalenceResult, EvalCheckedError, EvalCheckedOptions,
    Simplifier,
};
pub use error::{CasError, CasResult};
pub use eval::*;
// JSON API exports (canonical for CLI/FFI)
pub use assumptions::{
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionRecord, AssumptionReporting,
};
pub use domain::{can_cancel_factor, BlockedHint, CancelDecision, DomainMode, Proof};
pub use json::{
    eval_str_to_json, substitute_str_to_json, BudgetExceededJson, BudgetJsonInfo, BudgetOpts,
    EngineJsonError, EngineJsonResponse, EngineJsonStep, EngineJsonSubstep, EngineJsonWarning,
    JsonRunOptions, SpanJson, SubstituteJsonOptions, SubstituteJsonResponse, SCHEMA_VERSION,
};
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
pub use session::{resolve_session_refs, Entry, EntryId, EntryKind, ResolveError, SessionStore};
pub use session_state::SessionState;
pub use step::{DisplayEvalSteps, Step};
pub use visitors::{DepthVisitor, VariableCollector};
