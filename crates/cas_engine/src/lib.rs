// All clippy lints addressed: code fixes or local #[allow] attributes applied.
// See too_many_arguments allows in: inverse_trig.rs, gcd_zippel_modp.rs (×2), step.rs
// See arc_with_non_send_sync allows in: profile_cache.rs

pub mod assumptions;
pub mod auto_expand_scan;
pub mod budget;
pub mod canonical_forms;
pub mod collect;
pub mod const_eval;
pub mod const_fold;
pub mod cycle_detector;
pub mod didactic;
pub mod display_context;
pub mod domain;
pub mod engine;
pub mod env;
pub mod eval;
pub mod expand;
pub mod factor;
pub mod gcd_zippel_modp;
pub mod limits;
pub mod matrix;
pub mod modp;
pub mod mono;
pub mod multinomial_expand;
pub mod multipoly;
pub mod multipoly_modp;
pub mod options;
pub mod orchestrator;
pub mod ordering;
pub mod parent_context;
pub mod pattern_detection;
pub mod pattern_marks;
pub mod pattern_scanner;
pub mod phase;
pub mod poly_modp_conv;
pub mod polynomial;
pub mod profile_cache;
pub mod profiler;
pub mod rationalize;
pub mod rationalize_policy;
pub mod rule;
pub mod rules;
pub mod semantic_equality;
pub mod semantics;
pub mod session;
pub mod session_state;
pub mod solve_safety;
pub mod solver;
pub mod step;
pub mod step_optimization;
pub mod strategies;
pub mod substitute;
pub mod telescoping;
pub mod timeline;
pub mod timeline_templates;
pub mod unipoly_modp;
pub mod visualizer;

pub mod build;
pub mod error;
pub mod helpers;
pub mod json; // Canonical JSON API types (PR-B)
pub mod nary;
pub mod visitors;
#[macro_use]
pub mod macros;

pub use budget::{Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats};
pub use engine::{strip_all_holds, Simplifier};
pub use error::CasError;
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
pub use phase::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseStats, PipelineStats, SimplifyOptions,
    SimplifyPhase,
};
pub use rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome, RationalizeReason};
pub use rule::Rule;
pub use semantics::{
    AssumeScope, BranchPolicy, EvalConfig, InverseTrigPolicy, NormalFormGoal, ValueDomain,
};
pub use session::{resolve_session_refs, Entry, EntryId, EntryKind, ResolveError, SessionStore};
pub use session_state::SessionState;
pub use step::Step;
pub use visitors::{DepthVisitor, VariableCollector};
