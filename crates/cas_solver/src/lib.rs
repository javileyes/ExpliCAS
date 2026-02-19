//! Solver facade crate.
//!
//! During migration this crate re-exports the solver API from `cas_engine`.

pub mod json;
pub mod substitute;

pub use cas_engine::canonical_forms;
pub use cas_engine::normalize_and_dedupe_conditions;
pub use cas_engine::phase::{ExpandPolicy, SimplifyPhase};
pub use cas_engine::rationalize;
pub use cas_engine::rule::Rule;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
pub use cas_engine::solve_safety::*;
pub use cas_engine::solver::*;
pub use cas_engine::telescoping;
pub use cas_engine::visualizer;
pub use cas_engine::ConstFoldMode;
pub use cas_engine::ParentContext;
pub use cas_engine::{engine, error, expand, helpers, pattern_marks, phase, semantics};
pub use cas_engine::{limit, Approach, LimitOptions, PreSimplifyMode};
pub use cas_engine::{
    take_blocked_hints, AssumeScope, AssumptionKey, AssumptionReporting, AutoExpandBinomials,
    BlockedHint, BranchMode, Budget, CasError, ComplexMode, ContextMode, DomainMode, Engine,
    EquivalenceResult, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly,
    ImplicitCondition, PipelineStats, RequiresDisplayLevel, Simplifier, SimplifyOptions, StepsMode,
};
pub use cas_engine::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, substitute_str_to_json, EnvelopeEvalOptions,
};
pub use substitute::{substitute_power_aware, substitute_with_steps, SubstituteOptions};
