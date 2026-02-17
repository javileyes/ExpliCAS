//! Solver facade crate.
//!
//! During migration this crate re-exports the solver API from `cas_engine`.

pub use cas_engine::assumptions::AssumptionKey;
pub use cas_engine::canonical_forms;
pub use cas_engine::const_fold::ConstFoldMode;
pub use cas_engine::domain::take_blocked_hints;
pub use cas_engine::implicit_domain::{
    normalize_and_dedupe_conditions, ImplicitCondition, RequiresDisplayLevel,
};
pub use cas_engine::limits::{limit, Approach, LimitOptions, PreSimplifyMode};
pub use cas_engine::multipoly;
pub use cas_engine::options::{
    AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, EvalOptions, HeuristicPoly,
    StepsMode,
};
pub use cas_engine::parent_context::ParentContext;
pub use cas_engine::phase::{ExpandPolicy, SimplifyPhase};
pub use cas_engine::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_engine::rationalize;
pub use cas_engine::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_engine::rule::Rule;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
pub use cas_engine::solve_safety::*;
pub use cas_engine::solver::*;
pub use cas_engine::substitute::{
    substitute_power_aware, substitute_with_steps, SubstituteOptions,
};
pub use cas_engine::telescoping;
pub use cas_engine::visualizer;
pub use cas_engine::{
    substitute_str_to_json, AssumeScope, AssumptionReporting, BlockedHint, Budget, DomainMode,
    Engine, EquivalenceResult, EvalAction, EvalOutput, EvalRequest, EvalResult, PipelineStats,
    Simplifier, SimplifyOptions,
};
