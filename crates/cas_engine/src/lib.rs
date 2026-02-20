// All clippy lints addressed: code fixes or local #[allow] attributes applied.
// See too_many_arguments allows in: inverse_trig.rs, step.rs
// See arc_with_non_send_sync allows in: profile_cache.rs

pub(crate) mod assumptions;
pub(crate) mod best_so_far;
pub(crate) mod budget;
pub(crate) mod collect;
pub(crate) mod const_fold;
pub(crate) mod cycle_detector;
pub(crate) mod cycle_events;
pub(crate) mod diagnostics;
pub(crate) mod didactic;
pub(crate) mod domain;
pub(crate) mod domain_facts;
pub(crate) mod domain_oracle;
pub(crate) mod engine;
pub(crate) mod eval;
pub(crate) mod eval_step_pipeline;
pub(crate) mod expand;
pub(crate) mod helpers;
pub(crate) mod implicit_domain;
pub(crate) mod limits;
pub(crate) mod options;
pub(crate) mod orchestrator;
pub(crate) mod parent_context;
pub(crate) mod phase;
pub(crate) mod profile_cache;
pub(crate) mod profiler;
pub(crate) mod recursion_guard;
pub(crate) mod rule;
pub mod rules;
pub(crate) mod semantics;
pub(crate) mod solve_safety;
pub mod solver;
pub(crate) mod step;
pub(crate) mod step_optimization;
pub(crate) mod strategies;
pub(crate) mod telescoping;
pub(crate) mod timeline;

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
pub use cas_math::telescoping_dirichlet::DirichletKernelResult;

pub use cas_ast::ordering;
pub use cas_ast::target_kind;
pub(crate) use cas_ast::visitors;
pub use cas_formatter::visualizer;
pub(crate) use cas_math::pattern_scanner;

pub use assumptions::{
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionKind, AssumptionRecord,
    AssumptionReporting, ConditionClass,
};
pub use budget::{Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats};
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::rationalize_policy::{
    AutoRationalizeLevel, RationalizeOutcome, RationalizeReason,
};
pub use cas_math::substitute::{
    substitute_power_aware, substitute_with_trace, SubstituteOptions, SubstituteTraceResult,
    SubstituteTraceStep,
};
pub use const_fold::{fold_constants, ConstFoldMode, ConstFoldResult};
pub use cycle_events::{CycleEvent, CycleLevel};
pub use diagnostics::{Diagnostics, RequireOrigin, RequiredItem};
pub use didactic::{enrich_steps, get_standalone_substeps, EnrichedStep, SubStep};
pub use domain::{
    can_apply_analytic, can_cancel_factor, take_blocked_hints, BlockedHint, CancelDecision,
    DomainMode, Proof,
};
pub use domain_facts::{
    decide, decide_by_class, mode_allows_predicate, predicate_condition_class, proof_to_strength,
    strength_to_proof, DomainFact, DomainOracle, FactStrength, Predicate, Provenance,
};
pub use domain_oracle::{oracle_allows_with_hint, StandardOracle};
pub use engine::{
    eval_f64, eval_f64_checked, strip_all_holds, substitute_expr_by_id, EquivalenceResult,
    EvalCheckedError, EvalCheckedOptions, LoopConfig, Simplifier,
};
pub use error::{CasError, CasResult};
pub use eval::*;
pub use eval_step_pipeline::to_display_steps;
pub use expand::{
    eager_eval_expand_calls, estimate_expand_terms, expand, expand_div, expand_mul, expand_pow,
    expand_with_stats,
};
pub use helpers::{is_zero, prove_nonzero, prove_positive};
pub use implicit_domain::{
    check_analytic_expansion, classify_assumption, classify_assumptions_in_place,
    derive_requires_from_equation, domain_delta_check, expands_analytic_domain,
    expands_analytic_in_context, filter_requires_for_display, infer_domain_calls_get,
    infer_domain_calls_reset, infer_implicit_domain, normalize_and_dedupe_conditions,
    normalize_condition, normalize_condition_expr, render_conditions_normalized, witness_survives,
    witness_survives_in_context, AnalyticExpansionResult, DomainContext, DomainDelta,
    ImplicitCondition, ImplicitDomain, RequiresDisplayLevel, WitnessKind,
};
pub use limits::{limit, Approach, LimitOptions, LimitResult, PreSimplifyMode};
pub use options::{
    AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, EvalOptions, HeuristicPoly,
    StepsMode,
};
pub use orchestrator::Orchestrator;
pub use parent_context::ParentContext;
pub use phase::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseMask, PhaseStats, PipelineStats,
    SharedSemanticConfig, SimplifyOptions, SimplifyPhase,
};
pub use profile_cache::ProfileCache;
pub use profiler::{RuleProfiler, RuleStats};
pub use rule::{ChainedRewrite, Rewrite, Rule, SimpleRule, SoundnessLabel};
pub use semantics::{
    AssumeScope, BranchPolicy, EvalConfig, InverseTrigPolicy, NormalFormGoal, ValueDomain,
};
pub use solve_safety::{RequirementDescriptor, SimplifyPurpose, SolveSafety};
pub use step::{
    pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step, StepCategory,
};
pub use telescoping::{
    telescope, try_dirichlet_kernel_identity_pub, TelescopingResult, TelescopingStep,
};
pub use timeline::{html_escape, latex_escape, SolveTimelineHtml, TimelineHtml, VerbosityLevel};
pub use visitors::{DepthVisitor, VariableCollector};

// Equation-level primitives (not simplifier rules — used by solver pipeline)
pub use rules::cancel_common_terms::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult,
};
