// All clippy lints addressed: code fixes or local #[allow] attributes applied.
// See too_many_arguments allows in: inverse_trig.rs, step.rs
// See arc_with_non_send_sync allows in: profile_cache.rs

#[cfg(test)]
mod assumptions_tests;
pub(crate) mod best_so_far;
#[cfg(test)]
mod best_so_far_tests;
pub(crate) mod budget;
#[cfg(test)]
mod budget_tests;
mod cancel_runtime;
pub(crate) mod collect;
mod collect_by_var_support;
mod collect_focus_support;
mod collect_rule_support;
#[cfg(test)]
mod collect_tests;
pub(crate) mod const_fold;
pub(crate) mod diagnostics;
pub(crate) mod domain_oracle;
#[cfg(test)]
mod domain_oracle_tests;
pub(crate) mod engine;
pub(crate) mod eval;
#[cfg(test)]
mod events_tests;
pub(crate) mod expand;
#[cfg(test)]
mod expand_tests;
pub(crate) mod helpers;
pub(crate) mod implicit_domain;
mod integration_prep_support;
pub(crate) mod limits;
mod matrix_rule_support;
mod meta_functions_support;
pub(crate) mod options;
pub(crate) mod orchestrator;
pub(crate) mod parent_context;
#[cfg(test)]
mod parent_context_tests;
pub(crate) mod phase;
#[cfg(test)]
mod phase_tests;
mod poly_result_calls;
mod polynomial_identity_support;
pub(crate) mod profile_cache;
#[cfg(test)]
mod profile_cache_tests;
pub(crate) mod profiler;
#[cfg(test)]
mod profiler_tests;
pub(crate) mod recursion_guard;
#[cfg(test)]
mod recursion_guard_tests;
pub(crate) mod rule;
pub mod rules;
pub(crate) mod semantics;
#[cfg(test)]
mod semantics_tests;
mod solve_core_runtime;
mod solve_runtime_state_impl;
pub(crate) mod step;
#[cfg(test)]
mod step_tests;
mod symbolic_calculus_call_support;
pub(crate) mod telescoping;

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
#[cfg(test)]
mod error_tests;
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

pub use budget::{Budget, BudgetExceeded, BudgetScope, Metric, Operation, PassStats};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::substitute::{
    substitute_power_aware, substitute_with_trace, SubstituteOptions, SubstituteTraceResult,
    SubstituteTraceStep,
};
pub use cas_math::telescoping_dirichlet::try_dirichlet_kernel_identity as try_dirichlet_kernel_identity_pub;
pub use cas_solver_core::assumption_model::{
    collect_assumption_records, collect_assumption_records_from_iter, expr_fingerprint,
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionKind, AssumptionRecord,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::blocked_hint::BlockedHint;
pub use cas_solver_core::blocked_hint_store::{
    clear_blocked_hints, register_blocked_hint, take_blocked_hints,
};
pub use cas_solver_core::cycle_models::{CycleEvent, CycleLevel};
pub use cas_solver_core::domain_cancel_decision::CancelDecision;
pub use cas_solver_core::domain_facts_model::{
    predicate_condition_class, proof_to_strength, strength_to_proof, DomainFact, FactStrength,
    Predicate, Provenance,
};
pub use cas_solver_core::domain_gate::{can_apply_analytic, can_cancel_factor};
pub use cas_solver_core::domain_gate::{decide, decide_by_class};
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::domain_oracle_model::DomainOracle;
pub use cas_solver_core::domain_policy::mode_allows_predicate;
pub use cas_solver_core::domain_proof::Proof;
pub use cas_solver_core::engine_events::{EngineEvent, StepListener};
pub use cas_solver_core::eval_step_pipeline::to_display_eval_steps as to_display_steps;
pub use cas_solver_core::rationalize_policy::{
    AutoRationalizeLevel, RationalizeOutcome, RationalizeReason,
};
pub use cas_solver_core::solve_safety_policy::ConditionClass;
pub use cas_solver_core::solve_safety_policy::RequirementDescriptorKind as RequirementDescriptor;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use cas_solver_core::solve_safety_policy::SolveSafetyKind as SolveSafety;
pub use const_fold::{fold_constants, ConstFoldMode, ConstFoldResult};
pub use diagnostics::{Diagnostics, RequireOrigin, RequiredItem};
pub use domain_oracle::{oracle_allows_with_hint, StandardOracle};
pub use engine::{
    eval_f64, eval_f64_checked, strip_all_holds, substitute_expr_by_id, EquivalenceResult,
    EvalCheckedError, EvalCheckedOptions, LoopConfig, Simplifier,
};
pub use error::{CasError, CasResult};
pub use eval::*;
pub use expand::{
    eager_eval_expand_calls, estimate_expand_terms, expand, expand_div, expand_mul, expand_pow,
    expand_with_stats,
};
pub use helpers::{prove_nonzero, prove_positive};
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
pub use step::{
    pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step, StepCategory,
};
pub use telescoping::{telescope, TelescopingResult, TelescopingStep};
pub use visitors::{DepthVisitor, VariableCollector};

pub(crate) type SolverOptions = cas_solver_core::solver_options::SolverOptions;
pub(crate) type SolveCtx = cas_solver_core::solve_runtime_types::RuntimeSolveCtx;
pub(crate) type SolveStep = cas_solver_core::solve_runtime_types::RuntimeSolveStep;

// Equation-level primitives (not simplifier rules — used by solver pipeline)
pub use cancel_runtime::cancel_additive_terms_semantic;
pub use cas_solver_core::cancel_common_terms::{cancel_common_additive_terms, CancelResult};
