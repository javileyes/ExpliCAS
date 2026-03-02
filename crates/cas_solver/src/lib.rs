//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod assumption_model;
mod assumption_types;
mod blocked_hint;
pub mod check;
mod domain_types;
mod input_parse;
mod isolation;
pub mod json;
mod linear_system;
mod path_rewrite;
mod pipeline_display;
mod solution_display;
mod solve;
mod solve_core;
mod solve_safety;
pub mod substitute;
mod symbolic_transforms;
mod telescoping;
mod types;

/// Backward-compatible facade for former `cas_engine::strategies::substitute_expr` imports.
pub mod strategies {
    pub use cas_ast::substitute_expr_by_id as substitute_expr;
}

/// Backward-compatible facade for former `cas_engine::api::*` imports.
pub mod api {
    pub use cas_ast::{
        BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
    };
    pub use cas_formatter::{DisplayExpr, LaTeXExpr};
    pub use cas_solver_core::solve_budget::SolveBudget;

    pub use crate::{
        contains_var, infer_solve_variable, solve, solve_with_display_steps, verify_solution,
        verify_solution_set, verify_stats, DisplaySolveSteps, SolveDiagnostics, SolveStep,
        SolveSubStep, SolverOptions, VerifyResult, VerifyStatus, VerifySummary,
    };
}

pub use assumption_model::{
    assumption_events_from_engine, assumption_records_from_engine, blocked_hint_suggestion,
    classify_assumption, collect_assumption_records, collect_assumption_records_from_iter,
    collect_blocked_hint_items, format_assumption_records_conditions,
    format_assumption_records_section_lines, format_blocked_hint_condition,
    format_blocked_simplifications_section_lines, group_blocked_hint_conditions_by_rule,
    AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionKind,
};
pub use assumption_types::AssumptionRecord;
pub use blocked_hint::{
    blocked_hints_from_engine, clear_blocked_hints, register_blocked_hint, take_blocked_hints,
    BlockedHint,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_engine::error;
pub use cas_engine::expand;
pub use cas_engine::normalize_and_dedupe_conditions;
pub use cas_engine::normalize_condition;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::target_kind;
pub use cas_engine::ConstFoldMode;
pub use cas_engine::ConstFoldResult;
pub use cas_engine::Diagnostics;
pub use cas_engine::DomainContext;
pub use cas_engine::DomainWarning;
pub use cas_engine::EvalConfig;
pub use cas_engine::ImportanceLevel;
pub use cas_engine::Orchestrator;
pub use cas_engine::ParentContext;
pub use cas_engine::Proof;
pub use cas_engine::RequireOrigin;
pub use cas_engine::RequiredItem;
pub use cas_engine::Rewrite;
pub use cas_engine::Rule;
pub use cas_engine::SharedSemanticConfig;
pub use cas_engine::SimpleRule;
pub use cas_engine::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, expand_with_stats,
    fold_constants, infer_implicit_domain, is_zero, pathsteps_to_expr_path,
    render_conditions_normalized, to_display_steps, AssumeScope, AssumptionReporting,
    AutoExpandBinomials, BranchMode, Budget, CasError, ComplexMode, ContextMode, DisplayEvalSteps,
    DomainMode, Engine, EquivalenceResult, EvalAction, EvalOptions, EvalOutput, EvalRequest,
    EvalResult, HeuristicPoly, ImplicitCondition, Metric, Operation, PassStats, PathStep,
    PipelineStats, RequiresDisplayLevel, RuleProfiler, Simplifier, SimplifyOptions, Step,
    StepCategory, StepsMode,
};
pub use cas_engine::{
    derive_requires_from_equation, domain_delta_check, BudgetExceeded, DomainDelta, DomainFact,
    DomainOracle, FactStrength, ImplicitDomain, Predicate, StandardOracle,
};
pub use cas_engine::{eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions};
pub use cas_engine::{infer_domain_calls_get, infer_domain_calls_reset};
pub use cas_engine::{limit, Approach, LimitOptions, PreSimplifyMode};
pub use cas_engine::{prove_nonzero, prove_positive};
pub use cas_engine::{BranchPolicy, InverseTrigPolicy, ValueDomain};
pub use cas_engine::{ExpandBudget, PhaseBudgets, PhaseMask, PhaseStats};
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::expr_nary::{add_terms_no_sign, add_terms_signed, Sign};
pub use cas_math::factor::factor;
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_math::telescoping_dirichlet::{
    try_dirichlet_kernel_identity as try_dirichlet_kernel_identity_pub, DirichletKernelResult,
};
pub use cas_session_core::eval::{EvalSession, EvalStore};
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use check::{verify_solution, verify_solution_set, VerifyResult, VerifyStatus, VerifySummary};
pub use domain_types::{ConditionClass, Provenance};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, substitute_str_to_json,
    substitute_str_to_json_with_options, EnvelopeEvalOptions,
};
pub use linear_system::{
    solve_2x2_linear_system, solve_3x3_linear_system, solve_nxn_linear_system, LinSolveResult,
    LinearSystemError,
};
pub use path_rewrite::reconstruct_global_expr;
pub use pipeline_display::{display_expr_or_poly, format_pipeline_stats};
pub use solution_display::{display_interval, display_solution_set, is_pure_residual_otherwise};
pub use solve::{
    contains_var, evaluate_timeline_solve_with_eval_options, infer_solve_variable, solve,
    solve_with_display_steps, verify_stats, DisplaySolveSteps, SolveDiagnostics, SolvePrepareError,
    SolveStep, SolveSubStep, SolverOptions, TimelineSolveEvalError, TimelineSolveEvalOutput,
};
pub use solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use symbolic_transforms::{apply_weierstrass_recursive, expand_log_recursive};
pub use telescoping::{telescope, TelescopingResult, TelescopingStep};
pub use types::{SolveCtx, SolveDomainEnv};

/// Number-theory helpers exposed by the solver facade without pulling engine rule modules.
pub mod number_theory {
    pub use cas_math::number_theory_support::{compute_gcd, explain_gcd, GcdResult};
}
