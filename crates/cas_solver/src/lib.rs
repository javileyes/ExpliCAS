//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod assumption_model;
mod assumption_types;
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
    assumption_records_from_engine, blocked_hint_suggestion, classify_assumption,
    collect_assumed_conditions_from_steps, collect_assumption_records,
    collect_assumption_records_from_iter, collect_blocked_hint_items,
    filter_blocked_hints_for_eval, format_assumed_conditions_report_lines,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_assumption_records_summary, format_blocked_hint_condition, format_blocked_hint_lines,
    format_blocked_simplifications_section_lines, format_diagnostics_requires_lines,
    format_displayable_assumption_lines, format_domain_warning_lines,
    format_eval_blocked_hints_lines, format_normalized_condition_lines,
    format_required_condition_lines, format_solve_assumption_and_blocked_sections,
    group_assumed_conditions_by_rule, group_blocked_hint_conditions_by_rule, AssumptionCollector,
    AssumptionEvent, AssumptionKey, AssumptionKind, SolveAssumptionSectionConfig,
};
pub use assumption_types::AssumptionRecord;
pub use cas_engine::error;
pub use cas_engine::expand;
pub use cas_engine::normalize_and_dedupe_conditions;
pub use cas_engine::normalize_condition;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::target_kind;
pub use cas_engine::ConstFoldMode;
pub use cas_engine::ConstFoldResult;
pub use cas_engine::DomainContext;
pub use cas_engine::DomainWarning;
pub use cas_engine::EvalConfig;
pub use cas_engine::ImportanceLevel;
pub use cas_engine::Orchestrator;
pub use cas_engine::ParentContext;
pub use cas_engine::Proof;
pub use cas_engine::Rewrite;
pub use cas_engine::Rule;
pub use cas_engine::SharedSemanticConfig;
pub use cas_engine::SimpleRule;
pub use cas_engine::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, clear_blocked_hints,
    expand_with_stats, fold_constants, infer_implicit_domain, is_zero, pathsteps_to_expr_path,
    register_blocked_hint, render_conditions_normalized, take_blocked_hints, to_display_steps,
    AssumeScope, AssumptionReporting, AutoExpandBinomials, BlockedHint, BranchMode, Budget,
    CasError, ComplexMode, ContextMode, DisplayEvalSteps, DomainMode, Engine, EquivalenceResult,
    EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, ImplicitCondition,
    Metric, Operation, PassStats, PathStep, PipelineStats, RequiresDisplayLevel, RuleProfiler,
    Simplifier, SimplifyOptions, Step, StepCategory, StepsMode,
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
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
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
pub use pipeline_display::{clean_result_output_line, display_expr_or_poly, format_pipeline_stats};
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
pub use types::{SolveCtx, SolveDomainEnv};

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

/// Backward-compatible facade for former `cas_engine::factor::*` imports.
pub mod factor {
    pub use cas_engine::factor::*;
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

/// Backward-compatible facade for former `cas_engine::ordering::*` imports.
pub mod ordering {
    pub use cas_engine::ordering::*;
}

/// Backward-compatible facade for former `cas_engine::nary::*` imports.
pub mod nary {
    pub use cas_engine::nary::*;
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
