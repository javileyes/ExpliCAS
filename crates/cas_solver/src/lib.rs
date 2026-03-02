//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod analysis;
mod assumption_model;
mod assumption_types;
mod autoexpand_command;
mod cache_command;
pub mod check;
mod command_routing;
mod config_command;
mod context_command;
mod domain_types;
mod eval_command;
mod eval_json_command;
mod general_help;
mod health_command;
mod health_suite;
mod help_command;
mod help_topics;
mod input_parse;
mod isolation;
pub mod json;
mod limit_command;
mod linear_system;
mod panic_guard;
mod path_rewrite;
mod pipeline_display;
mod profile_command;
mod prompt_display;
mod semantics_command;
mod semantics_display;
mod semantics_presets;
mod semantics_set;
mod set_command;
mod simplifier_setup;
mod solution_display;
mod solve;
mod solve_core;
mod solve_display;
mod solve_safety;
mod steps_command;
pub mod substitute;
mod substitute_command;
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

pub use analysis::{
    build_visualize_cli_output, evaluate_det_command_lines, evaluate_det_command_lines_with_engine,
    evaluate_equiv_command_lines, evaluate_equiv_command_lines_with_engine, evaluate_equiv_input,
    evaluate_expand_command_wrapped_line, evaluate_expand_log_command_lines,
    evaluate_expand_log_command_lines_with_engine, evaluate_expand_log_input,
    evaluate_explain_command_lines, evaluate_explain_command_lines_with_engine,
    evaluate_explain_gcd_input, evaluate_full_simplify_command_lines,
    evaluate_full_simplify_command_lines_for_display_mode, evaluate_full_simplify_input,
    evaluate_limit_command_input, evaluate_limit_command_lines, evaluate_rationalize_command_lines,
    evaluate_rationalize_command_lines_with_engine, evaluate_rationalize_input,
    evaluate_substitute_and_simplify_input, evaluate_substitute_command_lines,
    evaluate_substitute_command_lines_for_display_mode,
    evaluate_substitute_command_lines_for_display_mode_with_engine, evaluate_substitute_input,
    evaluate_telescope_command_lines, evaluate_telescope_command_lines_with_engine,
    evaluate_telescope_input, evaluate_timeline_command_input, evaluate_timeline_command_line,
    evaluate_timeline_command_line_with_session_options, evaluate_timeline_simplify_aggressive_input,
    evaluate_timeline_simplify_input, evaluate_trace_command_lines,
    evaluate_trace_command_lines_with_engine, evaluate_transpose_command_lines,
    evaluate_transpose_command_lines_with_engine, evaluate_unary_command_lines,
    evaluate_unary_function_input, evaluate_visualize_command_output,
    evaluate_visualize_command_output_with_engine, evaluate_visualize_input,
    evaluate_weierstrass_command_lines, evaluate_weierstrass_command_lines_with_engine,
    evaluate_weierstrass_input, expand_log_usage_message, expand_usage_message,
    extract_equiv_command_tail, extract_explain_command_tail, extract_limit_command_tail,
    extract_simplify_command_tail, extract_solve_command_tail, extract_substitute_command_tail,
    extract_timeline_command_tail, extract_unary_command_tail, extract_visualize_command_tail,
    format_equivalence_result_lines, format_eval_metadata_lines, format_eval_result_line,
    format_eval_stored_entry_line, format_expand_log_eval_lines, format_explain_error_message,
    format_explain_gcd_eval_lines, format_expr_pair_parse_error_message,
    format_full_simplify_eval_lines, format_limit_command_error_message,
    format_limit_command_eval_lines, format_rationalize_eval_error_message,
    format_rationalize_eval_lines, format_substitute_eval_lines,
    format_substitute_parse_error_message, format_telescope_eval_lines,
    format_timeline_command_error_message, format_timeline_eval_error_message,
    format_timeline_simplify_info_lines, format_transform_eval_error_message,
    format_unary_function_eval_error_message, format_unary_function_eval_lines,
    format_weierstrass_eval_lines, history_eval_metadata_section_labels, limit_usage_message,
    parse_expand_command_input, parse_expand_log_command_input, parse_rationalize_command_input,
    parse_telescope_command_input, parse_weierstrass_command_input, rationalize_usage_message,
    should_show_simplify_step, substitute_render_mode_from_display_mode,
    substitute_usage_message, telescope_usage_message, timeline_no_steps_message,
    timeline_open_hint_message, unary_render_config_for_display_mode,
    visualize_output_hint_lines, weierstrass_usage_message, wrap_expand_eval_expression,
    EvalMetadataConfig, EvalMetadataLines, EvalResultLine, ExpandCommandInput, ExpandLogCommandInput,
    ExpandLogEvalOutput, ExplainEvalError, ExplainGcdEvalOutput, FullSimplifyEvalError,
    FullSimplifyEvalOutput, LimitCommandEvalError, LimitCommandEvalOutput, RationalizeCommandInput,
    RationalizeEvalError, RationalizeEvalOutcome, RationalizeEvalOutput, SubstituteEvalOutput,
    SubstituteRenderMode, TelescopeCommandInput, TelescopeEvalOutput, TimelineCommandEvalError,
    TimelineCommandEvalOutput, TimelineEvalError, TimelineSimplifyCommandEvalOutput,
    TimelineSimplifyEvalOutput, TransformEvalError, UnaryFunctionEvalError,
    UnaryFunctionEvalOutput, UnaryFunctionRenderConfig, VisualizeCliOutput, VisualizeEvalOutput,
    WeierstrassCommandInput, WeierstrassEvalOutput,
};
pub use assumption_model::{
    assumption_records_from_engine, blocked_hint_suggestion, classify_assumption,
    collect_assumed_conditions_from_steps, collect_assumption_records,
    collect_assumption_records_from_iter, collect_blocked_hint_items,
    filter_blocked_hints_for_eval, format_assumed_conditions_report_lines,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_assumption_records_summary, format_blocked_hint_condition, format_blocked_hint_lines,
    format_blocked_simplifications_section_lines, format_diagnostics_requires_lines,
    format_displayable_assumption_lines, format_domain_warning_lines,
    format_eval_blocked_hints_lines, format_eval_metadata_sections,
    format_normalized_condition_lines, format_required_condition_lines,
    format_solve_assumption_and_blocked_sections, format_text_requires_lines,
    group_assumed_conditions_by_rule, group_blocked_hint_conditions_by_rule, AssumptionCollector,
    AssumptionEvent, AssumptionKey, AssumptionKind, EvalMetadataSectionLabels,
    SolveAssumptionSectionConfig,
};
pub use assumption_types::AssumptionRecord;
pub use autoexpand_command::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message, parse_autoexpand_command_input, AutoexpandBudgetView,
    AutoexpandCommandApplyOutput, AutoexpandCommandInput, AutoexpandCommandResult,
    AutoexpandCommandState,
};
pub use cache_command::{
    apply_profile_cache_command, clear_engine_profile_cache, engine_profile_cache_len,
    evaluate_profile_cache_command_lines, format_profile_cache_command_lines,
    ProfileCacheCommandResult,
};
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
pub use check::{
    format_verify_summary_lines, verify_solution, verify_solution_set, VerifyResult, VerifyStatus,
    VerifySummary,
};
pub use command_routing::{
    parse_repl_command_input, preprocess_repl_function_syntax, ReplCommandInput,
};
pub use config_command::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    evaluate_config_command, parse_config_command_input, ConfigCommandInput, ConfigCommandResult,
};
pub use context_command::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input, format_context_current_message, format_context_set_message,
    format_context_unknown_message, parse_context_command_input, ContextCommandApplyOutput,
    ContextCommandInput, ContextCommandResult,
};
pub use domain_types::{ConditionClass, Provenance};
pub use eval_command::{
    build_eval_command_render_plan, evaluate_eval_command_output,
    evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
    EvalCommandRenderPlan, EvalDisplayMessage, EvalDisplayMessageKind,
};
pub use eval_json_command::{evaluate_eval_json_command_with_session, EvalJsonCommandConfig};
pub use general_help::general_help_text;
pub use health_command::{
    capture_health_report_if_enabled, clear_health_profiler, evaluate_health_command,
    evaluate_health_command_input, evaluate_health_command_with_engine,
    evaluate_health_status_lines, format_health_failed_tests_warning_line,
    format_health_invalid_category_message, format_health_missing_category_arg_message,
    format_health_report_lines, format_health_status_running_message,
    format_health_usage_message, health_clear_message, health_disable_message,
    health_enable_message, health_usage_message, parse_health_command_input,
    resolve_health_category_filter, HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
pub use health_suite::{
    category_names as health_suite_category_names, count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
    list_cases as list_health_suite_cases, run_suite_filtered as run_health_suite_filtered,
    Category as HealthSuiteCategory,
};
pub use help_command::{parse_help_command_input, HelpCommandInput};
pub use help_topics::help_topic_text;
pub use input_parse::{
    build_simplify_eval_request_from_statement, parse_cache_command_input,
    parse_expr_or_equation_as_expr, parse_expr_pair, parse_limit_command_input,
    parse_statement_or_session_ref, parse_substitute_args, parse_timeline_command_input,
    rsplit_ignoring_parens, split_by_comma_ignoring_parens, split_repl_statements,
    statement_to_expr_id, CacheCommandInput, LimitCommandInput, ParseExprPairError,
    ParseSubstituteArgsError, TimelineCommandInput,
};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, substitute_str_to_json,
    substitute_str_to_json_with_options, EnvelopeEvalOptions,
};
pub use limit_command::{
    evaluate_limit_subcommand_output, format_limit_subcommand_error, LimitSubcommandError,
    LimitSubcommandOutput,
};
pub use linear_system::{
    display_linear_system_solution, evaluate_linear_system_command_input,
    evaluate_linear_system_command_line, evaluate_linear_system_command_line_with_engine,
    format_linear_system_command_error_message, format_linear_system_result_message,
    is_valid_linear_system_var, parse_linear_system_invocation_input, parse_linear_system_spec,
    solve_2x2_linear_system, solve_3x3_linear_system, solve_linear_system_spec,
    solve_nxn_linear_system, split_semicolon_top_level, LinSolveResult,
    LinearSystemCommandEvalError, LinearSystemCommandEvalOutput, LinearSystemError,
    LinearSystemInvocationInput, LinearSystemSpec, LinearSystemSpecError,
};
pub use panic_guard::{
    format_panic_report_message, format_user_panic_message, generate_short_error_id,
    panic_payload_to_message,
};
pub use path_rewrite::reconstruct_global_expr;
pub use pipeline_display::{clean_result_output_line, display_expr_or_poly, format_pipeline_stats};
pub use profile_command::{
    apply_profile_command, apply_profile_command_with_engine, evaluate_profile_command_input,
    parse_profile_command_input, ProfileCommandInput, ProfileCommandResult,
};
pub use prompt_display::build_prompt_from_eval_options;
pub use semantics_command::{
    evaluate_semantics_command_line, parse_semantics_command_input, SemanticsCommandInput,
    SemanticsCommandOutput,
};
pub use semantics_display::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
    semantics_view_state_from_options, SemanticsViewState,
};
pub use semantics_presets::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    find_semantics_preset, format_semantics_preset_application_lines,
    format_semantics_preset_help_lines, format_semantics_preset_list_lines,
    semantics_preset_state_from_options, semantics_presets, SemanticsPreset,
    SemanticsPresetApplication, SemanticsPresetApplyError, SemanticsPresetCommandOutput,
    SemanticsPresetState,
};
pub use semantics_set::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
    semantics_set_state_from_options, SemanticsSetState,
};
pub use set_command::{
    apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
    format_set_option_value, parse_set_command_input, SetCommandApplyEffects, SetCommandInput,
    SetCommandPlan, SetCommandResult, SetCommandState, SetDisplayMode,
};
pub use simplifier_setup::{
    apply_simplifier_toggle_config, apply_simplifier_toggle_config_to_engine,
    build_simplifier_with_rule_config, format_simplifier_toggle_config,
    rebuild_engine_simplifier_with_profile, rebuild_engine_simplifier_with_rule_config,
    set_simplifier_toggle_rule, SimplifierRuleConfig, SimplifierToggleConfig,
};
pub use solution_display::{display_interval, display_solution_set, is_pure_residual_otherwise};
pub use solve::{
    contains_var, evaluate_parsed_solve_command_input, evaluate_solve_command_input,
    evaluate_solve_command_lines, evaluate_solve_command_lines_with_options,
    evaluate_solve_command_lines_with_session_options, evaluate_solve_invocation_input,
    evaluate_timeline_solve_command_input, evaluate_timeline_solve_with_eval_options,
    format_solve_command_error_message, format_solve_prepare_error_message,
    format_timeline_solve_error_message, infer_solve_variable, parse_solve_command_input,
    parse_solve_invocation_input, prepare_solve_eval_request, prepare_timeline_solve_input, solve,
    solve_with_display_steps, verify_stats, DisplaySolveSteps, PreparedSolveRequest,
    PreparedTimelineSolve, SolveCommandEvalError, SolveCommandEvalOutput, SolveCommandInput,
    SolveDiagnostics, SolveInvocationEvalOutput, SolveInvocationInput, SolvePrepareError,
    SolveStep, SolveSubStep, SolverOptions, TimelineSolveEvalError, TimelineSolveEvalOutput,
};
pub use solve_display::{
    format_solve_command_eval_lines, format_solve_result_line, format_solve_steps_lines,
    format_timeline_solve_no_steps_message, format_timeline_solve_result_line,
    requires_result_expr_anchor, solve_render_config_from_eval_options,
    solve_step_verbosity_from_display_mode, SolveCommandRenderConfig, SolveStepVerbosity,
};
pub use solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use steps_command::{
    apply_steps_command_update, evaluate_steps_command_input, format_steps_collection_set_message,
    format_steps_current_message, format_steps_display_set_message,
    format_steps_unknown_mode_message, parse_steps_command_input, set_engine_steps_mode,
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};
pub use substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use substitute_command::{
    evaluate_substitute_subcommand_json, evaluate_substitute_subcommand_text_lines,
    evaluate_substitute_subcommand_text_lines_with_mode, format_substitute_subcommand_text_lines,
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
