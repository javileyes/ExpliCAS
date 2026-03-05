//! Solver facade crate.
//!
//! During migration this crate hosts the solver entry points while still
//! re-exporting selected `cas_engine` APIs for compatibility.

mod algebra_command_eval;
mod algebra_command_parse;
#[cfg(test)]
mod algebra_command_tests;
#[cfg(test)]
mod analysis_command_eval_tests;
mod analysis_command_explain;
mod analysis_command_format_errors;
mod analysis_command_format_explain;
#[cfg(test)]
mod analysis_command_format_tests;
mod analysis_command_parse;
#[cfg(test)]
mod analysis_command_parse_tests;
mod analysis_command_types;
mod analysis_command_visualize;
mod analysis_input_parse;
#[cfg(test)]
mod analysis_input_parse_tests;
mod assignment_apply;
#[cfg(test)]
mod assignment_apply_tests;
mod assignment_command;
mod assignment_command_runtime;
#[cfg(test)]
mod assignment_command_runtime_tests;
#[cfg(test)]
mod assignment_command_tests;
mod assignment_format;
mod assignment_parse;
mod assignment_types;
mod assumption_format;
mod autoexpand_command_eval;
mod autoexpand_command_format;
mod autoexpand_command_parse;
#[cfg(test)]
mod autoexpand_command_tests;
mod autoexpand_command_types;
mod bindings_command;
mod bindings_command_runtime;
#[cfg(test)]
mod bindings_command_runtime_tests;
#[cfg(test)]
mod bindings_command_tests;
mod bindings_format;
mod bindings_types;
mod blocked_hint_format;
mod config_command_apply;
mod config_command_eval;
mod config_command_parse;
#[cfg(test)]
mod config_command_tests;
mod config_command_types;
mod const_fold_local;
mod context_command_eval;
mod context_command_format;
mod context_command_parse;
#[cfg(test)]
mod context_command_tests;
mod context_command_types;
mod equiv_command;
mod equiv_format;
mod eval_command_eval;
mod eval_command_format;
mod eval_command_format_metadata;
mod eval_command_format_result;
mod eval_command_render;
mod eval_command_request;
mod eval_command_text;
mod eval_command_types;
mod eval_json_command_runtime;
mod eval_json_finalize;
mod eval_json_finalize_expr;
mod eval_json_finalize_input;
mod eval_json_finalize_nonexpr;
mod eval_json_finalize_wire;
mod eval_json_input;
mod eval_json_input_special;
#[cfg(test)]
mod eval_json_input_tests;
mod eval_json_input_variable;
mod eval_json_options;
#[cfg(test)]
mod eval_json_options_tests;
mod eval_json_presentation;
mod eval_json_presentation_conditions;
mod eval_json_presentation_solution;
mod eval_json_presentation_solution_display;
mod eval_json_presentation_solution_latex;
mod eval_json_presentation_solve;
mod eval_json_presentation_solve_input;
mod eval_json_presentation_solve_steps;
mod eval_json_stats;
mod eval_json_stats_format;
mod eval_json_stats_hash;
mod eval_json_stats_metrics;
#[cfg(test)]
mod eval_json_stats_tests;
mod eval_output_adapters;
mod full_simplify_command;
mod full_simplify_display;
mod full_simplify_eval;
mod health_command_eval;
mod health_command_format;
mod health_command_messages;
mod health_command_parse;
#[cfg(test)]
mod health_command_tests;
mod health_command_types;
mod health_suite_catalog;
mod health_suite_catalog_core;
mod health_suite_catalog_stress;
mod health_suite_format_catalog;
mod health_suite_format_report;
mod health_suite_runner;
mod health_suite_types;
mod history_command_display;
#[cfg(test)]
mod history_command_display_tests;
mod history_command_runtime;
#[cfg(test)]
mod history_command_runtime_tests;
mod history_delete;
#[cfg(test)]
mod history_delete_tests;
mod history_format;
mod history_metadata_format;
#[cfg(test)]
mod history_metadata_format_tests;
mod history_overview;
#[cfg(test)]
mod history_overview_tests;
mod history_parse;
mod history_show_format;
mod history_types;
mod input_parse_common;
mod inspect_format;
mod inspect_parse;
mod inspect_runtime;
mod inspect_types;
mod json;
#[cfg(test)]
mod json_bridge_tests;
mod limit_command;
mod limit_command_core;
mod limit_command_eval;
#[cfg(test)]
mod limit_command_eval_tests;
mod limit_command_parse;
#[cfg(test)]
mod limit_command_tests;
mod limit_command_types;
mod limit_subcommand;
#[cfg(test)]
mod limit_subcommand_tests;
mod linear_system;
mod linear_system_command_entry;
mod linear_system_command_eval;
mod linear_system_command_format;
mod linear_system_command_parse;
#[cfg(test)]
mod linear_system_command_tests;
mod linear_system_command_types;
#[cfg(test)]
mod linear_system_tests;
mod options_budget_eval;
#[cfg(test)]
mod options_budget_eval_tests;
mod options_budget_format;
mod options_budget_types;
mod output_clean;
#[cfg(test)]
mod output_clean_tests;
mod parse_error_render;
#[cfg(test)]
mod parse_error_render_tests;
mod path_rewrite;
#[cfg(test)]
mod path_rewrite_tests;
mod pipeline_display;
#[cfg(test)]
mod pipeline_display_tests;
mod profile_cache_command;
mod profile_command;
mod prompt_display;
#[cfg(test)]
mod prompt_display_tests;
mod rationalize_command;
mod rationalize_command_eval;
mod rationalize_command_format;
mod rationalize_command_parse;
#[cfg(test)]
mod rationalize_command_tests;
mod rationalize_command_types;
mod repl_command_parse;
mod repl_command_parse_early;
mod repl_command_parse_routing;
mod repl_command_preprocess;
#[cfg(test)]
mod repl_command_routing_tests;
mod repl_command_types;
mod repl_config_runtime;
#[cfg(test)]
mod repl_config_runtime_tests;
mod repl_eval_runtime;
#[cfg(test)]
mod repl_eval_runtime_tests;
mod repl_health_runtime;
#[cfg(test)]
mod repl_health_runtime_tests;
mod repl_runtime_configured;
#[cfg(test)]
mod repl_runtime_configured_tests;
mod repl_runtime_state;
#[cfg(test)]
mod repl_runtime_state_tests;
mod repl_semantics_runtime;
#[cfg(test)]
mod repl_semantics_runtime_tests;
mod repl_session_runtime;
mod repl_set_runtime;
#[cfg(test)]
mod repl_set_runtime_tests;
mod repl_set_types;
mod repl_simplifier_runtime;
#[cfg(test)]
mod repl_simplifier_runtime_tests;
mod repl_solve_runtime;
mod repl_steps_runtime;
#[cfg(test)]
mod repl_steps_runtime_tests;
mod semantics_command_eval;
mod semantics_command_parse;
#[cfg(test)]
mod semantics_command_tests;
mod semantics_command_types;
#[cfg(test)]
mod semantics_display_tests;
mod semantics_preset_apply;
mod semantics_preset_catalog;
mod semantics_preset_format;
mod semantics_preset_labels;
mod semantics_preset_types;
#[cfg(test)]
mod semantics_presets_tests;
mod semantics_set_apply;
mod semantics_set_parse;
mod semantics_set_parse_apply;
mod semantics_set_parse_axis;
#[cfg(test)]
mod semantics_set_tests;
mod semantics_set_types;
mod semantics_view_format;
mod semantics_view_format_axis;
mod semantics_view_format_help;
mod semantics_view_format_overview;
mod semantics_view_types;
mod set_command_apply;
mod set_command_eval;
mod set_command_format;
mod set_command_options;
mod set_command_options_rules;
mod set_command_options_steps;
mod set_command_parse;
#[cfg(test)]
mod set_command_tests;
mod set_command_types;
mod show_command;
#[cfg(test)]
mod show_command_tests;
mod simplifier_setup_build;
mod simplifier_setup_toggle;
mod simplifier_setup_types;
mod solution_display;
mod solve_backend;
mod solve_command_errors;
mod solve_command_eval_core;
mod solve_command_session_eval;
mod solve_display_lines;
mod solve_display_result;
mod solve_display_steps;
mod solve_input_parse_parse;
mod solve_input_parse_prepare;
#[cfg(test)]
mod solve_input_parse_tests;
mod solve_input_types;
mod solve_render_config;
mod solve_safety;
mod solve_verify_display;
mod steps_command_eval;
mod steps_command_format;
mod steps_command_parse;
#[cfg(test)]
mod steps_command_tests;
mod steps_command_types;
pub mod substitute;
mod substitute_command_eval;
mod substitute_command_format;
mod substitute_command_parse;
#[cfg(test)]
mod substitute_command_tests;
mod substitute_command_types;
mod substitute_subcommand_eval;
mod substitute_subcommand_json;
#[cfg(test)]
mod substitute_subcommand_tests;
mod substitute_subcommand_text;
mod substitute_subcommand_types;
#[cfg(test)]
mod substitute_tests;
mod symbolic_transforms;
mod telescoping;
mod timeline_command_eval;
mod timeline_simplify_eval;
mod timeline_solve_eval;
mod timeline_types;
mod types;
mod unary_command_eval;
#[cfg(test)]
mod unary_command_tests;
mod unary_display;
mod vars_command_display;
#[cfg(test)]
mod vars_command_display_tests;
mod weierstrass_command;
#[cfg(test)]
mod weierstrass_command_tests;

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

pub use algebra_command_eval::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message, evaluate_expand_wrapped_expression,
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message,
};
pub use algebra_command_parse::{
    expand_log_usage_message, expand_usage_message, parse_expand_invocation_input,
    parse_expand_log_invocation_input, parse_telescope_invocation_input, telescope_usage_message,
    wrap_expand_eval_expression,
};
pub use analysis_command_explain::{
    evaluate_explain_command_lines, evaluate_explain_command_message,
    evaluate_explain_invocation_message,
};
pub use analysis_command_format_errors::{
    format_explain_command_error_message, format_timeline_command_error_message,
    format_visualize_command_error_message,
};
pub use analysis_command_format_explain::format_explain_gcd_eval_lines;
pub use analysis_command_parse::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_substitute_command_tail,
    extract_visualize_command_tail,
};
pub use analysis_command_types::{
    ExplainCommandEvalError, ExplainGcdEvalOutput, VisualizeCommandOutput, VisualizeEvalError,
};
pub use analysis_command_visualize::{
    evaluate_visualize_command_dot, evaluate_visualize_command_output,
    evaluate_visualize_invocation_output,
};
pub use analysis_input_parse::{parse_expr_pair, ParseExprPairError};
pub use assignment_apply::{apply_assignment_with_context, AssignmentApplyContext};
pub use assignment_command::{
    evaluate_assignment_command_message_with, evaluate_assignment_command_with,
    evaluate_let_assignment_command_message_with, evaluate_let_assignment_command_with,
    format_assignment_command_output_message, AssignmentCommandOutput,
};
pub use assignment_command_runtime::{
    evaluate_assignment_command_message_with_context, evaluate_assignment_command_with_context,
    evaluate_let_assignment_command_message_with_context,
    evaluate_let_assignment_command_with_context,
};
pub use assignment_format::{
    format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message,
};
pub use assignment_parse::{let_assignment_usage_message, parse_let_assignment_input};
pub use assignment_types::{AssignmentError, LetAssignmentParseError, ParsedLetAssignment};
pub use assumption_format::{
    collect_assumed_conditions_from_steps, format_assumed_conditions_report_lines,
    format_assumption_records_summary, format_blocked_hint_lines,
    format_diagnostics_requires_lines, format_displayable_assumption_lines,
    format_displayable_assumption_lines_for_step, format_displayable_assumption_lines_grouped,
    format_displayable_assumption_lines_grouped_for_step, format_domain_warning_lines,
    format_normalized_condition_lines, format_required_condition_lines,
    group_assumed_conditions_by_rule,
};
pub use autoexpand_command_eval::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
};
pub use autoexpand_command_format::{
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message,
};
pub use autoexpand_command_parse::parse_autoexpand_command_input;
pub use autoexpand_command_types::{
    AutoexpandBudgetView, AutoexpandCommandApplyOutput, AutoexpandCommandInput,
    AutoexpandCommandResult, AutoexpandCommandState,
};
pub use bindings_command::{binding_overview_entries, clear_bindings_command, BindingsContext};
pub use bindings_command_runtime::{
    evaluate_clear_bindings_command_lines, evaluate_vars_command_lines_from_bindings,
    evaluate_vars_command_lines_from_bindings_with_context,
};
pub use bindings_format::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use bindings_types::{BindingOverviewEntry, ClearBindingsResult};
pub use blocked_hint_format::{
    filter_blocked_hints_for_eval, format_eval_blocked_hints_lines,
    format_solve_assumption_and_blocked_sections, SolveAssumptionSectionConfig,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_ast::target_kind;
pub use cas_engine::error;
pub use cas_engine::expand;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::ImportanceLevel;
pub use cas_engine::Orchestrator;
pub use cas_engine::ParentContext;
pub use cas_engine::Rewrite;
pub use cas_engine::Rule;
pub use cas_engine::SharedSemanticConfig;
pub use cas_engine::SimpleRule;
pub use cas_engine::{
    AutoExpandBinomials, BranchMode, Budget, CasError, ComplexMode, ContextMode, DisplayEvalSteps,
    Engine, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, Metric,
    Operation, PassStats, PathStep, PipelineStats, RuleProfiler, Simplifier, SimplifyOptions, Step,
    StepCategory, StepsMode,
};
pub use cas_engine::{BudgetExceeded, StandardOracle};
pub use cas_engine::{ExpandBudget, PhaseBudgets, PhaseMask, PhaseStats};
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
pub use cas_formatter::visualizer;
pub use cas_math::canonical_forms;
pub use cas_math::evaluator_f64::{
    eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions,
};
pub use cas_math::expr_nary::{add_terms_no_sign, add_terms_signed, Sign};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub use cas_math::factor::factor;
pub use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};
pub use cas_math::number_theory_support::GcdResult;
pub use cas_math::pattern_marks;
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use cas_math::rationalize::{rationalize_denominator, RationalizeConfig, RationalizeResult};
pub use cas_math::rationalize_policy::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_math::telescoping_dirichlet::{
    try_dirichlet_kernel_identity as try_dirichlet_kernel_identity_pub, DirichletKernelResult,
};
pub use cas_session_core::eval::{EvalSession, EvalStore};
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::assumption_model::AssumptionRecord;
pub use cas_solver_core::assumption_model::{
    assumption_condition_text, assumption_key_dedupe_fingerprint, blocked_hint_suggestion,
    collect_assumption_records, collect_assumption_records_from_iter, collect_blocked_hint_items,
    format_assumption_records_conditions, format_assumption_records_section_lines,
    format_blocked_hint_condition, format_blocked_simplifications_section_lines,
    group_blocked_hint_conditions_by_rule, AssumptionCollector, AssumptionEvent, AssumptionKey,
    AssumptionKind,
};
pub use cas_solver_core::assumption_reporting::AssumptionReporting;
pub use cas_solver_core::blocked_hint::BlockedHint;
pub use cas_solver_core::blocked_hint_store::{
    clear_blocked_hints, register_blocked_hint, take_blocked_hints,
};
pub use cas_solver_core::const_fold_types::{ConstFoldMode, ConstFoldResult};
pub use cas_solver_core::diagnostics_model::{Diagnostics, RequireOrigin, RequiredItem};
pub use cas_solver_core::domain_assumption_classification::classify_assumption;
pub use cas_solver_core::domain_condition::{
    filter_requires_for_display, ImplicitCondition, ImplicitDomain, RequiresDisplayLevel,
};
pub use cas_solver_core::domain_context::DomainContext;
pub use cas_solver_core::domain_facts_model::{DomainFact, FactStrength, Predicate};
pub use cas_solver_core::domain_inference::{AnalyticExpansionResult, DomainDelta};
pub use cas_solver_core::domain_inference_counter::{
    get as infer_domain_calls_get, reset as infer_domain_calls_reset,
};
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::domain_normalization::{
    normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
    render_conditions_normalized,
};
pub use cas_solver_core::domain_oracle_model::DomainOracle;
pub use cas_solver_core::domain_proof::Proof;
pub use cas_solver_core::domain_warning::DomainWarning;
pub use cas_solver_core::engine_events::{EngineEvent, StepListener};
pub use cas_solver_core::equivalence::EquivalenceResult;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::solve_budget::SolveBudget;
pub use cas_solver_core::solve_infer::infer_solve_variable;
pub use cas_solver_core::solve_safety_policy::ConditionClass;
pub use cas_solver_core::solve_safety_policy::ProvenanceKind as Provenance;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};
pub use cas_solver_core::verify_stats;
pub use cas_solver_core::{branch_policy::BranchPolicy, value_domain::ValueDomain};
pub use config_command_apply::{
    evaluate_and_apply_config_command, ConfigCommandApplyContext, ConfigCommandApplyOutput,
};
pub use config_command_eval::evaluate_config_command;
pub use config_command_parse::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    format_simplifier_toggle_config, parse_config_command_input,
};
pub use config_command_types::{ConfigCommandInput, ConfigCommandResult};
pub use context_command_eval::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input,
};
pub use context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
pub use context_command_parse::parse_context_command_input;
pub use context_command_types::{
    ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult,
};
pub use equiv_command::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message, evaluate_equiv_invocation_message,
};
pub use equiv_format::{format_equivalence_result_lines, format_expr_pair_parse_error_message};
pub use eval_command_eval::evaluate_eval_command_output;
pub use eval_command_render::build_eval_command_render_plan;
pub use eval_command_text::evaluate_eval_text_simplify_with_session;
pub use eval_command_types::{
    EvalCommandError, EvalCommandOutput, EvalCommandRenderPlan, EvalDisplayMessage,
    EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
pub use eval_json_command_runtime::evaluate_eval_json_with_session;
pub use eval_json_input::build_eval_request_for_input;
pub use eval_output_adapters::{
    assumption_records_from_eval_output, blocked_hints_from_eval_output,
    diagnostics_from_eval_output, domain_warnings_from_eval_output, eval_output_view,
    output_scopes_from_eval_output, parsed_expr_from_eval_output,
    required_conditions_from_eval_output, resolved_expr_from_eval_output, result_from_eval_output,
    solve_steps_from_eval_output, steps_from_eval_output, stored_id_from_eval_output,
    EvalOutputView,
};
pub use full_simplify_command::{
    evaluate_full_simplify_command_lines_with_resolver, extract_simplify_command_tail,
};
pub use full_simplify_display::{format_full_simplify_eval_lines, FullSimplifyDisplayMode};
pub use full_simplify_eval::{
    evaluate_full_simplify_input_with_resolver, format_full_simplify_eval_error_message,
    FullSimplifyEvalError, FullSimplifyEvalOutput,
};
pub use health_command_eval::{evaluate_health_command, evaluate_health_status_lines};
pub use health_command_format::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
pub use health_command_messages::{
    capture_health_report_if_enabled, clear_health_profiler, health_clear_message,
    health_disable_message, health_enable_message,
};
pub use health_command_parse::{evaluate_health_command_input, parse_health_command_input};
pub use health_command_types::{HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput};
pub use health_suite_format_catalog::{
    category_names as health_suite_category_names, list_cases as list_health_suite_cases,
};
pub use health_suite_format_report::{
    count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
};
pub use health_suite_runner::run_suite_filtered as run_health_suite_filtered;
pub use health_suite_types::Category as HealthSuiteCategory;
pub use history_command_display::{
    evaluate_history_command_lines, evaluate_history_command_lines_with_context,
};
pub use history_command_runtime::{
    evaluate_history_command_lines_from_history,
    evaluate_history_command_lines_from_history_with_context,
};
pub use history_delete::{
    delete_history_entries, evaluate_delete_history_command_message, HistoryDeleteContext,
};
pub use history_format::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use history_metadata_format::format_history_eval_metadata_sections;
pub use history_overview::{
    history_overview_entries, HistoryEntryKindRaw, HistoryEntryRaw, HistoryOverviewContext,
};
pub use history_parse::parse_history_ids;
pub use history_show_format::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use history_types::{
    DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
};
pub use input_parse_common::{
    parse_statement_or_session_ref, rsplit_ignoring_parens, statement_to_expr_id,
};
pub use inspect_format::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use inspect_parse::parse_history_entry_id;
pub use inspect_runtime::{
    inspect_history_entry, inspect_history_entry_input, HistoryInspectEntryRaw,
    InspectHistoryContext,
};
pub use inspect_types::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, evaluate_envelope_json_command,
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
    substitute_str_to_json,
};
pub use limit_command::evaluate_limit_command_lines;
pub use limit_command_eval::{
    evaluate_limit_command_input, evaluate_limit_subcommand_output, format_limit_subcommand_error,
    parse_limit_command_input, LimitCommandEvalError, LimitCommandEvalOutput, LimitCommandInput,
    LimitSubcommandEvalError, LimitSubcommandEvalOutput,
};
pub use limit_subcommand::{
    evaluate_limit_subcommand, LimitCommandApproach, LimitCommandPreSimplify, LimitSubcommandOutput,
};
pub use linear_system::{
    solve_2x2_linear_system, solve_3x3_linear_system, solve_nxn_linear_system, LinSolveResult,
    LinearSystemError,
};
pub use linear_system_command_entry::evaluate_linear_system_command_message;
pub use linear_system_command_format::display_linear_system_solution;
pub use linear_system_command_parse::parse_linear_system_invocation_input;
pub use options_budget_eval::{
    apply_solve_budget_command, evaluate_solve_budget_command_message, SolveBudgetContext,
};
pub use options_budget_format::format_solve_budget_command_message;
pub use options_budget_types::SolveBudgetCommandResult;
pub use output_clean::clean_result_output_line;
pub use parse_error_render::{render_error_with_caret, render_parse_error};
pub use path_rewrite::reconstruct_global_expr;
pub use pipeline_display::{display_expr_or_poly, format_pipeline_stats};
pub use profile_cache_command::{
    apply_profile_cache_command, evaluate_profile_cache_command_lines,
    format_profile_cache_command_lines, ProfileCacheCommandResult,
};
pub use profile_command::{
    apply_profile_command, evaluate_profile_command_input, parse_profile_command_input,
    ProfileCommandInput, ProfileCommandResult,
};
pub use prompt_display::build_prompt_from_eval_options;
pub use rationalize_command::evaluate_rationalize_command_lines;
pub use repl_command_parse::parse_repl_command_input;
pub use repl_command_preprocess::{preprocess_repl_function_syntax, split_repl_statements};
pub use repl_command_types::ReplCommandInput;
pub use repl_config_runtime::evaluate_and_apply_config_command_on_runtime;
pub use repl_eval_runtime::{
    evaluate_eval_command_render_plan_on_runtime, evaluate_expand_command_render_plan_on_runtime,
    profile_cache_len_on_runtime, ReplEvalRuntimeContext,
};
pub use repl_health_runtime::{
    evaluate_health_command_message_on_runtime, update_health_report_on_runtime,
    ReplHealthRuntimeContext,
};
pub use repl_runtime_configured::{
    build_runtime_with_config, reset_runtime_full_with_config, reset_runtime_with_config,
    ReplConfiguredRuntimeContext,
};
pub use repl_runtime_state::{
    build_repl_prompt_on_runtime, clear_repl_profile_cache_on_runtime, eval_options_from_runtime,
    reset_repl_runtime_state_on_runtime, ReplRuntimeStateContext,
};
pub use repl_semantics_runtime::{
    apply_autoexpand_command_on_runtime, apply_context_command_on_runtime,
    apply_semantics_command_on_runtime, evaluate_autoexpand_command_on_runtime,
    evaluate_autoexpand_command_with_config_sync_on_runtime, evaluate_context_command_on_runtime,
    evaluate_context_command_with_config_sync_on_runtime, evaluate_semantics_command_on_runtime,
    evaluate_semantics_command_with_config_sync_on_runtime, ReplSemanticsApplyOutput,
    ReplSemanticsRuntimeContext,
};
pub use repl_session_runtime::{
    evaluate_assignment_command_message_on_runtime, evaluate_clear_command_lines_on_runtime,
    evaluate_delete_history_command_message_on_runtime,
    evaluate_history_command_message_on_runtime,
    evaluate_let_assignment_command_message_on_runtime,
    evaluate_profile_cache_command_lines_on_runtime, evaluate_show_command_lines_on_runtime,
    evaluate_solve_budget_command_message_on_runtime, evaluate_vars_command_message_on_runtime,
    ReplSessionRuntimeContext,
};
pub use repl_set_runtime::{
    apply_set_command_plan_on_runtime, evaluate_set_command_on_runtime,
    set_command_state_for_runtime, ReplSetRuntimeContext,
};
pub use repl_set_types::{ReplSetCommandOutput, ReplSetMessageKind};
pub use repl_simplifier_runtime::{
    apply_profile_command_on_runtime, evaluate_det_command_message_on_runtime,
    evaluate_equiv_invocation_message_on_runtime,
    evaluate_expand_log_invocation_message_on_runtime,
    evaluate_explain_invocation_message_on_runtime,
    evaluate_linear_system_command_message_on_runtime, evaluate_profile_command_message_on_runtime,
    evaluate_rationalize_command_lines_on_runtime,
    evaluate_substitute_invocation_user_message_on_runtime,
    evaluate_telescope_invocation_message_on_runtime, evaluate_trace_command_message_on_runtime,
    evaluate_transpose_command_message_on_runtime, evaluate_unary_command_message_on_runtime,
    evaluate_visualize_invocation_output_on_runtime,
    evaluate_weierstrass_invocation_message_on_runtime, ReplSimplifierRuntimeContext,
};
pub use repl_solve_runtime::{
    evaluate_full_simplify_command_lines_on_runtime, evaluate_solve_command_message_on_runtime,
    ReplSolveRuntimeContext,
};
pub use repl_steps_runtime::{
    apply_steps_command_update_on_runtime, steps_command_state_for_runtime, ReplStepsRuntimeContext,
};
pub use semantics_command_eval::evaluate_semantics_command_line;
pub use semantics_command_parse::parse_semantics_command_input;
pub use semantics_command_types::{SemanticsCommandInput, SemanticsCommandOutput};
pub use semantics_preset_apply::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    semantics_preset_state_from_options,
};
pub use semantics_preset_catalog::{find_semantics_preset, semantics_presets};
pub use semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
pub use semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
pub use semantics_set_apply::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args_to_overview_lines,
};
pub use semantics_set_parse::evaluate_semantics_set_args;
pub use semantics_set_types::{semantics_set_state_from_options, SemanticsSetState};
pub use semantics_view_format::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
};
pub use semantics_view_types::{semantics_view_state_from_options, SemanticsViewState};
pub use set_command_apply::apply_set_command_plan;
pub use set_command_eval::evaluate_set_command_input;
pub use set_command_format::{format_set_help_text, format_set_option_value};
pub use set_command_parse::parse_set_command_input;
pub use set_command_types::{
    SetCommandApplyEffects, SetCommandInput, SetCommandPlan, SetCommandResult, SetCommandState,
    SetDisplayMode,
};
pub use show_command::{
    evaluate_show_command_lines, evaluate_show_command_lines_with, ShowCommandContext,
};
pub use simplifier_setup_build::build_simplifier_with_rule_config;
pub use simplifier_setup_toggle::apply_simplifier_toggle_config;
pub use simplifier_setup_types::{
    set_simplifier_toggle_rule, SimplifierRuleConfig, SimplifierToggleConfig,
};
pub use solution_display::{display_interval, display_solution_set, is_pure_residual_otherwise};
pub use solve_command_errors::{
    format_solve_command_error_message, format_solve_prepare_error_message,
};
pub use solve_command_eval_core::{
    evaluate_solve_command_with_session, prepare_solve_eval_request, PreparedSolveEvalRequest,
    SolveCommandEvalError, SolveCommandEvalOutput,
};
pub use solve_command_session_eval::{
    evaluate_solve_command_lines_with_session, evaluate_solve_command_message_with_session,
};
pub use solve_display_lines::format_solve_command_eval_lines;
pub use solve_display_result::{format_solve_result_line, requires_result_expr_anchor};
pub use solve_display_steps::format_solve_steps_lines;
pub use solve_input_parse_parse::{
    parse_solve_command_input, parse_solve_invocation_check, parse_timeline_command_input,
};
pub use solve_input_parse_prepare::{
    prepare_solve_expr_and_var, prepare_timeline_solve_equation, resolve_solve_var,
};
pub use solve_input_types::{SolveCommandInput, SolvePrepareError, TimelineCommandInput};
pub use solve_render_config::{
    solve_render_config_from_eval_options, SolveCommandRenderConfig, SolveDisplayMode,
};
pub use solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use solve_verify_display::format_verify_summary_lines;
pub use steps_command_eval::{apply_steps_command_update, evaluate_steps_command_input};
pub use steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
pub use steps_command_parse::parse_steps_command_input;
pub use steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};
pub use substitute::{
    detect_substitute_strategy, substitute_auto, substitute_auto_with_strategy,
    substitute_power_aware, substitute_with_steps, SubstituteOptions, SubstituteStrategy,
};
pub use substitute_command_eval::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
};
pub use substitute_command_format::{
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode,
};
pub use substitute_command_types::{
    SubstituteEvalOutput, SubstituteParseError, SubstituteRenderMode, SubstituteSimplifyEvalOutput,
};
pub use substitute_subcommand_eval::evaluate_substitute_subcommand;
pub use substitute_subcommand_json::evaluate_substitute_subcommand_json_canonical;
pub use substitute_subcommand_text::parse_substitute_json_text_lines;
pub use substitute_subcommand_types::{SubstituteCommandMode, SubstituteSubcommandOutput};
pub use symbolic_transforms::{apply_weierstrass_recursive, expand_log_recursive};
pub use telescoping::{telescope, TelescopingResult, TelescopingStep};
pub use timeline_command_eval::evaluate_timeline_command_with_session;
pub use timeline_types::{
    TimelineCommandEvalError, TimelineCommandEvalOutput, TimelineSimplifyEvalError,
    TimelineSimplifyEvalOutput, TimelineSolveEvalError, TimelineSolveEvalOutput,
};
pub use types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverOptions,
};
pub use unary_command_eval::{
    evaluate_unary_command_lines, evaluate_unary_command_message,
    evaluate_unary_function_command_lines,
};
pub use unary_display::format_unary_function_eval_lines;
pub use vars_command_display::{
    evaluate_vars_command_lines, evaluate_vars_command_lines_with_context,
};
pub use weierstrass_command::{
    evaluate_weierstrass_command_lines, evaluate_weierstrass_invocation_lines,
    evaluate_weierstrass_invocation_message, parse_weierstrass_invocation_input,
    weierstrass_usage_message,
};

/// Result shape for equation-level additive cancellation.
pub type CancelResult = cas_engine::CancelResult;

/// Result of symbolic limit evaluation from solver facade.
#[derive(Debug, Clone)]
pub struct LimitResult {
    /// The computed limit expression (or residual `limit(...)` when unresolved).
    pub expr: cas_ast::ExprId,
    /// Steps emitted by limit evaluation (when requested).
    pub steps: Vec<Step>,
    /// Warning emitted when limit cannot be determined safely.
    pub warning: Option<String>,
}

/// Solve an equation for a variable.
pub fn solve(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(cas_ast::SolutionSet, Vec<SolveStep>), CasError> {
    let ctx = SolveCtx::default();
    solve_backend::solve_with_engine_backend(
        eq,
        var,
        simplifier,
        SolverOptions::default().to_core(),
        &ctx,
    )
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(cas_ast::SolutionSet, DisplaySolveSteps, SolveDiagnostics), CasError> {
    let ctx = SolveCtx::default();
    let result =
        solve_backend::solve_with_engine_backend(eq, var, simplifier, opts.to_core(), &ctx);
    cas_solver_core::solve_types::finalize_display_solve_with_ctx(
        &ctx,
        result,
        crate::collect_assumption_records,
        |raw_steps| {
            cas_solver_core::solve_types::cleanup_display_solve_steps(
                &mut simplifier.context,
                raw_steps,
                opts.detailed_steps,
                var,
            )
        },
    )
}

/// Convert raw eval steps to display-ready, cleaned steps.
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    let cleaned = cas_solver_core::eval_step_pipeline::clean_eval_steps(
        raw_steps,
        |s: &Step| s.before,
        |s: &Step| s.after,
        |s: &Step| s.before_local(),
        |s: &Step| s.after_local(),
        |s: &Step| s.global_after,
        |s: &mut Step, gb| s.global_before = Some(gb),
    );
    DisplayEvalSteps(cleaned)
}

/// Expand with budget tracking, returning pass stats for charging.
pub fn expand_with_stats(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> (cas_ast::ExprId, PassStats) {
    let nodes_snap = ctx.stats().nodes_created;
    let estimated_terms = cas_math::expand_estimate::estimate_expand_terms(ctx, expr).unwrap_or(0);
    let result = expand(ctx, expr);
    let nodes_delta = ctx.stats().nodes_created.saturating_sub(nodes_snap);

    let stats = PassStats {
        op: Operation::Expand,
        rewrite_count: 0,
        nodes_delta,
        terms_materialized: estimated_terms,
        poly_ops: 0,
        stop_reason: None,
    };

    (result, stats)
}

/// Fold constants under the given semantic config and mode.
pub fn fold_constants(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    const_fold_local::fold_constants_local(ctx, expr, cfg, mode, budget)
}

/// Equation-level additive cancellation primitives are delegated to `cas_engine`
/// while solver extraction is in progress.
pub use cas_engine::{cancel_additive_terms_semantic, cancel_common_additive_terms};

/// Evaluate a symbolic limit with the engine's current limit evaluator.
pub fn limit(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    var: cas_ast::ExprId,
    approach: Approach,
    opts: &LimitOptions,
    _budget: &mut Budget,
) -> Result<LimitResult, CasError> {
    let outcome = cas_math::limits_support::eval_limit_at_infinity(ctx, expr, var, approach, opts);
    Ok(LimitResult {
        expr: outcome.expr,
        steps: Vec::new(),
        warning: outcome.warning,
    })
}

/// Attempt to prove that an expression is non-zero.
pub fn prove_nonzero(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Proof {
    prove_nonzero_depth(ctx, expr, 50)
}

/// Attempt to prove that an expression is strictly positive.
pub fn prove_positive(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    value_domain: ValueDomain,
) -> Proof {
    prove_positive_depth(ctx, expr, value_domain, 50)
}

fn prove_nonzero_depth(ctx: &cas_ast::Context, expr: cas_ast::ExprId, depth: usize) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| prove_positive(core_ctx, inner, ValueDomain::RealOnly),
        try_ground_nonzero_for_proofs,
    )
}

fn prove_positive_depth(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
    value_domain: ValueDomain,
    depth: usize,
) -> Proof {
    cas_solver_core::predicate_proofs::prove_positive_depth_with(
        ctx,
        expr,
        value_domain,
        depth,
        prove_nonzero_depth,
    )
}

fn try_ground_nonzero_for_proofs(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<Proof> {
    cas_math::ground_nonzero::try_ground_nonzero_with(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = Simplifier::with_context(source_ctx.clone());
            simplifier.set_collect_steps(false);

            let opts = SimplifyOptions {
                collect_steps: false,
                expand_mode: false,
                shared: SharedSemanticConfig {
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                budgets: PhaseBudgets {
                    core_iters: 4,
                    transform_iters: 2,
                    rationalize_iters: 0,
                    post_iters: 2,
                    max_total_rewrites: 50,
                },
                ..Default::default()
            };

            let (result, _, _) = simplifier.simplify_with_stats(source_expr, opts);
            Some((simplifier.context, result))
        },
        |evaluated_ctx, evaluated_expr| match evaluated_ctx.get(evaluated_expr) {
            cas_ast::Expr::Number(n) => {
                if num_traits::Zero::is_zero(n) {
                    Some(Proof::Disproven)
                } else {
                    Some(Proof::Proven)
                }
            }
            _ => None,
        },
        |evaluated_ctx, evaluated_expr| {
            let proof = prove_nonzero_depth(
                evaluated_ctx,
                evaluated_expr,
                8, // shallow depth budget for structural fallback
            );
            if proof == Proof::Proven || proof == Proof::Disproven {
                Some(proof)
            } else {
                None
            }
        },
    )
}

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solution: cas_ast::ExprId,
) -> VerifyStatus {
    cas_solver_core::verification_flow::verify_solution_with_domain_modes_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            cas_solver_core::verify_substitution::substitute_equation_diff(
                &mut state.context,
                eq,
                solve_var,
                candidate,
            )
        },
        |state, expr, domain_mode| {
            let opts = verify_simplify_options_for_domain(domain_mode);
            state.simplify_with_stats(expr, opts).0
        },
        |state, expr| cas_math::expr_predicates::contains_variable(&state.context, expr),
        |state, expr| fold_numeric_islands_for_verify(&mut state.context, expr),
        |state, expr| cas_solver_core::isolation_utils::is_numeric_zero(&state.context, expr),
        |state, expr| cas_formatter::render_expr(&state.context, expr),
    )
}

/// Verify an entire solution set against the source equation.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &cas_ast::Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
) -> VerifyResult {
    cas_solver_core::verification_flow::verify_solution_set_for_equation_with_state(
        simplifier,
        equation,
        var,
        solutions,
        verify_solution,
    )
}

fn verify_simplify_options_for_domain(domain_mode: DomainMode) -> SimplifyOptions {
    SimplifyOptions {
        shared: SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fold_numeric_islands_for_verify(
    ctx: &mut cas_ast::Context,
    root: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let fold_opts = SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode: DomainMode::Generic,
                value_domain: ValueDomain::RealOnly,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: PhaseBudgets {
            core_iters: 4,
            transform_iters: 2,
            rationalize_iters: 0,
            post_iters: 2,
            max_total_rewrites: 50,
        },
        ..Default::default()
    };

    cas_solver_core::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        cas_math::ground_eval_guard::GroundEvalGuard::enter,
        |src_ctx, id| {
            let mut tmp = Simplifier::with_context(src_ctx.clone());
            tmp.set_collect_steps(false);
            let (result, _, _) = tmp.simplify_with_stats(id, fold_opts.clone());
            Some((tmp.context, result))
        },
    )
}

/// Infer implicit domain constraints from expression structure.
pub fn infer_implicit_domain(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    vd: ValueDomain,
) -> ImplicitDomain {
    cas_solver_core::domain_inference_counter::inc();
    cas_solver_core::domain_inference::infer_implicit_domain(ctx, root, vd == ValueDomain::RealOnly)
}

/// Derive additional required conditions from equation equality.
pub fn derive_requires_from_equation(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    existing: &ImplicitDomain,
    vd: ValueDomain,
) -> Vec<ImplicitCondition> {
    cas_solver_core::domain_inference::derive_requires_from_equation(
        ctx,
        lhs,
        rhs,
        existing,
        vd == ValueDomain::RealOnly,
        |ctx, expr| prove_positive(ctx, expr, vd),
    )
}

/// Check if a rewrite would expand the domain by removing implicit constraints.
pub fn domain_delta_check(
    ctx: &cas_ast::Context,
    input: cas_ast::ExprId,
    output: cas_ast::ExprId,
    vd: ValueDomain,
) -> DomainDelta {
    cas_solver_core::domain_inference::domain_delta_check(ctx, input, output, |ctx, expr| {
        infer_implicit_domain(ctx, expr, vd)
    })
}

/// Convert solver path steps to a compact AST expression path.
pub fn pathsteps_to_expr_path(steps: &[PathStep]) -> cas_ast::ExprPath {
    steps.iter().map(PathStep::to_child_index).collect()
}

/// Number-theory helpers exposed by the solver facade without pulling engine rule modules.
pub mod number_theory {
    pub use cas_math::number_theory_support::{compute_gcd, explain_gcd, GcdResult};
}
