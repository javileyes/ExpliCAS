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
mod assumption_format;
mod autoexpand_command_eval;
mod autoexpand_command_format;
mod autoexpand_command_parse;
#[cfg(test)]
mod autoexpand_command_tests;
mod bindings_command;
#[cfg(test)]
mod bindings_command_runtime_tests;
#[cfg(test)]
mod bindings_command_tests;
mod bindings_format;
mod blocked_hint_format;
mod cancel_runtime;
pub mod command_api;
mod config_command_apply;
mod config_command_eval;
mod config_command_parse;
#[cfg(test)]
mod config_command_tests;
mod const_fold_local;
mod context_command_eval;
mod context_command_format;
mod context_command_parse;
#[cfg(test)]
mod context_command_tests;
mod display_eval_steps;
mod domain_facade;
mod engine_event_display_steps;
mod equiv_command;
mod equiv_format;
mod eval_command_eval;
mod eval_command_format_metadata;
mod eval_command_format_result;
mod eval_command_render;
mod eval_command_request;
mod eval_command_runtime;
mod eval_command_text;
mod eval_input;
mod eval_input_special;
#[cfg(test)]
mod eval_input_tests;
mod eval_input_variable;
mod eval_option_axes;
#[cfg(test)]
mod eval_option_axes_tests;
mod eval_output_adapters;
mod eval_output_finalize;
mod eval_output_finalize_expr;
mod eval_output_finalize_input;
mod eval_output_finalize_nonexpr;
mod eval_output_finalize_wire;
mod eval_output_presentation;
mod eval_output_presentation_conditions;
mod eval_output_presentation_input;
mod eval_output_presentation_solution_display;
mod eval_output_presentation_solution_latex;
mod eval_output_presentation_solve_steps;
mod eval_output_stats;
mod eval_output_stats_format;
mod eval_output_stats_hash;
mod eval_output_stats_metrics;
#[cfg(test)]
mod eval_output_stats_tests;
mod eval_request_runtime;
mod full_simplify_command;
mod full_simplify_display;
mod full_simplify_eval;
mod health_command_eval;
mod health_command_format;
mod health_command_messages;
mod health_command_parse;
#[cfg(test)]
mod health_command_tests;
mod health_suite_catalog;
mod health_suite_catalog_core;
mod health_suite_catalog_stress;
mod health_suite_format_catalog;
mod health_suite_format_report;
mod health_suite_runner;
mod history_command_display;
#[cfg(test)]
mod history_command_display_tests;
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
mod input_parse_common;
mod inspect_format;
mod inspect_parse;
mod inspect_runtime;
mod limit_command;
mod limit_command_core;
#[cfg(test)]
mod limit_command_eval_tests;
mod limit_command_parse;
#[cfg(test)]
mod limit_command_tests;
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
#[cfg(test)]
mod linear_system_tests;
mod options_budget_eval;
#[cfg(test)]
mod options_budget_eval_tests;
mod options_budget_format;
mod output_clean;
#[cfg(test)]
mod output_clean_tests;
mod parse_error_render;
#[cfg(test)]
mod parse_error_render_tests;
#[cfg(test)]
mod path_rewrite_tests;
#[allow(dead_code)]
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
mod repl_command_parse;
mod repl_command_parse_early;
mod repl_command_parse_routing;
mod repl_command_preprocess;
#[cfg(test)]
mod repl_command_routing_tests;
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
mod repl_simplifier_runtime;
#[cfg(test)]
mod repl_simplifier_runtime_tests;
mod repl_solve_runtime;
mod repl_steps_runtime;
#[cfg(test)]
mod repl_steps_runtime_tests;
pub mod runtime;
#[path = "types/aliases.rs"]
mod runtime_aliases;
#[path = "types/session.rs"]
mod runtime_session;
mod semantics_command_eval;
mod semantics_command_parse;
#[cfg(test)]
mod semantics_command_tests;
#[cfg(test)]
mod semantics_display_tests;
mod semantics_preset_apply;
mod semantics_preset_catalog;
mod semantics_preset_format;
mod semantics_preset_labels;
#[cfg(test)]
mod semantics_presets_tests;
mod semantics_set_apply;
mod semantics_set_parse_apply;
mod semantics_set_parse_axis;
#[cfg(test)]
mod semantics_set_tests;
mod semantics_view_format;
mod semantics_view_format_axis;
mod semantics_view_format_help;
mod semantics_view_format_overview;
pub mod session_api;
mod set_command_apply;
mod set_command_eval;
mod set_command_format;
mod set_command_options;
mod set_command_options_rules;
mod set_command_options_steps;
mod set_command_parse;
#[cfg(test)]
mod set_command_tests;
mod show_command;
#[cfg(test)]
mod show_command_tests;
mod simplifier_setup_build;
mod simplifier_setup_toggle;
mod solution_display;
mod solve_backend_contract;
mod solve_backend_dispatch;
mod solve_backend_local;
mod solve_command_errors;
mod solve_command_eval_core;
mod solve_command_session_eval;
mod solve_core_runtime;
mod solve_display_lines;
mod solve_display_result;
mod solve_display_steps;
mod solve_event_steps;
mod solve_input_parse_parse;
mod solve_input_parse_prepare;
#[cfg(test)]
mod solve_input_parse_tests;
mod solve_render_config;
mod solve_safety;
mod solve_verify_display;
mod solver_entrypoints_eval;
mod solver_entrypoints_proof_verify;
mod solver_entrypoints_solve;
mod solver_number_theory;
#[path = "types/solver_options.rs"]
mod solver_options;
mod standard_oracle;
mod steps_command_eval;
mod steps_command_format;
mod steps_command_parse;
#[cfg(test)]
mod steps_command_tests;
mod substitute;
mod substitute_command_eval;
mod substitute_command_format;
mod substitute_command_parse;
#[cfg(test)]
mod substitute_command_tests;
mod substitute_subcommand_eval;
#[cfg(test)]
mod substitute_subcommand_tests;
mod substitute_subcommand_text;
mod substitute_subcommand_wire;
#[cfg(test)]
mod substitute_tests;
mod symbolic_transforms;
mod telescoping;
mod timeline_simplify_eval;
mod timeline_solve_eval;
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
pub mod wire;
#[cfg(test)]
mod wire_bridge_tests;

/// Backward-compatible facade for former `cas_engine::api::*` imports.
pub mod api;

#[allow(unused_imports)]
pub(crate) use crate::algebra_command_eval::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message, evaluate_expand_wrapped_expression,
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message,
};
#[allow(unused_imports)]
pub(crate) use crate::algebra_command_parse::{
    expand_log_usage_message, expand_usage_message, parse_expand_invocation_input,
    parse_expand_log_invocation_input, parse_telescope_invocation_input, telescope_usage_message,
    wrap_expand_eval_expression,
};
#[allow(unused_imports)]
pub(crate) use crate::analysis_command_explain::{
    evaluate_explain_command_lines, evaluate_explain_command_message,
    evaluate_explain_invocation_message,
};
#[allow(unused_imports)]
pub(crate) use crate::analysis_command_format_errors::{
    format_explain_command_error_message, format_timeline_command_error_message,
    format_visualize_command_error_message,
};
#[allow(unused_imports)]
pub(crate) use crate::analysis_command_format_explain::format_explain_gcd_eval_lines;
#[allow(unused_imports)]
pub(crate) use crate::analysis_command_parse::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_substitute_command_tail,
    extract_visualize_command_tail,
};
#[allow(unused_imports)]
pub(crate) use crate::analysis_command_visualize::{
    evaluate_visualize_command_dot, evaluate_visualize_command_output,
    evaluate_visualize_invocation_output,
};
#[allow(unused_imports)]
pub(crate) use crate::analysis_input_parse::parse_expr_pair;
pub(crate) use crate::api::{
    infer_solve_variable, solve_with_display_steps, telescope, ConditionClass, ConstFoldMode,
    ConstFoldResult, Diagnostics, DomainDelta, DomainWarning, ImplicitCondition, ImplicitDomain,
    Proof, RequireOrigin, RequiredItem, RequiresDisplayLevel, VerifyResult, VerifyStatus,
    VerifySummary,
};
#[allow(unused_imports)]
pub(crate) use crate::assignment_apply::{apply_assignment_with_context, AssignmentApplyContext};
#[allow(unused_imports)]
pub(crate) use crate::assignment_command::{
    evaluate_assignment_command_message_with, evaluate_assignment_command_with,
    evaluate_let_assignment_command_message_with, evaluate_let_assignment_command_with,
    format_assignment_command_output_message, AssignmentCommandOutput,
};
#[allow(unused_imports)]
pub(crate) use crate::assignment_command_runtime::{
    evaluate_assignment_command_message_with_context, evaluate_assignment_command_with_context,
    evaluate_let_assignment_command_message_with_context,
    evaluate_let_assignment_command_with_context,
};
#[allow(unused_imports)]
pub(crate) use crate::assignment_format::{
    format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message,
};
#[allow(unused_imports)]
pub(crate) use crate::assignment_parse::{
    let_assignment_usage_message, parse_let_assignment_input,
};
#[allow(unused_imports)]
pub(crate) use crate::assumption_format::format_assumption_records_summary;
#[allow(unused_imports)]
pub(crate) use crate::blocked_hint_format::{
    filter_blocked_hints_for_eval, format_eval_blocked_hints_lines,
    format_solve_assumption_and_blocked_sections, SolveAssumptionSectionConfig,
};
#[allow(unused_imports)]
pub(crate) use crate::domain_facade::{
    derive_requires_from_equation, domain_delta_check, infer_implicit_domain,
    pathsteps_to_expr_path,
};
#[allow(unused_imports)]
pub(crate) use crate::equiv_command::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message, evaluate_equiv_invocation_message,
};
#[allow(unused_imports)]
pub(crate) use crate::equiv_format::{
    format_equivalence_result_lines, format_expr_pair_parse_error_message,
};
#[allow(unused_imports)]
pub(crate) use crate::eval_output_adapters::{
    assumption_records_from_eval_output, blocked_hints_from_eval_output,
    diagnostics_from_eval_output, domain_warnings_from_eval_output, eval_output_view,
    output_scopes_from_eval_output, parsed_expr_from_eval_output, resolved_expr_from_eval_output,
    result_from_eval_output, solve_steps_from_eval_output, steps_from_eval_output,
    stored_id_from_eval_output, EvalOutputView,
};
#[allow(unused_imports)]
pub(crate) use crate::full_simplify_command::{
    evaluate_full_simplify_command_lines_with_resolver, extract_simplify_command_tail,
};
#[allow(unused_imports)]
pub(crate) use crate::full_simplify_display::{
    format_full_simplify_eval_lines, FullSimplifyDisplayMode,
};
#[allow(unused_imports)]
pub(crate) use crate::full_simplify_eval::{
    evaluate_full_simplify_input_with_resolver, format_full_simplify_eval_error_message,
    FullSimplifyEvalError, FullSimplifyEvalOutput,
};
#[allow(unused_imports)]
pub(crate) use crate::health_command_eval::{
    evaluate_health_command, evaluate_health_status_lines,
};
#[allow(unused_imports)]
pub(crate) use crate::health_command_format::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
#[allow(unused_imports)]
pub(crate) use crate::health_command_messages::{
    capture_health_report_if_enabled, clear_health_profiler, health_clear_message,
    health_disable_message, health_enable_message,
};
#[allow(unused_imports)]
pub(crate) use crate::health_command_parse::{
    evaluate_health_command_input, parse_health_command_input,
};
#[allow(unused_imports)]
pub(crate) use crate::health_suite_format_catalog::{
    category_names as health_suite_category_names, list_cases as list_health_suite_cases,
};
#[allow(unused_imports)]
pub(crate) use crate::health_suite_format_report::{
    count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
};
#[allow(unused_imports)]
pub(crate) use crate::health_suite_runner::run_suite_filtered as run_health_suite_filtered;
#[allow(unused_imports)]
pub(crate) use crate::history_command_display::{
    evaluate_history_command_lines, evaluate_history_command_lines_with_context,
};
#[allow(unused_imports)]
pub(crate) use crate::history_delete::{
    delete_history_entries, evaluate_delete_history_command_message, HistoryDeleteContext,
};
#[allow(unused_imports)]
pub(crate) use crate::history_format::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
#[allow(unused_imports)]
pub(crate) use crate::history_metadata_format::format_history_eval_metadata_sections;
#[allow(unused_imports)]
pub(crate) use crate::history_overview::{
    history_overview_entries, HistoryEntryKindRaw, HistoryEntryRaw, HistoryOverviewContext,
};
#[allow(unused_imports)]
pub(crate) use crate::history_parse::parse_history_ids;
#[allow(unused_imports)]
pub(crate) use crate::history_show_format::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
#[allow(unused_imports)]
pub(crate) use crate::input_parse_common::{
    parse_statement_or_session_ref, rsplit_ignoring_parens, statement_to_expr_id,
};
#[allow(unused_imports)]
pub(crate) use crate::inspect_format::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
#[allow(unused_imports)]
pub(crate) use crate::inspect_parse::parse_history_entry_id;
#[allow(unused_imports)]
pub(crate) use crate::limit_command_core::{
    evaluate_limit_command_input, evaluate_limit_subcommand_output, format_limit_subcommand_error,
};
#[allow(unused_imports)]
pub(crate) use crate::limit_command_parse::parse_limit_command_input;
#[allow(unused_imports)]
pub(crate) use crate::linear_system::{
    solve_2x2_linear_system, solve_3x3_linear_system, solve_nxn_linear_system, LinSolveResult,
    LinearSystemError,
};
#[allow(unused_imports)]
pub(crate) use crate::linear_system_command_entry::evaluate_linear_system_command_message;
#[allow(unused_imports)]
pub(crate) use crate::linear_system_command_format::display_linear_system_solution;
#[allow(unused_imports)]
pub(crate) use crate::linear_system_command_parse::parse_linear_system_invocation_input;
#[allow(unused_imports)]
pub(crate) use crate::output_clean::clean_result_output_line;
#[allow(unused_imports)]
pub(crate) use crate::parse_error_render::{render_error_with_caret, render_parse_error};
#[allow(unused_imports)]
pub(crate) use crate::pipeline_display::{display_expr_or_poly, format_pipeline_stats};
#[allow(unused_imports)]
pub(crate) use crate::rationalize_command::evaluate_rationalize_command_lines;
#[allow(unused_imports)]
pub(crate) use crate::repl_command_parse::parse_repl_command_input;
#[allow(unused_imports)]
pub(crate) use crate::repl_command_preprocess::{
    preprocess_repl_function_syntax, split_repl_statements,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_eval_runtime::{
    evaluate_eval_command_render_plan_on_runtime, evaluate_expand_command_render_plan_on_runtime,
    profile_cache_len_on_runtime, ReplEvalRuntimeContext,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_health_runtime::{
    evaluate_health_command_message_on_runtime, update_health_report_on_runtime,
    ReplHealthRuntimeContext,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_runtime_state::{
    build_repl_prompt_on_runtime, clear_repl_profile_cache_on_runtime, eval_options_from_runtime,
    reset_repl_runtime_state_on_runtime, ReplRuntimeStateContext,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_semantics_runtime::{
    apply_autoexpand_command_on_runtime, apply_context_command_on_runtime,
    apply_semantics_command_on_runtime, evaluate_autoexpand_command_on_runtime,
    evaluate_context_command_on_runtime, evaluate_semantics_command_on_runtime,
    ReplSemanticsApplyOutput, ReplSemanticsRuntimeContext,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_simplifier_runtime::{
    apply_profile_command_on_runtime, evaluate_det_command_message_on_runtime,
    evaluate_equiv_invocation_message_on_runtime,
    evaluate_expand_log_invocation_message_on_runtime,
    evaluate_explain_invocation_message_on_runtime,
    evaluate_linear_system_command_message_on_runtime, evaluate_profile_command_message_on_runtime,
    evaluate_rationalize_command_lines_on_runtime,
    evaluate_substitute_invocation_user_message_on_runtime,
    evaluate_telescope_invocation_message_on_runtime, evaluate_trace_command_message_on_runtime,
    evaluate_transpose_command_message_on_runtime, evaluate_visualize_invocation_output_on_runtime,
    evaluate_weierstrass_invocation_message_on_runtime, ReplSimplifierRuntimeContext,
};
#[allow(unused_imports)]
pub(crate) use crate::repl_solve_runtime::{
    evaluate_full_simplify_command_lines_on_runtime, evaluate_solve_command_message_on_runtime,
    ReplSolveRuntimeContext,
};
pub(crate) use crate::runtime::*;
#[allow(unused_imports)]
pub(crate) use crate::semantics_command_eval::evaluate_semantics_command_line;
#[allow(unused_imports)]
pub(crate) use crate::semantics_command_parse::parse_semantics_command_input;
#[allow(unused_imports)]
pub(crate) use crate::semantics_preset_apply::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    semantics_preset_state_from_options,
};
#[allow(unused_imports)]
pub(crate) use crate::semantics_preset_catalog::{find_semantics_preset, semantics_presets};
#[allow(unused_imports)]
pub(crate) use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
#[allow(unused_imports)]
pub(crate) use crate::semantics_set_apply::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args_to_overview_lines,
};
#[allow(unused_imports)]
pub(crate) use crate::semantics_set_parse_apply::evaluate_semantics_set_args;
#[allow(unused_imports)]
pub(crate) use crate::semantics_view_format::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
};
#[allow(unused_imports)]
pub(crate) use crate::session_api::timeline::{
    TimelineCommandEvalError, TimelineSimplifyEvalError, TimelineSimplifyEvalOutput,
    TimelineSolveEvalError, TimelineSolveEvalOutput,
};
#[allow(unused_imports)]
pub(crate) use crate::set_command_apply::apply_set_command_plan;
#[allow(unused_imports)]
pub(crate) use crate::set_command_eval::evaluate_set_command_input;
#[allow(unused_imports)]
pub(crate) use crate::set_command_format::{format_set_help_text, format_set_option_value};
#[allow(unused_imports)]
pub(crate) use crate::set_command_parse::parse_set_command_input;
#[allow(unused_imports)]
pub(crate) use crate::show_command::{
    evaluate_show_command_lines, evaluate_show_command_lines_with, ShowCommandContext,
};
#[allow(unused_imports)]
pub(crate) use crate::simplifier_setup_build::build_simplifier_with_rule_config;
#[allow(unused_imports)]
pub(crate) use crate::simplifier_setup_toggle::{
    apply_simplifier_toggle_config, set_simplifier_toggle_rule,
};
#[allow(unused_imports)]
pub(crate) use crate::solution_display::{display_solution_set, is_pure_residual_otherwise};
#[allow(unused_imports)]
pub(crate) use crate::solve_command_errors::format_solve_command_error_message;
#[allow(unused_imports)]
pub(crate) use crate::solve_command_eval_core::{
    evaluate_solve_command_with_session, SolveCommandEvalError,
};
#[allow(unused_imports)]
pub(crate) use crate::solve_command_session_eval::evaluate_solve_command_message_with_session;
#[allow(unused_imports)]
pub(crate) use crate::solve_display_lines::format_solve_command_eval_lines;
#[allow(unused_imports)]
pub(crate) use crate::solve_display_result::{
    format_solve_result_line, requires_result_expr_anchor,
};
#[allow(unused_imports)]
pub(crate) use crate::solve_display_steps::format_solve_steps_lines;
#[allow(unused_imports)]
pub(crate) use crate::solve_input_parse_parse::{
    parse_solve_command_input, parse_solve_invocation_check, parse_timeline_command_input,
};
#[allow(unused_imports)]
pub(crate) use crate::solve_input_parse_prepare::{
    prepare_solve_expr_and_var, prepare_timeline_solve_equation,
};
#[allow(unused_imports)]
pub(crate) use crate::solve_render_config::{
    solve_render_config_from_eval_options, SolveCommandRenderConfig, SolveDisplayMode,
};
#[allow(unused_imports)]
pub(crate) use crate::solve_verify_display::format_verify_summary_lines;
#[allow(unused_imports)]
pub(crate) use crate::steps_command_eval::{
    apply_steps_command_update, evaluate_steps_command_input,
};
#[allow(unused_imports)]
pub(crate) use crate::steps_command_format::{
    format_steps_collection_set_message, format_steps_current_message,
    format_steps_display_set_message, format_steps_unknown_mode_message,
};
#[allow(unused_imports)]
pub(crate) use crate::steps_command_parse::parse_steps_command_input;
#[allow(unused_imports)]
pub(crate) use crate::substitute::SubstituteStrategy;
#[allow(unused_imports)]
pub(crate) use crate::substitute_command_eval::evaluate_substitute_invocation_user_message;
#[allow(unused_imports)]
pub(crate) use crate::symbolic_transforms::{apply_weierstrass_recursive, expand_log_recursive};
#[allow(unused_imports)]
pub(crate) use crate::unary_command_eval::evaluate_unary_command_message;
#[allow(unused_imports)]
pub(crate) use crate::unary_display::format_unary_function_eval_lines;
#[allow(unused_imports)]
pub(crate) use crate::weierstrass_command::evaluate_weierstrass_invocation_message;
#[allow(unused_imports)]
pub(crate) use cas_session_core::eval::{EvalSession, EvalStore};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::analysis_command_types::ParseExprPairError;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::analysis_command_types::{
    ExplainCommandEvalError, ExplainGcdEvalOutput, VisualizeCommandOutput, VisualizeEvalError,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::assignment_command_types::{
    AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::assumption_model::AssumptionRecord;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::assumption_model::{
    blocked_hint_suggestion, collect_assumption_records, format_assumption_records_section_lines,
    format_blocked_hint_condition, format_blocked_simplifications_section_lines,
    group_blocked_hint_conditions_by_rule,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::assumption_reporting::AssumptionReporting;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::blocked_hint::BlockedHint;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::blocked_hint_store::{register_blocked_hint, take_blocked_hints};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::equivalence::EquivalenceResult;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::health_category::Category as HealthSuiteCategory;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::health_runtime::{
    HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::history_models::{
    DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::history_models::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::limit_command_types::{
    LimitCommandEvalError, LimitCommandEvalOutput, LimitCommandInput, LimitSubcommandEvalError,
    LimitSubcommandEvalOutput,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::repl_command_types::ReplCommandInput;
#[allow(unused_imports)]
pub(crate) use cas_solver_core::semantics_command_types::{
    SemanticsCommandInput, SemanticsCommandOutput,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::semantics_set_types::{
    semantics_set_state_from_options, SemanticsSetState,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::semantics_view_types::{
    semantics_view_state_from_options, SemanticsViewState,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::set_command_types::{
    SetCommandApplyEffects, SetCommandInput, SetCommandPlan, SetCommandResult, SetCommandState,
    SetDisplayMode,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::simplifier_config::{SimplifierRuleConfig, SimplifierToggleConfig};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::solve_command_types::{
    SolveCommandInput, SolvePrepareError, TimelineCommandInput,
};
#[allow(unused_imports)]
pub(crate) use cas_solver_core::steps_command_types::{
    StepsCommandApplyEffects, StepsCommandInput, StepsCommandResult, StepsCommandState,
    StepsDisplayMode,
};
