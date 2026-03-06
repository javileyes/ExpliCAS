//! Centralized public re-exports for the solver facade.

pub use crate::algebra_command_eval::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message, evaluate_expand_wrapped_expression,
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message,
};
pub use crate::algebra_command_parse::{
    expand_log_usage_message, expand_usage_message, parse_expand_invocation_input,
    parse_expand_log_invocation_input, parse_telescope_invocation_input, telescope_usage_message,
    wrap_expand_eval_expression,
};
pub use crate::analysis_command_explain::{
    evaluate_explain_command_lines, evaluate_explain_command_message,
    evaluate_explain_invocation_message,
};
pub use crate::analysis_command_format_errors::{
    format_explain_command_error_message, format_timeline_command_error_message,
    format_visualize_command_error_message,
};
pub use crate::analysis_command_format_explain::format_explain_gcd_eval_lines;
pub use crate::analysis_command_parse::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_substitute_command_tail,
    extract_visualize_command_tail,
};
pub use crate::analysis_command_types::{
    ExplainCommandEvalError, ExplainGcdEvalOutput, VisualizeCommandOutput, VisualizeEvalError,
};
pub use crate::analysis_command_visualize::{
    evaluate_visualize_command_dot, evaluate_visualize_command_output,
    evaluate_visualize_invocation_output,
};
pub use crate::analysis_input_parse::{parse_expr_pair, ParseExprPairError};
pub use crate::assignment_apply::{apply_assignment_with_context, AssignmentApplyContext};
pub use crate::assignment_command::{
    evaluate_assignment_command_message_with, evaluate_assignment_command_with,
    evaluate_let_assignment_command_message_with, evaluate_let_assignment_command_with,
    format_assignment_command_output_message, AssignmentCommandOutput,
};
pub use crate::assignment_command_runtime::{
    evaluate_assignment_command_message_with_context, evaluate_assignment_command_with_context,
    evaluate_let_assignment_command_message_with_context,
    evaluate_let_assignment_command_with_context,
};
pub use crate::assignment_format::{
    format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message,
};
pub use crate::assignment_parse::{let_assignment_usage_message, parse_let_assignment_input};
pub use crate::assignment_types::{AssignmentError, LetAssignmentParseError, ParsedLetAssignment};
pub use crate::assumption_format::{
    collect_assumed_conditions_from_steps, format_assumed_conditions_report_lines,
    format_assumption_records_summary, format_blocked_hint_lines,
    format_diagnostics_requires_lines, format_displayable_assumption_lines,
    format_displayable_assumption_lines_for_step, format_displayable_assumption_lines_grouped,
    format_displayable_assumption_lines_grouped_for_step, format_domain_warning_lines,
    format_normalized_condition_lines, format_required_condition_lines,
    group_assumed_conditions_by_rule,
};
pub use crate::autoexpand_command_eval::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
};
pub use crate::autoexpand_command_format::{
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message,
};
pub use crate::autoexpand_command_parse::parse_autoexpand_command_input;
pub use crate::autoexpand_command_types::{
    AutoexpandBudgetView, AutoexpandCommandApplyOutput, AutoexpandCommandInput,
    AutoexpandCommandResult, AutoexpandCommandState,
};
pub use crate::bindings_command::{
    binding_overview_entries, clear_bindings_command, BindingsContext,
};
pub use crate::bindings_command_runtime::{
    evaluate_clear_bindings_command_lines, evaluate_vars_command_lines_from_bindings,
    evaluate_vars_command_lines_from_bindings_with_context,
};
pub use crate::bindings_format::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use crate::bindings_types::{BindingOverviewEntry, ClearBindingsResult};
pub use crate::blocked_hint_format::{
    filter_blocked_hints_for_eval, format_eval_blocked_hints_lines,
    format_solve_assumption_and_blocked_sections, SolveAssumptionSectionConfig,
};
pub use crate::config_command_apply::{
    evaluate_and_apply_config_command, ConfigCommandApplyContext, ConfigCommandApplyOutput,
};
pub use crate::config_command_eval::evaluate_config_command;
pub use crate::config_command_parse::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    format_simplifier_toggle_config, parse_config_command_input,
};
pub use crate::config_command_types::{ConfigCommandInput, ConfigCommandResult};
pub use crate::context_command_eval::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input,
};
pub use crate::context_command_format::{
    format_context_current_message, format_context_set_message, format_context_unknown_message,
};
pub use crate::context_command_parse::parse_context_command_input;
pub use crate::context_command_types::{
    ContextCommandApplyOutput, ContextCommandInput, ContextCommandResult,
};
pub use crate::domain_facade::{
    derive_requires_from_equation, domain_delta_check, infer_implicit_domain,
    pathsteps_to_expr_path,
};
pub use crate::equiv_command::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message, evaluate_equiv_invocation_message,
};
pub use crate::equiv_format::{
    format_equivalence_result_lines, format_expr_pair_parse_error_message,
};
pub use crate::eval_command_eval::evaluate_eval_command_output;
pub use crate::eval_command_render::build_eval_command_render_plan;
pub use crate::eval_command_text::evaluate_eval_text_simplify_with_session;
pub use crate::eval_command_types::{
    EvalCommandError, EvalCommandOutput, EvalCommandRenderPlan, EvalDisplayMessage,
    EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};
pub use crate::eval_json_command_runtime::evaluate_eval_json_with_session;
pub use crate::eval_json_input::{
    build_eval_json_request_for_input, EvalJsonNonSolveAction, EvalJsonPreparedRequest,
};
pub use crate::eval_output_adapters::{
    assumption_records_from_eval_output, blocked_hints_from_eval_output,
    diagnostics_from_eval_output, domain_warnings_from_eval_output, eval_output_view,
    output_scopes_from_eval_output, parsed_expr_from_eval_output,
    required_conditions_from_eval_output, resolved_expr_from_eval_output, result_from_eval_output,
    solve_steps_from_eval_output, steps_from_eval_output, stored_id_from_eval_output,
    EvalOutputView,
};
pub use crate::full_simplify_command::{
    evaluate_full_simplify_command_lines_with_resolver, extract_simplify_command_tail,
};
pub use crate::full_simplify_display::{format_full_simplify_eval_lines, FullSimplifyDisplayMode};
pub use crate::full_simplify_eval::{
    evaluate_full_simplify_input_with_resolver, format_full_simplify_eval_error_message,
    FullSimplifyEvalError, FullSimplifyEvalOutput,
};
pub use crate::health_command_eval::{evaluate_health_command, evaluate_health_status_lines};
pub use crate::health_command_format::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
pub use crate::health_command_messages::{
    capture_health_report_if_enabled, clear_health_profiler, health_clear_message,
    health_disable_message, health_enable_message,
};
pub use crate::health_command_parse::{evaluate_health_command_input, parse_health_command_input};
pub use crate::health_command_types::{
    HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
pub use crate::health_suite_format_catalog::{
    category_names as health_suite_category_names, list_cases as list_health_suite_cases,
};
pub use crate::health_suite_format_report::{
    count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
};
pub use crate::health_suite_runner::run_suite_filtered as run_health_suite_filtered;
pub use crate::health_suite_types::Category as HealthSuiteCategory;
pub use crate::history_command_display::{
    evaluate_history_command_lines, evaluate_history_command_lines_with_context,
};
pub use crate::history_command_runtime::{
    evaluate_history_command_lines_from_history,
    evaluate_history_command_lines_from_history_with_context,
};
pub use crate::history_delete::{
    delete_history_entries, evaluate_delete_history_command_message, HistoryDeleteContext,
};
pub use crate::history_format::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use crate::history_metadata_format::format_history_eval_metadata_sections;
pub use crate::history_overview::{
    history_overview_entries, HistoryEntryKindRaw, HistoryEntryRaw, HistoryOverviewContext,
};
pub use crate::history_parse::parse_history_ids;
pub use crate::history_show_format::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use crate::history_types::{
    DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
};
pub use crate::input_parse_common::{
    parse_statement_or_session_ref, rsplit_ignoring_parens, statement_to_expr_id,
};
pub use crate::inspect_format::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use crate::inspect_parse::parse_history_entry_id;
pub use crate::inspect_runtime::{
    inspect_history_entry, inspect_history_entry_input, HistoryInspectEntryRaw,
    InspectHistoryContext,
};
pub use crate::inspect_types::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use crate::json::{
    eval_str_to_json, eval_str_to_output_envelope, evaluate_envelope_json_command,
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
    substitute_str_to_json,
};
pub use crate::limit_command::evaluate_limit_command_lines;
pub use crate::limit_command_eval::{
    evaluate_limit_command_input, evaluate_limit_subcommand_output, format_limit_subcommand_error,
    parse_limit_command_input, LimitCommandEvalError, LimitCommandEvalOutput, LimitCommandInput,
    LimitSubcommandEvalError, LimitSubcommandEvalOutput,
};
pub use crate::limit_subcommand::{
    evaluate_limit_subcommand, LimitCommandApproach, LimitCommandPreSimplify, LimitSubcommandOutput,
};
pub use crate::linear_system::{
    solve_2x2_linear_system, solve_3x3_linear_system, solve_nxn_linear_system, LinSolveResult,
    LinearSystemError,
};
pub use crate::linear_system_command_entry::evaluate_linear_system_command_message;
pub use crate::linear_system_command_format::display_linear_system_solution;
pub use crate::linear_system_command_parse::parse_linear_system_invocation_input;
pub use crate::options_budget_eval::{
    apply_solve_budget_command, evaluate_solve_budget_command_message, SolveBudgetContext,
};
pub use crate::options_budget_format::format_solve_budget_command_message;
pub use crate::options_budget_types::SolveBudgetCommandResult;
pub use crate::output_clean::clean_result_output_line;
pub use crate::parse_error_render::{render_error_with_caret, render_parse_error};
pub use crate::path_rewrite::reconstruct_global_expr;
pub use crate::pipeline_display::{display_expr_or_poly, format_pipeline_stats};
pub use crate::profile_cache_command::{
    apply_profile_cache_command, evaluate_profile_cache_command_lines,
    format_profile_cache_command_lines, ProfileCacheCommandResult,
};
pub use crate::profile_command::{
    apply_profile_command, evaluate_profile_command_input, parse_profile_command_input,
    ProfileCommandInput, ProfileCommandResult,
};
pub use crate::prompt_display::build_prompt_from_eval_options;
pub use crate::rationalize_command::evaluate_rationalize_command_lines;
