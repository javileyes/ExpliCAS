//! Re-exports of solver-facing command/runtime APIs for session clients.

pub use cas_solver::apply_assignment_with_context as apply_assignment;
pub use cas_solver::apply_autoexpand_command_on_runtime as apply_autoexpand_command_on_repl_core;
pub use cas_solver::apply_context_command_on_runtime as apply_context_command_on_repl_core;
pub use cas_solver::apply_profile_command_on_runtime as apply_profile_command_on_repl_core;
pub use cas_solver::apply_semantics_command_on_runtime as apply_semantics_command_on_repl_core;
pub use cas_solver::apply_set_command_plan_on_runtime as apply_set_command_plan_on_repl_core;
pub use cas_solver::apply_solve_budget_command;
pub use cas_solver::apply_steps_command_update_on_runtime as apply_steps_command_update_on_repl_core;
pub use cas_solver::build_repl_prompt_on_runtime as build_repl_prompt;
pub use cas_solver::clean_result_output_line;
pub use cas_solver::clear_repl_profile_cache_on_runtime as clear_repl_profile_cache;
pub use cas_solver::eval_options_from_runtime as eval_options_from_repl_core;
pub use cas_solver::evaluate_assignment_command_message_on_runtime as evaluate_assignment_command_message_on_repl_core;
pub use cas_solver::evaluate_assignment_command_message_with_context as evaluate_assignment_command_message_with_simplifier;
pub use cas_solver::evaluate_assignment_command_with_context as evaluate_assignment_command;
pub use cas_solver::evaluate_clear_bindings_command_lines as evaluate_clear_command_lines;
pub use cas_solver::evaluate_clear_command_lines_on_runtime as evaluate_clear_command_lines_on_repl_core;
pub use cas_solver::evaluate_delete_history_command_message;
pub use cas_solver::evaluate_delete_history_command_message_on_runtime as evaluate_delete_history_command_message_on_repl_core;
pub use cas_solver::evaluate_det_command_message_on_runtime as evaluate_det_command_message_on_repl_core;
pub use cas_solver::evaluate_equiv_invocation_message_on_runtime as evaluate_equiv_invocation_message_on_repl_core;
pub use cas_solver::evaluate_eval_command_render_plan_on_runtime as evaluate_eval_command_render_plan_on_repl_core;
pub use cas_solver::evaluate_expand_command_render_plan_on_runtime as evaluate_expand_command_render_plan_on_repl_core;
pub use cas_solver::evaluate_expand_log_invocation_message_on_runtime as evaluate_expand_log_invocation_message_on_repl_core;
pub use cas_solver::evaluate_explain_invocation_message_on_runtime as evaluate_explain_invocation_message_on_repl_core;
pub use cas_solver::evaluate_full_simplify_command_lines_on_runtime as evaluate_full_simplify_command_lines_on_repl_core;
pub use cas_solver::evaluate_health_command_message_on_runtime as evaluate_health_command_message_on_repl_core;
pub use cas_solver::evaluate_history_command_lines_from_history as evaluate_history_command_lines;
pub use cas_solver::evaluate_history_command_lines_from_history_with_context as evaluate_history_command_lines_with_context;
pub use cas_solver::evaluate_history_command_message_on_runtime as evaluate_history_command_message_on_repl_core;
pub use cas_solver::evaluate_let_assignment_command_message_on_runtime as evaluate_let_assignment_command_message_on_repl_core;
pub use cas_solver::evaluate_let_assignment_command_message_with_context as evaluate_let_assignment_command_message_with_simplifier;
pub use cas_solver::evaluate_let_assignment_command_with_context as evaluate_let_assignment_command;
pub use cas_solver::evaluate_linear_system_command_message;
pub use cas_solver::evaluate_linear_system_command_message_on_runtime as evaluate_linear_system_command_message_on_repl_core;
pub use cas_solver::evaluate_profile_cache_command_lines_on_runtime as evaluate_profile_cache_command_lines_on_repl_core;
pub use cas_solver::evaluate_profile_command_message_on_runtime as evaluate_profile_command_message_on_repl_core;
pub use cas_solver::evaluate_rationalize_command_lines;
pub use cas_solver::evaluate_rationalize_command_lines_on_runtime as evaluate_rationalize_command_lines_on_repl_core;
pub use cas_solver::evaluate_set_command_on_runtime as evaluate_set_command_on_repl_core;
pub use cas_solver::evaluate_show_command_lines;
pub use cas_solver::evaluate_show_command_lines_on_runtime as evaluate_show_command_lines_on_repl_core;
pub use cas_solver::evaluate_solve_budget_command_message;
pub use cas_solver::evaluate_solve_budget_command_message_on_runtime as evaluate_solve_budget_command_message_on_repl_core;
pub use cas_solver::evaluate_solve_command_message_on_runtime as evaluate_solve_command_message_on_repl_core;
pub use cas_solver::evaluate_substitute_invocation_user_message_on_runtime as evaluate_substitute_invocation_user_message_on_repl_core;
pub use cas_solver::evaluate_telescope_invocation_message_on_runtime as evaluate_telescope_invocation_message_on_repl_core;
pub use cas_solver::evaluate_timeline_command_with_session;
pub use cas_solver::evaluate_trace_command_message_on_runtime as evaluate_trace_command_message_on_repl_core;
pub use cas_solver::evaluate_transpose_command_message_on_runtime as evaluate_transpose_command_message_on_repl_core;
pub use cas_solver::evaluate_vars_command_lines_from_bindings as evaluate_vars_command_lines;
pub use cas_solver::evaluate_vars_command_lines_from_bindings_with_context as evaluate_vars_command_lines_with_context;
pub use cas_solver::evaluate_vars_command_message_on_runtime as evaluate_vars_command_message_on_repl_core;
pub use cas_solver::evaluate_visualize_invocation_output_on_runtime as evaluate_visualize_invocation_output_on_repl_core;
pub use cas_solver::evaluate_weierstrass_invocation_message_on_runtime as evaluate_weierstrass_invocation_message_on_repl_core;
pub use cas_solver::format_history_eval_metadata_sections;
pub use cas_solver::format_solve_command_eval_lines;
pub use cas_solver::profile_cache_len_on_runtime as profile_cache_len_on_repl_core;
pub use cas_solver::reset_repl_runtime_state_on_runtime as reset_repl_runtime_state;
pub use cas_solver::run_health_suite_filtered;
pub use cas_solver::set_command_state_for_runtime as set_command_state_for_repl_core;
pub use cas_solver::steps_command_state_for_runtime as steps_command_state_for_repl_core;
pub use cas_solver::update_health_report_on_runtime as update_health_report_on_repl_core;
pub use cas_solver::HealthSuiteCategory;
pub use cas_solver::ReplSemanticsApplyOutput;
pub use cas_solver::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message, parse_autoexpand_command_input, AutoexpandBudgetView,
    AutoexpandCommandApplyOutput, AutoexpandCommandInput, AutoexpandCommandResult,
    AutoexpandCommandState,
};
pub use cas_solver::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input, format_context_current_message, format_context_set_message,
    format_context_unknown_message, parse_context_command_input, ContextCommandApplyOutput,
    ContextCommandInput, ContextCommandResult,
};
pub use cas_solver::{
    apply_profile_cache_command, apply_profile_command, evaluate_profile_cache_command_lines,
    evaluate_profile_command_input, format_profile_cache_command_lines,
    parse_profile_command_input, ProfileCacheCommandResult, ProfileCommandInput,
    ProfileCommandResult,
};
pub use cas_solver::{
    apply_semantics_preset_by_name, apply_semantics_preset_by_name_to_options,
    apply_semantics_preset_state_to_options, evaluate_semantics_preset_args_to_options,
    find_semantics_preset, format_semantics_preset_application_lines,
    format_semantics_preset_help_lines, format_semantics_preset_list_lines,
    semantics_preset_state_from_options, semantics_presets, SemanticsPreset,
    SemanticsPresetApplication, SemanticsPresetApplyError, SemanticsPresetCommandOutput,
    SemanticsPresetState,
};
pub use cas_solver::{
    apply_semantics_set_args_to_options, apply_semantics_set_state_to_options,
    evaluate_semantics_set_args, evaluate_semantics_set_args_to_overview_lines,
    semantics_set_state_from_options, SemanticsSetState,
};
pub use cas_solver::{
    apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
    format_set_option_value, parse_set_command_input, SetCommandApplyEffects, SetCommandInput,
    SetCommandPlan, SetCommandResult, SetCommandState, SetDisplayMode,
};
pub use cas_solver::{apply_simplifier_toggle_config, build_simplifier_with_rule_config};
pub use cas_solver::{
    apply_steps_command_update, evaluate_steps_command_input, format_steps_current_message,
    format_steps_unknown_mode_message, parse_steps_command_input, StepsCommandApplyEffects,
    StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
};
pub use cas_solver::{binding_overview_entries, clear_bindings_command};
pub use cas_solver::{
    build_prompt_from_eval_options, parse_repl_command_input, preprocess_repl_function_syntax,
    split_repl_statements, ReplCommandInput,
};
pub use cas_solver::{
    capture_health_report_if_enabled, clear_health_profiler, health_clear_message,
    health_disable_message, health_enable_message,
};
pub use cas_solver::{
    collect_assumed_conditions_from_steps, filter_blocked_hints_for_eval,
    format_assumed_conditions_report_lines, format_assumption_records_summary,
    format_blocked_hint_lines, format_diagnostics_requires_lines,
    format_displayable_assumption_lines, format_domain_warning_lines,
    format_eval_blocked_hints_lines, format_normalized_condition_lines,
    format_required_condition_lines, format_solve_assumption_and_blocked_sections,
    group_assumed_conditions_by_rule, SolveAssumptionSectionConfig,
};
pub use cas_solver::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    evaluate_config_command, parse_config_command_input, ConfigCommandInput, ConfigCommandResult,
};
pub use cas_solver::{
    count_health_results, format_health_suite_report_filtered, health_suite_category_names,
    list_health_suite_cases,
};
pub use cas_solver::{delete_history_entries, history_overview_entries};
pub use cas_solver::{
    eval_str_to_json as evaluate_eval_json_canonical,
    substitute_str_to_json as evaluate_substitute_json_canonical,
};
pub use cas_solver::{evaluate_and_apply_config_command, ConfigCommandApplyOutput};
pub use cas_solver::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message, evaluate_equiv_invocation_message,
};
pub use cas_solver::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message, evaluate_expand_wrapped_expression,
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message, expand_log_usage_message, expand_usage_message,
    parse_expand_invocation_input, parse_expand_log_invocation_input,
    parse_telescope_invocation_input, telescope_usage_message, wrap_expand_eval_expression,
};
pub use cas_solver::{
    evaluate_explain_command_lines, evaluate_explain_command_message,
    evaluate_explain_invocation_message, evaluate_visualize_command_dot,
    evaluate_visualize_command_output, evaluate_visualize_invocation_output,
    format_explain_command_error_message, format_explain_gcd_eval_lines,
    format_timeline_command_error_message, format_visualize_command_error_message,
    ExplainCommandEvalError, ExplainGcdEvalOutput, VisualizeCommandOutput, VisualizeEvalError,
};
pub use cas_solver::{evaluate_health_command, evaluate_health_status_lines};
pub use cas_solver::{evaluate_health_command_input, parse_health_command_input};
pub use cas_solver::{
    evaluate_limit_command_lines, evaluate_limit_subcommand, LimitCommandApproach,
    LimitCommandPreSimplify, LimitSubcommandOutput,
};
pub use cas_solver::{
    evaluate_semantics_command_line, parse_semantics_command_input, SemanticsCommandInput,
    SemanticsCommandOutput,
};
pub use cas_solver::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode, SubstituteRenderMode,
};
pub use cas_solver::{
    evaluate_substitute_subcommand, evaluate_substitute_subcommand_json_canonical,
    SubstituteCommandMode, SubstituteSubcommandOutput,
};
pub use cas_solver::{
    evaluate_unary_command_lines, evaluate_unary_command_message,
    evaluate_unary_function_command_lines,
};
pub use cas_solver::{
    evaluate_weierstrass_command_lines, evaluate_weierstrass_invocation_lines,
    evaluate_weierstrass_invocation_message, format_solve_command_error_message,
    format_solve_prepare_error_message, format_verify_summary_lines,
    parse_weierstrass_invocation_input, weierstrass_usage_message,
};
pub use cas_solver::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_substitute_command_tail,
    extract_visualize_command_tail,
};
pub use cas_solver::{
    format_assignment_command_output_message, format_assignment_error_message,
    format_assignment_success_message, format_let_assignment_parse_error_message,
    let_assignment_usage_message, parse_let_assignment_input, AssignmentCommandOutput,
    AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use cas_solver::{
    format_binding_overview_lines, format_clear_bindings_result_lines, vars_empty_message,
};
pub use cas_solver::{
    format_delete_history_error_message, format_delete_history_result_message,
    format_history_overview_lines, history_empty_message,
};
pub use cas_solver::{format_equivalence_result_lines, format_expr_pair_parse_error_message};
pub use cas_solver::{
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_usage_message,
    resolve_health_category_filter,
};
pub use cas_solver::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
};
pub use cas_solver::{
    format_semantics_axis_lines, format_semantics_overview_lines,
    format_semantics_unknown_subcommand_message, semantics_help_message,
    semantics_view_state_from_options, SemanticsViewState,
};
pub use cas_solver::{
    format_show_history_command_lines, format_show_history_command_lines_with_context,
};
pub use cas_solver::{format_solve_budget_command_message, SolveBudgetCommandResult};
pub use cas_solver::{inspect_history_entry, inspect_history_entry_input};
pub use cas_solver::{parse_expr_pair, ParseExprPairError};
pub use cas_solver::{parse_history_entry_id, parse_history_ids};
pub use cas_solver::{render_error_with_caret, render_parse_error};
pub use cas_solver::{set_simplifier_toggle_rule, SimplifierRuleConfig, SimplifierToggleConfig};
pub use cas_solver::{BindingOverviewEntry, ClearBindingsResult};
pub use cas_solver::{
    DeleteHistoryError, DeleteHistoryResult, HistoryOverviewEntry, HistoryOverviewKind,
};
pub use cas_solver::{HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput};
pub use cas_solver::{
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use cas_solver::{ReplSetCommandOutput, ReplSetMessageKind};
pub use cas_solver::{
    SolveCommandEvalError, SolveCommandEvalOutput, SolveCommandInput, SolvePrepareError,
    TimelineCommandEvalError, TimelineCommandEvalOutput, TimelineCommandInput,
    TimelineSimplifyEvalError, TimelineSimplifyEvalOutput, TimelineSolveEvalError,
    TimelineSolveEvalOutput,
};
