//! Session-related components extracted from `cas_engine`.

use cas_solver::{Diagnostics, RequireOrigin, RequiredItem};

mod algebra_command;
mod algebra_command_eval;
mod algebra_command_parse;
#[cfg(test)]
mod algebra_command_tests;
mod analysis_command_equiv;
mod analysis_command_eval;
#[cfg(test)]
mod analysis_command_eval_tests;
mod analysis_command_explain;
mod analysis_command_format;
mod analysis_command_format_equivalence;
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
mod assignment;
mod assignment_command;
mod assignment_eval;
mod assignment_format;
mod assignment_parse;
#[cfg(test)]
mod assignment_tests;
mod assignment_types;
mod autoexpand_command;
mod autoexpand_command_eval;
mod autoexpand_command_format;
mod autoexpand_command_parse;
#[cfg(test)]
mod autoexpand_command_tests;
mod autoexpand_command_types;
mod bindings;
mod bindings_eval;
mod bindings_format;
#[cfg(test)]
mod bindings_tests;
mod bindings_types;
mod cache;
#[cfg(test)]
mod cache_tests;
mod commands;
#[cfg(test)]
mod commands_tests;
mod config;
mod config_command;
mod config_command_eval;
mod config_command_parse;
#[cfg(test)]
mod config_command_tests;
mod config_command_types;
#[cfg(test)]
mod config_tests;
mod context_command;
mod context_command_eval;
mod context_command_format;
mod context_command_parse;
#[cfg(test)]
mod context_command_tests;
mod context_command_types;
pub mod env;
#[cfg(test)]
mod envelope_json_command_tests;
mod eval_command;
mod eval_command_eval;
mod eval_command_format;
mod eval_command_format_metadata;
mod eval_command_format_result;
mod eval_command_render;
mod eval_command_request;
#[cfg(test)]
mod eval_command_tests;
mod eval_command_text;
mod eval_command_types;
mod eval_json_command;
mod eval_json_command_runtime;
#[cfg(test)]
mod eval_json_command_tests;
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
mod eval_text_command;
mod full_simplify_eval;
#[cfg(test)]
mod full_simplify_eval_tests;
mod full_simplify_render;
mod full_simplify_render_command;
mod full_simplify_render_steps;
#[cfg(test)]
mod full_simplify_render_tests;
mod health_command;
mod health_command_eval;
mod health_command_format;
mod health_command_messages;
mod health_command_parse;
#[cfg(test)]
mod health_command_tests;
mod health_command_types;
mod health_suite;
mod health_suite_catalog;
mod health_suite_catalog_core;
mod health_suite_catalog_stress;
mod health_suite_format;
mod health_suite_format_catalog;
mod health_suite_format_report;
mod health_suite_runner;
mod health_suite_types;
mod history;
mod history_eval;
mod history_format;
mod history_metadata_format;
#[cfg(test)]
mod history_metadata_format_tests;
#[cfg(test)]
mod history_tests;
mod history_types;
mod input_parse_common;
mod inspect;
mod inspect_eval;
mod inspect_format;
#[cfg(test)]
mod inspect_tests;
mod inspect_types;
mod json_bridge;
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
mod linear_system_command;
#[cfg(test)]
mod linear_system_command_tests;
mod linear_system_eval;
mod linear_system_format;
mod linear_system_parse;
mod linear_system_types;
mod options;
mod options_budget_eval;
mod options_budget_format;
mod options_budget_types;
#[cfg(test)]
mod options_tests;
mod output_clean;
#[cfg(test)]
mod output_clean_tests;
mod parse_error_render;
#[cfg(test)]
mod parse_error_render_tests;
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
mod repl_command_analysis_runtime;
mod repl_command_core_runtime;
mod repl_command_eval_runtime;
mod repl_command_parse;
mod repl_command_parse_early;
mod repl_command_parse_routing;
mod repl_command_preprocess;
mod repl_command_routing;
#[cfg(test)]
mod repl_command_routing_tests;
mod repl_command_runtime;
#[cfg(test)]
mod repl_command_runtime_tests;
mod repl_command_session_runtime;
mod repl_command_types;
mod repl_command_unary_runtime;
mod repl_config_runtime;
#[cfg(test)]
mod repl_config_runtime_tests;
mod repl_core;
mod repl_runtime;
mod repl_runtime_core;
#[cfg(test)]
mod repl_runtime_tests;
mod repl_semantics_runtime;
#[cfg(test)]
mod repl_semantics_runtime_tests;
mod repl_set_apply;
mod repl_set_eval;
mod repl_set_runtime;
#[cfg(test)]
mod repl_set_runtime_tests;
mod repl_set_types;
mod repl_steps_runtime;
#[cfg(test)]
mod repl_steps_runtime_tests;
mod semantics_command;
mod semantics_command_eval;
mod semantics_command_parse;
#[cfg(test)]
mod semantics_command_tests;
mod semantics_command_types;
mod semantics_display;
#[cfg(test)]
mod semantics_display_tests;
mod semantics_preset_apply;
mod semantics_preset_catalog;
mod semantics_preset_format;
mod semantics_preset_labels;
mod semantics_preset_types;
mod semantics_presets;
#[cfg(test)]
mod semantics_presets_tests;
mod semantics_set;
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
mod session_io;
mod session_state_command;
mod session_state_command_history;
mod session_state_command_show;
mod session_state_command_vars;
mod set_command;
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
mod simplifier_setup;
mod simplifier_setup_build;
#[cfg(test)]
mod simplifier_setup_tests;
mod simplifier_setup_toggle;
mod simplifier_setup_types;
mod snapshot;
mod snapshot_store_convert;
#[cfg(test)]
mod snapshot_tests;
mod solve_budget_command;
mod solve_command;
mod solve_command_eval;
mod solve_command_format;
mod solve_command_format_errors;
#[cfg(test)]
mod solve_command_format_tests;
mod solve_command_render;
mod solve_command_render_lines;
mod solve_command_render_result;
mod solve_command_render_steps;
#[cfg(test)]
mod solve_command_render_tests;
mod solve_command_render_types;
#[cfg(test)]
mod solve_command_tests;
mod solve_command_types;
mod solve_command_types_solve;
mod solve_command_types_timeline;
mod solve_command_weierstrass;
mod solve_input_parse;
mod solve_input_parse_parse;
mod solve_input_parse_prepare;
#[cfg(test)]
mod solve_input_parse_tests;
mod state;
mod state_bindings;
mod state_core;
mod state_eval_session;
mod state_eval_store;
mod state_history;
mod state_persistence;
mod state_resolution;
mod steps_command;
mod steps_command_eval;
mod steps_command_format;
mod steps_command_parse;
#[cfg(test)]
mod steps_command_tests;
mod steps_command_types;
mod substitute_command;
mod substitute_command_eval;
mod substitute_command_format;
mod substitute_command_parse;
#[cfg(test)]
mod substitute_command_tests;
mod substitute_command_types;
mod substitute_subcommand;
mod substitute_subcommand_eval;
mod substitute_subcommand_json;
#[cfg(test)]
mod substitute_subcommand_tests;
mod substitute_subcommand_text;
mod substitute_subcommand_types;
mod timeline_command;
mod timeline_command_eval;
mod timeline_command_simplify;
mod timeline_command_solve;
#[cfg(test)]
mod timeline_command_tests;
mod unary_command;
mod unary_command_eval;
#[cfg(test)]
mod unary_command_tests;

pub use algebra_command::{
    evaluate_expand_log_command_lines, evaluate_expand_log_invocation_lines,
    evaluate_expand_log_invocation_message, evaluate_expand_wrapped_expression,
    evaluate_telescope_command_lines, evaluate_telescope_invocation_lines,
    evaluate_telescope_invocation_message, expand_log_usage_message, expand_usage_message,
    parse_expand_invocation_input, parse_expand_log_invocation_input,
    parse_telescope_invocation_input, telescope_usage_message, wrap_expand_eval_expression,
};
pub use analysis_command_eval::{
    evaluate_equiv_command_lines, evaluate_equiv_command_message,
    evaluate_equiv_invocation_message, evaluate_explain_command_lines,
    evaluate_explain_command_message, evaluate_explain_invocation_message,
    evaluate_visualize_command_dot, evaluate_visualize_command_output,
    evaluate_visualize_invocation_output, ExplainCommandEvalError, ExplainGcdEvalOutput,
    VisualizeCommandOutput, VisualizeEvalError,
};
pub use analysis_command_format::{
    format_equivalence_result_lines, format_explain_command_error_message,
    format_explain_gcd_eval_lines, format_expr_pair_parse_error_message,
    format_timeline_command_error_message, format_visualize_command_error_message,
};
pub use analysis_command_parse::{
    extract_equiv_command_tail, extract_explain_command_tail, extract_substitute_command_tail,
    extract_visualize_command_tail,
};
pub use analysis_input_parse::{parse_expr_pair, ParseExprPairError};
pub use assignment::{
    apply_assignment, format_assignment_error_message, format_assignment_success_message,
    format_let_assignment_parse_error_message, let_assignment_usage_message,
    parse_let_assignment_input, AssignmentError, LetAssignmentParseError, ParsedLetAssignment,
};
pub use autoexpand_command::{
    apply_autoexpand_policy_to_options, autoexpand_budget_view_from_options,
    evaluate_and_apply_autoexpand_command, evaluate_autoexpand_command_input,
    format_autoexpand_current_message, format_autoexpand_set_message,
    format_autoexpand_unknown_mode_message, parse_autoexpand_command_input, AutoexpandBudgetView,
    AutoexpandCommandApplyOutput, AutoexpandCommandInput, AutoexpandCommandResult,
    AutoexpandCommandState,
};
pub use bindings::{
    binding_overview_entries, clear_bindings_command, format_binding_overview_lines,
    format_clear_bindings_result_lines, vars_empty_message, BindingOverviewEntry,
    ClearBindingsResult,
};
pub use cache::{CacheDomainMode, SimplifiedCache, SimplifyCacheKey};
pub use cas_solver::{
    collect_assumed_conditions_from_steps, filter_blocked_hints_for_eval,
    format_assumed_conditions_report_lines, format_assumption_records_summary,
    format_blocked_hint_lines, format_diagnostics_requires_lines,
    format_displayable_assumption_lines, format_domain_warning_lines,
    format_eval_blocked_hints_lines, format_normalized_condition_lines,
    format_required_condition_lines, format_solve_assumption_and_blocked_sections,
    group_assumed_conditions_by_rule, SolveAssumptionSectionConfig,
};
pub use commands::{
    apply_profile_cache_command, apply_profile_command, evaluate_assignment_command,
    evaluate_assignment_command_message_with_simplifier, evaluate_clear_command_lines,
    evaluate_delete_history_command_message, evaluate_history_command_lines,
    evaluate_history_command_lines_with_context, evaluate_let_assignment_command,
    evaluate_let_assignment_command_message_with_simplifier, evaluate_profile_cache_command_lines,
    evaluate_profile_command_input, evaluate_solve_budget_command_message,
    evaluate_vars_command_lines, evaluate_vars_command_lines_with_context,
    format_assignment_command_output_message, format_profile_cache_command_lines,
    format_show_history_command_lines, format_show_history_command_lines_with_context,
    parse_profile_command_input, AssignmentCommandOutput, ProfileCacheCommandResult,
    ProfileCommandInput, ProfileCommandResult,
};
pub use config::{
    apply_solver_toggle_to_cas_config, solver_rule_config_from_cas_config,
    solver_toggle_config_from_cas_config, CasConfig,
};
pub use config_command::{evaluate_and_apply_config_command, ConfigCommandApplyOutput};
pub use context_command::{
    apply_context_mode_to_options, evaluate_and_apply_context_command,
    evaluate_context_command_input, format_context_current_message, format_context_set_message,
    format_context_unknown_message, parse_context_command_input, ContextCommandApplyOutput,
    ContextCommandInput, ContextCommandResult,
};
pub use health_command::{
    capture_health_report_if_enabled, clear_health_profiler, evaluate_health_command,
    evaluate_health_command_input, evaluate_health_status_lines,
    format_health_failed_tests_warning_line, format_health_invalid_category_message,
    format_health_missing_category_arg_message, format_health_report_lines,
    format_health_status_running_message, format_health_usage_message, health_clear_message,
    health_disable_message, health_enable_message, health_usage_message,
    parse_health_command_input, resolve_health_category_filter, HealthCommandEvalOutput,
    HealthCommandInput, HealthStatusInput,
};
pub use health_suite::{
    category_names as health_suite_category_names, count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
    list_cases as list_health_suite_cases, run_suite_filtered as run_health_suite_filtered,
    Category as HealthSuiteCategory,
};
pub use history::{
    delete_history_entries, format_delete_history_error_message,
    format_delete_history_result_message, format_history_overview_lines, history_empty_message,
    history_overview_entries, parse_history_ids, DeleteHistoryError, DeleteHistoryResult,
    HistoryOverviewEntry, HistoryOverviewKind,
};
pub use history_metadata_format::format_history_eval_metadata_sections;
pub use inspect::{
    format_history_entry_inspection_lines, format_inspect_history_entry_error_message,
    inspect_history_entry, inspect_history_entry_input, parse_history_entry_id,
    HistoryEntryDetails, HistoryEntryInspection, HistoryExprInspection,
    InspectHistoryEntryInputError, ParseHistoryEntryIdError,
};
pub use json_bridge::{evaluate_eval_json_canonical, evaluate_substitute_json_canonical};
pub use limit_command::evaluate_limit_command_lines;
pub use limit_subcommand::{
    evaluate_limit_subcommand, LimitCommandApproach, LimitCommandPreSimplify, LimitSubcommandOutput,
};
pub use linear_system_command::evaluate_linear_system_command_message;
pub use options::{
    apply_solve_budget_command, format_solve_budget_command_message, SolveBudgetCommandResult,
};
pub use output_clean::clean_result_output_line;
pub use parse_error_render::{render_error_with_caret, render_parse_error};
pub use prompt_display::build_prompt_from_eval_options;
pub use rationalize_command::evaluate_rationalize_command_lines;
pub use repl_command_routing::{
    parse_repl_command_input, preprocess_repl_function_syntax, split_repl_statements,
    ReplCommandInput,
};
pub use repl_command_runtime::{
    evaluate_assignment_command_message_on_repl_core, evaluate_clear_command_lines_on_repl_core,
    evaluate_delete_history_command_message_on_repl_core,
    evaluate_det_command_message_on_repl_core, evaluate_equiv_invocation_message_on_repl_core,
    evaluate_eval_command_render_plan_on_repl_core,
    evaluate_expand_command_render_plan_on_repl_core,
    evaluate_expand_log_invocation_message_on_repl_core,
    evaluate_explain_invocation_message_on_repl_core,
    evaluate_full_simplify_command_lines_on_repl_core,
    evaluate_health_command_message_on_repl_core, evaluate_history_command_message_on_repl_core,
    evaluate_let_assignment_command_message_on_repl_core,
    evaluate_linear_system_command_message_on_repl_core,
    evaluate_profile_cache_command_lines_on_repl_core,
    evaluate_rationalize_command_lines_on_repl_core, evaluate_show_command_lines_on_repl_core,
    evaluate_solve_budget_command_message_on_repl_core,
    evaluate_solve_command_message_on_repl_core,
    evaluate_substitute_invocation_user_message_on_repl_core,
    evaluate_telescope_invocation_message_on_repl_core,
    evaluate_trace_command_message_on_repl_core, evaluate_transpose_command_message_on_repl_core,
    evaluate_vars_command_message_on_repl_core, evaluate_visualize_invocation_output_on_repl_core,
    evaluate_weierstrass_invocation_message_on_repl_core, profile_cache_len_on_repl_core,
    update_health_report_on_repl_core,
};
pub use repl_config_runtime::evaluate_and_apply_config_command_on_repl;
pub use repl_core::ReplCore;
pub use repl_runtime::{
    apply_profile_command_on_repl_core, build_repl_core_with_config, build_repl_prompt,
    clear_repl_profile_cache, eval_options_from_repl_core,
    evaluate_profile_command_message_on_repl_core, reset_repl_core_full_with_config,
    reset_repl_core_with_config, reset_repl_runtime_state,
};
pub use repl_semantics_runtime::{
    apply_autoexpand_command_on_repl_core, apply_context_command_on_repl_core,
    apply_semantics_command_on_repl_core, evaluate_autoexpand_command_on_repl,
    evaluate_context_command_on_repl, evaluate_semantics_command_on_repl, ReplSemanticsApplyOutput,
};
pub use repl_set_runtime::{apply_set_command_plan_on_repl_core, set_command_state_for_repl_core};
pub use repl_set_runtime::{
    evaluate_set_command_on_repl_core, ReplSetCommandOutput, ReplSetMessageKind,
};
pub use repl_steps_runtime::{
    apply_steps_command_update_on_repl_core, steps_command_state_for_repl_core,
};
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
pub use session_io::{
    load_or_new_session, run_with_domain_session, run_with_session, save_session,
};
pub use set_command::{
    apply_set_command_plan, evaluate_set_command_input, format_set_help_text,
    format_set_option_value, parse_set_command_input, SetCommandApplyEffects, SetCommandInput,
    SetCommandPlan, SetCommandResult, SetCommandState, SetDisplayMode,
};
pub use show_command::evaluate_show_command_lines;
pub use simplifier_setup::{
    apply_simplifier_toggle_config, build_simplifier_with_rule_config, set_simplifier_toggle_rule,
    sync_simplifier_with_cas_config, SimplifierRuleConfig, SimplifierToggleConfig,
};
pub use solve_command::{evaluate_solve_command_lines, evaluate_solve_command_message};
pub use solve_command_format::{
    evaluate_weierstrass_command_lines, evaluate_weierstrass_invocation_lines,
    evaluate_weierstrass_invocation_message, format_solve_command_error_message,
    format_solve_prepare_error_message, format_verify_summary_lines,
    parse_weierstrass_invocation_input, weierstrass_usage_message,
};
pub use solve_command_render::{
    format_solve_command_eval_lines, solve_render_config_from_eval_options,
    SolveCommandRenderConfig,
};
pub use solve_command_types::{
    PreparedSolveEvalRequest, SolveCommandEvalError, SolveCommandEvalOutput, SolveCommandInput,
    SolvePrepareError, TimelineCommandEvalError, TimelineCommandEvalOutput, TimelineCommandInput,
    TimelineSimplifyEvalError, TimelineSimplifyEvalOutput, TimelineSolveEvalError,
    TimelineSolveEvalOutput,
};
pub use steps_command::{
    apply_steps_command_update, evaluate_steps_command_input, format_steps_current_message,
    format_steps_unknown_mode_message, parse_steps_command_input, StepsCommandApplyEffects,
    StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
};
pub use substitute_command::{
    evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
    evaluate_substitute_invocation_message, evaluate_substitute_invocation_user_message,
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode, SubstituteRenderMode,
};
pub use substitute_subcommand::{
    evaluate_substitute_subcommand, evaluate_substitute_subcommand_json_canonical,
    SubstituteCommandMode, SubstituteSubcommandOutput,
};
pub use timeline_command::evaluate_timeline_command_with_session;
pub use unary_command::{
    evaluate_unary_command_lines, evaluate_unary_command_message,
    evaluate_unary_function_command_lines,
};
pub type CacheHitEntryId = u64;

pub type ResolvedExpr = cas_session_core::cache::ResolvedExpr<RequiredItem>;

pub type Entry = cas_session_core::store::Entry<Diagnostics, SimplifiedCache>;
pub type SessionStore = cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use cas_solver::evaluate_envelope_json_command;
pub use cas_solver::{
    BranchMode, ComplexMode, ContextMode, Engine, EvalOptions, ExpandPolicy, PipelineStats,
    SharedSemanticConfig, Simplifier, SimplifyOptions, Step, StepsMode,
};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use eval_command::{
    build_eval_command_render_plan, evaluate_eval_command_output,
    evaluate_eval_text_simplify_with_session, EvalCommandError, EvalCommandOutput,
    EvalCommandRenderPlan, EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines,
    EvalResultLine,
};
pub use eval_json_command::{
    evaluate_eval_json_command_pretty_with_session, evaluate_eval_json_command_with_session,
    EvalJsonCommandConfig,
};
pub use eval_text_command::evaluate_eval_text_command_with_session;
pub use full_simplify_eval::{
    evaluate_full_simplify_input, format_full_simplify_eval_error_message, FullSimplifyEvalError,
    FullSimplifyEvalOutput,
};
pub use full_simplify_render::{
    evaluate_full_simplify_command_lines, extract_simplify_command_tail,
    format_full_simplify_eval_lines,
};
pub use snapshot::SnapshotError;
pub use state::SessionState;

fn mode_entry_from_store_entry(
    entry: &Entry,
) -> cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem> {
    cas_session_core::resolve::ModeEntry {
        kind: entry.kind.clone(),
        requires: entry.diagnostics.requires.clone(),
        cache: entry
            .simplified
            .as_ref()
            .map(|cache| cas_session_core::resolve::ModeCacheEntry {
                key: cache.key.clone(),
                expr: cache.expr,
                requires: cache.requires.clone(),
            }),
    }
}

fn same_requirement(lhs: &RequiredItem, rhs: &RequiredItem) -> bool {
    lhs.cond == rhs.cond
}

fn mark_session_propagated(item: &mut RequiredItem) {
    item.merge_origin(RequireOrigin::SessionPropagated);
}

fn mode_resolve_config<'a>(
    mode: RefMode,
    cache_key: &'a SimplifyCacheKey,
    env: &'a Environment,
) -> cas_session_core::resolve::ModeResolveConfig<'a, SimplifyCacheKey> {
    cas_session_core::resolve::ModeResolveConfig {
        mode,
        cache_key,
        env,
    }
}

fn push_session_propagated_requirement(diagnostics: &mut Diagnostics, item: RequiredItem) {
    diagnostics.push_required(item.cond, RequireOrigin::SessionPropagated);
}

fn with_mode_resolution_plumbing<T, F>(store: &SessionStore, run: F) -> T
where
    F: FnOnce(
        &mut dyn FnMut(
            EntryId,
        ) -> Option<
            cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem>,
        >,
        &mut dyn FnMut(&RequiredItem, &RequiredItem) -> bool,
        &mut dyn FnMut(&mut RequiredItem),
    ) -> T,
{
    let mut lookup = |id: EntryId| store.get(id).map(mode_entry_from_store_entry);
    let mut same = same_requirement;
    let mut mark = mark_session_propagated;
    run(&mut lookup, &mut same, &mut mark)
}

pub(crate) fn simplify_cache_steps_len(cache: &SimplifiedCache) -> usize {
    cache.steps.as_ref().map(|s| s.len()).unwrap_or(0)
}

pub(crate) fn apply_simplified_light_cache(
    mut cache: SimplifiedCache,
    light_cache_threshold: Option<usize>,
) -> SimplifiedCache {
    if let Some(threshold) = light_cache_threshold {
        if simplify_cache_steps_len(&cache) > threshold {
            cache.steps = None;
        }
    }
    cache
}

pub(crate) fn session_store_with_cache_config(cache_config: CacheConfig) -> SessionStore {
    SessionStore::with_cache_config_and_policy(
        cache_config,
        simplify_cache_steps_len,
        apply_simplified_light_cache,
    )
}

/// Resolve all `Expr::SessionRef` in an expression tree.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup(ctx, expr, &mut lookup)
}

/// Resolve session refs and apply environment substitution.
pub fn resolve_session_refs_with_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_all_with_lookup_and_env(ctx, expr, &mut lookup, env)
}

/// Resolve session refs and accumulate inherited diagnostics.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<(cas_ast::ExprId, Diagnostics), ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup_accumulator(
        ctx,
        expr,
        &mut lookup,
        Diagnostics::new(),
        |inherited, id| {
            if let Some(entry) = store.get(id) {
                inherited.inherit_requires_from(&entry.diagnostics);
            }
        },
    )
}

/// Resolve session refs with mode selection and cache checking.
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
) -> Result<ResolvedExpr, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
            ctx,
            expr,
            mode,
            cache_key,
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode selection and apply environment substitution.
pub fn resolve_session_refs_with_mode_and_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<ResolvedExpr, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_all_with_mode_lookup_and_env(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode + env and return inherited diagnostics + cache hits.
pub fn resolve_session_refs_with_mode_and_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<(cas_ast::ExprId, Diagnostics, Vec<EntryId>), ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_mode_with_env_and_diagnostics(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
            Diagnostics::new(),
            push_session_propagated_requirement,
        )
    })
}
