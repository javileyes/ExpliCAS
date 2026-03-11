//! Session-facing solver helpers and support APIs.

pub use crate::evaluate_and_apply_config_command;
pub use crate::{binding_overview_entries, clear_bindings_command};
pub use crate::{
    build_prompt_from_eval_options, parse_repl_command_input, preprocess_repl_function_syntax,
    split_repl_statements,
};
pub use crate::{
    capture_health_report_if_enabled, clear_health_profiler, health_clear_message,
    health_disable_message, health_enable_message,
};
pub use crate::{
    config_rule_usage_message, config_unknown_subcommand_message, config_usage_message,
    evaluate_config_command, parse_config_command_input,
};
pub use crate::{
    count_health_results, format_health_suite_report_filtered, health_suite_category_names,
    list_health_suite_cases,
};
pub use crate::{delete_history_entries, history_overview_entries};
pub use crate::{
    filter_blocked_hints_for_eval, format_assumption_records_summary,
    format_eval_blocked_hints_lines, format_solve_assumption_and_blocked_sections,
};
pub use cas_solver_core::assumption_render::{
    format_blocked_hint_lines, format_diagnostics_requires_lines, format_domain_warning_lines,
    format_normalized_condition_lines, format_required_condition_lines,
};
pub use cas_solver_core::assumption_usage::{
    collect_assumed_conditions_from_steps, format_assumed_conditions_report_lines,
    group_assumed_conditions_by_rule,
};
pub use cas_solver_core::config_command_types::{ConfigCommandInput, ConfigCommandResult};
pub use cas_solver_core::config_runtime::ConfigCommandApplyOutput;
pub use cas_solver_core::repl_command_types::ReplCommandInput;
pub use cas_solver_core::solve_assumption_types::SolveAssumptionSectionConfig;
