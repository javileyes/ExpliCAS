//! Health command/session-facing API re-exported for session clients.

pub use crate::health_category::Category as HealthSuiteCategory;
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
pub use crate::health_suite_format_catalog::{
    category_names as health_suite_category_names, list_cases as list_health_suite_cases,
};
pub use crate::health_suite_format_report::{
    count_results as count_health_results,
    format_report_filtered as format_health_suite_report_filtered,
};
pub use crate::health_suite_runner::run_suite_filtered as run_health_suite_filtered;
pub use crate::repl_health_runtime::{
    evaluate_health_command_message_on_runtime as evaluate_health_command_message_on_repl_core,
    update_health_report_on_runtime as update_health_report_on_repl_core, ReplHealthRuntimeContext,
};
pub use cas_solver_core::health_runtime::{
    HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput,
};
