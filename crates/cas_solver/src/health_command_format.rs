mod category;
mod report;
mod usage;

use cas_solver_core::health_runtime::HealthStatusInput;

/// Usage message built from currently available health categories.
pub(crate) fn health_usage_message() -> String {
    self::usage::health_usage_message()
}

/// Resolve optional category filter from parsed `health status` args.
pub(crate) fn resolve_health_category_filter<T, E, F>(
    status: &HealthStatusInput,
    category_names: &str,
    parse_category: F,
) -> Result<Option<T>, String>
where
    F: Fn(&str) -> Result<T, E>,
    E: std::fmt::Display,
{
    self::category::resolve_health_category_filter(status, category_names, parse_category)
}

/// Format `health status` running banner for selected category.
pub(crate) fn format_health_status_running_message(category_label: &str) -> String {
    self::usage::format_health_status_running_message(category_label)
}

/// Optional warning line shown when health suite reports failures.
pub(crate) fn format_health_failed_tests_warning_line(failed: usize) -> Option<String> {
    self::usage::format_health_failed_tests_warning_line(failed)
}

/// Build `health` report lines from last pipeline stats and optional report text.
pub(crate) fn format_health_report_lines(
    last_stats: Option<&crate::PipelineStats>,
    report_text: Option<&str>,
) -> Vec<String> {
    self::report::format_health_report_lines(last_stats, report_text)
}
