use crate::health_command_format::{
    format_health_failed_tests_warning_line, format_health_status_running_message,
    resolve_health_category_filter,
};
use crate::health_command_types::HealthStatusInput;

pub fn evaluate_health_status_lines(
    simplifier: &mut crate::Simplifier,
    status: &HealthStatusInput,
) -> Result<Vec<String>, String> {
    if status.list_only {
        return Ok(vec![crate::health_suite_format_catalog::list_cases()]);
    }

    let category_names = crate::health_suite_format_catalog::category_names().join(", ");
    let category_filter = resolve_health_category_filter(status, &category_names, |raw| {
        raw.parse::<crate::health_suite_types::Category>()
    })?;

    let category_label = category_filter
        .as_ref()
        .map_or("all".to_string(), ToString::to_string);
    let mut lines = vec![format_health_status_running_message(&category_label)];

    let results = crate::health_suite_runner::run_suite_filtered(simplifier, category_filter);
    let report =
        crate::health_suite_format_report::format_report_filtered(&results, category_filter);
    lines.push(report);

    let (_passed, failed) = crate::health_suite_format_report::count_results(&results);
    if let Some(warning) = format_health_failed_tests_warning_line(failed) {
        lines.push(warning);
    }

    Ok(lines)
}
