use crate::health_command_format::{
    format_health_failed_tests_warning_line, format_health_report_lines,
    format_health_status_running_message, resolve_health_category_filter,
};
use crate::health_command_messages::{
    clear_health_profiler, health_clear_message, health_disable_message, health_enable_message,
};
use crate::health_command_parse::evaluate_health_command_input;
use crate::health_command_types::{HealthCommandEvalOutput, HealthCommandInput, HealthStatusInput};

/// Evaluate `health status` request and return display lines.
pub fn evaluate_health_status_lines(
    simplifier: &mut cas_solver::Simplifier,
    status: &HealthStatusInput,
) -> Result<Vec<String>, String> {
    if status.list_only {
        return Ok(vec![crate::health_suite::list_cases()]);
    }

    let category_names = crate::health_suite::category_names().join(", ");
    let category_filter = resolve_health_category_filter(status, &category_names, |raw| {
        raw.parse::<crate::health_suite::Category>()
    })?;

    let category_label = category_filter
        .as_ref()
        .map_or("all".to_string(), ToString::to_string);
    let mut lines = vec![format_health_status_running_message(&category_label)];

    let results = crate::health_suite::run_suite_filtered(simplifier, category_filter);
    let report = crate::health_suite::format_report_filtered(&results, category_filter);
    lines.push(report);

    let (_passed, failed) = crate::health_suite::count_results(&results);
    if let Some(warning) = format_health_failed_tests_warning_line(failed) {
        lines.push(warning);
    }

    Ok(lines)
}

/// Evaluate full `health ...` command and return both lines and side-effect intents.
///
/// The caller applies `set_enabled` and `clear_last_report` to its UI/session state.
pub fn evaluate_health_command(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    last_stats: Option<&cas_solver::PipelineStats>,
    last_health_report: Option<&str>,
) -> Result<HealthCommandEvalOutput, String> {
    match evaluate_health_command_input(line)? {
        HealthCommandInput::ShowLast => Ok(HealthCommandEvalOutput {
            lines: format_health_report_lines(last_stats, last_health_report),
            set_enabled: None,
            clear_last_report: false,
        }),
        HealthCommandInput::SetEnabled { enabled } => Ok(HealthCommandEvalOutput {
            lines: vec![if enabled {
                health_enable_message().to_string()
            } else {
                health_disable_message().to_string()
            }],
            set_enabled: Some(enabled),
            clear_last_report: false,
        }),
        HealthCommandInput::Clear => {
            clear_health_profiler(simplifier);
            Ok(HealthCommandEvalOutput {
                lines: vec![health_clear_message().to_string()],
                set_enabled: None,
                clear_last_report: true,
            })
        }
        HealthCommandInput::Status(status) => {
            let lines = evaluate_health_status_lines(simplifier, &status)?;
            Ok(HealthCommandEvalOutput {
                lines,
                set_enabled: None,
                clear_last_report: false,
            })
        }
        HealthCommandInput::Invalid => {
            unreachable!("invalid is handled in evaluate_health_command_input")
        }
    }
}
