/// Parsed input for the `health` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthCommandInput {
    ShowLast,
    SetEnabled { enabled: bool },
    Clear,
    Status(HealthStatusInput),
    Invalid,
}

/// Parsed options for `health status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStatusInput {
    pub list_only: bool,
    pub category: Option<String>,
    pub category_missing_arg: bool,
}

/// Evaluated output for a `health ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthCommandEvalOutput {
    pub lines: Vec<String>,
    pub set_enabled: Option<bool>,
    pub clear_last_report: bool,
}

/// Parse raw `health ...` command input.
pub fn parse_health_command_input(line: &str) -> HealthCommandInput {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() == 1 {
        return HealthCommandInput::ShowLast;
    }

    match parts[1] {
        "on" | "enable" => HealthCommandInput::SetEnabled { enabled: true },
        "off" | "disable" => HealthCommandInput::SetEnabled { enabled: false },
        "reset" | "clear" => HealthCommandInput::Clear,
        "status" => {
            let opts: Vec<&str> = parts.iter().skip(2).copied().collect();
            let list_only = opts.contains(&"--list") || opts.contains(&"-l");
            let mut category = None;
            let mut category_missing_arg = false;

            if let Some(idx) = opts.iter().position(|&x| x == "--category" || x == "-c") {
                if let Some(cat) = opts.get(idx + 1) {
                    category = Some((*cat).to_string());
                } else {
                    category_missing_arg = true;
                }
            }

            HealthCommandInput::Status(HealthStatusInput {
                list_only,
                category,
                category_missing_arg,
            })
        }
        _ => HealthCommandInput::Invalid,
    }
}

/// Parse and validate `health ...` command input.
///
/// Returns a preformatted usage message when command input is invalid.
pub fn evaluate_health_command_input(line: &str) -> Result<HealthCommandInput, String> {
    match parse_health_command_input(line) {
        HealthCommandInput::Invalid => Err(health_usage_message()),
        parsed => Ok(parsed),
    }
}

pub fn health_enable_message() -> &'static str {
    "Health tracking ENABLED (metrics captured after each simplify)"
}

pub fn health_disable_message() -> &'static str {
    "Health tracking DISABLED"
}

pub fn health_clear_message() -> &'static str {
    "Health statistics cleared."
}

/// Clear health profiling counters for a simplifier.
pub fn clear_health_profiler(simplifier: &mut crate::Simplifier) {
    simplifier.profiler.clear_run();
}

pub fn format_health_usage_message(category_names: &str) -> String {
    format!(
        "Usage: health [on|off|reset|status]\n\
         \n\
               health               Show last health report\n\
               health on            Enable health tracking\n\
               health off           Disable health tracking\n\
               health reset         Clear health statistics\n\
               health status        Run diagnostic test suite\n\
               health status --list List available test cases\n\
               health status --category <cat>  Run only category\n\
                                            Categories: {}",
        category_names
    )
}

/// Usage message built from currently available health categories.
pub fn health_usage_message() -> String {
    format_health_usage_message(&crate::health_suite::category_names().join(", "))
}

pub fn format_health_missing_category_arg_message(category_names: &str) -> String {
    format!(
        "Error: --category requires an argument\nAvailable categories: {}",
        category_names
    )
}

pub fn format_health_invalid_category_message(error: &str, category_names: &str) -> String {
    format!("Error: {}\nAvailable categories: {}", error, category_names)
}

/// Resolve optional category filter from parsed `health status` args.
pub fn resolve_health_category_filter<T, E, F>(
    status: &HealthStatusInput,
    category_names: &str,
    parse_category: F,
) -> Result<Option<T>, String>
where
    F: Fn(&str) -> Result<T, E>,
    E: std::fmt::Display,
{
    if status.category_missing_arg {
        return Err(format_health_missing_category_arg_message(category_names));
    }

    if let Some(cat_str) = status.category.as_deref() {
        if cat_str == "all" {
            return Ok(None);
        }

        match parse_category(cat_str) {
            Ok(category) => Ok(Some(category)),
            Err(error) => Err(format_health_invalid_category_message(
                &error.to_string(),
                category_names,
            )),
        }
    } else {
        Ok(None)
    }
}

/// Format `health status` running banner for selected category.
pub fn format_health_status_running_message(category_label: &str) -> String {
    format!(
        "Running health status suite [category={}]...\n",
        category_label
    )
}

/// Optional warning line shown when health suite reports failures.
pub fn format_health_failed_tests_warning_line(failed: usize) -> Option<String> {
    if failed == 0 {
        None
    } else {
        Some(format!(
            "\n⚠ {} tests failed. Check Transform rules for churn.",
            failed
        ))
    }
}

/// Build `health` report lines from last pipeline stats and optional report text.
pub fn format_health_report_lines(
    last_stats: Option<&crate::PipelineStats>,
    report_text: Option<&str>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    if let Some(stats) = last_stats {
        let cycles: Vec<_> = [
            (&stats.core.cycle, "Core"),
            (&stats.transform.cycle, "Transform"),
            (&stats.rationalize.cycle, "Rationalize"),
            (&stats.post_cleanup.cycle, "PostCleanup"),
        ]
        .iter()
        .filter_map(|(c, name)| c.as_ref().map(|info| (*name, info)))
        .collect();

        for (phase_name, cycle) in &cycles {
            lines.push(format!(
                "⚠ Cycle detected in {}: period={} at rewrite={} (stopped early)",
                phase_name, cycle.period, cycle.at_step
            ));
        }
        if !cycles.is_empty() {
            lines.push(String::new());
        }
    }

    if let Some(report) = report_text {
        lines.push(report.to_string());
    } else {
        lines.push("No health report available.".to_string());
        lines.push(
            "Run a simplification first (health is captured when debug mode or health mode is on)."
                .to_string(),
        );
        lines.push("Enable with: health on".to_string());
    }

    lines
}

/// Evaluate `health status` request and return display lines.
pub fn evaluate_health_status_lines(
    simplifier: &mut crate::Simplifier,
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
    simplifier: &mut crate::Simplifier,
    line: &str,
    last_stats: Option<&crate::PipelineStats>,
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

#[cfg(test)]
mod tests {
    use super::{
        clear_health_profiler, evaluate_health_command, evaluate_health_command_input,
        evaluate_health_status_lines, format_health_failed_tests_warning_line,
        format_health_missing_category_arg_message, format_health_report_lines,
        format_health_status_running_message, format_health_usage_message, health_usage_message,
        parse_health_command_input, resolve_health_category_filter, HealthCommandInput,
        HealthStatusInput,
    };

    #[test]
    fn parse_health_command_input_status_list() {
        assert_eq!(
            parse_health_command_input("health status --list"),
            HealthCommandInput::Status(HealthStatusInput {
                list_only: true,
                category: None,
                category_missing_arg: false,
            })
        );
    }

    #[test]
    fn parse_health_command_input_status_category() {
        assert_eq!(
            parse_health_command_input("health status --category algebra"),
            HealthCommandInput::Status(HealthStatusInput {
                list_only: false,
                category: Some("algebra".to_string()),
                category_missing_arg: false,
            })
        );
    }

    #[test]
    fn format_health_usage_message_mentions_status() {
        let text = format_health_usage_message("a,b,c");
        assert!(text.contains("health status"));
        assert!(text.contains("Categories: a,b,c"));
    }

    #[test]
    fn health_usage_message_includes_categories() {
        let text = health_usage_message();
        assert!(text.contains("Categories:"));
    }

    #[test]
    fn format_health_missing_category_arg_message_mentions_argument() {
        let text = format_health_missing_category_arg_message("a,b,c");
        assert!(text.contains("--category requires an argument"));
    }

    #[test]
    fn format_health_report_lines_without_report_includes_hint() {
        let lines = format_health_report_lines(None, None);
        assert!(lines
            .iter()
            .any(|line| line.contains("No health report available.")));
        assert!(lines
            .iter()
            .any(|line| line.contains("Enable with: health on")));
    }

    #[test]
    fn resolve_health_category_filter_handles_all_and_missing() {
        let names = "algebra,analysis";
        let status_all = HealthStatusInput {
            list_only: false,
            category: Some("all".to_string()),
            category_missing_arg: false,
        };
        let all = resolve_health_category_filter::<u8, String, _>(&status_all, names, |_raw| {
            Err("unused".to_string())
        })
        .expect("all");
        assert!(all.is_none());

        let status_missing = HealthStatusInput {
            list_only: false,
            category: None,
            category_missing_arg: true,
        };
        let err = resolve_health_category_filter::<u8, String, _>(&status_missing, names, |_raw| {
            Err("unused".to_string())
        })
        .expect_err("missing");
        assert!(err.contains("--category requires an argument"));
    }

    #[test]
    fn format_health_status_running_message_includes_category() {
        let line = format_health_status_running_message("algebra");
        assert!(line.contains("category=algebra"));
    }

    #[test]
    fn format_health_failed_tests_warning_line_only_for_nonzero() {
        assert!(format_health_failed_tests_warning_line(0).is_none());
        let line = format_health_failed_tests_warning_line(2).expect("warning");
        assert!(line.contains("2 tests failed"));
    }

    #[test]
    fn evaluate_health_command_input_maps_invalid_to_usage() {
        let err = evaluate_health_command_input("health nope").expect_err("invalid");
        assert!(err.contains("Usage: health"));
    }

    #[test]
    fn evaluate_health_status_lines_list_only() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_health_status_lines(
            &mut simplifier,
            &HealthStatusInput {
                list_only: true,
                category: None,
                category_missing_arg: false,
            },
        )
        .expect("list");
        assert!(lines
            .first()
            .is_some_and(|line| line.contains("Available health cases")));
    }

    #[test]
    fn evaluate_health_status_lines_invalid_category_returns_error() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let err = evaluate_health_status_lines(
            &mut simplifier,
            &HealthStatusInput {
                list_only: false,
                category: Some("unknown".to_string()),
                category_missing_arg: false,
            },
        )
        .expect_err("invalid category");
        assert!(err.contains("Available categories"));
    }

    #[test]
    fn clear_health_profiler_resets_profiler_state() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        clear_health_profiler(&mut simplifier);
    }

    #[test]
    fn evaluate_health_command_show_last_uses_report_lines() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_health_command(&mut simplifier, "health", None, None).expect("health");
        assert!(out.set_enabled.is_none());
        assert!(!out.clear_last_report);
        assert!(out
            .lines
            .iter()
            .any(|line| line.contains("No health report available.")));
    }

    #[test]
    fn evaluate_health_command_enable_sets_flag() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out =
            evaluate_health_command(&mut simplifier, "health on", None, None).expect("health on");
        assert_eq!(out.set_enabled, Some(true));
        assert!(!out.clear_last_report);
        assert!(out
            .lines
            .first()
            .is_some_and(|line| line.contains("ENABLED")));
    }

    #[test]
    fn evaluate_health_command_clear_requests_report_clear() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let out = evaluate_health_command(&mut simplifier, "health clear", None, None)
            .expect("health clear");
        assert!(out.set_enabled.is_none());
        assert!(out.clear_last_report);
        assert!(out
            .lines
            .first()
            .is_some_and(|line| line.contains("Health statistics cleared")));
    }
}
