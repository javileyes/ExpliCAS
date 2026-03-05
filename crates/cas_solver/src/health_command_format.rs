use crate::health_command_types::HealthStatusInput;

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
    format_health_usage_message(&crate::health_suite_format_catalog::category_names().join(", "))
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
