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

pub fn health_enable_message() -> &'static str {
    "Health tracking ENABLED (metrics captured after each simplify)"
}

pub fn health_disable_message() -> &'static str {
    "Health tracking DISABLED"
}

pub fn health_clear_message() -> &'static str {
    "Health statistics cleared."
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

#[cfg(test)]
mod tests {
    use super::{
        format_health_missing_category_arg_message, format_health_report_lines,
        format_health_status_running_message, format_health_usage_message,
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
}
