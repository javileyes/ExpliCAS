use cas_solver::json::{SubstituteEvalMode, SubstituteEvalResult, SubstituteEvalStep};

/// Evaluate substitute subcommand in JSON mode.
pub fn evaluate_substitute_subcommand_json(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: &str,
    steps: bool,
) -> String {
    cas_solver::substitute_str_to_json_with_options(expr, target, replacement, mode, steps, true)
}

/// Evaluate substitute subcommand in text mode and return lines for CLI output.
pub fn evaluate_substitute_subcommand_text_lines(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteEvalMode,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let output =
        cas_solver::json::eval_substitute_from_str(expr, target, replacement, mode, steps_enabled)
            .map_err(|error| error.to_string())?;
    Ok(format_substitute_subcommand_text_lines(
        &output,
        steps_enabled,
    ))
}

/// Evaluate substitute subcommand in text mode from string mode flag (`exact|power`).
pub fn evaluate_substitute_subcommand_text_lines_with_mode(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let parsed_mode = parse_substitute_eval_mode(mode);
    evaluate_substitute_subcommand_text_lines(expr, target, replacement, parsed_mode, steps_enabled)
}

/// Format text output lines for substitute subcommand.
pub fn format_substitute_subcommand_text_lines(
    output: &SubstituteEvalResult,
    steps_enabled: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    if steps_enabled && !output.steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(output.steps.iter().map(format_substitute_step_line));
    }
    lines.push(output.result.clone());
    lines
}

fn format_substitute_step_line(step: &SubstituteEvalStep) -> String {
    match &step.note {
        Some(note) => format!(
            "  {} → {} [{}] ({})",
            step.before, step.after, step.rule, note
        ),
        None => format!("  {} → {} [{}]", step.before, step.after, step.rule),
    }
}

fn parse_substitute_eval_mode(mode: &str) -> SubstituteEvalMode {
    match mode {
        "exact" => SubstituteEvalMode::Exact,
        _ => SubstituteEvalMode::Power,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_substitute_subcommand_json, evaluate_substitute_subcommand_text_lines,
        evaluate_substitute_subcommand_text_lines_with_mode,
        format_substitute_subcommand_text_lines,
    };
    use cas_solver::json::{SubstituteEvalMode, SubstituteEvalResult, SubstituteEvalStep};

    #[test]
    fn evaluate_substitute_subcommand_json_returns_ok_contract() {
        let out = evaluate_substitute_subcommand_json("x^2+1", "x", "y", "exact", false);
        let json: serde_json::Value = match serde_json::from_str(&out) {
            Ok(json) => json,
            Err(err) => panic!("invalid json: {err}"),
        };
        assert_eq!(json["ok"], true);
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_success() {
        let lines = match evaluate_substitute_subcommand_text_lines(
            "x^4 + x^2 + 1",
            "x^2",
            "y",
            SubstituteEvalMode::Power,
            true,
        ) {
            Ok(lines) => lines,
            Err(err) => panic!("text output failed: {err}"),
        };

        assert!(!lines.is_empty());
        assert_eq!(lines.first().map(String::as_str), Some("Steps:"));
        assert!(lines.last().is_some_and(|line| line.contains('y')));
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_with_mode_exact() {
        let lines = match evaluate_substitute_subcommand_text_lines_with_mode(
            "x + 1", "x", "y", "exact", false,
        ) {
            Ok(lines) => lines,
            Err(err) => panic!("text output failed: {err}"),
        };
        assert!(lines.last().is_some_and(|line| line.contains('y')));
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_parse_error_passthrough() {
        let err = match evaluate_substitute_subcommand_text_lines(
            "x^2 + 1",
            "invalid(((",
            "y",
            SubstituteEvalMode::Power,
            false,
        ) {
            Ok(_) => panic!("expected parse error"),
            Err(err) => err,
        };

        assert!(err.contains("Parse error in target"));
    }

    #[test]
    fn format_substitute_subcommand_text_lines_renders_steps_and_result() {
        let output = SubstituteEvalResult {
            result: "2 * y + 1".to_string(),
            steps: vec![
                SubstituteEvalStep {
                    rule: "subst".to_string(),
                    before: "x^2".to_string(),
                    after: "y".to_string(),
                    note: None,
                },
                SubstituteEvalStep {
                    rule: "simplify".to_string(),
                    before: "y + y".to_string(),
                    after: "2 * y".to_string(),
                    note: Some("combine like terms".to_string()),
                },
            ],
        };

        let lines = format_substitute_subcommand_text_lines(&output, true);
        assert_eq!(lines[0], "Steps:");
        assert_eq!(lines[1], "  x^2 → y [subst]");
        assert_eq!(lines[2], "  y + y → 2 * y [simplify] (combine like terms)");
        assert_eq!(lines[3], "2 * y + 1");
    }
}
