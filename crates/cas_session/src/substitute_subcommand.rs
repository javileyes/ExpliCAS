//! Stateless CLI-subcommand helpers for `substitute`.

/// Substitution mode for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteCommandMode {
    Exact,
    Power,
}

/// CLI-friendly output contract for `substitute` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteSubcommandOutput {
    Json(String),
    TextLines(Vec<String>),
}

/// Evaluate substitute subcommand in canonical JSON mode.
///
/// Uses `cas_solver::substitute_str_to_json` as the canonical serializer.
pub fn evaluate_substitute_subcommand_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
) -> String {
    let mode = match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    };
    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":true}}",
        mode, steps_enabled
    );
    cas_solver::substitute_str_to_json(expr, target, replacement, Some(&opts))
}

/// Evaluate substitute subcommand and map solver contracts to session-layer output.
pub fn evaluate_substitute_subcommand(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteCommandMode,
    steps_enabled: bool,
    json_output: bool,
) -> Result<SubstituteSubcommandOutput, String> {
    if json_output {
        let out = evaluate_substitute_subcommand_json_canonical(
            expr,
            target,
            replacement,
            mode,
            steps_enabled,
        );
        return Ok(SubstituteSubcommandOutput::Json(out));
    }

    let mode = match mode {
        SubstituteCommandMode::Exact => "exact",
        SubstituteCommandMode::Power => "power",
    };

    let opts = format!(
        "{{\"mode\":\"{}\",\"steps\":{},\"pretty\":false}}",
        mode, steps_enabled
    );
    let payload = cas_solver::substitute_str_to_json(expr, target, replacement, Some(&opts));
    let lines = parse_substitute_json_text_lines(&payload, steps_enabled)?;
    Ok(SubstituteSubcommandOutput::TextLines(lines))
}

fn parse_substitute_json_text_lines(
    payload: &str,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let value: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| format!("Invalid substitute JSON: {e}"))?;

    let ok = value.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
    if !ok {
        let error_message = value
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .or_else(|| value.get("error").and_then(|e| e.as_str()))
            .unwrap_or("Substitute evaluation failed");
        return Err(error_message.to_string());
    }

    let result = value
        .get("result")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "Missing result in substitute JSON".to_string())?;

    let mut lines = Vec::new();
    if steps_enabled {
        if let Some(steps) = value.get("steps").and_then(|s| s.as_array()) {
            if !steps.is_empty() {
                lines.push("Steps:".to_string());
                for step in steps {
                    let before = step.get("before").and_then(|v| v.as_str()).unwrap_or("");
                    let after = step.get("after").and_then(|v| v.as_str()).unwrap_or("");
                    let rule = step
                        .get("rule")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Substitute");
                    let note = step.get("note").and_then(|v| v.as_str());
                    let line = match note {
                        Some(note) => format!("  {} → {} [{}] ({})", before, after, rule, note),
                        None => format!("  {} → {} [{}]", before, after, rule),
                    };
                    lines.push(line);
                }
            }
        }
    }
    lines.push(result.to_string());
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_substitute_subcommand, parse_substitute_json_text_lines, SubstituteCommandMode,
        SubstituteSubcommandOutput,
    };

    #[test]
    fn evaluate_substitute_subcommand_json_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            false,
            true,
        )
        .expect("substitute json");

        match out {
            SubstituteSubcommandOutput::Json(payload) => {
                assert!(payload.contains("\"ok\""));
            }
            _ => panic!("expected json output"),
        }
    }

    #[test]
    fn evaluate_substitute_subcommand_text_contract() {
        let out = evaluate_substitute_subcommand(
            "x^2+1",
            "x",
            "y",
            SubstituteCommandMode::Exact,
            true,
            false,
        )
        .expect("substitute text");

        match out {
            SubstituteSubcommandOutput::TextLines(lines) => {
                assert!(!lines.is_empty());
                assert!(lines.iter().any(|line| line.contains('y')));
            }
            _ => panic!("expected text output"),
        }
    }

    #[test]
    fn parse_substitute_json_text_lines_maps_error_message() {
        let payload = r#"{"ok":false,"error":{"message":"Parse error in target: bad token"}}"#;
        let err = parse_substitute_json_text_lines(payload, false).expect_err("should fail");
        assert_eq!(err, "Parse error in target: bad token");
    }
}
