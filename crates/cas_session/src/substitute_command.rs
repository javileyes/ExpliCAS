//! Session-level substitute command rendering helpers.

type SubstituteEvalOutput = cas_solver::SubstituteSimplifyEvalOutput;

const SUBSTITUTE_USAGE_MESSAGE: &str = "Usage: subst <expr>, <target>, <replacement>\n\n\
                     Examples:\n\
                       subst x^2 + x, x, 3              -> 12\n\
                       subst x^4 + x^2 + 1, x^2, y      -> y^2 + y + 1\n\
                       subst x^3, x^2, y                -> y*x";

/// Render policy for substitute command step output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteRenderMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Format substitute parse errors into user-facing messages.
pub fn format_substitute_parse_error_message(error: &cas_solver::SubstituteParseError) -> String {
    match error {
        cas_solver::SubstituteParseError::InvalidArity => SUBSTITUTE_USAGE_MESSAGE.to_string(),
        cas_solver::SubstituteParseError::Expression(e) => {
            format!("Error parsing expression: {e}")
        }
        cas_solver::SubstituteParseError::Target(e) => {
            format!("Error parsing target: {e}")
        }
        cas_solver::SubstituteParseError::Replacement(e) => {
            format!("Error parsing replacement: {e}")
        }
    }
}

fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    parts.push(&s[start..]);
    parts
}

/// Convert REPL display mode into substitute render mode.
pub fn substitute_render_mode_from_display_mode(
    mode: crate::SetDisplayMode,
) -> SubstituteRenderMode {
    match mode {
        crate::SetDisplayMode::None => SubstituteRenderMode::None,
        crate::SetDisplayMode::Succinct => SubstituteRenderMode::Succinct,
        crate::SetDisplayMode::Normal => SubstituteRenderMode::Normal,
        crate::SetDisplayMode::Verbose => SubstituteRenderMode::Verbose,
    }
}

fn should_render_substitute_step(step: &cas_solver::Step, mode: SubstituteRenderMode) -> bool {
    match mode {
        SubstituteRenderMode::None => false,
        SubstituteRenderMode::Verbose => true,
        SubstituteRenderMode::Succinct | SubstituteRenderMode::Normal => {
            if step.get_importance() < cas_solver::ImportanceLevel::Medium {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

/// Format substitute command eval output into display lines.
pub fn format_substitute_eval_lines(
    context: &cas_ast::Context,
    input: &str,
    output: &SubstituteEvalOutput,
    mode: SubstituteRenderMode,
) -> Vec<String> {
    let display_parts = split_by_comma_ignoring_parens(input);
    let expr_str = display_parts.first().map(|s| s.trim()).unwrap_or_default();
    let target_str = display_parts.get(1).map(|s| s.trim()).unwrap_or_default();
    let replacement_str = display_parts.get(2).map(|s| s.trim()).unwrap_or_default();

    let mut lines = Vec::new();
    if mode != SubstituteRenderMode::None {
        let label = match output.strategy {
            cas_solver::SubstituteStrategy::Variable => "Variable substitution",
            cas_solver::SubstituteStrategy::PowerAware => "Expression substitution",
        };
        lines.push(format!(
            "{label}: {} → {} in {}",
            target_str, replacement_str, expr_str
        ));
    }

    if mode != SubstituteRenderMode::None && !output.steps.is_empty() {
        if mode != SubstituteRenderMode::Succinct {
            lines.push("Steps:".to_string());
        }
        for step in &output.steps {
            if should_render_substitute_step(step, mode) {
                if mode == SubstituteRenderMode::Succinct {
                    lines.push(format!(
                        "-> {}",
                        cas_formatter::DisplayExpr {
                            context,
                            id: step.global_after.unwrap_or(step.after)
                        }
                    ));
                } else {
                    lines.push(format!("  {}  [{}]", step.description, step.rule_name));
                }
            }
        }
    }

    lines.push(format!(
        "Result: {}",
        cas_formatter::DisplayExpr {
            context,
            id: output.simplified_expr
        }
    ));
    lines
}

/// Evaluate and format a `subst` command for CLI display.
pub fn evaluate_substitute_command_lines(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    mode: SubstituteRenderMode,
) -> Result<Vec<String>, cas_solver::SubstituteParseError> {
    let output = cas_solver::evaluate_substitute_and_simplify(
        simplifier,
        input,
        cas_solver::SubstituteOptions::default(),
    )?;
    Ok(format_substitute_eval_lines(
        &simplifier.context,
        input,
        &output,
        mode,
    ))
}

/// Evaluate full `subst ...` invocation line and return cleaned display lines.
pub fn evaluate_substitute_invocation_lines(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<Vec<String>, cas_solver::SubstituteParseError> {
    let input = crate::extract_substitute_command_tail(line);
    let mode = substitute_render_mode_from_display_mode(display_mode);
    let mut lines = evaluate_substitute_command_lines(simplifier, input, mode)?;
    crate::clean_result_output_line(&mut lines);
    Ok(lines)
}

/// Evaluate full `subst ...` invocation line and return cleaned message text.
pub fn evaluate_substitute_invocation_message(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<String, cas_solver::SubstituteParseError> {
    Ok(evaluate_substitute_invocation_lines(simplifier, line, display_mode)?.join("\n"))
}

/// Evaluate full `subst ...` invocation and return user-facing text message.
pub fn evaluate_substitute_invocation_user_message(
    simplifier: &mut cas_solver::Simplifier,
    line: &str,
    display_mode: crate::SetDisplayMode,
) -> Result<String, String> {
    evaluate_substitute_invocation_message(simplifier, line, display_mode)
        .map_err(|error| format_substitute_parse_error_message(&error))
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_substitute_command_lines, evaluate_substitute_invocation_lines,
        format_substitute_parse_error_message, substitute_render_mode_from_display_mode,
    };

    #[test]
    fn substitute_render_mode_maps_from_set_display_mode() {
        assert_eq!(
            substitute_render_mode_from_display_mode(crate::SetDisplayMode::Verbose),
            super::SubstituteRenderMode::Verbose
        );
    }

    #[test]
    fn evaluate_substitute_command_lines_runs() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = evaluate_substitute_command_lines(
            &mut simplifier,
            "x^2 + x, x, 3",
            super::SubstituteRenderMode::Normal,
        )
        .expect("subst eval");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn evaluate_substitute_invocation_lines_trims_prefix() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let lines = evaluate_substitute_invocation_lines(
            &mut simplifier,
            "subst x^2 + x, x, 3",
            crate::SetDisplayMode::Normal,
        )
        .expect("subst eval");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }

    #[test]
    fn format_substitute_parse_error_message_usage_is_human_readable() {
        let msg =
            format_substitute_parse_error_message(&cas_solver::SubstituteParseError::InvalidArity);
        assert!(msg.contains("Usage: subst"));
    }

    #[test]
    fn evaluate_substitute_invocation_message_joins_lines() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let message = super::evaluate_substitute_invocation_message(
            &mut simplifier,
            "subst x^2 + x, x, 3",
            crate::SetDisplayMode::Normal,
        )
        .expect("subst eval");
        assert!(message.contains("Result:"));
    }

    #[test]
    fn evaluate_substitute_invocation_user_message_formats_parse_errors() {
        let mut simplifier = cas_solver::Simplifier::with_default_rules();
        let message = super::evaluate_substitute_invocation_user_message(
            &mut simplifier,
            "subst x^2 + x",
            crate::SetDisplayMode::Normal,
        )
        .expect_err("invalid arity");
        assert!(message.contains("Usage: subst"));
    }
}
