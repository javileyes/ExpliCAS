//! Session-level substitute command rendering helpers.

use cas_ast::ExprId;

type SubstituteEvalOutput = SubstituteSimplifyEvalOutput;

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

/// Parse/eval errors for `subst <expr>, <target>, <replacement>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteParseError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}

/// Evaluated payload for REPL-style `subst` followed by simplify.
#[derive(Debug, Clone)]
pub struct SubstituteSimplifyEvalOutput {
    pub simplified_expr: ExprId,
    pub strategy: cas_solver::SubstituteStrategy,
    pub steps: Vec<cas_solver::Step>,
}

/// Format substitute parse errors into user-facing messages.
pub fn format_substitute_parse_error_message(error: &SubstituteParseError) -> String {
    match error {
        SubstituteParseError::InvalidArity => SUBSTITUTE_USAGE_MESSAGE.to_string(),
        SubstituteParseError::Expression(e) => {
            format!("Error parsing expression: {e}")
        }
        SubstituteParseError::Target(e) => {
            format!("Error parsing target: {e}")
        }
        SubstituteParseError::Replacement(e) => {
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

/// Parse REPL-like substitute arguments (`expr, target, replacement`) into ids.
fn parse_substitute_args(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<(ExprId, ExprId, ExprId), SubstituteParseError> {
    let parts = split_by_comma_ignoring_parens(input);
    if parts.len() != 3 {
        return Err(SubstituteParseError::InvalidArity);
    }

    let expr_str = parts[0].trim();
    let target_str = parts[1].trim();
    let replacement_str = parts[2].trim();

    let expr = cas_parser::parse(expr_str, ctx)
        .map_err(|e| SubstituteParseError::Expression(e.to_string()))?;
    let target = cas_parser::parse(target_str, ctx)
        .map_err(|e| SubstituteParseError::Target(e.to_string()))?;
    let replacement = cas_parser::parse(replacement_str, ctx)
        .map_err(|e| SubstituteParseError::Replacement(e.to_string()))?;

    Ok((expr, target, replacement))
}

fn evaluate_substitute_and_simplify(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    options: cas_solver::SubstituteOptions,
) -> Result<SubstituteSimplifyEvalOutput, SubstituteParseError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    let (substituted_expr, strategy) = cas_solver::substitute_auto_with_strategy(
        &mut simplifier.context,
        expr,
        target,
        replacement,
        options,
    );
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteSimplifyEvalOutput {
        simplified_expr,
        strategy,
        steps,
    })
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
) -> Result<Vec<String>, SubstituteParseError> {
    let output = evaluate_substitute_and_simplify(
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
) -> Result<Vec<String>, SubstituteParseError> {
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
) -> Result<String, SubstituteParseError> {
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
        let msg = format_substitute_parse_error_message(&super::SubstituteParseError::InvalidArity);
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
