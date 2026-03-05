use crate::substitute_command_parse::{split_by_comma_ignoring_parens, substitute_usage_message};
use crate::substitute_command_types::{
    SubstituteEvalOutput, SubstituteParseError, SubstituteRenderMode,
};

/// Format substitute parse errors into user-facing messages.
pub fn format_substitute_parse_error_message(error: &SubstituteParseError) -> String {
    match error {
        SubstituteParseError::InvalidArity => substitute_usage_message().to_string(),
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

fn should_render_substitute_step(step: &crate::Step, mode: SubstituteRenderMode) -> bool {
    match mode {
        SubstituteRenderMode::None => false,
        SubstituteRenderMode::Verbose => true,
        SubstituteRenderMode::Succinct | SubstituteRenderMode::Normal => {
            if step.get_importance() < crate::ImportanceLevel::Medium {
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
            crate::SubstituteStrategy::Variable => "Variable substitution",
            crate::SubstituteStrategy::PowerAware => "Expression substitution",
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
