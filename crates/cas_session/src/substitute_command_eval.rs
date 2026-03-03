use crate::substitute_command_format::{
    format_substitute_eval_lines, format_substitute_parse_error_message,
    substitute_render_mode_from_display_mode,
};
use crate::substitute_command_parse::parse_substitute_args;
use crate::substitute_command_types::{
    SubstituteParseError, SubstituteRenderMode, SubstituteSimplifyEvalOutput,
};

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
