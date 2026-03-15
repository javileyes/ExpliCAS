use crate::substitute::{evaluate_substitute_and_simplify, SubstituteOptions};
use crate::substitute_command_format::format_substitute_eval_lines;
use cas_api_models::{SubstituteParseError, SubstituteRenderMode};

/// Evaluate and format a `subst` command for CLI display.
pub fn evaluate_substitute_command_lines(
    simplifier: &mut crate::Simplifier,
    input: &str,
    mode: SubstituteRenderMode,
) -> Result<Vec<String>, SubstituteParseError> {
    let output = evaluate_substitute_and_simplify(simplifier, input, SubstituteOptions::default())?;
    Ok(format_substitute_eval_lines(
        &simplifier.context,
        input,
        &output,
        mode,
    ))
}
