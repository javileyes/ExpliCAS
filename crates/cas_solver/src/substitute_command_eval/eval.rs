use crate::substitute_command_format::format_substitute_eval_lines;
use crate::substitute_command_parse::parse_substitute_args;
use crate::substitute_command_types::{
    SubstituteParseError, SubstituteRenderMode, SubstituteSimplifyEvalOutput,
};

fn evaluate_substitute_and_simplify(
    simplifier: &mut crate::Simplifier,
    input: &str,
    options: crate::SubstituteOptions,
) -> Result<SubstituteSimplifyEvalOutput, SubstituteParseError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    let (substituted_expr, strategy) = crate::substitute_auto_with_strategy(
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
    simplifier: &mut crate::Simplifier,
    input: &str,
    mode: SubstituteRenderMode,
) -> Result<Vec<String>, SubstituteParseError> {
    let output =
        evaluate_substitute_and_simplify(simplifier, input, crate::SubstituteOptions::default())?;
    Ok(format_substitute_eval_lines(
        &simplifier.context,
        input,
        &output,
        mode,
    ))
}
