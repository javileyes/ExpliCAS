use super::{
    parse_substitute_args, substitute_auto_with_strategy, SubstituteOptions, SubstituteParseError,
    SubstituteSimplifyEvalOutput,
};

/// Parse, substitute, and simplify REPL-style `subst` input.
#[cfg_attr(not(test), allow(dead_code))]
pub fn evaluate_substitute_and_simplify(
    simplifier: &mut crate::Simplifier,
    input: &str,
    options: SubstituteOptions,
) -> Result<SubstituteSimplifyEvalOutput, SubstituteParseError> {
    let (expr, target, replacement) = parse_substitute_args(&mut simplifier.context, input)?;
    let (substituted_expr, strategy) =
        substitute_auto_with_strategy(&mut simplifier.context, expr, target, replacement, options);
    let (simplified_expr, steps) = simplifier.simplify(substituted_expr);
    Ok(SubstituteSimplifyEvalOutput {
        simplified_expr,
        strategy,
        steps,
    })
}
