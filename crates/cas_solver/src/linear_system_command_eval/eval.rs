use cas_ast::Context;

use crate::linear_system_command_eval::{
    LinearSystemCommandEvalError, LinearSystemCommandEvalOutput,
};
use crate::linear_system_command_parse::parse_linear_system_spec;

use super::solve::solve_linear_system_spec;

pub(crate) fn evaluate_linear_system_command_input(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemCommandEvalOutput, LinearSystemCommandEvalError> {
    let spec = parse_linear_system_spec(ctx, input).map_err(LinearSystemCommandEvalError::Parse)?;
    let result =
        solve_linear_system_spec(ctx, &spec).map_err(LinearSystemCommandEvalError::Solve)?;
    Ok(LinearSystemCommandEvalOutput {
        vars: spec.vars,
        result,
    })
}

/// Same as [`evaluate_linear_system_command_input`], but with the full
/// simplifier available: a NotLinear decline gets the S2 nonlinear
/// composition shot (isolate → substitute → solve → verify) before erroring.
pub(crate) fn evaluate_linear_system_command_input_with_simplifier(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<LinearSystemCommandEvalOutput, LinearSystemCommandEvalError> {
    let spec = parse_linear_system_spec(&mut simplifier.context, input)
        .map_err(LinearSystemCommandEvalError::Parse)?;
    let result = match solve_linear_system_spec(&mut simplifier.context, &spec) {
        Ok(result) => result,
        Err(error) => {
            match super::nonlinear::try_solve_nonlinear_2x2(simplifier, &spec.exprs, &spec.vars) {
                Some((result, _narration)) => result,
                None => return Err(LinearSystemCommandEvalError::Solve(error)),
            }
        }
    };
    Ok(LinearSystemCommandEvalOutput {
        vars: spec.vars,
        result,
    })
}
