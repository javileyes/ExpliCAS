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
