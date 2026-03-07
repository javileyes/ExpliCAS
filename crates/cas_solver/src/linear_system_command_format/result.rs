use cas_ast::Context;

use super::display_linear_system_solution;
use crate::linear_system_command_types::LinearSystemCommandEvalOutput;

pub(crate) fn format_linear_system_result_message(
    ctx: &mut Context,
    output: &LinearSystemCommandEvalOutput,
) -> String {
    match &output.result {
        crate::LinSolveResult::Unique(solution) => {
            display_linear_system_solution(ctx, &output.vars, solution)
        }
        crate::LinSolveResult::Infinite => "System has infinitely many solutions.\n\
                 The equations are dependent."
            .to_string(),
        crate::LinSolveResult::Inconsistent => "System has no solution.\n\
                 The equations are inconsistent."
            .to_string(),
    }
}
