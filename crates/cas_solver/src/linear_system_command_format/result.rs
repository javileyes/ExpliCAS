mod degenerate;
mod unique;

use cas_ast::Context;

use crate::linear_system_command_eval::LinearSystemCommandEvalOutput;

pub(crate) fn format_linear_system_result_message(
    ctx: &mut Context,
    output: &LinearSystemCommandEvalOutput,
) -> String {
    match &output.result {
        crate::LinSolveResult::Unique(solution) => {
            unique::format_unique_result(ctx, &output.vars, solution)
        }
        crate::LinSolveResult::Infinite => degenerate::format_infinite_result(),
        crate::LinSolveResult::Inconsistent => degenerate::format_inconsistent_result(),
    }
}
