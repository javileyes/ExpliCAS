mod degenerate;
mod unique;

use cas_ast::Context;
use cas_formatter::latex_escape;

use crate::linear_system_command_eval::LinearSystemCommandEvalOutput;

pub(crate) fn format_linear_system_result_message(
    ctx: &mut Context,
    output: &LinearSystemCommandEvalOutput,
) -> String {
    render_linear_system_result(ctx, output).0
}

pub(crate) fn render_linear_system_result(
    ctx: &mut Context,
    output: &LinearSystemCommandEvalOutput,
) -> (String, Option<String>) {
    match &output.result {
        crate::LinSolveResult::Unique(solution) => (
            unique::format_unique_result(ctx, &output.vars, solution),
            Some(super::display_linear_system_solution_latex(
                ctx,
                &output.vars,
                solution,
            )),
        ),
        crate::LinSolveResult::Infinite => {
            let plain = degenerate::format_infinite_result();
            let latex = format!("\\text{{{}}}", latex_escape(&plain.replace('\n', " ")));
            (plain, Some(latex))
        }
        crate::LinSolveResult::Inconsistent => {
            let plain = degenerate::format_inconsistent_result();
            let latex = format!("\\text{{{}}}", latex_escape(&plain.replace('\n', " ")));
            (plain, Some(latex))
        }
    }
}
