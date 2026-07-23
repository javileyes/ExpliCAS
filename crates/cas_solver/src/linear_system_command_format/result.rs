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
        crate::LinSolveResult::UniqueExpr {
            values,
            nonzero_conditions,
        } => {
            let mut plain = super::display_linear_system_solution_exprs(ctx, &output.vars, values);
            let mut latex =
                super::display_linear_system_solution_exprs_latex(ctx, &output.vars, values);
            // The validity requirement is part of the RESULT, not a footnote:
            // a symbolic Cramer solution without its det ≠ 0 clause would be a
            // wrong answer at det = 0 (result-as-contract).
            for &cond in nonzero_conditions {
                let shown = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: cond,
                };
                plain.push_str(&format!("\n  requires: {shown} != 0"));
                let cond_latex = cas_formatter::LaTeXExpr {
                    context: ctx,
                    id: cond,
                }
                .to_latex();
                latex.push_str(&format!("\\quad\\left({cond_latex} \\neq 0\\right)"));
            }
            (plain, Some(latex))
        }
        crate::LinSolveResult::SolutionPairs(pairs) => {
            let rendered: Vec<String> = pairs
                .iter()
                .map(|values| {
                    super::display_linear_system_solution_exprs(ctx, &output.vars, values)
                })
                .collect();
            let latex_rendered: Vec<String> = pairs
                .iter()
                .map(|values| {
                    super::display_linear_system_solution_exprs_latex(ctx, &output.vars, values)
                })
                .collect();
            (
                rendered.join(" or "),
                Some(latex_rendered.join("\\ \\text{or}\\ ")),
            )
        }
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
