mod final_result;
mod steps;

use super::solve_render::{render_solve_final_result_html, render_solve_step_html};
use super::solve_solution_latex::render_solution_set_to_latex;
use cas_ast::{Context, SolutionSet};
use cas_solver::SolveStep;

pub(super) fn render_solve_timeline_body(
    context: &Context,
    steps: &[SolveStep],
    solution_set: &SolutionSet,
    var: &str,
) -> String {
    let mut html = String::from("        <div class=\"timeline\">\n");

    html.push_str(&steps::render_solve_timeline_steps(
        context,
        steps,
        render_solve_step_html,
    ));
    html.push_str(&final_result::render_solve_timeline_final_result(
        context,
        solution_set,
        var,
        render_solution_set_to_latex,
        render_solve_final_result_html,
    ));
    html
}
