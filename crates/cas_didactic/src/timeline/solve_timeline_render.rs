mod final_result;
mod steps;

use super::solve_render::{render_solve_final_result_html, render_solve_step_html};
use super::solve_solution_latex::render_solution_set_to_latex;
use crate::runtime::SolveStep;
use cas_ast::{Context, SolutionSet};

pub(super) fn render_solve_timeline_body(
    context: &Context,
    steps: &[SolveStep],
    solution_set: &SolutionSet,
    var: &str,
) -> String {
    let mut html = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/assets/timeline/solve_render/timeline_open.html"
    ))
    .to_string();

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
