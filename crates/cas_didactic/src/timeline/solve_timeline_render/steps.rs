use crate::cas_solver::SolveStep;
use cas_ast::Context;

pub(super) fn render_solve_timeline_steps(
    context: &Context,
    steps: &[SolveStep],
    render_solve_step_html: fn(&Context, usize, &SolveStep) -> String,
) -> String {
    let mut html = String::new();
    for (index, step) in steps.iter().enumerate() {
        html.push_str(&render_solve_step_html(context, index + 1, step));
    }
    html
}
