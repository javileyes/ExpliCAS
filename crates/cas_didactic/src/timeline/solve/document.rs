use crate::cas_solver::SolveStep;
use cas_ast::{Context, Equation, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(super) fn render_solve_timeline_document(
    context: &mut Context,
    steps: &[SolveStep],
    original_eq: &Equation,
    solution_set: &SolutionSet,
    var: &str,
    title: &str,
    render_equation_latex: fn(&Context, &Equation) -> String,
    render_solve_timeline_html_header: fn(&str, &str, &str) -> String,
    render_solve_timeline_body: fn(&Context, &[SolveStep], &SolutionSet, &str) -> String,
    solve_timeline_html_footer: fn() -> String,
    clean_latex_identities: fn(&str) -> String,
) -> String {
    let original_latex = render_equation_latex(context, original_eq);
    let mut html = render_solve_timeline_html_header(title, var, &original_latex);
    html.push_str(&render_solve_timeline_body(
        context,
        steps,
        solution_set,
        var,
    ));
    html.push_str(&solve_timeline_html_footer());
    clean_latex_identities(&html)
}
