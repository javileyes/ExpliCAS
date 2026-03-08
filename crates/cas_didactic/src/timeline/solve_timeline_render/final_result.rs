use cas_ast::{Context, SolutionSet};

pub(super) fn render_solve_timeline_final_result(
    context: &Context,
    solution_set: &SolutionSet,
    var: &str,
    render_solution_set_to_latex: fn(&Context, &SolutionSet) -> String,
    render_solve_final_result_html: fn(&str, &str) -> String,
) -> String {
    let solution_latex = render_solution_set_to_latex(context, solution_set);
    render_solve_final_result_html(var, &solution_latex)
}
