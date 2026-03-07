use cas_ast::{Case, Context, SolutionSet};

pub(super) fn render_conditional_case_line(
    context: &Context,
    case: &Case,
    render_inner_solution_set_to_latex: fn(&Context, &SolutionSet) -> String,
) -> String {
    let solution_latex = render_inner_solution_set_to_latex(context, &case.then.solutions);
    if case.when.is_otherwise() {
        format!("{} & \\text{{otherwise}}", solution_latex)
    } else {
        let condition_latex = cas_formatter::condition_set_to_latex(&case.when, context);
        format!("{} & \\text{{if }} {}", solution_latex, condition_latex)
    }
}
