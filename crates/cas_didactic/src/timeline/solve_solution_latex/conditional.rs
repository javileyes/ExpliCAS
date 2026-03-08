mod cases;
mod render;

use cas_ast::{Context, SolutionSet};

pub(super) fn render_conditional_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    let SolutionSet::Conditional(cases) = solution_set else {
        unreachable!("conditional renderer only supports conditional solution sets");
    };

    let case_strs = cases::collect_conditional_case_lines(context, cases);
    render::render_conditional_case_lines(&case_strs)
}
