mod conditional;
mod expr;
mod interval;
mod non_nested;

use cas_ast::{Context, SolutionSet};

pub(super) fn render_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Conditional(_) => {
            conditional::render_conditional_solution_set_to_latex(context, solution_set)
        }
        _ => non_nested::render_non_nested_solution_set_to_latex(context, solution_set),
    }
}

pub(super) fn render_inner_solution_set_to_latex(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    match solution_set {
        SolutionSet::Conditional(_) => r"\text{(nested conditional)}".to_string(),
        _ => non_nested::render_non_nested_solution_set_to_latex(context, solution_set),
    }
}
