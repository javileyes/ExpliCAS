use cas_ast::{Context, SolutionSet};

pub(super) fn format_timeline_solve_result_line(
    context: &Context,
    solution_set: &SolutionSet,
    display_solution_set: fn(&Context, &SolutionSet) -> String,
) -> String {
    format!("Result: {}", display_solution_set(context, solution_set))
}

pub(super) fn format_timeline_solve_no_steps_message(
    context: &Context,
    solution_set: &SolutionSet,
    format_timeline_solve_result_line: fn(&Context, &SolutionSet) -> String,
) -> String {
    format!(
        "No solving steps to visualize.\n{}",
        format_timeline_solve_result_line(context, solution_set)
    )
}
