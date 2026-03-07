mod conditional;
mod discrete;
mod display;
mod interval;
mod messages;

use cas_ast::{Context, SolutionSet};

pub(super) fn format_timeline_solve_result_line(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    messages::format_timeline_solve_result_line(
        context,
        solution_set,
        display::display_solution_set,
    )
}

pub(super) fn format_timeline_solve_no_steps_message(
    context: &Context,
    solution_set: &SolutionSet,
) -> String {
    messages::format_timeline_solve_no_steps_message(
        context,
        solution_set,
        format_timeline_solve_result_line,
    )
}
