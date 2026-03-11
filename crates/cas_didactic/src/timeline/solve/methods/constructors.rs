use super::super::super::solve_init::build_solve_timeline_title;
use super::super::build;
use super::super::SolveTimelineHtml;
use crate::runtime::SolveStep;
use cas_ast::{Context, Equation, SolutionSet};

pub(super) fn build_solve_timeline_html<'a>(
    context: &'a mut Context,
    steps: &'a [SolveStep],
    original_eq: &'a Equation,
    solution_set: &'a SolutionSet,
    var: &str,
) -> SolveTimelineHtml<'a> {
    build::build_solve_timeline_html(
        context,
        steps,
        original_eq,
        solution_set,
        var,
        build_solve_timeline_title,
    )
}
