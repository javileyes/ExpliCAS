use super::SolveTimelineHtml;
use crate::cas_solver::SolveStep;
use cas_ast::{Context, Equation, SolutionSet};

pub(super) fn build_solve_timeline_html<'a>(
    context: &'a mut Context,
    steps: &'a [SolveStep],
    original_eq: &'a Equation,
    solution_set: &'a SolutionSet,
    var: &str,
    build_solve_timeline_title: fn(&Context, &Equation) -> String,
) -> SolveTimelineHtml<'a> {
    let title = build_solve_timeline_title(context, original_eq);
    SolveTimelineHtml {
        context,
        steps,
        original_eq,
        solution_set,
        var: var.to_string(),
        title,
    }
}
