use super::simplify::{TimelineHtml, VerbosityLevel};
use super::solve::SolveTimelineHtml;
use crate::cas_solver::{SolveStep, Step};
use cas_ast::{Context, Equation, ExprId, SolutionSet};

/// Render full HTML for simplification timeline.
pub fn render_simplify_timeline_html(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    verbosity: VerbosityLevel,
    input_string: Option<&str>,
) -> String {
    let mut timeline = TimelineHtml::new_with_result_and_style(
        context,
        steps,
        original_expr,
        simplified_result,
        verbosity,
        input_string,
    );
    timeline.to_html()
}

/// Render full HTML for solve timeline.
pub fn render_solve_timeline_html(
    context: &mut Context,
    steps: &[SolveStep],
    original_eq: &Equation,
    solution_set: &SolutionSet,
    var: &str,
) -> String {
    let mut timeline = SolveTimelineHtml::new(context, steps, original_eq, solution_set, var);
    timeline.to_html()
}
