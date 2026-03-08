mod constructors;
mod html;

use super::SolveTimelineHtml;

impl<'a> SolveTimelineHtml<'a> {
    pub fn new(
        context: &'a mut cas_ast::Context,
        steps: &'a [cas_solver::SolveStep],
        original_eq: &'a cas_ast::Equation,
        solution_set: &'a cas_ast::SolutionSet,
        var: &str,
    ) -> Self {
        constructors::build_solve_timeline_html(context, steps, original_eq, solution_set, var)
    }

    /// Generate complete HTML document for solve steps
    pub fn to_html(&mut self) -> String {
        html::render_solve_timeline_html(self)
    }
}
