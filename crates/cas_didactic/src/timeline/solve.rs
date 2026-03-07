use super::solve_init::build_solve_timeline_title;
use super::solve_page::{render_solve_timeline_html_header, solve_timeline_html_footer};
use super::solve_render::render_equation_latex;
use super::solve_timeline_render::render_solve_timeline_body;
use cas_ast::{Context, Equation, SolutionSet};
use cas_formatter::clean_latex_identities;
use cas_solver::SolveStep;

/// Timeline HTML generator for equation solving steps
pub struct SolveTimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [SolveStep],
    original_eq: &'a Equation,
    solution_set: &'a SolutionSet,
    var: String,
    title: String,
}

impl<'a> SolveTimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [SolveStep],
        original_eq: &'a Equation,
        solution_set: &'a SolutionSet,
        var: &str,
    ) -> Self {
        let title = build_solve_timeline_title(context, original_eq);
        Self {
            context,
            steps,
            original_eq,
            solution_set,
            var: var.to_string(),
            title,
        }
    }

    /// Generate complete HTML document for solve steps
    pub fn to_html(&mut self) -> String {
        let original_latex = render_equation_latex(self.context, self.original_eq);
        let mut html = render_solve_timeline_html_header(&self.title, &self.var, &original_latex);
        html.push_str(&self.render_solve_timeline());
        html.push_str(&solve_timeline_html_footer());

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
    }

    fn render_solve_timeline(&mut self) -> String {
        render_solve_timeline_body(self.context, self.steps, self.solution_set, &self.var)
    }
}
