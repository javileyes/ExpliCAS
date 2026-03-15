use super::super::super::solve_page::{
    render_solve_timeline_html_header, solve_timeline_html_footer,
};
use super::super::super::solve_render::render_equation_latex;
use super::super::super::solve_timeline_render::render_solve_timeline_body;
use super::super::document;
use super::super::SolveTimelineHtml;
use cas_formatter::clean_latex_identities;

pub(super) fn render_solve_timeline_html(timeline: &mut SolveTimelineHtml<'_>) -> String {
    document::render_solve_timeline_document(
        timeline.context,
        timeline.steps,
        timeline.original_eq,
        timeline.solution_set,
        timeline.var.as_str(),
        timeline.title.as_str(),
        render_equation_latex,
        render_solve_timeline_html_header,
        render_solve_timeline_body,
        solve_timeline_html_footer,
        clean_latex_identities,
    )
}
