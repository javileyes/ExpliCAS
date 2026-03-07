use super::super::simplify_render::render_timeline_filtered_enriched;
use super::{document, prepare, TimelineHtml};

impl TimelineHtml<'_> {
    /// Generate complete HTML document
    pub fn to_html(&mut self) -> String {
        let render_data = prepare::prepare_timeline_render_data(
            self.context,
            self.steps,
            self.original_expr,
            self.verbosity_level,
        );
        let body = render_timeline_filtered_enriched(
            self.context,
            self.steps,
            self.original_expr,
            self.simplified_result,
            &self.global_requires,
            &self.style_prefs,
            &render_data.filtered_steps,
            &render_data.enriched_steps,
        );
        document::render_simplify_timeline_document(&self.title, &body)
    }
}
