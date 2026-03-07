mod document;
mod prepare;
mod verbosity;

use super::simplify_init::build_simplify_timeline_init;
use super::simplify_render::render_timeline_filtered_enriched;
use cas_ast::{Context, ExprId};
use cas_solver::{ImplicitCondition, Step};
pub use verbosity::VerbosityLevel;

/// Timeline HTML generator - exports simplification steps to interactive HTML
pub struct TimelineHtml<'a> {
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>, // Optional: the final simplified result
    title: String,
    verbosity_level: VerbosityLevel,
    /// V2.12.13: Global requires inferred from input expression.
    /// Shown at the end of the timeline, after final result.
    global_requires: Vec<ImplicitCondition>,
    /// V2.14.40: Style preferences derived from input string for consistent root rendering
    style_prefs: cas_formatter::root_style::StylePreferences,
}

impl<'a> TimelineHtml<'a> {
    pub fn new(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self::new_with_result(context, steps, original_expr, None, verbosity)
    }

    /// Create a new TimelineHtml with a known simplified result
    pub fn new_with_result(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self::new_with_result_and_style(
            context,
            steps,
            original_expr,
            simplified_result,
            verbosity,
            None,
        )
    }

    /// Create a new TimelineHtml with style preferences derived from input string
    /// V2.14.40: Enables consistent root rendering (exponential vs radical)
    pub fn new_with_result_and_style(
        context: &'a mut Context,
        steps: &'a [Step],
        original_expr: ExprId,
        simplified_result: Option<ExprId>,
        verbosity: VerbosityLevel,
        input_string: Option<&str>,
    ) -> Self {
        let init = build_simplify_timeline_init(
            context,
            steps,
            original_expr,
            simplified_result,
            input_string,
        );

        Self {
            context,
            steps,
            original_expr,
            simplified_result,
            title: init.title,
            verbosity_level: verbosity,
            global_requires: init.global_requires,
            style_prefs: init.style_prefs,
        }
    }

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
