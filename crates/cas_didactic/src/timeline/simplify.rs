use super::simplify_init::build_simplify_timeline_init;
use super::simplify_page::{render_simplify_timeline_html_header, simplify_timeline_html_footer};
use super::simplify_render::render_timeline_filtered_enriched;
use cas_ast::{Context, ExprId};
use cas_formatter::clean_latex_identities;
use cas_solver::{ImplicitCondition, Step};

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

#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only high-importance steps (Factor, Expand, Integrate, etc.)
    Normal,  // Medium+ importance steps (most transformations)
    Verbose, // All steps including trivial ones
}

impl VerbosityLevel {
    fn step_visibility(&self) -> crate::didactic::StepVisibility {
        match self {
            VerbosityLevel::Verbose => crate::didactic::StepVisibility::All,
            VerbosityLevel::Low => crate::didactic::StepVisibility::HighOrHigher,
            VerbosityLevel::Normal => crate::didactic::StepVisibility::MediumOrHigher,
        }
    }
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
        // Filter steps based on verbosity level
        let filtered_steps: Vec<&Step> = self
            .steps
            .iter()
            .filter(|step| {
                crate::didactic::step_matches_visibility(
                    step,
                    self.verbosity_level.step_visibility(),
                )
            })
            .collect();

        // Enrich steps with didactic sub-steps
        let enriched_steps =
            crate::didactic::enrich_steps(self.context, self.original_expr, self.steps.to_vec());

        let mut html = render_simplify_timeline_html_header(&self.title);
        html.push_str(&render_timeline_filtered_enriched(
            self.context,
            self.steps,
            self.original_expr,
            self.simplified_result,
            &self.global_requires,
            &self.style_prefs,
            &filtered_steps,
            &enriched_steps,
        ));
        html.push_str(&simplify_timeline_html_footer());

        // Clean up identity patterns like "\cdot 1" for better display
        clean_latex_identities(&html)
    }
}
