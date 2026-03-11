use super::super::super::simplify_init::build_simplify_timeline_init;
use super::super::{TimelineHtml, VerbosityLevel};
use super::build;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

impl<'a> TimelineHtml<'a> {
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
        build::build_timeline_html(
            context,
            steps,
            original_expr,
            simplified_result,
            verbosity,
            input_string,
            build_simplify_timeline_init,
        )
    }
}
