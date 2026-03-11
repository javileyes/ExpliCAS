use super::super::{TimelineHtml, VerbosityLevel};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

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
}
