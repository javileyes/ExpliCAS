use super::super::{TimelineHtml, VerbosityLevel};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

#[allow(clippy::type_complexity)]
pub(super) fn build_timeline_html<'a>(
    context: &'a mut Context,
    steps: &'a [Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    verbosity: VerbosityLevel,
    input_string: Option<&str>,
    build_simplify_timeline_init: fn(
        &mut Context,
        &[Step],
        ExprId,
        Option<ExprId>,
        Option<&str>,
    )
        -> super::super::super::simplify_init::SimplifyTimelineInit,
) -> TimelineHtml<'a> {
    let init = build_simplify_timeline_init(
        context,
        steps,
        original_expr,
        simplified_result,
        input_string,
    );

    TimelineHtml {
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
