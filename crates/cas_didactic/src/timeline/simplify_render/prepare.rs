mod display;
mod filter;

use crate::runtime::Step;
use cas_ast::{Context, ExprId};
use cas_formatter::DisplayContext;
use std::collections::HashSet;

pub(super) const TIMELINE_CLOSING_HTML: &str = "    </div>\n";

pub(super) struct TimelineRenderPreparation {
    pub(super) display_hints: DisplayContext,
    pub(super) filtered_indices: HashSet<*const Step>,
}

pub(super) fn prepare_timeline_render(
    context: &mut Context,
    _html: &mut String,
    original_expr: ExprId,
    steps: &[Step],
    simplified_result: Option<ExprId>,
    filtered_steps: &[&Step],
) -> TimelineRenderPreparation {
    TimelineRenderPreparation {
        display_hints: display::build_timeline_display_hints(
            context,
            original_expr,
            steps,
            simplified_result,
        ),
        filtered_indices: filter::collect_filtered_indices(filtered_steps),
    }
}
