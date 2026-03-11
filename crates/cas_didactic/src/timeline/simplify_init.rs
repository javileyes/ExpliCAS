mod requires;
mod title;

use crate::cas_solver::{ImplicitCondition, Step};
use cas_ast::{Context, ExprId};

pub(super) struct SimplifyTimelineInit {
    pub title: String,
    pub global_requires: Vec<ImplicitCondition>,
    pub style_prefs: cas_formatter::root_style::StylePreferences,
}

pub(super) fn build_simplify_timeline_init(
    context: &mut Context,
    _steps: &[Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    input_string: Option<&str>,
) -> SimplifyTimelineInit {
    let (title, style_prefs) =
        title::build_timeline_title_and_style_prefs(context, original_expr, input_string);
    let global_requires =
        requires::collect_timeline_global_requires(context, original_expr, simplified_result);

    SimplifyTimelineInit {
        title,
        global_requires,
        style_prefs,
    }
}
