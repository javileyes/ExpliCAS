mod assumptions;
mod display;

use cas_ast::{Context, ExprId};

pub(crate) fn push_detailed_step_lines(
    lines: &mut Vec<String>,
    ctx: &mut Context,
    step: &crate::Step,
    style_prefs: &cas_formatter::StylePreferences,
    step_count: usize,
    current_root: ExprId,
) -> ExprId {
    let next_root = display::push_detailed_display_lines(
        lines,
        ctx,
        step,
        style_prefs,
        step_count,
        current_root,
    );
    assumptions::push_detailed_assumption_lines(lines, step);
    next_root
}
