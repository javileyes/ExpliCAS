use super::super::super::render::visibility::render_rule_visible_change;
use super::super::super::state::StepLoopState;
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn extend_with_visible_rule_lines(
    lines: &mut Vec<String>,
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    state: &mut StepLoopState,
    render_step_postrule: fn(
        &mut Context,
        &Step,
        &cas_formatter::root_style::StylePreferences,
        &mut StepLoopState,
    ) -> Vec<String>,
) {
    if !render_rule_visible_change(ctx, step, style_prefs) {
        state.advance(ctx, step);
        return;
    }

    lines.extend(render_step_postrule(ctx, step, style_prefs, state));
}
