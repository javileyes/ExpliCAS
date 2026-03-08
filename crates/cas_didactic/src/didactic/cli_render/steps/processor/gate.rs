use super::super::super::super::display_policy::StepDisplayMode;
use super::super::super::super::step_visibility::should_show_simplify_step;
use super::super::render::visibility::render_step_visible_change;
use super::super::state::StepLoopState;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn begin_step_render(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    display_mode: StepDisplayMode,
    state: &mut StepLoopState,
) -> Option<usize> {
    if !should_show_simplify_step(step, display_mode) {
        state.advance(ctx, step);
        return None;
    }

    if !render_step_visible_change(ctx, step, style_prefs) {
        state.advance(ctx, step);
        return None;
    }

    Some(state.next_step_number())
}
