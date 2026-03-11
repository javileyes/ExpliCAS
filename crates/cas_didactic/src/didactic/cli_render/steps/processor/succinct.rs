use super::super::render::lines::render_succinct_step_line;
use super::super::state::StepLoopState;
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn render_succinct_step_lines(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    state: &mut StepLoopState,
) -> Vec<String> {
    state.advance(ctx, step);
    vec![render_succinct_step_line(
        ctx,
        state.current_root(),
        style_prefs,
    )]
}
