use super::super::super::render::lines::{
    render_after_line, render_assumption_lines, render_engine_substeps_lines,
    render_rule_with_scope_line,
};
use super::super::super::state::StepLoopState;
use crate::runtime::Step;
use cas_ast::Context;

pub(super) fn render_step_postrule(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    state: &mut StepLoopState,
) -> Vec<String> {
    let mut lines = vec![render_rule_with_scope_line(ctx, step, style_prefs)];
    lines.extend(render_engine_substeps_lines(step));

    state.advance(ctx, step);
    lines.push(render_after_line(ctx, state.current_root(), style_prefs));
    lines.extend(render_assumption_lines(step));
    lines
}
