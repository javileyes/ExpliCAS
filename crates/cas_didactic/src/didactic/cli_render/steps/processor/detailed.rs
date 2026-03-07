use super::super::super::super::display_policy::CliSubstepsRenderState;
use super::super::render::lines::{
    render_after_line, render_assumption_lines, render_before_line, render_engine_substeps_lines,
    render_rule_with_scope_line, render_step_header,
};
use super::super::state::StepLoopState;
use crate::EnrichedStep;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn render_step_prelude(
    ctx: &mut Context,
    step: &Step,
    enriched_step: Option<&EnrichedStep>,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    step_number: usize,
    state: &mut StepLoopState,
    render_cli_enriched_substeps_lines: fn(
        &EnrichedStep,
        &mut CliSubstepsRenderState,
    ) -> Vec<String>,
) -> Vec<String> {
    let mut lines = vec![
        render_step_header(step_number, step),
        render_before_line(
            ctx,
            step.global_before.unwrap_or(state.current_root()),
            style_prefs,
        ),
    ];

    if let Some(enriched_step) = enriched_step {
        lines.extend(render_cli_enriched_substeps_lines(
            enriched_step,
            state.cli_substeps_state_mut(),
        ));
    }

    lines
}

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
