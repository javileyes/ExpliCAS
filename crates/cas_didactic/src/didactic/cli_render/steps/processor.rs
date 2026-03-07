mod detailed;
mod detailed_flow;
mod gate;
mod succinct;

use super::super::super::display_policy::{render_cli_enriched_substeps_lines, StepDisplayMode};
use super::super::super::EnrichedStep;
use super::state::StepLoopState;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn render_step_lines(
    ctx: &mut Context,
    step: &Step,
    enriched_step: Option<&EnrichedStep>,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    display_mode: StepDisplayMode,
    state: &mut StepLoopState,
) -> Option<Vec<String>> {
    let step_number = gate::begin_step_render(ctx, step, style_prefs, display_mode, state)?;

    if display_mode == StepDisplayMode::Succinct {
        return Some(succinct::render_succinct_step_lines(
            ctx,
            step,
            style_prefs,
            state,
        ));
    }

    Some(detailed_flow::render_detailed_step_lines(
        ctx,
        step,
        enriched_step,
        style_prefs,
        step_number,
        state,
        render_cli_enriched_substeps_lines,
        detailed::render_step_prelude,
        detailed::render_step_postrule,
    ))
}
