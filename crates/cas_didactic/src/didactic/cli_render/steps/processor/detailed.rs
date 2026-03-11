mod postrule;
mod prelude;

use super::super::super::super::display_policy::CliSubstepsRenderState;
use super::super::state::StepLoopState;
use crate::runtime::Step;
use crate::EnrichedStep;
use cas_ast::Context;

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
    prelude::render_step_prelude(
        ctx,
        step,
        enriched_step,
        style_prefs,
        step_number,
        state,
        render_cli_enriched_substeps_lines,
    )
}

pub(super) fn render_step_postrule(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    state: &mut StepLoopState,
) -> Vec<String> {
    postrule::render_step_postrule(ctx, step, style_prefs, state)
}
