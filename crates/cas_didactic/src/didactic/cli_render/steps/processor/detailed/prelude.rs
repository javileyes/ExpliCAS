use super::super::super::super::super::display_policy::CliSubstepsRenderState;
use super::super::super::render::lines::{render_before_line, render_step_header};
use super::super::super::state::StepLoopState;
use crate::cas_solver::Step;
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
