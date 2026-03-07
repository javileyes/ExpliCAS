mod visible;

use super::super::super::super::display_policy::CliSubstepsRenderState;
use super::super::super::super::EnrichedStep;
use super::super::state::StepLoopState;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn render_detailed_step_lines(
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
    render_step_prelude: fn(
        &mut Context,
        &Step,
        Option<&EnrichedStep>,
        &cas_formatter::root_style::StylePreferences,
        usize,
        &mut StepLoopState,
        fn(&EnrichedStep, &mut CliSubstepsRenderState) -> Vec<String>,
    ) -> Vec<String>,
    render_step_postrule: fn(
        &mut Context,
        &Step,
        &cas_formatter::root_style::StylePreferences,
        &mut StepLoopState,
    ) -> Vec<String>,
) -> Vec<String> {
    let mut lines = render_step_prelude(
        ctx,
        step,
        enriched_step,
        style_prefs,
        step_number,
        state,
        render_cli_enriched_substeps_lines,
    );

    visible::extend_with_visible_rule_lines(
        &mut lines,
        ctx,
        step,
        style_prefs,
        state,
        render_step_postrule,
    );
    lines
}
