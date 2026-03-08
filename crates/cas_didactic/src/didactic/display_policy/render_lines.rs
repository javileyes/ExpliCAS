mod lines;
mod state;

use super::plans::build_cli_substeps_render_plan;
use super::EnrichedStep;
pub(crate) use state::CliSubstepsRenderState;

/// Render enriched CLI sub-steps, applying the shared header/deduplication policy.
pub(crate) fn render_cli_enriched_substeps_lines(
    enriched_step: &EnrichedStep,
    state: &mut CliSubstepsRenderState,
) -> Vec<String> {
    if enriched_step.sub_steps.is_empty() {
        return Vec::new();
    }

    let render_plan = build_cli_substeps_render_plan(&enriched_step.sub_steps);
    let should_show = if render_plan.dedupe_once {
        !state.dedupe_shown
    } else {
        true
    };

    if !should_show {
        return Vec::new();
    }

    if render_plan.dedupe_once {
        state.dedupe_shown = true;
    }

    lines::render_cli_enriched_substeps_lines(enriched_step, render_plan.header)
}
