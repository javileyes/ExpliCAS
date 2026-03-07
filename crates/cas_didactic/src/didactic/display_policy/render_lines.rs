use super::plans::build_cli_substeps_render_plan;
use super::{latex_to_plain_text, EnrichedStep};

/// Mutable rendering state for CLI enriched sub-step blocks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CliSubstepsRenderState {
    dedupe_shown: bool,
}

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

    let mut lines = Vec::new();
    if let Some(header) = render_plan.header {
        lines.push(format!("   {}", header));
    }
    for sub in &enriched_step.sub_steps {
        lines.push(format!("      → {}", sub.description));
        if !sub.before_expr.is_empty() {
            lines.push(format!(
                "        {} → {}",
                latex_to_plain_text(&sub.before_expr),
                latex_to_plain_text(&sub.after_expr)
            ));
        }
    }

    lines
}
