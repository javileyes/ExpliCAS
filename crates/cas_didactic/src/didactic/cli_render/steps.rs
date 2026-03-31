mod processor;
mod render;
mod state;

use self::processor::render_step_lines;
use self::state::StepLoopState;
use super::super::enrich_steps;
use super::super::{display_policy::StepDisplayMode, StepVisibility};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(super) fn render_simplification_step_lines(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_prefs: &cas_formatter::root_style::StylePreferences,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    let visibility = match display_mode {
        StepDisplayMode::Verbose => StepVisibility::All,
        StepDisplayMode::Succinct | StepDisplayMode::Normal => StepVisibility::MediumOrHigher,
        StepDisplayMode::None => StepVisibility::All,
    };

    let mut lines = Vec::new();
    if display_mode != StepDisplayMode::Succinct {
        lines.push("Steps:".to_string());
    }

    let enriched_steps = enrich_steps(ctx, expr, steps.to_vec());
    let mut state = StepLoopState::new(expr);

    for (step_idx, step) in steps.iter().enumerate() {
        if crate::didactic::should_absorb_preparatory_step_at(steps, step_idx, visibility) {
            state.advance(ctx, step);
            continue;
        }
        if let Some(step_lines) = render_step_lines(
            ctx,
            step,
            enriched_steps.get(step_idx),
            style_prefs,
            display_mode,
            &mut state,
        ) {
            lines.extend(step_lines);
        }
    }

    lines
}
