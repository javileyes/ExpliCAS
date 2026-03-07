mod detailed;
mod succinct;

use super::super::super::display_policy::{render_cli_enriched_substeps_lines, StepDisplayMode};
use super::super::super::step_visibility::should_show_simplify_step;
use super::super::super::EnrichedStep;
use super::render::visibility::{render_rule_visible_change, render_step_visible_change};
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
    if !should_show_simplify_step(step, display_mode) {
        state.advance(ctx, step);
        return None;
    }

    if !render_step_visible_change(ctx, step, style_prefs) {
        state.advance(ctx, step);
        return None;
    }

    let step_number = state.next_step_number();

    if display_mode == StepDisplayMode::Succinct {
        return Some(succinct::render_succinct_step_lines(
            ctx,
            step,
            style_prefs,
            state,
        ));
    }

    let mut lines = detailed::render_step_prelude(
        ctx,
        step,
        enriched_step,
        style_prefs,
        step_number,
        state,
        render_cli_enriched_substeps_lines,
    );

    if !render_rule_visible_change(ctx, step, style_prefs) {
        state.advance(ctx, step);
        return Some(lines);
    }

    lines.extend(detailed::render_step_postrule(
        ctx,
        step,
        style_prefs,
        state,
    ));
    Some(lines)
}
