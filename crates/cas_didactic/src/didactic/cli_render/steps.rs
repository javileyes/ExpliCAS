mod render;
mod state;

use self::render::lines::{
    render_after_line, render_assumption_lines, render_before_line, render_engine_substeps_lines,
    render_rule_with_scope_line, render_step_header, render_succinct_step_line,
};
use self::render::visibility::{render_rule_visible_change, render_step_visible_change};
use self::state::advance_current_root;
use super::super::display_policy::{
    render_cli_enriched_substeps_lines, CliSubstepsRenderState, StepDisplayMode,
};
use super::super::enrich_steps;
use super::super::step_visibility::should_show_simplify_step;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn render_simplification_step_lines(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_prefs: &cas_formatter::root_style::StylePreferences,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    let mut lines = Vec::new();
    if display_mode != StepDisplayMode::Succinct {
        lines.push("Steps:".to_string());
    }

    let enriched_steps = enrich_steps(ctx, expr, steps.to_vec());
    let mut current_root = expr;
    let mut step_count = 0;
    let mut cli_substeps_state = CliSubstepsRenderState::default();

    for (step_idx, step) in steps.iter().enumerate() {
        if should_show_simplify_step(step, display_mode) {
            if !render_step_visible_change(ctx, step, style_prefs) {
                current_root = advance_current_root(ctx, current_root, step);
                continue;
            }

            step_count += 1;

            if display_mode == StepDisplayMode::Succinct {
                current_root = advance_current_root(ctx, current_root, step);
                lines.push(render_succinct_step_line(ctx, current_root, style_prefs));
                continue;
            }

            lines.push(render_step_header(step_count, step));
            lines.push(render_before_line(
                ctx,
                step.global_before.unwrap_or(current_root),
                style_prefs,
            ));

            if let Some(enriched_step) = enriched_steps.get(step_idx) {
                lines.extend(render_cli_enriched_substeps_lines(
                    enriched_step,
                    &mut cli_substeps_state,
                ));
            }

            if !render_rule_visible_change(ctx, step, style_prefs) {
                current_root = advance_current_root(ctx, current_root, step);
                continue;
            }

            lines.push(render_rule_with_scope_line(ctx, step, style_prefs));
            lines.extend(render_engine_substeps_lines(step));

            current_root = advance_current_root(ctx, current_root, step);
            lines.push(render_after_line(ctx, current_root, style_prefs));
            lines.extend(render_assumption_lines(step));
        } else {
            current_root = advance_current_root(ctx, current_root, step);
        }
    }

    lines
}
