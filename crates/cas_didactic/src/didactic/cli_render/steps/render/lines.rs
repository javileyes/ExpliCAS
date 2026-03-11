mod details;
mod display;
mod succinct;

use super::local_rule_expr_ids;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(crate) fn render_succinct_step_line(
    ctx: &mut Context,
    current_root: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    succinct::render_succinct_step_line(ctx, current_root, style_prefs)
}

pub(crate) fn render_step_header(step_count: usize, step: &Step) -> String {
    display::render_step_header(step_count, step)
}

pub(crate) fn render_before_line(
    ctx: &mut Context,
    before_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    display::render_before_line(ctx, before_expr, style_prefs)
}

pub(crate) fn render_rule_with_scope_line(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    display::render_rule_with_scope_line(ctx, step, style_prefs, local_rule_expr_ids)
}

pub(crate) fn render_engine_substeps_lines(step: &Step) -> Vec<String> {
    details::render_engine_substeps_lines(step)
}

pub(crate) fn render_after_line(
    ctx: &mut Context,
    after_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    display::render_after_line(ctx, after_expr, style_prefs)
}

pub(crate) fn render_assumption_lines(step: &Step) -> Vec<String> {
    details::render_assumption_lines(step)
}
