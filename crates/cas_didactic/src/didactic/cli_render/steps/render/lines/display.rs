mod around;
mod expr;
mod header;
mod rule;

use crate::cas_solver::Step;
use cas_ast::{Context, ExprId};

pub(super) fn render_step_header(step_count: usize, step: &Step) -> String {
    header::render_step_header(step_count, step)
}

pub(super) fn render_before_line(
    ctx: &mut Context,
    before_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    around::render_before_line(ctx, before_expr, style_prefs, expr::display_expr_styled)
}

pub(super) fn render_rule_with_scope_line(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    local_rule_expr_ids: fn(&Step) -> (ExprId, ExprId),
) -> String {
    rule::render_rule_with_scope_line(
        ctx,
        step,
        style_prefs,
        local_rule_expr_ids,
        expr::display_expr_styled,
    )
}

pub(super) fn render_after_line(
    ctx: &mut Context,
    after_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    around::render_after_line(ctx, after_expr, style_prefs, expr::display_expr_styled)
}
