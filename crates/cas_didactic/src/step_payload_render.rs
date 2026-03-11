mod path;
mod rule;
mod substeps;

use crate::cas_solver::Step;
use cas_api_models::SubStepWire;
use cas_ast::{ExprId, ExprPath};

pub(crate) fn collect_step_payload_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepWire> {
    substeps::collect_step_payload_substeps(step, enriched)
}

pub(crate) fn render_step_path_latex(
    ctx: &cas_ast::Context,
    expr_id: ExprId,
    expr_path: ExprPath,
    color: cas_formatter::HighlightColor,
) -> String {
    path::render_step_path_latex(ctx, expr_id, expr_path, color)
}

pub(crate) fn render_local_rule_latex(
    ctx: &cas_ast::Context,
    before_expr: ExprId,
    after_expr: ExprId,
) -> String {
    rule::render_local_rule_latex(ctx, before_expr, after_expr)
}
