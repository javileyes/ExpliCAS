mod rule;
mod substeps;

use crate::runtime::Step;
use cas_api_models::SubStepWire;
use cas_ast::ExprId;

pub(crate) fn collect_step_payload_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepWire> {
    substeps::collect_step_payload_substeps(step, enriched)
}

pub(crate) fn render_local_rule_latex(
    ctx: &cas_ast::Context,
    before_expr: ExprId,
    after_expr: ExprId,
) -> String {
    rule::render_local_rule_latex(ctx, before_expr, after_expr)
}
