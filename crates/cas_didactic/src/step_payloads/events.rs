use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

use super::build::render_human_expr;

pub(super) fn collect_event_step_payloads(events: &[EngineEvent], ctx: &Context) -> Vec<StepWire> {
    events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| build_event_step_payload(index + 1, event, ctx))
        .collect()
}

fn build_event_step_payload(index: usize, event: &EngineEvent, ctx: &Context) -> Option<StepWire> {
    match event {
        EngineEvent::RuleApplied {
            rule_name,
            before,
            after,
            global_before,
            global_after,
            ..
        } => {
            let before_expr = global_before.unwrap_or(*before);
            let after_expr = global_after.unwrap_or(*after);

            Some(StepWire {
                index,
                rule: crate::didactic::visible_rule_name(rule_name).to_string(),
                rule_latex: rule_name.clone(),
                before: render_human_expr(ctx, before_expr),
                after: render_human_expr(ctx, after_expr),
                before_latex: cas_formatter::LaTeXExpr {
                    context: ctx,
                    id: before_expr,
                }
                .to_latex()
                .to_string(),
                after_latex: cas_formatter::LaTeXExpr {
                    context: ctx,
                    id: after_expr,
                }
                .to_latex()
                .to_string(),
                substeps: Vec::new(),
            })
        }
    }
}
