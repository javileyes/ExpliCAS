use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

use super::build::render_human_expr;

pub(super) fn collect_event_step_payloads(events: &[EngineEvent], ctx: &Context) -> Vec<StepWire> {
    let mut wires: Vec<StepWire> = events
        .iter()
        .filter_map(|event| build_event_step_payload(event, ctx))
        .collect();
    // Re-number after dropping no-op steps so the displayed indices stay 1..n with no gaps.
    for (display_index, wire) in wires.iter_mut().enumerate() {
        wire.index = display_index + 1;
    }
    wires
}

fn build_event_step_payload(event: &EngineEvent, ctx: &Context) -> Option<StepWire> {
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
            let before_human = render_human_expr(ctx, before_expr);
            let after_human = render_human_expr(ctx, after_expr);

            // Drop a no-op step: when the displayed expression is unchanged the step teaches nothing
            // (the event/equiv path previously emitted ~10/19 such canonicalization no-ops, e.g. "Quitar
            // paréntesis"/"Reescribir la división" that render identical before -> after). The normal
            // step path already filters these; mirror it here. Always-keep rules (domain assumptions,
            // etc.) are never dropped.
            if before_human == after_human
                && !cas_solver_core::step_rules::is_always_keep_step_rule_name(rule_name)
            {
                return None;
            }

            Some(StepWire {
                index: 0, // reassigned in collect_event_step_payloads after the no-op filter
                rule: crate::didactic::visible_rule_name(rule_name).to_string(),
                // Build the colored `before -> after` transform like the normal step path. The raw
                // `rule_name` string used to land here and the web wraps `rule_latex` in `$...$`, so
                // the English rule name rendered as garbled MathJax. Use the LOCAL change.
                rule_latex: crate::step_payload_render::render_local_rule_latex(
                    ctx, *before, *after,
                ),
                before: before_human,
                after: after_human,
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
