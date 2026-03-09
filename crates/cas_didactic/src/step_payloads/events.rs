use cas_api_models::StepJson;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

pub(super) fn collect_event_step_payloads(events: &[EngineEvent], ctx: &Context) -> Vec<StepJson> {
    events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| build_event_step_payload(index + 1, event, ctx))
        .collect()
}

fn build_event_step_payload(index: usize, event: &EngineEvent, ctx: &Context) -> Option<StepJson> {
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

            Some(StepJson {
                index,
                rule: rule_name.clone(),
                rule_latex: rule_name.clone(),
                before: format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: before_expr
                    }
                ),
                after: format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: after_expr
                    }
                ),
                before_latex: format!(
                    "{}",
                    cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: before_expr
                    }
                    .to_latex()
                ),
                after_latex: format!(
                    "{}",
                    cas_formatter::LaTeXExpr {
                        context: ctx,
                        id: after_expr
                    }
                    .to_latex()
                ),
                substeps: Vec::new(),
            })
        }
    }
}
