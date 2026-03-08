use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

pub(crate) fn build_display_eval_steps_from_events(
    events: &[EngineEvent],
    ctx: &Context,
) -> crate::DisplayEvalSteps {
    let raw_steps = events
        .iter()
        .filter_map(|event| build_step_from_event(event, ctx))
        .collect();
    cas_solver_core::eval_step_pipeline::to_display_eval_steps(raw_steps)
}

fn build_step_from_event(event: &EngineEvent, ctx: &Context) -> Option<crate::Step> {
    match event {
        EngineEvent::RuleApplied {
            rule_name,
            before,
            after,
            global_before,
            global_after,
            is_chained,
        } => {
            let mut step = match (global_before, global_after) {
                (Some(global_before), Some(global_after)) => crate::Step::with_snapshots(
                    rule_name,
                    rule_name,
                    *before,
                    *after,
                    Vec::new(),
                    Some(ctx),
                    *global_before,
                    *global_after,
                ),
                _ => crate::Step::new(rule_name, rule_name, *before, *after, Vec::new(), Some(ctx)),
            };
            step.importance = crate::ImportanceLevel::Medium;
            if *is_chained {
                step.meta_mut().is_chained = true;
            }
            Some(step)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::build_display_eval_steps_from_events;

    #[test]
    fn builds_medium_snapshot_steps_from_rule_events() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);

        let steps = build_display_eval_steps_from_events(
            &[cas_solver_core::engine_events::EngineEvent::RuleApplied {
                rule_name: "TestRule".to_string(),
                before: one,
                after: two,
                global_before: Some(one),
                global_after: Some(two),
                is_chained: true,
            }],
            &ctx,
        );

        assert_eq!(steps.len(), 1);
        let step = &steps[0];
        assert_eq!(step.description, "TestRule");
        assert_eq!(step.rule_name, "TestRule");
        assert_eq!(step.global_before, Some(one));
        assert_eq!(step.global_after, Some(two));
        assert_eq!(step.importance, crate::ImportanceLevel::Medium);
        assert!(step.is_chained());
    }
}
