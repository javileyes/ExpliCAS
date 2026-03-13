use cas_ast::ExprId;
use cas_solver_core::engine_event_collector::EngineEventCollector;
use cas_solver_core::engine_events::EngineEvent;

use super::{error::FullSimplifyEvalError, FullSimplifyEvalOutput};

pub fn evaluate_full_simplify_input_with_resolver<F>(
    simplifier: &mut crate::Simplifier,
    input: &str,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
    resolve_expr: F,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError>
where
    F: FnOnce(&mut cas_ast::Context, ExprId) -> Result<ExprId, String>,
{
    let mut temp_simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let resolved_expr = resolve_expr(&mut temp_simplifier.context, parsed_expr)
            .map_err(FullSimplifyEvalError::Resolve)?;

        simplify_options.collect_steps = collect_steps;
        let collector = collect_steps.then(EngineEventCollector::new);
        let previous_listener = collector.as_ref().map(|collector| {
            temp_simplifier.replace_step_listener(Some(Box::new(collector.clone())))
        });

        let (simplified_expr, mut steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, simplify_options);

        if let Some(previous_listener) = previous_listener {
            let _ = temp_simplifier.replace_step_listener(previous_listener);
        }

        if steps.is_empty() {
            if let Some(collector) = collector.as_ref() {
                steps = fallback_steps_from_events(&collector.events(), &temp_simplifier.context);
            }
        }

        let _ = stats;
        Ok(FullSimplifyEvalOutput {
            resolved_expr,
            simplified_expr,
            steps,
        })
    })();

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);
    result
}

fn fallback_steps_from_events(events: &[EngineEvent], ctx: &cas_ast::Context) -> Vec<crate::Step> {
    crate::engine_event_display_steps::build_display_eval_steps_from_events(events, ctx)
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::fallback_steps_from_events;

    #[test]
    fn builds_full_simplify_steps_from_rule_events() {
        let mut ctx = cas_ast::Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);

        let steps = fallback_steps_from_events(
            &[cas_solver_core::engine_events::EngineEvent::RuleApplied {
                rule_name: "EventRule".to_string(),
                before: one,
                after: two,
                global_before: Some(one),
                global_after: Some(two),
                is_chained: false,
            }],
            &ctx,
        );

        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].description, "EventRule");
        assert_eq!(steps[0].global_before, Some(one));
        assert_eq!(steps[0].global_after, Some(two));
    }
}
