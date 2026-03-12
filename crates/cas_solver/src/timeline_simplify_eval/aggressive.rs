pub(super) fn evaluate_timeline_simplify_aggressive(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError> {
    let mut temp_simplifier = crate::Simplifier::with_default_rules();
    let collector = cas_solver_core::engine_event_collector::EngineEventCollector::new();
    let previous_listener =
        temp_simplifier.replace_step_listener(Some(Box::new(collector.clone())));
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| crate::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        let mut steps = crate::runtime::to_display_steps(steps);
        if steps.is_empty() {
            let fallback_steps =
                crate::engine_event_display_steps::build_display_eval_steps_from_events(
                    &collector.events(),
                    &temp_simplifier.context,
                );
            if !fallback_steps.is_empty() {
                steps = fallback_steps;
            }
        }
        Ok(crate::TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps,
        })
    })();
    let _ = temp_simplifier.replace_step_listener(previous_listener);
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}
