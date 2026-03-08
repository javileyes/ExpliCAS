pub(super) fn evaluate_timeline_simplify_standard<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: crate::SolverEvalSession,
{
    let was_collecting = engine.simplifier.collect_steps();
    let collector = cas_solver_core::engine_event_collector::EngineEventCollector::new();
    let previous_listener = engine
        .simplifier
        .replace_step_listener(Some(Box::new(collector.clone())));
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| crate::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let req = crate::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: crate::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| crate::TimelineSimplifyEvalError::Eval(e.to_string()))?;
        let output_view = crate::eval_output_view(&output);
        let simplified_expr = match output_view.result {
            crate::EvalResult::Expr(e) => e,
            _ => parsed_expr,
        };
        let mut steps = output_view.steps;
        if steps.is_empty() {
            let fallback_steps =
                crate::engine_event_display_steps::build_display_eval_steps_from_events(
                    &collector.events(),
                    &engine.simplifier.context,
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
    engine.simplifier.set_collect_steps(was_collecting);
    let _ = engine.simplifier.replace_step_listener(previous_listener);
    result
}
