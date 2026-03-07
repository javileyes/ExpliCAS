pub(super) fn evaluate_timeline_simplify_standard<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: crate::SolverEvalSession,
{
    let was_collecting = engine.simplifier.collect_steps();
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
        Ok(crate::TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: output_view.steps,
        })
    })();
    engine.simplifier.set_collect_steps(was_collecting);
    result
}
