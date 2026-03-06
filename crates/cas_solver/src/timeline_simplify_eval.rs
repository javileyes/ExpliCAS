fn evaluate_timeline_simplify_aggressive(
    simplifier: &mut crate::Simplifier,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError> {
    let mut temp_simplifier = crate::Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| crate::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(crate::TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: crate::to_display_steps(steps),
        })
    })();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}

fn evaluate_timeline_simplify_standard<S>(
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

pub(crate) fn evaluate_timeline_simplify_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    aggressive: bool,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: crate::SolverEvalSession,
{
    if aggressive {
        evaluate_timeline_simplify_aggressive(&mut engine.simplifier, input)
    } else {
        evaluate_timeline_simplify_standard(engine, session, input)
    }
}
