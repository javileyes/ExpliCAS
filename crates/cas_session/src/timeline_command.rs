//! Session-level orchestration for `timeline` command dispatch.

fn evaluate_timeline_solve_command_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    opts: cas_solver::SolverOptions,
) -> Result<crate::TimelineSolveEvalOutput, crate::TimelineSolveEvalError> {
    let parsed_input = crate::solve_input_parse::parse_solve_command_input(input);
    let (equation, var) = crate::solve_input_parse::prepare_timeline_solve_equation(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
    )
    .map_err(crate::TimelineSolveEvalError::Prepare)?;

    let (solution_set, display_steps, diagnostics) =
        cas_solver::solve_with_display_steps(&equation, &var, simplifier, opts)
            .map_err(|e| crate::TimelineSolveEvalError::Solve(e.to_string()))?;

    Ok(crate::TimelineSolveEvalOutput {
        equation,
        var,
        solution_set,
        display_steps,
        diagnostics,
    })
}

fn evaluate_timeline_solve_with_eval_options(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<crate::TimelineSolveEvalOutput, crate::TimelineSolveEvalError> {
    simplifier.set_collect_steps(true);
    let opts = cas_solver::SolverOptions::from_eval_options(eval_options);
    evaluate_timeline_solve_command_input(simplifier, input, opts)
}

fn evaluate_timeline_simplify_aggressive(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError> {
    let mut temp_simplifier = cas_solver::Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| crate::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(crate::TimelineSimplifyEvalOutput {
            parsed_expr,
            simplified_expr,
            steps: cas_solver::to_display_steps(steps),
        })
    })();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    result
}

fn evaluate_timeline_simplify_standard<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    let was_collecting = engine.simplifier.collect_steps();
    engine.simplifier.set_collect_steps(true);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut engine.simplifier.context)
            .map_err(|e| crate::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let req = cas_solver::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_solver::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| crate::TimelineSimplifyEvalError::Eval(e.to_string()))?;
        let output_view = cas_solver::eval_output_view(&output);
        let simplified_expr = match output_view.result {
            cas_solver::EvalResult::Expr(e) => e,
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

fn evaluate_timeline_simplify_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    aggressive: bool,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    if aggressive {
        evaluate_timeline_simplify_aggressive(&mut engine.simplifier, input)
    } else {
        evaluate_timeline_simplify_standard(engine, session, input)
    }
}

/// Evaluate REPL `timeline` command (solve/simplify) and return typed output.
pub fn evaluate_timeline_command_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<crate::TimelineCommandEvalOutput, crate::TimelineCommandEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    match crate::solve_input_parse::parse_timeline_command_input(input) {
        crate::TimelineCommandInput::Solve(solve_rest) => {
            evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(crate::TimelineCommandEvalOutput::Solve)
            .map_err(crate::TimelineCommandEvalError::Solve)
        }
        crate::TimelineCommandInput::Simplify { expr, aggressive } => {
            evaluate_timeline_simplify_with_session(engine, session, &expr, aggressive)
                .map(|output| crate::TimelineCommandEvalOutput::Simplify {
                    expr_input: expr,
                    aggressive,
                    output,
                })
                .map_err(crate::TimelineCommandEvalError::Simplify)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let opts = cas_solver::EvalOptions::default();
        let out = super::evaluate_timeline_command_with_session(
            &mut engine,
            &mut session,
            "x + x",
            &opts,
        )
        .expect("timeline command simplify");
        match out {
            crate::TimelineCommandEvalOutput::Simplify { .. } => {}
            _ => panic!("expected simplify"),
        }
    }
}
