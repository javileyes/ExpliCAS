pub(crate) fn evaluate_timeline_solve_command_input(
    simplifier: &mut crate::Simplifier,
    input: &str,
    opts: crate::SolverOptions,
) -> Result<crate::TimelineSolveEvalOutput, crate::TimelineSolveEvalError> {
    let parsed_input = crate::parse_solve_command_input(input);
    let (equation, var) = crate::prepare_timeline_solve_equation(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
    )
    .map_err(crate::TimelineSolveEvalError::Prepare)?;

    let (solution_set, display_steps, diagnostics) =
        crate::solve_with_display_steps(&equation, &var, simplifier, opts)
            .map_err(|e| crate::TimelineSolveEvalError::Solve(e.to_string()))?;

    Ok(crate::TimelineSolveEvalOutput {
        equation,
        var,
        solution_set,
        display_steps,
        diagnostics,
    })
}

pub(crate) fn evaluate_timeline_solve_with_eval_options(
    simplifier: &mut crate::Simplifier,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<crate::TimelineSolveEvalOutput, crate::TimelineSolveEvalError> {
    simplifier.set_collect_steps(true);
    let opts = crate::SolverOptions::from_eval_options(eval_options);
    evaluate_timeline_solve_command_input(simplifier, input, opts)
}
