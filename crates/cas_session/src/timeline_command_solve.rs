pub(crate) fn evaluate_timeline_solve_command_input(
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

pub(crate) fn evaluate_timeline_solve_with_eval_options(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<crate::TimelineSolveEvalOutput, crate::TimelineSolveEvalError> {
    simplifier.set_collect_steps(true);
    let opts = cas_solver::SolverOptions::from_eval_options(eval_options);
    evaluate_timeline_solve_command_input(simplifier, input, opts)
}
