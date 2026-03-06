pub fn evaluate_solve_command_lines_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    line: &str,
    eval_options: &crate::EvalOptions,
    display_mode: crate::SolveDisplayMode,
    debug_mode: bool,
) -> Result<Vec<String>, String>
where
    S: crate::SolverEvalSession,
{
    let rest = line.strip_prefix("solve").unwrap_or(line).trim();
    let (check_enabled, solve_tail) =
        crate::parse_solve_invocation_check(rest, eval_options.check_solutions);
    let parsed = crate::parse_solve_command_input(solve_tail);

    let eval_output = crate::evaluate_solve_command_with_session(simplifier, session, parsed, true)
        .map_err(|error| crate::format_solve_command_error_message(&error))?;

    let mut render_config =
        crate::solve_render_config_from_eval_options(eval_options, display_mode, debug_mode);
    render_config.check_solutions = check_enabled;

    Ok(crate::format_solve_command_eval_lines(
        simplifier,
        &eval_output.var,
        eval_output.original_equation.as_ref(),
        &eval_output.output,
        render_config,
    ))
}

pub fn evaluate_solve_command_message_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    line: &str,
    eval_options: &crate::EvalOptions,
    display_mode: crate::SolveDisplayMode,
    debug_mode: bool,
) -> Result<String, String>
where
    S: crate::SolverEvalSession,
{
    Ok(evaluate_solve_command_lines_with_session(
        simplifier,
        session,
        line,
        eval_options,
        display_mode,
        debug_mode,
    )?
    .join("\n"))
}
