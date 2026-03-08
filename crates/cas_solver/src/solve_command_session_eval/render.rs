use super::parse::parse_solve_command_session_request;

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
    let request = parse_solve_command_session_request(line, eval_options);

    let eval_output =
        crate::evaluate_solve_command_with_session(simplifier, session, request.parsed, true)
            .map_err(|error| crate::format_solve_command_error_message(&error))?;

    let mut render_config =
        crate::solve_render_config_from_eval_options(eval_options, display_mode, debug_mode);
    render_config.check_solutions = request.check_enabled;

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
