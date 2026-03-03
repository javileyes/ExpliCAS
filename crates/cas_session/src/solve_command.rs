//! Session-level orchestration for `solve` command execution.

/// Evaluate REPL `solve ...` input against engine/session state and render lines.
pub fn evaluate_solve_command_lines(
    engine: &mut cas_solver::Engine,
    session: &mut crate::SessionState,
    line: &str,
    display_mode: crate::SetDisplayMode,
    debug_mode: bool,
) -> Result<Vec<String>, String> {
    let options = session.options().clone();
    let rest = line.strip_prefix("solve").unwrap_or(line).trim();
    let (check_enabled, solve_tail) =
        cas_solver::parse_solve_invocation_check(rest, options.check_solutions);
    let parsed = cas_solver::parse_solve_command_input(solve_tail);

    let eval_output =
        cas_solver::evaluate_solve_command_with_session(engine, session, parsed, true)
            .map_err(|error| crate::format_solve_command_error_message(&error))?;

    let mut render_config =
        crate::solve_render_config_from_eval_options(&options, display_mode, debug_mode);
    render_config.check_solutions = check_enabled;

    Ok(crate::format_solve_command_eval_lines(
        &mut engine.simplifier,
        &eval_output,
        render_config,
    ))
}

/// Evaluate REPL `solve ...` input and return rendered message text.
pub fn evaluate_solve_command_message(
    engine: &mut cas_solver::Engine,
    session: &mut crate::SessionState,
    line: &str,
    display_mode: crate::SetDisplayMode,
    debug_mode: bool,
) -> Result<String, String> {
    Ok(evaluate_solve_command_lines(engine, session, line, display_mode, debug_mode)?.join("\n"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_solve_command_lines_reports_ambiguous_variables() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let out = super::evaluate_solve_command_lines(
            &mut engine,
            &mut session,
            "solve x+y=0",
            crate::SetDisplayMode::Normal,
            false,
        )
        .expect_err("expected ambiguous-variable error");
        assert!(out.contains("ambiguous variables"));
    }

    #[test]
    fn evaluate_solve_command_message_joins_lines() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let message = super::evaluate_solve_command_message(
            &mut engine,
            &mut session,
            "solve x+2=5",
            crate::SetDisplayMode::Normal,
            false,
        )
        .expect("solve should succeed");
        assert!(message.contains("x"));
        assert!(message.contains("3"));
    }
}
