fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<crate::PreparedSolveEvalRequest, crate::SolvePrepareError> {
    let (parsed_expr, original_equation, var) =
        crate::solve_input_parse::prepare_solve_expr_and_var(ctx, input, explicit_var)?;

    Ok(crate::PreparedSolveEvalRequest {
        request: cas_solver::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_solver::EvalAction::Solve { var: var.clone() },
            auto_store,
        },
        var,
        original_equation,
    })
}

fn evaluate_solve_command_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    parsed_input: crate::SolveCommandInput,
    auto_store: bool,
) -> Result<crate::SolveCommandEvalOutput, crate::SolveCommandEvalError>
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
    let crate::PreparedSolveEvalRequest {
        request,
        var,
        original_equation,
    } = prepare_solve_eval_request(
        &mut engine.simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(crate::SolveCommandEvalError::Prepare)?;

    let output = engine
        .eval(session, request)
        .map_err(|e| crate::SolveCommandEvalError::Eval(e.to_string()))?;
    let output = cas_solver::eval_output_view(&output);

    Ok(crate::SolveCommandEvalOutput {
        var,
        original_equation,
        output,
    })
}

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
        crate::solve_input_parse::parse_solve_invocation_check(rest, options.check_solutions);
    let parsed = crate::solve_input_parse::parse_solve_command_input(solve_tail);

    let eval_output = evaluate_solve_command_with_session(engine, session, parsed, true)
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
