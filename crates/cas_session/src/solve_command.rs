//! Session-level orchestration for `solve` command execution.

fn rsplit_ignoring_parens(s: &str, delimiter: char) -> Option<(&str, &str)> {
    let mut balance = 0;
    let mut split_idx = None;

    for (i, c) in s.char_indices().rev() {
        if c == ')' {
            balance += 1;
        } else if c == '(' {
            balance -= 1;
        } else if c == delimiter && balance == 0 {
            split_idx = Some(i);
            break;
        }
    }

    split_idx.map(|idx| (&s[..idx], &s[idx + 1..]))
}

fn parse_solve_command_input(input: &str) -> cas_solver::SolveCommandInput {
    if let Some((eq, var)) = rsplit_ignoring_parens(input, ',') {
        return cas_solver::SolveCommandInput {
            equation: eq.trim().to_string(),
            variable: Some(var.trim().to_string()),
        };
    }

    if let Some((eq, var)) = rsplit_ignoring_parens(input, ' ') {
        let eq_trim = eq.trim();
        let var_trim = var.trim();

        let has_operators_after_eq = if let Some(eq_pos) = eq_trim.find('=') {
            let after_eq = &eq_trim[eq_pos + 1..];
            after_eq.contains('+')
                || after_eq.contains('-')
                || after_eq.contains('*')
                || after_eq.contains('/')
                || after_eq.contains('^')
        } else {
            false
        };

        if !var_trim.is_empty()
            && var_trim.chars().all(char::is_alphabetic)
            && !eq_trim.ends_with('=')
            && !has_operators_after_eq
        {
            return cas_solver::SolveCommandInput {
                equation: eq_trim.to_string(),
                variable: Some(var_trim.to_string()),
            };
        }
    }

    cas_solver::SolveCommandInput {
        equation: input.to_string(),
        variable: None,
    }
}

fn parse_solve_invocation_check(input: &str, default_check_enabled: bool) -> (bool, &str) {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    }
}

fn parse_statement_or_session_ref(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<cas_parser::Statement, String> {
    if input.starts_with('#') && input[1..].chars().all(char::is_numeric) {
        Ok(cas_parser::Statement::Expression(ctx.var(input)))
    } else {
        cas_parser::parse_statement(input, ctx).map_err(|e| e.to_string())
    }
}

fn statement_to_expr_id(
    ctx: &mut cas_ast::Context,
    stmt: cas_parser::Statement,
) -> cas_ast::ExprId {
    match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => expr,
    }
}

fn resolve_solve_var(
    ctx: &mut cas_ast::Context,
    parsed_expr: cas_ast::ExprId,
    explicit_var: Option<String>,
) -> Result<String, cas_solver::SolvePrepareError> {
    if let Some(v) = explicit_var {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    match cas_solver::infer_solve_variable(ctx, parsed_expr) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(cas_solver::SolvePrepareError::NoVariable),
        Err(vars) => Err(cas_solver::SolvePrepareError::AmbiguousVariables(vars)),
    }
}

fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<cas_solver::PreparedSolveEvalRequest, cas_solver::SolvePrepareError> {
    let stmt = parse_statement_or_session_ref(ctx, input)
        .map_err(cas_solver::SolvePrepareError::ParseError)?;

    let original_equation = match &stmt {
        cas_parser::Statement::Equation(eq) => Some(eq.clone()),
        cas_parser::Statement::Expression(_) => None,
    };
    let parsed_expr = statement_to_expr_id(ctx, stmt);
    let var = resolve_solve_var(ctx, parsed_expr, explicit_var)?;

    Ok(cas_solver::PreparedSolveEvalRequest {
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
    parsed_input: cas_solver::SolveCommandInput,
    auto_store: bool,
) -> Result<cas_solver::SolveCommandEvalOutput, cas_solver::SolveCommandEvalError>
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
    let cas_solver::PreparedSolveEvalRequest {
        request,
        var,
        original_equation,
    } = prepare_solve_eval_request(
        &mut engine.simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(cas_solver::SolveCommandEvalError::Prepare)?;

    let output = engine
        .eval(session, request)
        .map_err(|e| cas_solver::SolveCommandEvalError::Eval(e.to_string()))?;
    let output = cas_solver::eval_output_view(&output);

    Ok(cas_solver::SolveCommandEvalOutput {
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
    let (check_enabled, solve_tail) = parse_solve_invocation_check(rest, options.check_solutions);
    let parsed = parse_solve_command_input(solve_tail);

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

#[cfg(test)]
mod tests {
    #[test]
    fn parse_solve_invocation_check_honors_flag() {
        let (check, tail) = super::parse_solve_invocation_check("--check x+1=2, x", false);
        assert!(check);
        assert_eq!(tail, "x+1=2, x");
    }

    #[test]
    fn parse_solve_command_input_accepts_comma_form() {
        let parsed = super::parse_solve_command_input("x + 2 = 5, x");
        assert_eq!(parsed.equation, "x + 2 = 5");
        assert_eq!(parsed.variable.as_deref(), Some("x"));
    }

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
