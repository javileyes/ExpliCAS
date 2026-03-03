//! Session-level orchestration for `timeline` command dispatch.

struct PreparedTimelineSolve {
    equation: cas_ast::Equation,
    var: String,
}

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

fn parse_timeline_command_input(rest: &str) -> cas_solver::TimelineCommandInput {
    if let Some(solve_rest) = rest.strip_prefix("solve ") {
        return cas_solver::TimelineCommandInput::Solve(solve_rest.trim().to_string());
    }

    if let Some(inner) = rest
        .strip_prefix("simplify(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return cas_solver::TimelineCommandInput::Simplify {
            expr: inner.trim().to_string(),
            aggressive: true,
        };
    }

    if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
        return cas_solver::TimelineCommandInput::Simplify {
            expr: simplify_rest.trim().to_string(),
            aggressive: true,
        };
    }

    cas_solver::TimelineCommandInput::Simplify {
        expr: rest.trim().to_string(),
        aggressive: false,
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

fn prepare_timeline_solve_input(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
) -> Result<PreparedTimelineSolve, cas_solver::SolvePrepareError> {
    let stmt = parse_statement_or_session_ref(ctx, input)
        .map_err(cas_solver::SolvePrepareError::ParseError)?;

    let equation = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        cas_parser::Statement::Expression(_) => {
            return Err(cas_solver::SolvePrepareError::ExpectedEquation);
        }
    };

    let eq_expr = ctx.add(cas_ast::Expr::Sub(equation.lhs, equation.rhs));
    let var = resolve_solve_var(ctx, eq_expr, explicit_var)?;

    Ok(PreparedTimelineSolve { equation, var })
}

fn evaluate_timeline_solve_command_input(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    opts: cas_solver::SolverOptions,
) -> Result<cas_solver::TimelineSolveEvalOutput, cas_solver::TimelineSolveEvalError> {
    let parsed_input = parse_solve_command_input(input);
    let prepared = prepare_timeline_solve_input(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
    )
    .map_err(cas_solver::TimelineSolveEvalError::Prepare)?;

    let (solution_set, display_steps, diagnostics) =
        cas_solver::solve_with_display_steps(&prepared.equation, &prepared.var, simplifier, opts)
            .map_err(|e| cas_solver::TimelineSolveEvalError::Solve(e.to_string()))?;

    Ok(cas_solver::TimelineSolveEvalOutput {
        equation: prepared.equation,
        var: prepared.var,
        solution_set,
        display_steps,
        diagnostics,
    })
}

fn evaluate_timeline_solve_with_eval_options(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<cas_solver::TimelineSolveEvalOutput, cas_solver::TimelineSolveEvalError> {
    simplifier.set_collect_steps(true);
    let opts = cas_solver::SolverOptions::from_eval_options(eval_options);
    evaluate_timeline_solve_command_input(simplifier, input, opts)
}

fn evaluate_timeline_simplify_aggressive(
    simplifier: &mut cas_solver::Simplifier,
    input: &str,
) -> Result<cas_solver::TimelineSimplifyEvalOutput, cas_solver::TimelineSimplifyEvalError> {
    let mut temp_simplifier = cas_solver::Simplifier::with_default_rules();
    temp_simplifier.set_collect_steps(true);

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    let result = (|| {
        let parsed_expr = cas_parser::parse(input.trim(), &mut temp_simplifier.context)
            .map_err(|e| cas_solver::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let (simplified_expr, steps) = temp_simplifier.simplify(parsed_expr);
        Ok(cas_solver::TimelineSimplifyEvalOutput {
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
) -> Result<cas_solver::TimelineSimplifyEvalOutput, cas_solver::TimelineSimplifyEvalError>
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
            .map_err(|e| cas_solver::TimelineSimplifyEvalError::Parse(e.to_string()))?;
        let req = cas_solver::EvalRequest {
            raw_input: input.to_string(),
            parsed: parsed_expr,
            action: cas_solver::EvalAction::Simplify,
            auto_store: false,
        };
        let output = engine
            .eval(session, req)
            .map_err(|e| cas_solver::TimelineSimplifyEvalError::Eval(e.to_string()))?;
        let output_view = cas_solver::eval_output_view(&output);
        let simplified_expr = match output_view.result {
            cas_solver::EvalResult::Expr(e) => e,
            _ => parsed_expr,
        };
        Ok(cas_solver::TimelineSimplifyEvalOutput {
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
) -> Result<cas_solver::TimelineSimplifyEvalOutput, cas_solver::TimelineSimplifyEvalError>
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
) -> Result<cas_solver::TimelineCommandEvalOutput, cas_solver::TimelineCommandEvalError>
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
    match parse_timeline_command_input(input) {
        cas_solver::TimelineCommandInput::Solve(solve_rest) => {
            evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(cas_solver::TimelineCommandEvalOutput::Solve)
            .map_err(cas_solver::TimelineCommandEvalError::Solve)
        }
        cas_solver::TimelineCommandInput::Simplify { expr, aggressive } => {
            evaluate_timeline_simplify_with_session(engine, session, &expr, aggressive)
                .map(|output| cas_solver::TimelineCommandEvalOutput::Simplify {
                    expr_input: expr,
                    aggressive,
                    output,
                })
                .map_err(cas_solver::TimelineCommandEvalError::Simplify)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_solve_command_input_accepts_comma_form() {
        let parsed = super::parse_solve_command_input("x + 2 = 5, x");
        assert_eq!(
            parsed,
            cas_solver::SolveCommandInput {
                equation: "x + 2 = 5".to_string(),
                variable: Some("x".to_string()),
            }
        );
    }

    #[test]
    fn parse_timeline_command_input_routes_solve() {
        let parsed = super::parse_timeline_command_input("solve x + 2 = 5, x");
        assert_eq!(
            parsed,
            cas_solver::TimelineCommandInput::Solve("x + 2 = 5, x".to_string())
        );
    }

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
            cas_solver::TimelineCommandEvalOutput::Simplify { .. } => {}
            _ => panic!("expected simplify"),
        }
    }
}
