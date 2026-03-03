//! Shared parsing helpers for `solve` and `timeline` command families.

/// Parse optional solve invocation flags and return (check_enabled, remaining_tail).
pub(crate) fn parse_solve_invocation_check(
    input: &str,
    default_check_enabled: bool,
) -> (bool, &str) {
    let trimmed = input.trim();
    if let Some(rest) = trimmed.strip_prefix("--check") {
        (true, rest.trim_start())
    } else {
        (default_check_enabled, trimmed)
    }
}

/// Parse REPL `solve` argument shape.
pub(crate) fn parse_solve_command_input(input: &str) -> crate::SolveCommandInput {
    if let Some((eq, var)) = crate::input_parse_common::rsplit_ignoring_parens(input, ',') {
        return crate::SolveCommandInput {
            equation: eq.trim().to_string(),
            variable: Some(var.trim().to_string()),
        };
    }

    if let Some((eq, var)) = crate::input_parse_common::rsplit_ignoring_parens(input, ' ') {
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
            return crate::SolveCommandInput {
                equation: eq_trim.to_string(),
                variable: Some(var_trim.to_string()),
            };
        }
    }

    crate::SolveCommandInput {
        equation: input.to_string(),
        variable: None,
    }
}

/// Parse REPL `timeline` command shape.
pub(crate) fn parse_timeline_command_input(rest: &str) -> crate::TimelineCommandInput {
    if let Some(solve_rest) = rest.strip_prefix("solve ") {
        return crate::TimelineCommandInput::Solve(solve_rest.trim().to_string());
    }

    if let Some(inner) = rest
        .strip_prefix("simplify(")
        .and_then(|s| s.strip_suffix(')'))
    {
        return crate::TimelineCommandInput::Simplify {
            expr: inner.trim().to_string(),
            aggressive: true,
        };
    }

    if let Some(simplify_rest) = rest.strip_prefix("simplify ") {
        return crate::TimelineCommandInput::Simplify {
            expr: simplify_rest.trim().to_string(),
            aggressive: true,
        };
    }

    crate::TimelineCommandInput::Simplify {
        expr: rest.trim().to_string(),
        aggressive: false,
    }
}

pub(crate) fn resolve_solve_var(
    ctx: &mut cas_ast::Context,
    parsed_expr: cas_ast::ExprId,
    explicit_var: Option<String>,
) -> Result<String, crate::SolvePrepareError> {
    if let Some(v) = explicit_var {
        if !v.trim().is_empty() {
            return Ok(v);
        }
    }

    match cas_solver::infer_solve_variable(ctx, parsed_expr) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Err(crate::SolvePrepareError::NoVariable),
        Err(vars) => Err(crate::SolvePrepareError::AmbiguousVariables(vars)),
    }
}

/// Parse solve input into expression + optional original equation + resolved variable.
pub(crate) fn prepare_solve_expr_and_var(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
) -> Result<(cas_ast::ExprId, Option<cas_ast::Equation>, String), crate::SolvePrepareError> {
    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, input)
        .map_err(crate::SolvePrepareError::ParseError)?;

    let original_equation = match &stmt {
        cas_parser::Statement::Equation(eq) => Some(eq.clone()),
        cas_parser::Statement::Expression(_) => None,
    };
    let parsed_expr = crate::input_parse_common::statement_to_expr_id(ctx, stmt);
    let var = resolve_solve_var(ctx, parsed_expr, explicit_var)?;

    Ok((parsed_expr, original_equation, var))
}

/// Parse timeline-solve input as equation and resolve variable.
pub(crate) fn prepare_timeline_solve_equation(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
) -> Result<(cas_ast::Equation, String), crate::SolvePrepareError> {
    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, input)
        .map_err(crate::SolvePrepareError::ParseError)?;

    let equation = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        cas_parser::Statement::Expression(_) => {
            return Err(crate::SolvePrepareError::ExpectedEquation);
        }
    };

    let eq_expr = ctx.add(cas_ast::Expr::Sub(equation.lhs, equation.rhs));
    let var = resolve_solve_var(ctx, eq_expr, explicit_var)?;
    Ok((equation, var))
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
        assert_eq!(
            parsed,
            crate::SolveCommandInput {
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
            crate::TimelineCommandInput::Solve("x + 2 = 5, x".to_string())
        );
    }

    #[test]
    fn prepare_timeline_solve_equation_requires_equation() {
        let mut ctx = cas_ast::Context::new();
        let err = super::prepare_timeline_solve_equation(&mut ctx, "x + 2", Some("x".to_string()))
            .expect_err("expected equation requirement");
        assert_eq!(err, crate::SolvePrepareError::ExpectedEquation);
    }
}
