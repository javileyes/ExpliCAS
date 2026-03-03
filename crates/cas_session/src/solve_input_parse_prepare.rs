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
