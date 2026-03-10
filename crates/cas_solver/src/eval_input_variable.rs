use cas_ast::{Context, Expr, ExprId};

fn preferred_variable_fallback(vars: &[String]) -> String {
    if vars.iter().any(|v| v == "x") {
        return "x".to_string();
    }

    for preferred in ["y", "z", "t", "n", "a", "b", "c"] {
        if vars.iter().any(|v| v == preferred) {
            return preferred.to_string();
        }
    }

    vars.first().cloned().unwrap_or_else(|| "x".to_string())
}

pub(crate) fn detect_solve_variable_for_eval_request(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> String {
    let equation_residual = ctx.add(Expr::Sub(lhs, rhs));

    match crate::infer_solve_variable(ctx, equation_residual) {
        Ok(Some(var)) => var,
        Ok(None) => "x".to_string(),
        Err(vars) => preferred_variable_fallback(&vars),
    }
}
