use cas_ast::ExprId;

/// Infer the variable to solve for when the caller doesn't provide one.
///
/// Returns:
/// - `Ok(Some(var))` if exactly one free variable exists
/// - `Ok(None)` if no free variables exist
/// - `Err(vars)` if multiple free variables exist (ambiguous)
///
/// Filters out known constants (`pi`, `π`, `e`, `i`) and internal symbols
/// (`_*`, `#*`).
pub fn infer_solve_variable(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Result<Option<String>, Vec<String>> {
    let all_vars = cas_ast::collect_variables(ctx, expr);

    let free_vars: Vec<String> = all_vars
        .into_iter()
        .filter(|v| {
            let is_constant = matches!(v.as_str(), "pi" | "π" | "e" | "i");
            let is_internal = v.starts_with('_') || v.starts_with('#');
            !is_constant && !is_internal
        })
        .collect();

    match free_vars.len() {
        0 => Ok(None),
        1 => Ok(free_vars.into_iter().next()),
        _ => {
            let mut sorted = free_vars;
            sorted.sort();
            Err(sorted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::infer_solve_variable;
    use cas_ast::Expr;
    use num_rational::Ratio;

    #[test]
    fn infers_single_variable() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(x, one));
        assert_eq!(infer_solve_variable(&ctx, expr), Ok(Some("x".to_string())));
    }

    #[test]
    fn returns_none_for_constant_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.add(Expr::Number(Ratio::from_integer(42.into())));
        assert_eq!(infer_solve_variable(&ctx, expr), Ok(None));
    }

    #[test]
    fn returns_sorted_ambiguity_for_multiple_variables() {
        let mut ctx = cas_ast::Context::new();
        let y = ctx.var("y");
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Mul(y, x));
        assert_eq!(
            infer_solve_variable(&ctx, expr),
            Err(vec!["x".to_string(), "y".to_string()])
        );
    }

    #[test]
    fn filters_constants_and_internal_symbols() {
        let mut ctx = cas_ast::Context::new();
        let pi = ctx.var("pi");
        let internal = ctx.var("_tmp");
        let z = ctx.var("z");
        let tail = ctx.add(Expr::Add(internal, z));
        let expr = ctx.add(Expr::Add(pi, tail));
        assert_eq!(infer_solve_variable(&ctx, expr), Ok(Some("z".to_string())));
    }
}
