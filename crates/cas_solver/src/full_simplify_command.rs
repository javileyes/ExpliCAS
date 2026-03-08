/// Extract expression tail from a `simplify` command line.
pub fn extract_simplify_command_tail(line: &str) -> &str {
    line.strip_prefix("simplify").unwrap_or(line).trim()
}

/// Evaluate a full `simplify ...` invocation and return final display lines.
pub fn evaluate_full_simplify_command_lines_with_resolver<F>(
    simplifier: &mut crate::Simplifier,
    line: &str,
    display_mode: crate::FullSimplifyDisplayMode,
    simplify_options: crate::SimplifyOptions,
    resolve_expr: F,
) -> Result<Vec<String>, String>
where
    F: FnOnce(&mut cas_ast::Context, cas_ast::ExprId) -> Result<cas_ast::ExprId, String>,
{
    let expr_input = extract_simplify_command_tail(line);
    let output = crate::evaluate_full_simplify_input_with_resolver(
        simplifier,
        expr_input,
        !matches!(display_mode, crate::FullSimplifyDisplayMode::None),
        simplify_options,
        resolve_expr,
    )
    .map_err(|error| crate::format_full_simplify_eval_error_message(&error))?;

    Ok(crate::format_full_simplify_eval_lines(
        &mut simplifier.context,
        expr_input,
        output.resolved_expr,
        output.simplified_expr,
        &output.steps,
        display_mode,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_full_simplify_command_lines_with_resolver, extract_simplify_command_tail,
    };

    #[test]
    fn extract_simplify_command_tail_trims_prefix() {
        assert_eq!(extract_simplify_command_tail("simplify x+1"), "x+1");
    }

    #[test]
    fn evaluate_full_simplify_command_lines_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let lines = evaluate_full_simplify_command_lines_with_resolver(
            &mut simplifier,
            "simplify x + 0",
            crate::FullSimplifyDisplayMode::Normal,
            crate::SimplifyOptions::default(),
            |_ctx, expr| Ok(expr),
        )
        .expect("simplify");
        assert!(lines.iter().any(|line| line.starts_with("Result:")));
    }
}
