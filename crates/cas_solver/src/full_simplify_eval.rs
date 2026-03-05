use cas_ast::ExprId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

#[derive(Debug, Clone)]
pub struct FullSimplifyEvalOutput {
    pub resolved_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: Vec<crate::Step>,
}

/// Format full-simplify evaluation errors for user-facing output.
pub fn format_full_simplify_eval_error_message(error: &FullSimplifyEvalError) -> String {
    match error {
        FullSimplifyEvalError::Parse(message) => format!("Error: {message}"),
        FullSimplifyEvalError::Resolve(message) => format!("Error resolving variables: {message}"),
    }
}

pub fn evaluate_full_simplify_input_with_resolver<F>(
    simplifier: &mut crate::Simplifier,
    input: &str,
    collect_steps: bool,
    mut simplify_options: crate::SimplifyOptions,
    resolve_expr: F,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError>
where
    F: FnOnce(&mut cas_ast::Context, ExprId) -> Result<ExprId, String>,
{
    let mut temp_simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let resolved_expr = resolve_expr(&mut temp_simplifier.context, parsed_expr)
            .map_err(FullSimplifyEvalError::Resolve)?;

        simplify_options.collect_steps = collect_steps;
        let (simplified_expr, steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, simplify_options);
        let _ = stats;
        Ok(FullSimplifyEvalOutput {
            resolved_expr,
            simplified_expr,
            steps,
        })
    })();

    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);
    result
}

#[cfg(test)]
mod tests {
    use super::{evaluate_full_simplify_input_with_resolver, FullSimplifyEvalError};

    #[test]
    fn evaluate_full_simplify_input_parse_error_is_typed() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let simplify_options = crate::SimplifyOptions::default();
        let err = evaluate_full_simplify_input_with_resolver(
            &mut simplifier,
            "x+",
            true,
            simplify_options,
            |_ctx, expr| Ok(expr),
        )
        .expect_err("parse error");
        assert!(matches!(err, FullSimplifyEvalError::Parse(_)));
    }

    #[test]
    fn evaluate_full_simplify_input_runs() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let simplify_options = crate::SimplifyOptions::default();
        let out = evaluate_full_simplify_input_with_resolver(
            &mut simplifier,
            "x + 0",
            true,
            simplify_options,
            |_ctx, expr| Ok(expr),
        )
        .expect("full simplify");
        let shown = cas_formatter::render_expr(&simplifier.context, out.simplified_expr);
        assert_eq!(shown, "x");
    }
}
