use cas_ast::ExprId;

use crate::SessionState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullSimplifyEvalError {
    Parse(String),
    Resolve(String),
}

#[derive(Debug, Clone)]
pub struct FullSimplifyEvalOutput {
    pub resolved_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: Vec<cas_solver::Step>,
}

/// Format full-simplify evaluation errors for user-facing output.
pub fn format_full_simplify_eval_error_message(error: &FullSimplifyEvalError) -> String {
    match error {
        FullSimplifyEvalError::Parse(message) => format!("Error: {message}"),
        FullSimplifyEvalError::Resolve(message) => format!("Error resolving variables: {message}"),
    }
}

pub fn evaluate_full_simplify_input(
    simplifier: &mut cas_solver::Simplifier,
    session: &SessionState,
    input: &str,
    collect_steps: bool,
) -> Result<FullSimplifyEvalOutput, FullSimplifyEvalError> {
    let mut temp_simplifier = cas_solver::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, &mut temp_simplifier.context);
    std::mem::swap(&mut simplifier.profiler, &mut temp_simplifier.profiler);

    let result = (|| {
        let parsed_expr = cas_parser::parse(input, &mut temp_simplifier.context)
            .map_err(|e| FullSimplifyEvalError::Parse(e.to_string()))?;
        let (resolved_expr, _diag, _cache_hits) = session
            .resolve_state_refs_with_diagnostics(&mut temp_simplifier.context, parsed_expr)
            .map_err(|e| FullSimplifyEvalError::Resolve(e.to_string()))?;

        let mut opts = session.options().to_simplify_options();
        opts.collect_steps = collect_steps;
        let (simplified_expr, steps, stats) =
            temp_simplifier.simplify_with_stats(resolved_expr, opts);
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
