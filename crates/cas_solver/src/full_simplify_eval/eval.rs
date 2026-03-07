use cas_ast::ExprId;

use super::{error::FullSimplifyEvalError, types::FullSimplifyEvalOutput};

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
