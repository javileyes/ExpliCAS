use cas_ast::{Context, ExprId};
use cas_formatter::display_transforms::{DisplayTransformRegistry, ScopeTag, ScopedRenderer};

pub fn format_solve_result_line(
    ctx: &Context,
    result: &crate::EvalResult,
    output_scopes: &[ScopeTag],
) -> String {
    match result {
        crate::EvalResult::SolutionSet(solution_set) => {
            format!("Result: {}", crate::display_solution_set(ctx, solution_set))
        }
        crate::EvalResult::Set(solutions) => {
            let sol_strs: Vec<String> = {
                let registry = DisplayTransformRegistry::with_defaults();
                let style = cas_formatter::root_style::StylePreferences::default();
                let renderer = ScopedRenderer::new(ctx, output_scopes, &registry, &style);
                solutions.iter().map(|id| renderer.render(*id)).collect()
            };
            if sol_strs.is_empty() {
                "Result: No solution".to_string()
            } else {
                format!("Result: {{ {} }}", sol_strs.join(", "))
            }
        }
        _ => format!("Result: {:?}", result),
    }
}

pub fn requires_result_expr_anchor(result: &crate::EvalResult, resolved: ExprId) -> ExprId {
    match result {
        crate::EvalResult::Expr(expr) => *expr,
        crate::EvalResult::Set(values) => *values.first().unwrap_or(&resolved),
        _ => resolved,
    }
}
