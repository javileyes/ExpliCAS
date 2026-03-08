use cas_ast::{Context, ExprId};
use cas_solver::{infer_implicit_domain, ImplicitCondition, ValueDomain};

pub(super) fn collect_timeline_global_requires(
    context: &mut Context,
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
) -> Vec<ImplicitCondition> {
    let domain_expr = simplified_result.unwrap_or(original_expr);
    let input_domain = infer_implicit_domain(context, domain_expr, ValueDomain::RealOnly);
    input_domain.conditions().iter().cloned().collect()
}
