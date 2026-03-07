use super::super::SubStep;
use super::render;
use cas_ast::{Context, ExprId};

pub(super) fn build_factor_substep(
    context: &Context,
    local_before: ExprId,
    numerator_str: &str,
    gcd_str: &str,
) -> Option<SubStep> {
    let local_before_str = render::render_expr(context, local_before);
    if !local_before_str.contains(gcd_str) || !local_before_str.contains('*') {
        return None;
    }

    Some(SubStep {
        description: format!("Factor: {} contains factor {}", numerator_str, gcd_str),
        before_expr: numerator_str.to_string(),
        after_expr: local_before_str
            .split('/')
            .next()
            .unwrap_or(&local_before_str)
            .trim()
            .to_string(),
        before_latex: None,
        after_latex: None,
    })
}
