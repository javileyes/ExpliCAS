use super::super::SubStep;
use super::render;

pub(super) fn build_cancel_substep(
    gcd_str: &str,
    numerator_str: &str,
    denominator_str: &str,
    after_str: String,
) -> SubStep {
    SubStep {
        description: format!("Cancel common factor: {}", gcd_str),
        before_expr: render::format_division_expr(numerator_str, denominator_str),
        after_expr: after_str,
        before_latex: None,
        after_latex: None,
    }
}
