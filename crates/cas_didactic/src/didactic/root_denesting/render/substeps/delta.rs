use super::super::super::super::SubStep;
use super::super::super::analysis::RootDenestingAnalysis;
use super::super::latex::{denesting_latex, format_rational_latex};
use cas_ast::Context;
use num_rational::BigRational;

pub(super) fn build_denesting_delta_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    delta: &BigRational,
) -> SubStep {
    let a_str = denesting_latex(ctx, analysis.a_expr);
    let d_str = denesting_latex(ctx, analysis.d_expr);
    let c_str = format_rational_latex(&analysis.c_coeff);

    SubStep {
        description: "Calcular Δ = a² - c²d".to_string(),
        before_expr: format!("({})^2 - ({})^2 \\cdot {}", a_str, c_str, d_str),
        after_expr: format_rational_latex(delta),
        before_latex: None,
        after_latex: None,
    }
}
