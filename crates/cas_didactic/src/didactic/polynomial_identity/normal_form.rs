use super::{
    display_expr, format_poly_stats, latex_expr, normal_form_summary, PolynomialProofData,
};
use crate::didactic::SubStep;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn generate_polynomial_identity_normal_form_substeps(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
) -> Vec<SubStep> {
    if let (Some(lhs_stats), Some(rhs_stats)) = (&proof.lhs_stats, &proof.rhs_stats) {
        return generate_identity_comparison_substeps(ctx, lhs_stats, rhs_stats);
    }

    generate_normal_form_conversion_substeps(ctx, step, proof)
}

fn generate_identity_comparison_substeps(
    ctx: &Context,
    lhs_stats: &cas_math::multipoly_display::PolyNormalFormStats,
    rhs_stats: &cas_math::multipoly_display::PolyNormalFormStats,
) -> Vec<SubStep> {
    vec![
        SubStep {
            description: "Expandir lado izquierdo".to_string(),
            before_expr: "(a + b + c)³".to_string(),
            after_expr: format_poly_stats(ctx, lhs_stats),
            before_latex: None,
            after_latex: None,
        },
        SubStep {
            description: "Expandir lado derecho".to_string(),
            before_expr: "a³ + b³ + c³ + ...".to_string(),
            after_expr: format_poly_stats(ctx, rhs_stats),
            before_latex: None,
            after_latex: None,
        },
        SubStep {
            description: "Comparar formas normales".to_string(),
            before_expr: format!(
                "LHS: {} monomios | RHS: {} monomios",
                lhs_stats.monomials, rhs_stats.monomials
            ),
            after_expr: "Coinciden ⇒ diferencia = 0".to_string(),
            before_latex: Some(format!(
                "\\text{{LHS: {} monomios}} \\;|\\; \\text{{RHS: {} monomios}}",
                lhs_stats.monomials, rhs_stats.monomials
            )),
            after_latex: Some("\\text{Coinciden} \\Rightarrow \\text{diferencia} = 0".to_string()),
        },
    ]
}

fn generate_normal_form_conversion_substeps(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
) -> Vec<SubStep> {
    let (normal_form_plain, normal_form_latex) = normal_form_summary(ctx, proof);

    vec![
        SubStep {
            description: "Convertir a forma normal polinómica".to_string(),
            before_expr: display_expr(ctx, step.before),
            after_expr: normal_form_plain,
            before_latex: Some(latex_expr(ctx, step.before)),
            after_latex: normal_form_latex,
        },
        SubStep {
            description: "Cancelar términos semejantes".to_string(),
            before_expr: "todos los coeficientes".to_string(),
            after_expr: "0".to_string(),
            before_latex: Some("\\text{todos los coeficientes}".to_string()),
            after_latex: Some("0".to_string()),
        },
    ]
}
