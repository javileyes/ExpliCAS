use super::super::format_poly_stats;
use crate::didactic::SubStep;
use cas_ast::Context;

pub(super) fn generate_identity_comparison_substeps(
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
