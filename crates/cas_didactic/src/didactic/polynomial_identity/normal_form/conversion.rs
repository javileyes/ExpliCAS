use super::super::{display_expr, latex_expr, normal_form_summary, PolynomialProofData};
use crate::didactic::SubStep;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn generate_normal_form_conversion_substeps(
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
