use super::{display_expr, latex_expr, PolynomialProofData};
use crate::didactic::SubStep;
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn generate_polynomial_identity_substitution_substeps(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    let subs_desc_plain: Vec<String> = proof
        .opaque_substitutions
        .iter()
        .map(|(name, expr_id)| format!("{} = {}", name, display_expr(ctx, *expr_id)))
        .collect();

    let subs_desc_latex: Vec<String> = proof
        .opaque_substitutions
        .iter()
        .map(|(name, expr_id)| format!("{} = {}", name, latex_expr(ctx, *expr_id)))
        .collect();

    sub_steps.push(SubStep {
        description: "Sustitución para simplificar".to_string(),
        before_expr: display_expr(ctx, step.before),
        after_expr: format!("Sea {}", subs_desc_plain.join(", ")),
        before_latex: Some(latex_expr(ctx, step.before)),
        after_latex: Some(format!("\\text{{Sea }} {}", subs_desc_latex.join(", \\; "))),
    });

    if let Some(display_id) = proof.normal_form_expr {
        let (after_plain, after_latex) = if let Some(expanded_id) = proof.expanded_form_expr {
            (display_expr(ctx, expanded_id), latex_expr(ctx, expanded_id))
        } else {
            (
                "Polinomio en la(s) variable(s) sustituida(s)".to_string(),
                "\\text{Polinomio en la(s) variable(s) sustituida(s)}".to_string(),
            )
        };

        sub_steps.push(SubStep {
            description: "Expresión sustituida".to_string(),
            before_expr: display_expr(ctx, display_id),
            after_expr: after_plain,
            before_latex: Some(latex_expr(ctx, display_id)),
            after_latex: Some(after_latex),
        });
    }

    sub_steps.push(SubStep {
        description: "Todos los términos se cancelan".to_string(),
        before_expr: "Expandir y agrupar".to_string(),
        after_expr: "= 0".to_string(),
        before_latex: Some("\\text{Expandir y agrupar}".to_string()),
        after_latex: Some("= 0".to_string()),
    });

    sub_steps
}
