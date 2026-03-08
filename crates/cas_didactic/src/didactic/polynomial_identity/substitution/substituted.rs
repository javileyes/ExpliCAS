use super::{PolynomialProofData, SubStep};
use cas_ast::Context;

pub(super) fn build_substituted_expression_substep(
    ctx: &Context,
    proof: &PolynomialProofData,
    display_id: cas_ast::ExprId,
    display_expr: fn(&Context, cas_ast::ExprId) -> String,
    latex_expr: fn(&Context, cas_ast::ExprId) -> String,
) -> SubStep {
    let (after_plain, after_latex) = if let Some(expanded_id) = proof.expanded_form_expr {
        (display_expr(ctx, expanded_id), latex_expr(ctx, expanded_id))
    } else {
        (
            "Polinomio en la(s) variable(s) sustituida(s)".to_string(),
            "\\text{Polinomio en la(s) variable(s) sustituida(s)}".to_string(),
        )
    };

    SubStep {
        description: "Expresión sustituida".to_string(),
        before_expr: display_expr(ctx, display_id),
        after_expr: after_plain,
        before_latex: Some(latex_expr(ctx, display_id)),
        after_latex: Some(after_latex),
    }
}
