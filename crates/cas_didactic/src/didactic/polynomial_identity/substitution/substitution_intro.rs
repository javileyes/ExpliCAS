use super::{PolynomialProofData, SubStep};
use cas_ast::Context;
use cas_solver::Step;

pub(super) fn build_substitution_intro_substep(
    ctx: &Context,
    step: &Step,
    proof: &PolynomialProofData,
    display_expr: fn(&Context, cas_ast::ExprId) -> String,
    latex_expr: fn(&Context, cas_ast::ExprId) -> String,
) -> SubStep {
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

    SubStep {
        description: "Sustitución para simplificar".to_string(),
        before_expr: display_expr(ctx, step.before),
        after_expr: format!("Sea {}", subs_desc_plain.join(", ")),
        before_latex: Some(latex_expr(ctx, step.before)),
        after_latex: Some(format!("\\text{{Sea }} {}", subs_desc_latex.join(", \\; "))),
    }
}
