mod normal_form;
mod substitution;

use super::SubStep;
use cas_ast::Context;
use cas_math::multipoly_display::{PolyNormalFormStats, PolynomialProofData};
use cas_solver::Step;

/// Generate sub-steps explaining polynomial identity normalization (PolyZero airbag)
pub(crate) fn generate_polynomial_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let proof = match step.poly_proof() {
        Some(p) => p,
        None => return Vec::new(),
    };

    if !proof.opaque_substitutions.is_empty() {
        return substitution::generate_polynomial_identity_substitution_substeps(ctx, step, proof);
    }

    normal_form::generate_polynomial_identity_normal_form_substeps(ctx, step, proof)
}

fn display_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr_id
        }
    )
}

fn latex_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr_id,
    }
    .to_latex()
}

fn format_poly_stats(ctx: &Context, stats: &PolyNormalFormStats) -> String {
    if let Some(expr_id) = stats.expr {
        display_expr(ctx, expr_id)
    } else {
        format!("{} monomios, grado {}", stats.monomials, stats.degree)
    }
}

fn normal_form_summary(ctx: &Context, proof: &PolynomialProofData) -> (String, Option<String>) {
    if let Some(expr_id) = proof.normal_form_expr {
        (display_expr(ctx, expr_id), Some(latex_expr(ctx, expr_id)))
    } else {
        let vars_str = if proof.vars.is_empty() {
            "constante".to_string()
        } else {
            proof.vars.join(", ")
        };
        (
            format!(
                "{} monomios, grado {}, vars: {}",
                proof.monomials, proof.degree, vars_str
            ),
            None,
        )
    }
}
