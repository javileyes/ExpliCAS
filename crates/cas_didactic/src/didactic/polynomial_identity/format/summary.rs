use cas_ast::Context;
use cas_math::multipoly_display::{PolyNormalFormStats, PolynomialProofData};

pub(super) fn format_poly_stats(
    ctx: &Context,
    stats: &PolyNormalFormStats,
    display_expr: fn(&Context, cas_ast::ExprId) -> String,
) -> String {
    if let Some(expr_id) = stats.expr {
        display_expr(ctx, expr_id)
    } else {
        format!("{} monomios, grado {}", stats.monomials, stats.degree)
    }
}

pub(super) fn normal_form_summary(
    ctx: &Context,
    proof: &PolynomialProofData,
    display_expr: fn(&Context, cas_ast::ExprId) -> String,
    latex_expr: fn(&Context, cas_ast::ExprId) -> String,
) -> (String, Option<String>) {
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
