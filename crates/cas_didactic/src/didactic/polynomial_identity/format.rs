mod expr;
mod summary;

use cas_ast::Context;
use cas_math::multipoly_display::{PolyNormalFormStats, PolynomialProofData};

pub(super) fn display_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    expr::display_expr(ctx, expr_id)
}

pub(super) fn latex_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    expr::latex_expr(ctx, expr_id)
}

pub(super) fn format_poly_stats(ctx: &Context, stats: &PolyNormalFormStats) -> String {
    summary::format_poly_stats(ctx, stats, display_expr)
}

pub(super) fn normal_form_summary(
    ctx: &Context,
    proof: &PolynomialProofData,
) -> (String, Option<String>) {
    summary::normal_form_summary(ctx, proof, display_expr, latex_expr)
}
