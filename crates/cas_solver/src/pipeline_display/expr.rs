use cas_ast::{Context, ExprId};

/// Display an expression, preferring formatted poly output when available.
pub fn display_expr_or_poly(ctx: &Context, id: ExprId) -> String {
    if let Some(poly_str) = crate::try_render_poly_result(ctx, id) {
        return poly_str;
    }
    cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr { context: ctx, id }
    ))
}
