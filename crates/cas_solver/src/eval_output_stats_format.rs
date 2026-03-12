use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::poly_store::try_render_poly_result;

/// Render expression for output with max length truncation.
pub(crate) fn format_limited_output_expr(
    ctx: &Context,
    expr: ExprId,
    max_chars: usize,
) -> (String, bool, usize) {
    if let Some(poly_str) = try_render_poly_result(ctx, expr) {
        let len = poly_str.chars().count();
        if len <= max_chars {
            return (poly_str, false, len);
        }
        let truncated: String = poly_str.chars().take(max_chars).collect();
        return (format!("{truncated} … <truncated>"), true, len);
    }

    let full = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let len = full.chars().count();

    if len <= max_chars {
        return (full, false, len);
    }

    let truncated: String = full.chars().take(max_chars).collect();
    (format!("{truncated} … <truncated>"), true, len)
}
