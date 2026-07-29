use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayExpr, ParseStyleSignals, RootStyle};
use cas_math::poly_store::try_render_poly_result;

/// Render expression for output with max length truncation.
///
/// `signals` decide la notación de raíz por la MISMA función que el LaTeX del
/// resultado (`result_root_style`): el texto plano y el LaTeX del mismo resultado
/// no pueden decir cosas distintas, que es exactamente la queja que abrió este
/// frente (`sqrt(3+4i)` devolvía `\sqrt{3 + 4i}` y `(3 + 4·i)^(1/2)`).
pub(crate) fn format_limited_output_expr(
    ctx: &Context,
    expr: ExprId,
    max_chars: usize,
    signals: &ParseStyleSignals,
) -> (String, bool, usize) {
    if let Some(poly_str) = try_render_poly_result(ctx, expr) {
        let len = poly_str.chars().count();
        if len <= max_chars {
            return (poly_str, false, len);
        }
        let truncated: String = poly_str.chars().take(max_chars).collect();
        return (format!("{truncated} … <truncated>"), true, len);
    }

    // `DisplayExpr` no tiene perilla de estilo (se construye en ~2000 sitios), así
    // que la raíz se consigue reescribiendo el ÁRBOL a los nodos que ese
    // renderizador ya imprime como raíz, sobre un contexto de scratch.
    let full = match crate::eval_output_latex_style::result_root_style(signals) {
        RootStyle::Exponential => format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: expr
            }
        ),
        RootStyle::Radical | RootStyle::Auto => {
            let mut scratch = ctx.clone();
            let as_roots = cas_formatter::root_display_rewrite::rewrite_fractional_powers_as_roots(
                &mut scratch,
                expr,
            );
            format!(
                "{}",
                DisplayExpr {
                    context: &scratch,
                    id: as_roots
                }
            )
        }
    };
    let full = crate::pipeline_display::compact_subtracted_difference_display(full);
    let len = full.chars().count();

    if len <= max_chars {
        return (full, false, len);
    }

    let truncated: String = full.chars().take(max_chars).collect();
    (format!("{truncated} … <truncated>"), true, len)
}
