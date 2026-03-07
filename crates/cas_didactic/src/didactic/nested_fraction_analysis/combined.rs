use cas_ast::{Context, Expr, ExprId};

/// Extract the combined fraction string from an `Add` expression containing a fraction.
/// Example: `1 + 1/x -> "\\frac{x + 1}{x}"` in LaTeX.
pub(crate) fn extract_combined_fraction_str(ctx: &Context, add_expr: ExprId) -> String {
    use cas_formatter::DisplayContext;
    use cas_formatter::LaTeXExprWithHints;

    let hints = DisplayContext::default();
    let to_latex = |id: ExprId| -> String {
        LaTeXExprWithHints {
            context: ctx,
            id,
            hints: &hints,
        }
        .to_latex()
    };

    if let Expr::Add(l, r) = ctx.get(add_expr) {
        let (frac_id, other_id) = if matches!(ctx.get(*l), Expr::Div(_, _)) {
            (*l, *r)
        } else if matches!(ctx.get(*r), Expr::Div(_, _)) {
            (*r, *l)
        } else {
            return "\\text{(combinado)}".to_string();
        };

        if let Expr::Div(frac_num, frac_den) = ctx.get(frac_id) {
            let frac_num_latex = to_latex(*frac_num);
            let frac_den_latex = to_latex(*frac_den);
            let other_latex = to_latex(other_id);

            return format!(
                "\\frac{{{} \\cdot {} + {}}}{{{}}}",
                other_latex, frac_den_latex, frac_num_latex, frac_den_latex
            );
        }
    }

    "\\text{(combinado)}".to_string()
}
