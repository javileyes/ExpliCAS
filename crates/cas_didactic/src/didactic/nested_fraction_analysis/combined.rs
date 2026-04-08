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
        if let (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) =
            (ctx.get(*l), ctx.get(*r))
        {
            let left_num_latex = to_latex(*left_num);
            let left_den_latex = to_latex(*left_den);
            let right_num_latex = to_latex(*right_num);
            let right_den_latex = to_latex(*right_den);

            let left_scaled = latex_product(&left_num_latex, &right_den_latex);
            let right_scaled = latex_product(&right_num_latex, &left_den_latex);
            let common_den = latex_product(&left_den_latex, &right_den_latex);

            return format!(
                "\\frac{{{} + {}}}{{{}}}",
                left_scaled, right_scaled, common_den
            );
        }

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
                "\\frac{{{} + {}}}{{{}}}",
                latex_product(&other_latex, &frac_den_latex),
                frac_num_latex,
                frac_den_latex
            );
        }
    }

    "\\text{(combinado)}".to_string()
}

fn latex_product(lhs: &str, rhs: &str) -> String {
    match (lhs.trim(), rhs.trim()) {
        ("1", other) => other.to_string(),
        (other, "1") => other.to_string(),
        (left, right) => format!("{left} \\cdot {right}"),
    }
}
