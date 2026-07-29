use cas_ast::Context;
use cas_formatter::{LaTeXExprStyled, StylePreferences};

pub(super) fn render_discrete_solution_set(
    ctx: &Context,
    exprs: &[cas_ast::ExprId],
    style: &StylePreferences,
) -> String {
    if exprs.is_empty() {
        r"\emptyset".to_string()
    } else {
        let solutions: Vec<String> = exprs
            .iter()
            .map(|e| {
                LaTeXExprStyled {
                    context: ctx,
                    id: *e,
                    style_prefs: style,
                }
                .to_latex()
            })
            .collect();
        format!(r"\left\{{ {} \right\}}", solutions.join(", "))
    }
}
