use cas_ast::Context;
use cas_formatter::LaTeXExpr;

pub(super) fn render_discrete_solution_set(ctx: &Context, exprs: &[cas_ast::ExprId]) -> String {
    if exprs.is_empty() {
        r"\emptyset".to_string()
    } else {
        let solutions: Vec<String> = exprs
            .iter()
            .map(|e| {
                LaTeXExpr {
                    context: ctx,
                    id: *e,
                }
                .to_latex()
            })
            .collect();
        format!(r"\left\{{ {} \right\}}", solutions.join(", "))
    }
}
