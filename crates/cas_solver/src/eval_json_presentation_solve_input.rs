use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

pub(crate) fn format_eval_input_latex(ctx: &Context, parsed: ExprId) -> String {
    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
    }
}
