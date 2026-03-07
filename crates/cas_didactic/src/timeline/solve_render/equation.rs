use super::solve_relop_to_latex;
use cas_ast::{Context, Equation};
use cas_formatter::LaTeXExpr;

pub(super) fn render_equation_latex(context: &Context, equation: &Equation) -> String {
    format!(
        "{} {} {}",
        LaTeXExpr {
            context,
            id: equation.lhs
        }
        .to_latex(),
        solve_relop_to_latex(&equation.op),
        LaTeXExpr {
            context,
            id: equation.rhs
        }
        .to_latex()
    )
}
