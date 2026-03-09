use cas_ast::{Context, Equation, RelOp};
use cas_formatter::{DisplayExpr, LaTeXExpr};

pub(super) struct RenderedEquationStrings {
    pub(super) equation: String,
    pub(super) lhs_latex: String,
    pub(super) rhs_latex: String,
}

pub(super) fn relop_to_latex(op: &RelOp) -> String {
    match op {
        RelOp::Eq => "=".to_string(),
        RelOp::Lt => "<".to_string(),
        RelOp::Leq => r"\leq".to_string(),
        RelOp::Gt => ">".to_string(),
        RelOp::Geq => r"\geq".to_string(),
        RelOp::Neq => r"\neq".to_string(),
    }
}

pub(super) fn render_equation_strings(
    ctx: &Context,
    equation: &Equation,
) -> RenderedEquationStrings {
    let lhs_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: equation.lhs
        }
    );
    let rhs_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: equation.rhs
        }
    );
    let relop_str = format!("{}", equation.op);
    let equation_str = format!("{lhs_str} {relop_str} {rhs_str}");

    let lhs_latex = LaTeXExpr {
        context: ctx,
        id: equation.lhs,
    }
    .to_latex();
    let rhs_latex = LaTeXExpr {
        context: ctx,
        id: equation.rhs,
    }
    .to_latex();

    RenderedEquationStrings {
        equation: equation_str,
        lhs_latex,
        rhs_latex,
    }
}
