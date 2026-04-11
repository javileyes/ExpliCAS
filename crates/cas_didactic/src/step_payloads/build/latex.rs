mod rule;

use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::LaTeXExpr;

pub(super) struct RenderedStepWireLatex {
    pub(super) before_latex: String,
    pub(super) after_latex: String,
    pub(super) rule_latex: String,
}

pub(super) fn render_step_wire_latex(context: &Context, step: &Step) -> RenderedStepWireLatex {
    let (before_latex, after_latex) = if matches!(
        step.rule_name.as_str(),
        "Product-to-Sum Identity" | "Hyperbolic Product-to-Sum Identity"
    ) {
        let before_expr = step.global_before.unwrap_or(step.before);
        let after_expr = step.global_after.unwrap_or(step.after);
        (
            LaTeXExpr {
                context,
                id: before_expr,
            }
            .to_latex(),
            LaTeXExpr {
                context,
                id: after_expr,
            }
            .to_latex(),
        )
    } else {
        crate::timeline::simplify_highlights::render_step_wire_global_before_after_latex(
            context, step,
        )
    };
    RenderedStepWireLatex {
        before_latex,
        after_latex,
        rule_latex: rule::render_step_rule_latex(context, step),
    }
}
