mod rule;

use crate::runtime::Step;
use cas_ast::Context;

pub(super) struct RenderedStepWireLatex {
    pub(super) before_latex: String,
    pub(super) after_latex: String,
    pub(super) rule_latex: String,
}

pub(super) fn render_step_wire_latex(context: &Context, step: &Step) -> RenderedStepWireLatex {
    let (before_latex, after_latex) =
        crate::timeline::simplify_highlights::render_step_wire_global_before_after_latex(
            context, step,
        );
    RenderedStepWireLatex {
        before_latex,
        after_latex,
        rule_latex: rule::render_step_rule_latex(context, step),
    }
}
