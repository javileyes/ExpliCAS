mod path;
mod rule;

use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) struct RenderedStepWireLatex {
    pub(super) before_latex: String,
    pub(super) after_latex: String,
    pub(super) rule_latex: String,
}

pub(super) fn render_step_wire_latex(context: &Context, step: &Step) -> RenderedStepWireLatex {
    RenderedStepWireLatex {
        before_latex: path::render_step_before_latex(context, step),
        after_latex: path::render_step_after_latex(context, step),
        rule_latex: rule::render_step_rule_latex(context, step),
    }
}
