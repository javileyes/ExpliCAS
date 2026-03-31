use crate::runtime::Step;
use cas_ast::Context;

pub(super) struct RenderedStepWireExprs {
    pub(super) before: String,
    pub(super) after: String,
}

pub(super) fn render_step_wire_exprs(context: &Context, step: &Step) -> RenderedStepWireExprs {
    let before_expr = step.global_before.unwrap_or(step.before);
    let after_expr = step.global_after.unwrap_or(step.after);

    RenderedStepWireExprs {
        before: render_human_expr(context, before_expr),
        after: render_human_expr(context, after_expr),
    }
}

pub(crate) fn render_human_expr(context: &Context, expr: cas_ast::ExprId) -> String {
    let latex = cas_formatter::LaTeXExpr { context, id: expr }.to_latex();
    let human = crate::didactic::latex_to_plain_text(&latex);
    if human.trim().is_empty() {
        format!("{}", cas_formatter::DisplayExpr { context, id: expr })
    } else {
        human
    }
}
