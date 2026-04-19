use crate::runtime::Step;
use cas_ast::Context;

pub(super) struct RenderedStepWireExprs {
    pub(super) before: String,
    pub(super) after: String,
}

pub(super) fn render_step_wire_exprs(context: &Context, step: &Step) -> RenderedStepWireExprs {
    let mut temp_ctx = context.clone();
    let snapshots =
        crate::timeline::simplify_highlights::step_wire_presentation_snapshots(&mut temp_ctx, step);

    RenderedStepWireExprs {
        before: render_human_expr(&temp_ctx, snapshots.global_before_expr),
        after: render_human_expr(&temp_ctx, snapshots.global_after_expr),
    }
}

pub(crate) fn render_human_expr(context: &Context, expr: cas_ast::ExprId) -> String {
    let mut temp_ctx = context.clone();
    let normalized =
        cas_solver_core::eval_step_pipeline::normalize_expr_for_display(&mut temp_ctx, expr);
    let latex = cas_formatter::LaTeXExpr {
        context: &temp_ctx,
        id: normalized,
    }
    .to_latex();
    let human = cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(&latex));
    if human.trim().is_empty() {
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &temp_ctx,
                id: normalized
            }
        ))
    } else {
        human
    }
}
