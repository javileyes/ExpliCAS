use crate::cas_solver::Step;
use cas_ast::{Context, ExprId};
use cas_formatter::DisplayContext;

pub(super) fn build_timeline_display_hints(
    context: &mut Context,
    original_expr: ExprId,
    steps: &[Step],
    simplified_result: Option<ExprId>,
) -> DisplayContext {
    cas_formatter::build_display_context_with_result(
        context,
        original_expr,
        steps,
        simplified_result,
    )
}
