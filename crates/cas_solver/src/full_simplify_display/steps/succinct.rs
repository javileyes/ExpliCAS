use cas_ast::{Context, ExprId};

pub(crate) fn push_succinct_step_line(lines: &mut Vec<String>, ctx: &mut Context, root: ExprId) {
    lines.push(format!(
        "-> {}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: root
        }
    ));
}
