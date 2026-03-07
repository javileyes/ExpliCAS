use cas_ast::{Context, ExprId};

/// Filter blocked hints for eval display.
///
/// When the resolved result is `Undefined`, drops `defined` hints because
/// they are often cycle-artifacts and not actionable.
pub fn filter_blocked_hints_for_eval(
    ctx: &Context,
    resolved: ExprId,
    hints: &[crate::BlockedHint],
) -> Vec<crate::BlockedHint> {
    let result_is_undefined = matches!(
        ctx.get(resolved),
        cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
    );

    hints
        .iter()
        .filter(|hint| !(result_is_undefined && hint.key.kind() == "defined"))
        .cloned()
        .collect()
}
