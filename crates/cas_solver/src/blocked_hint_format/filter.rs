use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;

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

    let result_is_zero = matches!(ctx.get(resolved), Expr::Number(value) if value.is_zero());

    hints
        .iter()
        .filter(|hint| {
            !(hint.key.kind() == "defined"
                && (result_is_undefined
                    || (result_is_zero && hint.suggestion.starts_with("cycle detected"))))
        })
        .cloned()
        .collect()
}
