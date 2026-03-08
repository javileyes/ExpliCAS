use cas_ast::{symbol::SymbolId, Context, Expr, ExprId};

use super::reconstruct_global_expr;

pub(super) fn reconstruct_function(
    context: &mut Context,
    root: ExprId,
    name: SymbolId,
    args: Vec<ExprId>,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Arg(idx) => {
            let mut new_args = args;
            if *idx < new_args.len() {
                new_args[*idx] =
                    reconstruct_global_expr(context, new_args[*idx], remaining_path, replacement);
                context.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        _ => root,
    }
}
