use cas_ast::Expr;

use super::super::IterativeFolder;
use crate::const_fold_local::folds::{fold_sqrt, is_constant_literal};

pub(super) fn try_fold_function(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    name: cas_ast::symbol::SymbolId,
    args: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    let args_folded: Vec<_> = args.iter().map(|a| folder.get_folded(*a)).collect();

    if folder.ctx.is_builtin(name, cas_ast::BuiltinFn::Sqrt)
        && args_folded.len() == 1
        && is_constant_literal(folder.ctx, args_folded[0])
    {
        if let Some(result) = fold_sqrt(folder.ctx, args_folded[0], folder.cfg.value_domain) {
            folder.nodes_created += 1;
            return result;
        }
    }

    if args_folded != args {
        folder.nodes_created += 1;
        folder.ctx.add(Expr::Function(name, args_folded))
    } else {
        id
    }
}
