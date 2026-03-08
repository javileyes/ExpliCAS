use cas_ast::Expr;

use super::super::IterativeFolder;
use crate::const_fold_local::folds::{fold_neg, is_constant_literal};

pub(super) fn try_fold_neg(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    inner: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let inner_folded = folder.get_folded(inner);
    if is_constant_literal(folder.ctx, inner_folded) {
        if let Some(result) = fold_neg(folder.ctx, inner_folded) {
            folder.nodes_created += 1;
            return result;
        }
    }
    if inner_folded != inner {
        folder.nodes_created += 1;
        folder.ctx.add(Expr::Neg(inner_folded))
    } else {
        id
    }
}

pub(super) fn try_fold_hold(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    inner: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let inner_folded = folder.get_folded(inner);
    if inner_folded != inner {
        folder.nodes_created += 1;
        folder.ctx.add(Expr::Hold(inner_folded))
    } else {
        id
    }
}
