use cas_ast::Expr;

use super::super::IterativeFolder;
use crate::const_fold_local::folds::{fold_mul_imaginary, fold_pow};

pub(super) fn try_fold_mul(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    a: cas_ast::ExprId,
    b: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let a_folded = folder.get_folded(a);
    let b_folded = folder.get_folded(b);

    if let Some(result) =
        fold_mul_imaginary(folder.ctx, a_folded, b_folded, folder.cfg.value_domain)
    {
        folder.nodes_created += 1;
        return result;
    }

    rebuild_binary_node(folder, id, Expr::Mul, a, b, a_folded, b_folded)
}

pub(super) fn try_fold_pow(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    base: cas_ast::ExprId,
    exp: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let base_folded = folder.get_folded(base);
    let exp_folded = folder.get_folded(exp);

    if let Some(result) = fold_pow(
        folder.ctx,
        base_folded,
        exp_folded,
        folder.cfg.value_domain,
        folder.cfg.branch,
    ) {
        folder.nodes_created += 1;
        return result;
    }

    rebuild_binary_node(folder, id, Expr::Pow, base, exp, base_folded, exp_folded)
}

fn rebuild_binary_node(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    build: fn(cas_ast::ExprId, cas_ast::ExprId) -> Expr,
    left: cas_ast::ExprId,
    right: cas_ast::ExprId,
    left_folded: cas_ast::ExprId,
    right_folded: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if left_folded != left || right_folded != right {
        folder.nodes_created += 1;
        folder.ctx.add(build(left_folded, right_folded))
    } else {
        id
    }
}
