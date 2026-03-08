use cas_ast::Expr;

use super::super::IterativeFolder;

pub(super) fn try_fold_matrix(
    folder: &mut IterativeFolder<'_>,
    id: cas_ast::ExprId,
    rows: usize,
    cols: usize,
    data: &[cas_ast::ExprId],
) -> cas_ast::ExprId {
    let data_folded: Vec<_> = data.iter().map(|d| folder.get_folded(*d)).collect();
    if data_folded != data {
        folder.nodes_created += 1;
        folder.ctx.add(Expr::Matrix {
            rows,
            cols,
            data: data_folded,
        })
    } else {
        id
    }
}
