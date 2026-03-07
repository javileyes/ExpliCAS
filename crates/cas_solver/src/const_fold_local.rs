//! Local constant-fold implementation for solver facade.
//!
//! Mirrors engine allowlist folding behavior so solver can expose the same
//! API without routing through `cas_engine::fold_constants`.

mod folder;
mod folds;
mod tree;

use crate::{Budget, CasError, ConstFoldMode, ConstFoldResult, EvalConfig, Metric, Operation};
use cas_ast::{Context, ExprId};

use self::folder::IterativeFolder;

/// Fold constants in an expression using allowlist-only operations.
pub fn fold_constants_local(
    ctx: &mut Context,
    expr: ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    if mode == ConstFoldMode::Off {
        return Ok(ConstFoldResult {
            expr,
            nodes_created: 0,
            folds_performed: 0,
        });
    }

    budget.charge(Operation::ConstFold, Metric::Iterations, 1)?;

    let mut folder = IterativeFolder::new(ctx, cfg);
    let out_expr = folder.fold(expr, budget)?;

    Ok(ConstFoldResult {
        expr: out_expr,
        nodes_created: folder.nodes_created(),
        folds_performed: folder.folds_performed(),
    })
}
