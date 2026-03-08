use crate::{Budget, CasError, ConstFoldMode, ConstFoldResult, EvalConfig};
use cas_ast::{Context, ExprId};

/// Fold constants under the given semantic config and mode.
pub fn fold_constants(
    ctx: &mut Context,
    expr: ExprId,
    cfg: &EvalConfig,
    mode: ConstFoldMode,
    budget: &mut Budget,
) -> Result<ConstFoldResult, CasError> {
    crate::const_fold_local::fold_constants_local(ctx, expr, cfg, mode, budget)
}
