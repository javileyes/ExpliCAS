mod apply;
mod detect;

use cas_ast::{Context, ExprId};

use super::{SubstituteOptions, SubstituteStrategy};

/// Perform power-aware substitution.
pub fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    apply::substitute_power_aware(ctx, root, target, replacement, opts)
}

/// Same as [`substitute_auto`], returning the strategy that was applied.
pub(crate) fn substitute_auto_with_strategy(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> (ExprId, SubstituteStrategy) {
    apply::substitute_auto_with_strategy(ctx, root, target, replacement, opts)
}
