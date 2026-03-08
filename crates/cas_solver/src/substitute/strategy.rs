mod apply;
mod detect;

use cas_ast::{Context, ExprId};

use super::{SubstituteOptions, SubstituteStrategy};

/// Detect which substitution strategy should be used for a parsed target.
pub fn detect_substitute_strategy(ctx: &Context, target: ExprId) -> SubstituteStrategy {
    detect::detect_substitute_strategy(ctx, target)
}

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

/// Perform substitution selecting strategy from target shape.
///
/// - variable target => direct substitute by id
/// - expression target => power-aware substitution
pub fn substitute_auto(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    apply::substitute_auto(ctx, root, target, replacement, opts)
}

/// Same as [`substitute_auto`], returning the strategy that was applied.
pub fn substitute_auto_with_strategy(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> (ExprId, SubstituteStrategy) {
    apply::substitute_auto_with_strategy(ctx, root, target, replacement, opts)
}
