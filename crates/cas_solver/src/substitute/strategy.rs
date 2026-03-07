use cas_ast::{Context, Expr, ExprId};

use super::{SubstituteOptions, SubstituteStrategy};

/// Detect which substitution strategy should be used for a parsed target.
pub fn detect_substitute_strategy(ctx: &Context, target: ExprId) -> SubstituteStrategy {
    match ctx.get(target) {
        Expr::Variable(_) => SubstituteStrategy::Variable,
        _ => SubstituteStrategy::PowerAware,
    }
}

/// Perform power-aware substitution.
pub fn substitute_power_aware(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> ExprId {
    cas_math::substitute::substitute_power_aware(ctx, root, target, replacement, opts)
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
    match detect_substitute_strategy(ctx, target) {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    }
}

/// Same as [`substitute_auto`], returning the strategy that was applied.
pub fn substitute_auto_with_strategy(
    ctx: &mut Context,
    root: ExprId,
    target: ExprId,
    replacement: ExprId,
    opts: SubstituteOptions,
) -> (ExprId, SubstituteStrategy) {
    let strategy = detect_substitute_strategy(ctx, target);
    let expr = match strategy {
        SubstituteStrategy::Variable => {
            cas_ast::substitute_expr_by_id(ctx, root, target, replacement)
        }
        SubstituteStrategy::PowerAware => {
            substitute_power_aware(ctx, root, target, replacement, opts)
        }
    };
    (expr, strategy)
}
